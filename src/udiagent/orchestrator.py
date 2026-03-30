"""Orchestrator — routes user requests to the appropriate tool handlers."""

import copy
import json
import logging
from dataclasses import dataclass, field

from udiagent.grammar import Skill, load_skills, load_grammar
from udiagent.messages import normalize_tool_calls, strip_tool_calls, split_tool_calls
from udiagent.schema import parse_schema_from_dict, simplify_data_domains
from udiagent.structured_functions import (
    validate_structured_text,
    segment_structured_text,
    get_function_signatures,
)
from udiagent.tools import (
    ORCHESTRATOR_TOOLS,
    function_call_filter,
    function_call_render_visualization_legacy,
    function_call_render_visualization_pipeline,
)
from udiagent.vis_generate import _render_template

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Result of an orchestration run."""

    tool_calls: list[dict] = field(default_factory=list)
    orchestrator_choice: str = "render-visualization"


class Orchestrator:
    """Routes user requests to visualization, filter, explanation, and other tool handlers.

    Args:
        agent: A ``UDIAgent`` instance.
        skills: Skill registry.  Loaded from bundled data if ``None``.
        tools: Tool definitions list.  Defaults to ``ORCHESTRATOR_TOOLS``.
        use_vis_pipeline: Use the multi-stage pipeline for visualization generation.
        grammar: Preloaded grammar dict.  Auto-loaded if ``None`` and *use_vis_pipeline* is set.
    """

    def __init__(
        self,
        agent,
        skills: dict[str, Skill] | None = None,
        tools: list[dict] | None = None,
        use_vis_pipeline: bool = False,
        grammar: dict | None = None,
    ):
        self.agent = agent
        self.skills = skills if skills is not None else load_skills()
        self.tools = tools if tools is not None else ORCHESTRATOR_TOOLS
        self.use_vis_pipeline = use_vis_pipeline
        if use_vis_pipeline and grammar is None:
            self.grammar = load_grammar("udi")
        else:
            self.grammar = grammar

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        messages: list[dict],
        data_schema: str,
        data_domains: str,
        openai_api_key: str | None = None,
    ) -> OrchestratorResult:
        """Run the orchestrator on a user request.

        Args:
            messages: Chat history (OpenAI message format).
            data_schema: JSON string describing dataset entities and fields.
            data_domains: JSON string (or list) describing field domains.
            openai_api_key: Optional per-request OpenAI key override.

        Returns:
            An ``OrchestratorResult`` with tool_calls and orchestrator_choice.
        """
        msgs = split_tool_calls(messages)
        tool_calls, choice = self._orchestrate_tool_calls(
            msgs, data_schema, data_domains,
            use_pipeline=self.use_vis_pipeline,
            openai_api_key=openai_api_key,
        )
        return OrchestratorResult(tool_calls=tool_calls, orchestrator_choice=choice)

    def run_legacy(
        self,
        messages: list[dict],
        data_schema: str,
        data_domains: str,
        calls_to_make: str,
        use_pipeline: bool | None = None,
        openai_api_key: str | None = None,
    ) -> list[dict]:
        """Run legacy if/else orchestration for backward-compatible benchmark overrides."""
        if use_pipeline is None:
            use_pipeline = self.use_vis_pipeline
        msgs = split_tool_calls(messages)
        return self._run_legacy_orchestration(
            msgs, data_schema, data_domains, calls_to_make, use_pipeline,
            openai_api_key=openai_api_key,
        )

    # ------------------------------------------------------------------
    # Tool dispatch handlers
    # ------------------------------------------------------------------

    def _handle_create_visualization(
        self, tool_args, messages, data_schema, data_domains, use_pipeline, openai_api_key=None,
    ):
        description = tool_args.get("description", "")
        if description:
            focused_messages = [
                msg for msg in messages if msg.get("role") != "user"
            ] + [{"role": "user", "content": description}]
        else:
            focused_messages = list(messages)

        if use_pipeline:
            result = function_call_render_visualization_pipeline(
                self.agent, focused_messages, data_schema, self.grammar,
                openai_api_key=openai_api_key,
            )
        else:
            result = function_call_render_visualization_legacy(
                self.agent, focused_messages, data_schema,
                openai_api_key=openai_api_key,
            )

        title = tool_args.get("title", "")
        if title:
            result["arguments"]["title"] = title

        return result

    def _handle_rebuff(
        self, tool_args, messages, data_schema, data_domains, use_pipeline, openai_api_key=None,
    ):
        available_capabilities = [
            f"{t['function']['name']}: {t['function']['description']}"
            for t in self.tools
            if t["function"]["name"] != "Rebuff"
        ]

        rebuff_skill = self.skills.get("rebuff")
        if rebuff_skill:
            rendered = _render_template(
                rebuff_skill.instructions,
                {
                    "user_request": tool_args.get("user_request", ""),
                    "reason": tool_args.get("reason", ""),
                    "available_tools": "\n".join(
                        f"- {cap}" for cap in available_capabilities
                    ),
                },
            )
            gpt_client = self.agent._get_gpt_client(openai_api_key)
            msgs = copy.deepcopy(messages)
            strip_tool_calls(msgs)
            msgs.insert(0, {"role": "system", "content": rendered})
            resp = gpt_client.chat.completions.create(
                model=self.agent.gpt_model_name,
                messages=msgs,
                temperature=0.0,
                max_completion_tokens=1024,
            )
            try:
                response_data = json.loads(resp.choices[0].message.content)
            except (json.JSONDecodeError, IndexError):
                response_data = {
                    "message": f"Sorry, I cannot fulfill this request. {tool_args.get('reason', '')}",
                    "suggestions": [],
                }
        else:
            response_data = {
                "message": f"Sorry, I cannot fulfill this request. {tool_args.get('reason', '')}",
                "suggestions": [],
            }

        return {
            "name": "Rebuff",
            "arguments": response_data,
        }

    def _handle_free_text_explain(
        self, tool_args, messages, data_schema, data_domains, use_pipeline, openai_api_key=None,
    ):
        available_tools = "\n".join(
            f"- {t['function']['name']}: {t['function']['description']}"
            for t in self.tools
            if t["function"]["name"] not in ("Rebuff", "FreeTextExplain")
        )

        data_schema_simple = simplify_data_domains(data_domains)

        explain_skill = self.skills.get("free_text_explain")
        if explain_skill:
            rendered = _render_template(
                explain_skill.instructions,
                {
                    "user_request": tool_args.get("user_request", ""),
                    "response_type": tool_args.get("response_type", "general"),
                    "available_tools": available_tools,
                    "data_schema": data_schema_simple,
                    "structured_functions": get_function_signatures(),
                },
            )
            gpt_client = self.agent._get_gpt_client(openai_api_key)
            msgs = copy.deepcopy(messages)
            strip_tool_calls(msgs)
            msgs.insert(0, {"role": "system", "content": rendered})
            resp = gpt_client.chat.completions.create(
                model=self.agent.gpt_model_name,
                messages=msgs,
                temperature=0.0,
                max_completion_tokens=1024,
            )
            text_response = resp.choices[0].message.content
        else:
            text_response = "I can help you explore and visualize data. Try asking for a specific chart or data summary."

        # Validate structured function references
        validation_errors = validate_structured_text(text_response)

        if not validation_errors:
            try:
                schema_dict = (
                    json.loads(data_schema)
                    if isinstance(data_schema, str)
                    else data_schema
                )
                schema_parsed = parse_schema_from_dict(schema_dict)
                text_segments, has_structured = segment_structured_text(
                    text_response, schema_parsed
                )
            except Exception:
                text_segments = [text_response]
                has_structured = False
        else:
            text_segments = [text_response]
            has_structured = False

        return {
            "name": "FreeTextExplain",
            "arguments": {
                "response_type": tool_args.get("response_type", "general"),
                "text": text_segments,
                "has_structured_elements": has_structured,
            },
        }

    def _handle_clarify_variable(
        self, tool_args, messages, data_schema, data_domains, use_pipeline, openai_api_key=None,
    ):
        try:
            schema_raw = (
                json.loads(data_schema)
                if isinstance(data_schema, str)
                else data_schema
            )
        except (json.JSONDecodeError, TypeError):
            schema_raw = {}

        field_meta = {}
        for resource in schema_raw.get("resources", []):
            entity_name = resource.get("name", "")
            for field in resource.get("schema", {}).get("fields", []):
                fname = field.get("name", "")
                field_meta[(entity_name, fname)] = {
                    "data_type": field.get("udi:data_type", "unknown"),
                    "description": field.get("description", "").strip(),
                }

        ambiguous_variables = tool_args.get("ambiguous_variables", [])
        for var in ambiguous_variables:
            for candidate in var.get("candidates", []):
                key = (candidate.get("entity", ""), candidate.get("field_name", ""))
                meta = field_meta.get(key, {})
                candidate["data_type"] = meta.get("data_type", "unknown")
                candidate["description"] = meta.get("description", "")

        return {
            "name": "ClarifyVariable",
            "arguments": {
                "message": tool_args.get("message", ""),
                "ambiguous_variables": ambiguous_variables,
            },
        }

    def _handle_filter_data(
        self, tool_args, messages, data_schema, data_domains, use_pipeline, openai_api_key=None,
    ):
        filter_obj = {
            "filterType": tool_args["filterType"],
            "intervalRange": tool_args.get("intervalRange", {"min": 0, "max": 0}),
            "pointValues": tool_args.get("pointValues", [""]),
        }
        return {
            "name": "FilterData",
            "arguments": {
                "title": tool_args.get("title", ""),
                "entity": tool_args["entity"],
                "field": tool_args["field"],
                "filter": filter_obj,
            },
        }

    # ------------------------------------------------------------------
    # Core orchestration logic
    # ------------------------------------------------------------------

    def _orchestrate_tool_calls(
        self, messages, data_schema, data_domains, use_pipeline=False, openai_api_key=None,
    ):
        msgs = normalize_tool_calls(copy.deepcopy(messages))

        orchestrate_skill = self.skills["orchestrate"]
        rendered = _render_template(
            orchestrate_skill.instructions,
            {"data_domains": simplify_data_domains(data_domains)},
        )
        msgs.insert(0, {"role": "system", "content": rendered})

        gpt_client = self.agent._get_gpt_client(openai_api_key)
        resp = gpt_client.chat.completions.create(
            model=self.agent.gpt_model_name,
            messages=msgs,
            tools=self.tools,
            tool_choice="required",
            temperature=0.0,
            max_completion_tokens=1024,
        )

        choice = resp.choices[0]
        if not choice.message.tool_calls:
            return [], "render-visualization"

        tool_dispatch = {
            "Rebuff": self._handle_rebuff,
            "ClarifyVariable": self._handle_clarify_variable,
            "FreeTextExplain": self._handle_free_text_explain,
            "CreateVisualization": self._handle_create_visualization,
            "FilterData": self._handle_filter_data,
        }

        tool_calls = []
        has_vis = False
        has_filter = False
        has_rebuff = False
        has_clarify = False
        has_explain = False

        for tc in choice.message.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)

            handler = tool_dispatch.get(tool_name)
            if handler is None:
                logger.warning("Unknown tool: %s, skipping", tool_name)
                continue

            result = handler(
                tool_args, messages, data_schema, data_domains, use_pipeline,
                openai_api_key=openai_api_key,
            )
            tool_calls.append(result)

            if tool_name == "CreateVisualization":
                has_vis = True
            elif tool_name == "FilterData":
                has_filter = True
            elif tool_name == "Rebuff":
                has_rebuff = True
            elif tool_name == "ClarifyVariable":
                has_clarify = True
            elif tool_name == "FreeTextExplain":
                has_explain = True

        # Derive orchestrator_choice for backward compatibility
        if has_explain:
            orchestrator_choice = "explain"
        elif has_clarify:
            orchestrator_choice = "clarify-variable"
        elif has_rebuff:
            orchestrator_choice = "rebuff"
        elif has_vis and has_filter:
            orchestrator_choice = "both"
        elif has_filter:
            orchestrator_choice = "get-subset-of-data"
        else:
            orchestrator_choice = "render-visualization"

        return tool_calls, orchestrator_choice

    def _run_legacy_orchestration(
        self, messages, data_schema, data_domains, calls_to_make, use_pipeline,
        openai_api_key=None,
    ):
        tool_calls = []
        if calls_to_make in ("both", "get-subset-of-data"):
            tool_calls.extend(
                function_call_filter(
                    self.agent, messages, data_domains,
                    openai_api_key=openai_api_key,
                )
            )
        if calls_to_make in ("both", "render-visualization"):
            if use_pipeline:
                tool_calls.append(
                    function_call_render_visualization_pipeline(
                        self.agent, messages, data_schema, self.grammar,
                        openai_api_key=openai_api_key,
                    )
                )
            else:
                tool_calls.append(
                    function_call_render_visualization_legacy(
                        self.agent, messages, data_schema,
                        openai_api_key=openai_api_key,
                    )
                )
        return tool_calls
