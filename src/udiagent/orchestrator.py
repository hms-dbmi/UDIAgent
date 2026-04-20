"""Orchestrator — routes user requests to the appropriate tool handlers."""

import copy
import json
import logging
from dataclasses import dataclass, field

from udiagent.skills import Skill, load_skills, render_template
from udiagent.grammar import load_grammar
from udiagent.messages import normalize_tool_calls, split_tool_calls
from udiagent.schema import parse_schema_from_dict, simplify_data_domains
from udiagent.structured_functions import (
    validate_structured_text,
    segment_structured_text,
    get_function_signatures,
)
from udiagent.tools import (
    ORCHESTRATOR_TOOLS,
    function_call_render_visualization,
)

logger = logging.getLogger(__name__)


def _build_schema_field_meta(schema_raw: dict) -> dict:
    """Map (entity, field_name) → {data_type, description} from a raw schema dict."""
    meta: dict = {}
    for resource in schema_raw.get("resources", []):
        entity_name = resource.get("name", "")
        for field_def in resource.get("schema", {}).get("fields", []):
            fname = field_def.get("name", "")
            meta[(entity_name, fname)] = {
                "data_type": field_def.get("udi:data_type", "unknown"),
                "description": field_def.get("description", "").strip(),
            }
    return meta


def _validate_variable_candidates(
    ambiguous_variables: list, valid_keys: set
) -> list[str]:
    """Return a list of human-readable errors for candidates that don't match the schema."""
    errors: list[str] = []
    for var in ambiguous_variables:
        query_term = var.get("query_term", "?")
        for candidate in var.get("candidates", []):
            entity = candidate.get("entity", "")
            field = candidate.get("field_name", "")
            if (entity, field) not in valid_keys:
                errors.append(
                    f"Candidate for '{query_term}': (entity={entity!r}, field_name={field!r}) "
                    f"is not in the schema. Use the bare field name only."
                )
    return errors


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
        grammar: Preloaded grammar dict.  Auto-loaded if ``None``.
    """

    def __init__(
        self,
        agent,
        skills: dict[str, Skill] | None = None,
        tools: list[dict] | None = None,
        grammar: dict | None = None,
    ):
        self.agent = agent
        self.skills = skills if skills is not None else load_skills()
        self.tools = tools if tools is not None else ORCHESTRATOR_TOOLS
        self.grammar = grammar if grammar is not None else load_grammar("udi")

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
            msgs,
            data_schema,
            data_domains,
            openai_api_key=openai_api_key,
        )
        return OrchestratorResult(tool_calls=tool_calls, orchestrator_choice=choice)

    # ------------------------------------------------------------------
    # Tool dispatch handlers
    # ------------------------------------------------------------------

    def _handle_create_visualization(
        self,
        tool_args,
        messages,
        data_schema,
        data_domains,
        openai_api_key=None,
    ):
        description = tool_args.get("description", "")
        if description:
            focused_messages = [
                msg for msg in messages if msg.get("role") != "user"
            ] + [{"role": "user", "content": description}]
        else:
            focused_messages = list(messages)

        result = function_call_render_visualization(
            self.agent,
            focused_messages,
            data_schema,
            self.grammar,
            openai_api_key=openai_api_key,
        )

        title = tool_args.get("title", "")
        if title:
            result["arguments"]["title"] = title

        return result

    def _handle_rebuff(
        self,
        tool_args,
        messages,
        data_schema,
        data_domains,
        openai_api_key=None,
    ):
        available_capabilities = [
            f"{t['function']['name']}: {t['function']['description']}"
            for t in self.tools
            if t["function"]["name"] != "Rebuff"
        ]

        rebuff_skill = self.skills.get("rebuff")
        if rebuff_skill:
            rendered = render_template(
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
            msgs = normalize_tool_calls(copy.deepcopy(messages))
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
        self,
        tool_args,
        messages,
        data_schema,
        data_domains,
        openai_api_key=None,
    ):
        available_tools = "\n".join(
            f"- {t['function']['name']}: {t['function']['description']}"
            for t in self.tools
            if t["function"]["name"] not in ("Rebuff", "FreeTextExplain")
        )

        data_schema_simple = simplify_data_domains(data_domains)

        explain_skill = self.skills.get("free_text_explain")
        if explain_skill:
            rendered = render_template(
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
            msgs = normalize_tool_calls(copy.deepcopy(messages))
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
        self,
        tool_args,
        messages,
        data_schema,
        data_domains,
        openai_api_key=None,
    ):
        clarification_type = tool_args.get("clarification_type", "variable")

        if clarification_type != "variable":
            return {
                "name": "ClarifyVariable",
                "arguments": {
                    "clarification_type": clarification_type,
                    "message": tool_args.get("message", ""),
                    "ambiguous_variables": tool_args.get("ambiguous_variables", []),
                    "validation_retries": 0,
                },
            }

        try:
            schema_raw = (
                json.loads(data_schema) if isinstance(data_schema, str) else data_schema
            )
        except (json.JSONDecodeError, TypeError):
            schema_raw = {}

        field_meta = _build_schema_field_meta(schema_raw)
        valid_keys = set(field_meta.keys())

        validation_retries = 0
        ambiguous_variables = tool_args.get("ambiguous_variables", [])
        message = tool_args.get("message", "")

        errors = _validate_variable_candidates(ambiguous_variables, valid_keys)
        if errors:
            retried = self._retry_clarify_variable(
                prev_tool_args={
                    "clarification_type": "variable",
                    "message": message,
                    "ambiguous_variables": ambiguous_variables,
                },
                errors=errors,
                messages=messages,
                data_domains=data_domains,
                openai_api_key=openai_api_key,
            )
            validation_retries = 1
            if retried is not None:
                ambiguous_variables = retried.get(
                    "ambiguous_variables", ambiguous_variables
                )
                message = retried.get("message", message)

        for var in ambiguous_variables:
            for candidate in var.get("candidates", []):
                key = (candidate.get("entity", ""), candidate.get("field_name", ""))
                meta = field_meta.get(key, {})
                candidate["data_type"] = meta.get("data_type", "unknown")
                candidate["description"] = meta.get("description", "")

        return {
            "name": "ClarifyVariable",
            "arguments": {
                "clarification_type": "variable",
                "message": message,
                "ambiguous_variables": ambiguous_variables,
                "validation_retries": validation_retries,
            },
        }

    def _retry_clarify_variable(
        self,
        prev_tool_args,
        errors,
        messages,
        data_domains,
        openai_api_key,
    ):
        """Re-invoke the LLM constrained to ClarifyVariable with error feedback.

        Returns the corrected tool arguments dict, or None on failure.
        """
        orchestrate_skill = self.skills.get("orchestrate")
        if orchestrate_skill is None:
            return None

        clarify_tool = next(
            (t for t in self.tools if t["function"]["name"] == "ClarifyVariable"),
            None,
        )
        if clarify_tool is None:
            return None

        rendered = render_template(
            orchestrate_skill.instructions,
            {"data_domains": simplify_data_domains(data_domains)},
        )
        retry_msgs = normalize_tool_calls(copy.deepcopy(messages))
        retry_msgs.insert(0, {"role": "system", "content": rendered})
        hint = "The previous tool call had errors:\n" + "\n".join(
            f"- {e}" for e in errors
        )
        retry_msgs.append(
            {
                "role": "assistant",
                "content": f"Tool call: ClarifyVariable({json.dumps(prev_tool_args)})",
            }
        )
        retry_msgs.append(
            {
                "role": "user",
                "content": hint
                + "\n\nReturn a corrected ClarifyVariable call. Use bare field names from the schema, never qualified names like 'entity.field'.",
            }
        )

        gpt_client = self.agent._get_gpt_client(openai_api_key)
        try:
            resp = gpt_client.chat.completions.create(
                model=self.agent.gpt_model_name,
                messages=retry_msgs,
                tools=[clarify_tool],
                tool_choice={
                    "type": "function",
                    "function": {"name": "ClarifyVariable"},
                },
                temperature=0.0,
                max_completion_tokens=1024,
            )
            tc = resp.choices[0].message.tool_calls[0]
            return json.loads(tc.function.arguments)
        except Exception as exc:
            logger.warning("ClarifyVariable retry failed: %s", exc)
            return None

    def _handle_filter_data(
        self,
        tool_args,
        messages,
        data_schema,
        data_domains,
        openai_api_key=None,
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
        self,
        messages,
        data_schema,
        data_domains,
        openai_api_key=None,
    ):
        msgs = normalize_tool_calls(copy.deepcopy(messages))

        orchestrate_skill = self.skills["orchestrate"]
        rendered = render_template(
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
                tool_args,
                messages,
                data_schema,
                data_domains,
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
