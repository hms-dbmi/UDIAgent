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


def _format_current_filters(current_filters: list[dict]) -> str:
    """Render the current-filters list into a compact, LLM-friendly block."""
    if not current_filters:
        return "(none — no filters are currently applied)"
    lines = []
    for idx, f in enumerate(current_filters, start=1):
        entity = f.get("entity", "?")
        field = f.get("field", "?")
        filt = f.get("filter", {}) or {}
        ftype = filt.get("filterType", "?")
        if ftype == "interval":
            rng = filt.get("intervalRange", {}) or {}
            detail = f"range [{rng.get('min', '?')}, {rng.get('max', '?')}]"
        elif ftype == "point":
            vals = filt.get("pointValues", []) or []
            detail = f"values {vals}"
        else:
            detail = ""
        title = f.get("title", "").strip()
        title_suffix = f" — {title}" if title else ""
        lines.append(
            f"{idx}. entity={entity}, field={field}, type={ftype}, {detail}{title_suffix}"
        )
    return "\n".join(lines)


def _validate_filter_action(
    tool_args: dict, schema_keys: set, current_filters: list[dict]
) -> list[str]:
    """Validate a FilterData tool call against the schema and current filters."""
    errors: list[str] = []
    action = tool_args.get("action", "add")
    entity = tool_args.get("entity", "")
    field = tool_args.get("field", "")

    if action not in ("add", "modify", "remove"):
        errors.append(f"Unknown action {action!r}; expected 'add', 'modify', or 'remove'.")
        return errors

    if schema_keys and (entity, field) not in schema_keys:
        errors.append(
            f"({entity!r}, {field!r}) is not a valid (entity, field) pair in the schema."
        )

    current_keys = {(f.get("entity", ""), f.get("field", "")) for f in current_filters}
    target_key = (entity, field)

    if action == "add" and target_key in current_keys:
        errors.append(
            f"Filter on ({entity!r}, {field!r}) already exists. "
            f"Use action='modify' to change it instead of 'add'."
        )
    if action in ("modify", "remove") and target_key not in current_keys:
        errors.append(
            f"No existing filter on ({entity!r}, {field!r}); cannot {action}. "
            f"Use action='add' for a new filter."
        )

    if action in ("add", "modify"):
        ftype = tool_args.get("filterType")
        if ftype not in ("point", "interval"):
            errors.append(
                f"filterType must be 'point' or 'interval' for action={action!r}; got {ftype!r}."
            )
        elif ftype == "point" and not tool_args.get("pointValues"):
            errors.append(
                f"action={action!r} with filterType='point' requires non-empty pointValues."
            )
        elif ftype == "interval" and not tool_args.get("intervalRange"):
            errors.append(
                f"action={action!r} with filterType='interval' requires intervalRange."
            )

    return errors


def _build_filter_payload(tool_args: dict, validation_retries: int) -> dict:
    """Shape the outgoing FilterData tool-call arguments for the frontend."""
    action = tool_args.get("action", "add")
    payload = {
        "action": action,
        "title": tool_args.get("title", ""),
        "entity": tool_args.get("entity", ""),
        "field": tool_args.get("field", ""),
        "validation_retries": validation_retries,
    }
    if action == "remove":
        return payload
    payload["filter"] = {
        "filterType": tool_args.get("filterType", "point"),
        "intervalRange": tool_args.get("intervalRange", {"min": 0, "max": 0}),
        "pointValues": tool_args.get("pointValues", [""]),
    }
    return payload


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
        current_filters: list[dict] | None = None,
        openai_api_key: str | None = None,
    ) -> OrchestratorResult:
        """Run the orchestrator on a user request.

        Args:
            messages: Chat history (OpenAI message format).
            data_schema: JSON string describing dataset entities and fields.
            data_domains: JSON string (or list) describing field domains.
            current_filters: The currently-applied filters (as a list of dicts
                with at least ``entity``, ``field``, and ``filter``). Passed
                explicitly into the orchestrate prompt so the LLM can decide
                between add/modify/remove actions without inferring state
                from message history.
            openai_api_key: Optional per-request OpenAI key override.

        Returns:
            An ``OrchestratorResult`` with tool_calls and orchestrator_choice.
        """
        msgs = split_tool_calls(messages)
        tool_calls, choice = self._orchestrate_tool_calls(
            msgs,
            data_schema,
            data_domains,
            current_filters=current_filters or [],
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
        try:
            schema_raw = (
                json.loads(data_schema) if isinstance(data_schema, str) else data_schema
            )
        except (json.JSONDecodeError, TypeError):
            schema_raw = {}

        field_meta = {}
        for resource in schema_raw.get("resources", []):
            entity_name = resource.get("name", "")
            for field_def in resource.get("schema", {}).get("fields", []):
                fname = field_def.get("name", "")
                field_meta[(entity_name, fname)] = {
                    "data_type": field_def.get("udi:data_type", "unknown"),
                    "description": field_def.get("description", "").strip(),
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
        self,
        tool_args,
        messages,
        data_schema,
        data_domains,
        current_filters=None,
        openai_api_key=None,
    ):
        current_filters = current_filters or []
        validation_retries = 0

        try:
            schema_raw = (
                json.loads(data_schema) if isinstance(data_schema, str) else data_schema
            )
        except (json.JSONDecodeError, TypeError):
            schema_raw = {}
        schema_keys = set(_build_schema_field_meta(schema_raw).keys())

        errors = _validate_filter_action(tool_args, schema_keys, current_filters)
        if errors:
            retried = self._retry_filter_data(
                prev_tool_args=tool_args,
                errors=errors,
                messages=messages,
                data_domains=data_domains,
                current_filters=current_filters,
                openai_api_key=openai_api_key,
            )
            validation_retries = 1
            if retried is not None:
                tool_args = retried

        return {
            "name": "FilterData",
            "arguments": _build_filter_payload(tool_args, validation_retries),
        }

    def _retry_filter_data(
        self,
        prev_tool_args,
        errors,
        messages,
        data_domains,
        current_filters,
        openai_api_key,
    ):
        """Re-invoke the LLM constrained to FilterData with error feedback."""
        orchestrate_skill = self.skills.get("orchestrate")
        if orchestrate_skill is None:
            return None
        filter_tool = next(
            (t for t in self.tools if t["function"]["name"] == "FilterData"),
            None,
        )
        if filter_tool is None:
            return None

        rendered = render_template(
            orchestrate_skill.instructions,
            {
                "data_domains": simplify_data_domains(data_domains),
                "current_filters": _format_current_filters(current_filters),
            },
        )
        retry_msgs = normalize_tool_calls(copy.deepcopy(messages))
        retry_msgs.insert(0, {"role": "system", "content": rendered})
        hint = "The previous tool call had errors:\n" + "\n".join(
            f"- {e}" for e in errors
        )
        retry_msgs.append(
            {
                "role": "assistant",
                "content": f"Tool call: FilterData({json.dumps(prev_tool_args)})",
            }
        )
        retry_msgs.append(
            {
                "role": "user",
                "content": hint
                + "\n\nReturn a corrected FilterData call. Pick 'add', 'modify', or 'remove' according to the Current Filters list.",
            }
        )

        gpt_client = self.agent._get_gpt_client(openai_api_key)
        try:
            resp = gpt_client.chat.completions.create(
                model=self.agent.gpt_model_name,
                messages=retry_msgs,
                tools=[filter_tool],
                tool_choice={
                    "type": "function",
                    "function": {"name": "FilterData"},
                },
                temperature=0.0,
                max_completion_tokens=1024,
            )
            tc = resp.choices[0].message.tool_calls[0]
            return json.loads(tc.function.arguments)
        except Exception as exc:
            logger.warning("FilterData retry failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Core orchestration logic
    # ------------------------------------------------------------------

    def _orchestrate_tool_calls(
        self,
        messages,
        data_schema,
        data_domains,
        current_filters=None,
        openai_api_key=None,
    ):
        msgs = normalize_tool_calls(copy.deepcopy(messages))

        orchestrate_skill = self.skills["orchestrate"]
        rendered = render_template(
            orchestrate_skill.instructions,
            {
                "data_domains": simplify_data_domains(data_domains),
                "current_filters": _format_current_filters(current_filters or []),
            },
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

            handler_kwargs = {"openai_api_key": openai_api_key}
            if tool_name == "FilterData":
                handler_kwargs["current_filters"] = current_filters or []

            result = handler(
                tool_args,
                messages,
                data_schema,
                data_domains,
                **handler_kwargs,
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
