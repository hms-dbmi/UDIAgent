"""UDIAgent — LLM-powered data visualization orchestration library."""

from udiagent.agent import UDIAgent
from udiagent.orchestrator import Orchestrator, OrchestratorResult
from udiagent.grammar import Skill, load_grammar, load_skills
from udiagent.vis_generate import generate_vis_spec, _render_template
from udiagent.vis_pipeline import run_vis_pipeline
from udiagent.schema import parse_schema_from_dict, simplify_data_domains, simplify_data_schema
from udiagent.messages import split_tool_calls, normalize_tool_calls, strip_tool_calls
from udiagent.structured_functions import (
    validate_structured_text,
    segment_structured_text,
    get_function_signatures,
    export_registry_json,
)
from udiagent.tools import ORCHESTRATOR_TOOLS

__all__ = [
    "UDIAgent",
    "Orchestrator",
    "OrchestratorResult",
    "Skill",
    "load_grammar",
    "load_skills",
    "generate_vis_spec",
    "run_vis_pipeline",
    "parse_schema_from_dict",
    "simplify_data_domains",
    "simplify_data_schema",
    "split_tool_calls",
    "normalize_tool_calls",
    "strip_tool_calls",
    "validate_structured_text",
    "segment_structured_text",
    "get_function_signatures",
    "export_registry_json",
    "ORCHESTRATOR_TOOLS",
    "_render_template",
]
