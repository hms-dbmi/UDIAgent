"""Backward-compatibility shim — imports from the new package location."""
from udiagent.vis_generate import *  # noqa: F401, F403
from udiagent.vis_generate import (  # noqa: F401
    generate_vis_spec,
    _render_template,
    _load_examples,
    run_skills,
    _call_llm,
    _call_llm_with_tools,
    _parse_and_validate,
    instantiate_template,
    validate_bindings,
    _load_generated_tools,
)
from udiagent.grammar import load_grammar, load_skills  # noqa: F401
from udiagent.schema import simplify_data_domains, simplify_data_schema  # noqa: F401
