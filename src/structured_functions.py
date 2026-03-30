"""Backward-compatibility shim — imports from the new package location."""
from udiagent.structured_functions import *  # noqa: F401, F403
from udiagent.structured_functions import (  # noqa: F401
    validate_structured_text,
    segment_structured_text,
    get_function_signatures,
    export_registry_json,
    resolve_structured_text,
    FUNCTION_REGISTRY,
    _FUNC_REF_PATTERN,
    _ARG_PATTERN,
)
