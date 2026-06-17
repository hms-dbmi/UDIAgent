"""Tests that visualization validation uses the request-provided schema,
not a baked-in one from ``generated_vis_tools``.
"""

import json

from udiagent.schema import parse_schema_from_dict
from udiagent.vis_generate import (
    _load_generated_tools,
    _request_schema_for_vis,
    validate_bindings,
)


PENGUIN_SCHEMA = {
    "udi:path": "./",
    "resources": [
        {
            "name": "penguins",
            "path": "penguins.csv",
            "udi:row_count": 344,
            "schema": {
                "fields": [
                    {
                        "name": "species",
                        "udi:data_type": "nominal",
                        "udi:cardinality": 3,
                    },
                    {
                        "name": "body_mass_g",
                        "udi:data_type": "quantitative",
                        "udi:cardinality": 94,
                    },
                ]
            },
        }
    ],
}


def test_parse_schema_emits_url_and_relationships_keys():
    parsed = parse_schema_from_dict(PENGUIN_SCHEMA)
    assert "entities" in parsed and "relationships" in parsed
    assert parsed["entities"]["penguins"]["url"] == "./penguins.csv"
    assert parsed["relationships"] == []


def test_request_schema_helper_accepts_string():
    parsed = _request_schema_for_vis(json.dumps(PENGUIN_SCHEMA))
    assert "penguins" in parsed["entities"]


def test_generated_tools_no_longer_export_schema():
    """SCHEMA was the only dataset-specific export; it must be gone."""
    import udiagent.generated_vis_tools as g

    assert hasattr(g, "TOOL_DEFS")
    assert hasattr(g, "TOOL_DISPATCH")
    assert hasattr(g, "TEMPLATES")
    assert not hasattr(g, "SCHEMA")


def test_load_generated_tools_returns_three_tuple():
    result = _load_generated_tools()
    assert result is not None
    assert len(result) == 3


def test_validate_bindings_accepts_unknown_dataset():
    """A previously-unknown dataset should validate fine once its schema is
    parsed at request time — this is what the refactor enables."""
    parsed = parse_schema_from_dict(PENGUIN_SCHEMA)
    spec_template = '{"source": {"name": "<E>"}, "representation": {"mapping": [{"encoding": "x", "field": "<F:n>"}]}}'
    bindings = {"E": "penguins", "F": "species"}
    errors = validate_bindings(spec_template, bindings, parsed)
    assert errors == []


def test_validate_bindings_rejects_unknown_entity_in_request_schema():
    parsed = parse_schema_from_dict(PENGUIN_SCHEMA)
    spec_template = '{"source": {"name": "<E>"}, "representation": {"mapping": [{"encoding": "x", "field": "<F:n>"}]}}'
    bindings = {"E": "donors", "F": "species"}
    errors = validate_bindings(spec_template, bindings, parsed)
    assert errors
    assert "donors" in errors[0]
