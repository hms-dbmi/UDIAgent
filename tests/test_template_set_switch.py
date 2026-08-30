"""Tests for the hard-coded template-set switch and the data-cube template set.

These tests exercise template loading, the switch, and template instantiation
without calling the OpenAI API.
"""

import json
from pathlib import Path

import jsonschema
import pytest

from udiagent import vis_generate as vg
from udiagent.skills import _package_data_path

# Chart types that require per-record data and must never appear in a cube set.
EXCLUDED_CHART_TYPES = {
    "scatterplot",
    "grouped_scatter",
    "dot",
    "grouped_dot",
    "histogram",
    "area",  # KDE / density
    "grouped_area",  # grouped KDE / density
    "grouped_line",  # grouped CDF
}


@pytest.fixture
def use_cube(monkeypatch):
    """Point the pipeline at the cube template set for the duration of a test."""
    monkeypatch.setattr(vg, "ACTIVE_TEMPLATE_SET", "cube")


def _cube_templates():
    path = _package_data_path() / "skills" / "template_visualizations_cube.json"
    return json.loads(Path(path).read_text())


class TestSwitch:
    def test_default_is_line_item(self):
        assert vg.ACTIVE_TEMPLATE_SET == "line_item"

    def test_unknown_set_raises(self, monkeypatch):
        monkeypatch.setattr(vg, "ACTIVE_TEMPLATE_SET", "nope")
        with pytest.raises(ValueError):
            vg._active_template_set()

    def test_examples_path_follows_switch(self, use_cube):
        assert vg._default_examples_path().endswith("template_visualizations_cube.json")

    def test_generated_tools_follow_switch(self, use_cube):
        loaded = vg._load_generated_tools()
        assert loaded is not None
        tool_defs, tool_dispatch, templates, schema = loaded
        assert "encounter_counts" in schema["entities"]
        assert len(tool_defs) == len(templates) == 25

    def test_line_item_tools_still_load(self):
        loaded = vg._load_generated_tools()
        assert loaded is not None
        tool_defs, _, templates, _ = loaded
        assert len(tool_defs) == len(templates) > 0

    def test_cube_few_shot_examples_load(self, use_cube):
        examples = vg._load_examples(None)
        assert examples.strip()
        assert "marginal" in examples.lower()


class TestCubeTemplateSet:
    def test_no_excluded_chart_types(self):
        chart_types = {t["chart_type"] for t in _cube_templates()}
        assert not (chart_types & EXCLUDED_CHART_TYPES)

    def test_all_specs_validate_against_grammar(self):
        grammar = json.loads(
            (_package_data_path() / "UDIGrammarSchema.json").read_text()
        )
        for t in _cube_templates():
            spec = json.loads(t["spec_template"])
            jsonschema.validate(instance=spec, schema=grammar)

    def test_every_template_filters_all_dimensions(self):
        # A cube read must be a marginal: every dimension appears in the filter,
        # exactly one clause per dimension (present-or-empty), never a re-count.
        schema = json.loads(
            (Path(__file__).resolve().parent.parent
             / "data" / "data_domains" / "encounter_cube_schema.json").read_text()
        )
        resource = schema["resources"][0]
        dims = resource["udi:dimensions"]
        for t in _cube_templates():
            spec = json.loads(t["spec_template"])
            transforms = spec.get("transformation", [])
            filters = [tr["filter"] for tr in transforms if "filter" in tr]
            assert filters, f"{t['chart_type']} template has no marginal filter"
            filter_expr = filters[0]
            for d in dims:
                assert f"d['{d}']" in filter_expr, (
                    f"dimension {d} missing from filter of {t['chart_type']}"
                )

    def test_no_count_rollup_over_raw_rows(self):
        # The cube is pre-aggregated: count() must never be used. Sums of the
        # measure (after a marginal filter) are allowed.
        for t in _cube_templates():
            assert '"op": "count"' not in t["spec_template"], (
                f"{t['chart_type']} re-counts raw rows"
            )


class TestCubeInstantiation:
    def test_templates_instantiate_and_validate_bindings(self, use_cube):
        tool_defs, tool_dispatch, templates, schema = vg._load_generated_tools()
        for tool_def in tool_defs:
            name = tool_def["function"]["name"]
            template_idx, param_map = tool_dispatch[name]
            # The cube tools take only an entity argument.
            bindings = {"E": "encounter_counts"}
            errors = vg.validate_bindings(templates[template_idx], bindings, schema)
            assert errors == [], f"{name}: {errors}"
            spec = vg.instantiate_template(templates[template_idx], bindings, schema)
            assert spec["source"]["source"].endswith(
                "core__count_encounter_month.csv"
            )
