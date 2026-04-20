"""Tests for action-aware FilterData validation, retry, and payload shape."""

import json
from unittest.mock import MagicMock

import pytest

from udiagent.orchestrator import (
    Orchestrator,
    _build_filter_payload,
    _format_current_filters,
    _validate_filter_action,
)


PENGUIN_SCHEMA = {
    "resources": [
        {
            "name": "penguins",
            "schema": {
                "fields": [
                    {"name": "sex", "udi:data_type": "nominal"},
                    {"name": "species", "udi:data_type": "nominal"},
                    {"name": "body_mass_g", "udi:data_type": "quantitative"},
                ]
            },
        }
    ]
}

SCHEMA_KEYS = {
    ("penguins", "sex"),
    ("penguins", "species"),
    ("penguins", "body_mass_g"),
}


def _existing_filter(entity, field, ftype="point", values=None, rng=None):
    filt = {"filterType": ftype}
    if ftype == "point":
        filt["pointValues"] = values or ["Male"]
    else:
        filt["intervalRange"] = rng or {"min": 0, "max": 1}
    return {"entity": entity, "field": field, "filter": filt}


class TestValidate:
    def test_add_new_field_passes(self):
        args = {
            "action": "add",
            "entity": "penguins",
            "field": "sex",
            "filterType": "point",
            "pointValues": ["Male"],
        }
        assert _validate_filter_action(args, SCHEMA_KEYS, []) == []

    def test_add_already_filtered_is_rejected(self):
        args = {
            "action": "add",
            "entity": "penguins",
            "field": "sex",
            "filterType": "point",
            "pointValues": ["Male"],
        }
        current = [_existing_filter("penguins", "sex")]
        errors = _validate_filter_action(args, SCHEMA_KEYS, current)
        assert any("already exists" in e for e in errors)

    def test_modify_requires_existing_filter(self):
        args = {
            "action": "modify",
            "entity": "penguins",
            "field": "sex",
            "filterType": "point",
            "pointValues": ["Female"],
        }
        errors = _validate_filter_action(args, SCHEMA_KEYS, [])
        assert any("No existing filter" in e for e in errors)

    def test_modify_on_existing_passes(self):
        args = {
            "action": "modify",
            "entity": "penguins",
            "field": "sex",
            "filterType": "point",
            "pointValues": ["Female"],
        }
        current = [_existing_filter("penguins", "sex")]
        assert _validate_filter_action(args, SCHEMA_KEYS, current) == []

    def test_remove_requires_existing_filter(self):
        args = {"action": "remove", "entity": "penguins", "field": "sex"}
        errors = _validate_filter_action(args, SCHEMA_KEYS, [])
        assert any("No existing filter" in e for e in errors)

    def test_remove_on_existing_passes_without_filter_type(self):
        args = {"action": "remove", "entity": "penguins", "field": "sex"}
        current = [_existing_filter("penguins", "sex")]
        assert _validate_filter_action(args, SCHEMA_KEYS, current) == []

    def test_unknown_entity_field_is_rejected(self):
        args = {
            "action": "add",
            "entity": "penguins",
            "field": "wingspan",
            "filterType": "point",
            "pointValues": ["big"],
        }
        errors = _validate_filter_action(args, SCHEMA_KEYS, [])
        assert any("not a valid" in e for e in errors)

    def test_point_action_requires_values(self):
        args = {
            "action": "add",
            "entity": "penguins",
            "field": "sex",
            "filterType": "point",
            "pointValues": [],
        }
        errors = _validate_filter_action(args, SCHEMA_KEYS, [])
        assert any("pointValues" in e for e in errors)


class TestPayloadShape:
    def test_remove_payload_omits_filter(self):
        args = {"action": "remove", "entity": "penguins", "field": "sex"}
        payload = _build_filter_payload(args, validation_retries=0)
        assert payload["action"] == "remove"
        assert "filter" not in payload
        assert payload["validation_retries"] == 0

    def test_add_payload_includes_filter(self):
        args = {
            "action": "add",
            "entity": "penguins",
            "field": "sex",
            "filterType": "point",
            "pointValues": ["Male"],
        }
        payload = _build_filter_payload(args, validation_retries=0)
        assert payload["filter"]["filterType"] == "point"
        assert payload["filter"]["pointValues"] == ["Male"]


class TestFormatCurrentFilters:
    def test_empty_returns_none_marker(self):
        assert "none" in _format_current_filters([]).lower()

    def test_renders_each_filter(self):
        current = [
            _existing_filter("penguins", "sex", "point", ["Male"]),
            _existing_filter(
                "penguins",
                "body_mass_g",
                "interval",
                rng={"min": 3000, "max": 4500},
            ),
        ]
        rendered = _format_current_filters(current)
        assert "entity=penguins, field=sex" in rendered
        assert "entity=penguins, field=body_mass_g" in rendered
        assert "range [3000, 4500]" in rendered


class TestHandlerRetry:
    def _make_orchestrator(self):
        from udiagent.agent import UDIAgent
        from udiagent.skills import load_skills
        from udiagent.tools import ORCHESTRATOR_TOOLS

        agent = UDIAgent.__new__(UDIAgent)
        agent.gpt_model = MagicMock(name="default_gpt_model")
        agent.gpt_model_name = "gpt-4.1"
        agent._get_gpt_client = MagicMock(return_value=agent.gpt_model)
        orch = Orchestrator.__new__(Orchestrator)
        orch.agent = agent
        orch.skills = load_skills()
        orch.tools = ORCHESTRATOR_TOOLS
        return orch, agent

    def test_add_on_existing_filter_triggers_retry_to_modify(self):
        orch, agent = self._make_orchestrator()

        corrected_args = {
            "action": "modify",
            "entity": "penguins",
            "field": "sex",
            "filterType": "point",
            "pointValues": ["Female"],
        }
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(
                    tool_calls=[
                        MagicMock(
                            function=MagicMock(
                                arguments=json.dumps(corrected_args)
                            )
                        )
                    ]
                )
            )
        ]
        agent.gpt_model.chat.completions.create = MagicMock(return_value=response)

        initial_args = {
            "action": "add",
            "entity": "penguins",
            "field": "sex",
            "filterType": "point",
            "pointValues": ["Female"],
        }
        result = orch._handle_filter_data(
            initial_args,
            messages=[{"role": "user", "content": "filter to females"}],
            data_schema=json.dumps(PENGUIN_SCHEMA),
            data_domains="[]",
            current_filters=[_existing_filter("penguins", "sex")],
            openai_api_key=None,
        )

        args = result["arguments"]
        assert args["action"] == "modify"
        assert args["validation_retries"] == 1
        assert args["filter"]["pointValues"] == ["Female"]

    def test_valid_add_does_not_retry(self):
        orch, agent = self._make_orchestrator()
        agent.gpt_model.chat.completions.create = MagicMock(
            side_effect=AssertionError("retry should not be invoked")
        )

        args = {
            "action": "add",
            "entity": "penguins",
            "field": "sex",
            "filterType": "point",
            "pointValues": ["Male"],
        }
        result = orch._handle_filter_data(
            args,
            messages=[],
            data_schema=json.dumps(PENGUIN_SCHEMA),
            data_domains="[]",
            current_filters=[],
            openai_api_key=None,
        )
        assert result["arguments"]["action"] == "add"
        assert result["arguments"]["validation_retries"] == 0

    def test_remove_on_existing_passes(self):
        orch, agent = self._make_orchestrator()
        agent.gpt_model.chat.completions.create = MagicMock(
            side_effect=AssertionError("retry should not be invoked")
        )

        args = {"action": "remove", "entity": "penguins", "field": "sex"}
        result = orch._handle_filter_data(
            args,
            messages=[],
            data_schema=json.dumps(PENGUIN_SCHEMA),
            data_domains="[]",
            current_filters=[_existing_filter("penguins", "sex")],
            openai_api_key=None,
        )
        assert result["arguments"]["action"] == "remove"
        assert "filter" not in result["arguments"]
