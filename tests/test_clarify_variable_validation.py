"""Tests for variable clarification schema validation and retry."""

import json
from unittest.mock import MagicMock, patch

import pytest

from udiagent.orchestrator import (
    Orchestrator,
    _validate_variable_candidates,
    _build_schema_field_meta,
)


PENGUIN_SCHEMA = {
    "resources": [
        {
            "name": "penguins",
            "schema": {
                "fields": [
                    {"name": "sex", "udi:data_type": "nominal", "description": "Sex of the penguin."},
                    {"name": "species", "udi:data_type": "nominal", "description": "Species name."},
                    {"name": "body_mass_g", "udi:data_type": "quantitative", "description": "Body mass in grams."},
                ]
            },
        }
    ]
}


class TestCandidateValidation:
    def test_valid_candidates_produce_no_errors(self):
        schema_meta = _build_schema_field_meta(PENGUIN_SCHEMA)
        candidates = [
            {
                "query_term": "sex",
                "candidates": [{"entity": "penguins", "field_name": "sex"}],
            }
        ]
        assert _validate_variable_candidates(candidates, set(schema_meta.keys())) == []

    def test_qualified_name_is_rejected(self):
        schema_meta = _build_schema_field_meta(PENGUIN_SCHEMA)
        candidates = [
            {
                "query_term": "sex",
                "candidates": [{"entity": "penguins", "field_name": "penguins.sex"}],
            }
        ]
        errors = _validate_variable_candidates(candidates, set(schema_meta.keys()))
        assert len(errors) == 1
        assert "penguins.sex" in errors[0]

    def test_wrong_entity_is_rejected(self):
        schema_meta = _build_schema_field_meta(PENGUIN_SCHEMA)
        candidates = [
            {
                "query_term": "sex",
                "candidates": [{"entity": "donors", "field_name": "sex"}],
            }
        ]
        errors = _validate_variable_candidates(candidates, set(schema_meta.keys()))
        assert len(errors) == 1


class TestHandlerRetryFlow:
    def _make_orchestrator(self):
        from udiagent.agent import UDIAgent

        agent = UDIAgent.__new__(UDIAgent)
        agent.gpt_model = MagicMock(name="default_gpt_model")
        agent.gpt_model_name = "gpt-4.1"
        agent._get_gpt_client = MagicMock(return_value=agent.gpt_model)
        orch = Orchestrator.__new__(Orchestrator)
        orch.agent = agent
        # Minimal orchestrate skill for retry prompt rendering
        from udiagent.skills import load_skills
        orch.skills = load_skills()
        from udiagent.tools import ORCHESTRATOR_TOOLS
        orch.tools = ORCHESTRATOR_TOOLS
        return orch, agent

    def test_malformed_qualified_name_triggers_retry(self):
        orch, agent = self._make_orchestrator()

        retry_corrected_args = {
            "clarification_type": "variable",
            "message": "Which 'sex' did you mean?",
            "ambiguous_variables": [
                {
                    "query_term": "sex",
                    "candidates": [
                        {"entity": "penguins", "field_name": "sex"},
                    ],
                }
            ],
        }
        retry_response = MagicMock()
        retry_response.choices = [
            MagicMock(
                message=MagicMock(
                    tool_calls=[
                        MagicMock(
                            function=MagicMock(
                                arguments=json.dumps(retry_corrected_args)
                            )
                        )
                    ]
                )
            )
        ]
        agent.gpt_model.chat.completions.create = MagicMock(return_value=retry_response)

        initial_tool_args = {
            "clarification_type": "variable",
            "message": "Which one?",
            "ambiguous_variables": [
                {
                    "query_term": "sex",
                    "candidates": [
                        {"entity": "penguins", "field_name": "penguins.sex"},
                    ],
                }
            ],
        }

        result = orch._handle_clarify_variable(
            initial_tool_args,
            messages=[{"role": "user", "content": "show sex"}],
            data_schema=json.dumps(PENGUIN_SCHEMA),
            data_domains="[]",
            openai_api_key=None,
        )

        args = result["arguments"]
        assert args["validation_retries"] == 1
        candidates = args["ambiguous_variables"][0]["candidates"]
        assert candidates[0]["field_name"] == "sex"
        # Enrichment happened on the corrected candidate
        assert candidates[0]["data_type"] == "nominal"

    def test_valid_input_does_not_retry(self):
        orch, agent = self._make_orchestrator()
        agent.gpt_model.chat.completions.create = MagicMock(
            side_effect=AssertionError("retry should not be invoked")
        )

        tool_args = {
            "clarification_type": "variable",
            "message": "Which one?",
            "ambiguous_variables": [
                {
                    "query_term": "sex",
                    "candidates": [{"entity": "penguins", "field_name": "sex"}],
                }
            ],
        }
        result = orch._handle_clarify_variable(
            tool_args,
            messages=[],
            data_schema=json.dumps(PENGUIN_SCHEMA),
            data_domains="[]",
            openai_api_key=None,
        )
        assert result["arguments"]["validation_retries"] == 0

    def test_general_type_skips_validation(self):
        orch, agent = self._make_orchestrator()
        agent.gpt_model.chat.completions.create = MagicMock(
            side_effect=AssertionError("retry should not be invoked for general")
        )

        tool_args = {
            "clarification_type": "general",
            "message": "Which chart type?",
            "ambiguous_variables": [
                {
                    "query_term": "chart",
                    "candidates": [
                        {"entity": "", "field_name": "bar chart"},
                        {"entity": "", "field_name": "scatter plot"},
                    ],
                }
            ],
        }
        result = orch._handle_clarify_variable(
            tool_args,
            messages=[],
            data_schema=json.dumps(PENGUIN_SCHEMA),
            data_domains="[]",
            openai_api_key=None,
        )
        args = result["arguments"]
        assert args["clarification_type"] == "general"
        assert args["validation_retries"] == 0
        # No enrichment for general type
        candidate = args["ambiguous_variables"][0]["candidates"][0]
        assert "data_type" not in candidate
