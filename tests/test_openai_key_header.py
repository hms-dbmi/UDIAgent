"""Tests for X-OpenAI-Key header propagation through the UDIAgent API."""

from unittest.mock import patch, MagicMock
import pytest

from udi_agent import UDIAgent, _make_openai_client


# ---------------------------------------------------------------------------
# Unit tests for _make_openai_client cache
# ---------------------------------------------------------------------------


class TestMakeOpenaiClientCache:
    def setup_method(self):
        _make_openai_client.cache_clear()

    def test_same_key_returns_cached_client(self):
        client_a = _make_openai_client("sk-test-key-1")
        client_b = _make_openai_client("sk-test-key-1")
        assert client_a is client_b

    def test_different_keys_return_different_clients(self):
        client_a = _make_openai_client("sk-test-key-1")
        client_b = _make_openai_client("sk-test-key-2")
        assert client_a is not client_b


# ---------------------------------------------------------------------------
# Unit tests for UDIAgent._get_gpt_client
# ---------------------------------------------------------------------------


class TestGetGptClient:
    def setup_method(self):
        _make_openai_client.cache_clear()

    def _make_agent(self):
        """Create a UDIAgent without initializing model connections."""
        agent = UDIAgent.__new__(UDIAgent)
        agent.gpt_model = MagicMock(name="default_gpt_model")
        agent.gpt_model_name = "gpt-4.1"
        return agent

    def test_none_key_returns_default_client(self):
        agent = self._make_agent()
        client = agent._get_gpt_client(None)
        assert client is agent.gpt_model

    def test_custom_key_returns_different_client(self):
        agent = self._make_agent()
        client = agent._get_gpt_client("sk-custom-key")
        assert client is not agent.gpt_model


# ---------------------------------------------------------------------------
# Integration tests for API header extraction
# ---------------------------------------------------------------------------


class TestApiHeaderExtraction:
    @pytest.fixture(autouse=True)
    def setup_app(self):
        """Patch UDIAgent to avoid real model initialization, then import the app."""
        with patch.object(UDIAgent, "__init__", lambda self, **kwargs: None):
            import udi_api

            # Set required attributes on the agent singleton
            udi_api.agent = UDIAgent.__new__(UDIAgent)
            udi_api.agent.gpt_model = MagicMock(name="default_gpt_model")
            udi_api.agent.gpt_model_name = "gpt-4.1"
            udi_api.agent.model_name = "test-model"

            from starlette.testclient import TestClient

            self.client = TestClient(udi_api.app)
            self.udi_api = udi_api
            yield

    def _make_request_body(self):
        return {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "show a bar chart"}],
            "dataSchema": "{}",
            "dataDomains": "{}",
        }

    def test_header_passed_to_determine_function_calls(self):
        """When X-OpenAI-Key is sent, it should reach completions_guided_choice."""
        with patch.object(
            self.udi_api.agent,
            "completions_guided_choice",
            return_value="render-visualization",
        ) as mock_choice, patch(
            "udi_api.function_call_render_visualization",
            return_value={"name": "RenderVisualization", "arguments": {"spec": {}}},
        ):
            self.client.post(
                "/v1/yac/completions",
                json=self._make_request_body(),
                headers={
                    "Authorization": "Bearer test",
                    "X-OpenAI-Key": "sk-user-provided",
                },
            )
            mock_choice.assert_called_once()
            assert mock_choice.call_args.kwargs.get("openai_api_key") == "sk-user-provided"

    def test_no_header_passes_none(self):
        """When X-OpenAI-Key is absent, openai_api_key should be None."""
        with patch.object(
            self.udi_api.agent,
            "completions_guided_choice",
            return_value="render-visualization",
        ) as mock_choice, patch(
            "udi_api.function_call_render_visualization",
            return_value={"name": "RenderVisualization", "arguments": {"spec": {}}},
        ):
            self.client.post(
                "/v1/yac/completions",
                json=self._make_request_body(),
                headers={"Authorization": "Bearer test"},
            )
            mock_choice.assert_called_once()
            assert mock_choice.call_args.kwargs.get("openai_api_key") is None

    def test_header_passed_to_filter(self):
        """When both filter and vis are needed, the key reaches gpt_completions_guided_json."""
        with patch.object(
            self.udi_api.agent,
            "completions_guided_choice",
            return_value="both",
        ), patch.object(
            self.udi_api.agent,
            "gpt_completions_guided_json",
            return_value=[{"entity": "donors", "field": "age", "filter": {}}],
        ) as mock_filter, patch(
            "udi_api.function_call_render_visualization",
            return_value={"name": "RenderVisualization", "arguments": {"spec": {}}},
        ):
            self.client.post(
                "/v1/yac/completions",
                json=self._make_request_body(),
                headers={
                    "Authorization": "Bearer test",
                    "X-OpenAI-Key": "sk-user-key",
                },
            )
            mock_filter.assert_called_once()
            assert mock_filter.call_args.kwargs.get("openai_api_key") == "sk-user-key"
