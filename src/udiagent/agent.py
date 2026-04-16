"""UDIAgent — OpenAI client wrapper."""

import json
import logging
from functools import lru_cache

from udiagent._compat import get_openai_class

logger = logging.getLogger(__name__)

_OpenAI = get_openai_class()


@lru_cache(maxsize=128)
def _make_openai_client(api_key: str):
    """Cached OpenAI client factory — preserves httpx connection pooling across requests."""
    return _OpenAI(api_key=api_key)


class UDIAgent:
    """UDIAgent for requesting UDI grammar via OpenAI."""

    def __init__(
        self,
        gpt_model_name: str,
        openai_api_key: str | None = None,
    ):
        self.gpt_model_name = gpt_model_name
        self._init_server_model_connection(openai_api_key)

    def _init_server_model_connection(self, openai_api_key: str | None = None):
        """Instantiate the OpenAI client for GPT-based features.

        Uses the explicitly provided *openai_api_key* if given.
        """
        if openai_api_key is None:
            logger.info(
                "No OpenAI API key provided; GPT-based features will require per-request keys."
            )
            self.gpt_model = None
        else:
            logger.info(
                "OpenAI API key provided; GPT-based features will use this key by default."
            )
            self.gpt_model = _OpenAI(api_key=openai_api_key)

    def _get_gpt_client(self, openai_api_key: str | None = None):
        """Return a per-request OpenAI client if a custom key is provided, otherwise the default."""
        if openai_api_key:
            return _make_openai_client(openai_api_key)
        if self.gpt_model is None:
            raise RuntimeError(
                "No OpenAI API key available. Provide openai_api_key to UDIAgent() "
                "or pass a per-request key."
            )
        return self.gpt_model

    def gpt_completions_guided_json(
        self,
        messages: list[dict],
        json_schema: str,
        n=1,
        openai_api_key: str | None = None,
    ):
        # Normalize schema to dict
        if isinstance(json_schema, str):
            try:
                schema_obj = json.loads(json_schema)
            except json.JSONDecodeError as e:
                raise ValueError(f"json_schema must be a valid JSON string: {e}")
        else:
            schema_obj = json_schema

        # Wrap for Structured Outputs (required shape)
        schema_wrapper = {
            "name": "GuidedJSON",
            "schema": schema_obj,
            "strict": True,
        }

        client = self._get_gpt_client(openai_api_key)
        resp = client.chat.completions.create(
            model=self.gpt_model_name,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": schema_wrapper,
            },
            n=n,
            temperature=0.0,
            max_completion_tokens=16_384,
        )

        outputs = [json.loads(choice.message.content) for choice in resp.choices]
        return outputs
