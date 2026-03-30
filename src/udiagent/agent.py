"""UDIAgent — LLM client abstraction for OpenAI and vLLM backends."""

import json
import logging
from functools import lru_cache

from jinja2 import Template

from udiagent._compat import get_openai_class

logger = logging.getLogger(__name__)

_OpenAI = get_openai_class()


@lru_cache(maxsize=128)
def _make_openai_client(api_key: str):
    """Cached OpenAI client factory — preserves httpx connection pooling across requests."""
    return _OpenAI(api_key=api_key)


class UDIAgent:
    """UDIAgent for requesting UDI grammar given a prompt."""

    def __init__(
        self,
        model_name: str,
        gpt_model_name: str,
        vllm_server_url: str | None = None,
        vllm_server_port: int | None = None,
        tokenizer_name: str | None = None,
        use_vis_pipeline: bool = False,
        openai_api_key: str | None = None,
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.gpt_model_name = gpt_model_name
        self.use_vis_pipeline = use_vis_pipeline
        self._init_server_model_connection(openai_api_key)
        if (vllm_server_port is not None and vllm_server_url is not None) and not use_vis_pipeline:
            self.vllm_server_url = vllm_server_url
            self.vllm_server_port = vllm_server_port
            self._init_model_connection()

    def _init_model_connection(self):
        base_url = f"{self.vllm_server_url}:{self.vllm_server_port}/v1"
        self.model = _OpenAI(
            api_key="EMPTY",
            base_url=base_url,
        )
        logger.info("Local Model connections initialized")

        # Cache tokenizer + chat template once (avoid reloading per request)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer = tokenizer
        self.chat_template = Template(tokenizer.chat_template)

    def _init_server_model_connection(self, openai_api_key: str | None = None):
        """Instantiate the OpenAI client for GPT-based features.

        Uses the explicitly provided *openai_api_key* if given.
        """
        if openai_api_key is None:
            logger.info("No OpenAI API key provided; GPT-based features will require per-request keys.")
            self.gpt_model = None
        else:
            logger.info("OpenAI API key provided; GPT-based features will use this key by default.")
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

    def chat_completions(self, messages: list[dict]):
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=120_000,
            temperature=0.0,
        )
        return response

    def completions_guided_choice(
        self,
        messages: list[dict],
        tools: list[dict],
        choices: list[str],
        openai_api_key: str | None = None,
    ):
        schema = {
            "name": "ChoiceSelection",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"choice": {"type": "string", "enum": choices}},
                "required": ["choice"],
            },
            "strict": True,
        }

        client = self._get_gpt_client(openai_api_key)
        resp = client.chat.completions.create(
            model=self.gpt_model_name,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": schema,
            },
            max_completion_tokens=10,
            temperature=0.0,
        )
        content = resp.choices[0].message.content
        return json.loads(content)["choice"]

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

    def completions_guided_json(
        self, messages: list[dict], tools: list[dict], json_schema: str, n=1
    ):
        prompt = self.chat_template.render(
            messages=messages, tools=tools, add_generation_prompt=True
        )

        response = self.model.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_completion_tokens=16_384,
            temperature=0.0,
            n=n,
            extra_body={
                "guided_json": json_schema,
            },
        )
        return response

    def completions(self, messages: list[dict], tools: list[dict]):
        logger.debug("Messages: %s", messages)

        for message in messages:
            if "tool_calls" in message:
                del message["tool_calls"]

        prompt = self.chat_template.render(
            messages=messages, tools=tools, add_generation_prompt=True
        )

        logger.debug("Prompt: %s", prompt)

        response = self.model.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_completion_tokens=16_384,
            temperature=0.0,
        )
        return response
