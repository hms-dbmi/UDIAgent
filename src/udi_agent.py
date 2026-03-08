import json
from langfuse.openai import OpenAI
from langfuse import observe
import anthropic
from jinja2 import Template
from transformers import AutoTokenizer

# from vllm import LLM, SamplingParams
import os
from dotenv import load_dotenv

load_dotenv()  # automatically loads from .env

# Use multiprocess backend for workers
# os.environ["VLLM_WORKER_MULTIPROC"] = "1"

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class UDIAgent:
    """UDIAgent for requesting UDI grammar given a prompt."""

    def __init__(
        self,
        model_name: str,
        claude_model_name: str,
        vllm_server_url=None,
        vllm_server_port=None,
        tokenizer_name: str = None,
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.claude_model_name = claude_model_name
        if vllm_server_port is not None and vllm_server_url is not None:
            self.vllm_server_url = vllm_server_url
            self.vllm_server_port = vllm_server_port
            self.init_model_connection()

    def init_model_connection(self):
        base_url = f"{self.vllm_server_url}:{self.vllm_server_port}/v1"
        # vLLM client stays as OpenAI-compatible (unchanged)
        self.model = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
        )

        # Anthropic Claude client (replaces OpenAI GPT)
        self.claude_model = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        # Cache tokenizer + chat template once (avoid reloading per request)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer = tokenizer
        self.chat_template = Template(tokenizer.chat_template)

    @staticmethod
    def _split_system_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        """Extract system messages from the messages list.

        Anthropic requires system messages as a separate parameter,
        not in the messages array.

        Returns (system_text, non_system_messages).
        """
        system_parts = []
        other_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg["content"])
            else:
                other_messages.append(msg)
        return "\n\n".join(system_parts), other_messages

    def chat_completions(self, messages: list[dict]):
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=120_000,
            temperature=0.0,
        )
        return response

    @observe()
    def completions_guided_choice(
        self, messages: list[dict], tools: list[dict], choices: list[str]
    ):
        """Use Anthropic tool use to force a structured choice selection."""
        system_text, user_messages = self._split_system_messages(messages)

        # Define a tool that forces the LLM to return one of the choices
        choice_tool = {
            "name": "ChoiceSelection",
            "description": "Select the appropriate choice based on the user's request.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "string",
                        "enum": choices,
                        "description": "The selected choice.",
                    }
                },
                "required": ["choice"],
            },
        }

        resp = self.claude_model.messages.create(
            model=self.claude_model_name,
            system=system_text,
            messages=user_messages,
            tools=[choice_tool],
            tool_choice={"type": "tool", "name": "ChoiceSelection"},
            max_tokens=256,
            temperature=0.0,
        )

        # Extract tool use result from response
        for block in resp.content:
            if block.type == "tool_use":
                return block.input["choice"]

        raise ValueError("No tool_use block in Claude response")

    @observe()
    def claude_completions_guided_json(
        self, messages: list[dict], json_schema: str, n=1
    ):
        """Use Anthropic tool use to get structured JSON output.

        Replaces gpt_completions_guided_json. Uses a tool definition to
        constrain the output to the given JSON schema.
        """
        # Normalize schema to dict
        if isinstance(json_schema, str):
            try:
                schema_obj = json.loads(json_schema)
            except json.JSONDecodeError as e:
                raise ValueError(f"json_schema must be a valid JSON string: {e}")
        else:
            schema_obj = json_schema

        system_text, user_messages = self._split_system_messages(messages)

        # Define a tool whose input_schema matches the desired output
        guided_tool = {
            "name": "GuidedJSON",
            "description": "Return the structured JSON output matching the required schema.",
            "input_schema": schema_obj,
        }

        outputs = []
        for _ in range(n):
            resp = self.claude_model.messages.create(
                model=self.claude_model_name,
                system=system_text,
                messages=user_messages,
                tools=[guided_tool],
                tool_choice={"type": "tool", "name": "GuidedJSON"},
                max_tokens=16_384,
                temperature=0.0,
            )

            for block in resp.content:
                if block.type == "tool_use":
                    outputs.append(block.input)
                    break
            else:
                outputs.append({})

        return outputs

    # Keep old method name as alias for backward compatibility
    def gpt_completions_guided_json(self, messages: list[dict], json_schema: str, n=1):
        return self.claude_completions_guided_json(messages, json_schema, n)

    def completions_guided_json(
        self, messages: list[dict], tools: list[dict], json_schema: str, n=1
    ):
        prompt = self.chat_template.render(
            messages=messages, tools=tools, add_generation_prompt=True
        )

        # todo, add prompt engineering here?
        response = self.model.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=16_384,
            temperature=0.0,
            n=n,
            extra_body={
                "guided_json": json_schema,
            },
        )
        return response

    def completions(self, messages: list[dict], tools: list[dict]):
        print(f"Messages: {messages}")

        # Debugging, remove all the tool_calls from the messages
        # TODO: try reformatting as strings.
        for message in messages:
            if "tool_calls" in message:
                del message["tool_calls"]

        prompt = self.chat_template.render(
            messages=messages, tools=tools, add_generation_prompt=True
        )

        print(f"Prompt: {prompt}")

        response = self.model.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=16_384,
            temperature=0.0,
        )
        return response
