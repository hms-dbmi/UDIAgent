from openai import OpenAI
from jinja2 import Template
from transformers import AutoTokenizer

class UDIAgent:
    """UDIAgent for requesting UDI grammar given a prompt."""


    def __init__(self,
                 model_name: str,
                 vllm_server_url: int,
                 vllm_server_port: int,
                ):
        self.model_name = model_name
        self.vllm_server_url = vllm_server_url
        self.vllm_server_port = vllm_server_port
        self.init_models()

    def init_models(self):
        base_url = f"{self.vllm_server_url}:{self.vllm_server_port}/v1"
        self.model = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
        )

    def chat_completions(self, messages: list[dict]):
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            # max_tokens=40960,
            max_tokens=120_000,
            temperature=0.0,
            # top_p=1.0,
        )
        return response


    def completions(self, messages: list[dict], tools: list[dict]):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        chat_template = Template(tokenizer.chat_template)

        print(f"Messages: {messages}")

        # Debugging, remove all the tool_calls from the messages
        # TODO: try reformatting as strings.
        for message in messages:
            if 'tool_calls' in message:
                del message['tool_calls']

        prompt = chat_template.render(
            messages=messages, tools=tools, add_generation_prompt=True)


        print(f"Prompt: {prompt}")

        response = self.model.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=16_384,
            temperature=0.0,
            # top_p=1.0,
        )
        return response