from openai import OpenAI

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
        """Request chat completions from the model."""
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            # max_tokens=40960,
            max_tokens=120_000,
            # temperature=0.7,
            # top_p=1.0,
        )
        return response