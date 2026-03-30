"""Server configuration from environment variables."""

import os
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    """Configuration for the UDIAgent FastAPI server.

    All values can be set via environment variables (see field names).
    Call ``ServerConfig.from_env()`` to load from the current environment.
    """

    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    udi_model_name: str = ""
    udi_tokenizer_name: str | None = None
    insecure_dev_mode: bool = False
    vllm_server_url: str = "http://localhost"
    vllm_server_port: int = 55001
    gpt_model_name: str = "gpt-5.4"
    use_vis_pipeline: bool = False
    openai_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create a config by reading environment variables."""
        model_name = os.getenv("UDI_MODEL_NAME", "")
        return cls(
            jwt_secret_key=os.getenv("JWT_SECRET_KEY", ""),
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            udi_model_name=model_name,
            udi_tokenizer_name=os.getenv("UDI_TOKENIZER_NAME", model_name or None),
            insecure_dev_mode=int(os.getenv("INSECURE_DEV_MODE", "0")) == 1,
            vllm_server_url=os.getenv("VLLM_SERVER_URL", "http://localhost"),
            vllm_server_port=int(os.getenv("VLLM_SERVER_PORT", "55001")),
            gpt_model_name=os.getenv("GPT_MODEL_NAME", "gpt-5.4"),
            use_vis_pipeline=int(os.getenv("USE_VIS_PIPELINE", "0")) == 1,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
