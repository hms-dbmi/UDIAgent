"""Server configuration from environment variables."""

import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for the UDIAgent FastAPI server.

    All values can be set via environment variables (see field names).
    Call ``ServerConfig.from_env()`` to load from the current environment.
    """

    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    insecure_dev_mode: bool = False
    gpt_model_name: str = "gpt-5.4"
    openai_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create a config by reading environment variables."""
        return cls(
            jwt_secret_key=os.getenv("JWT_SECRET_KEY", ""),
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            insecure_dev_mode=int(os.getenv("INSECURE_DEV_MODE", "0")) == 1,
            gpt_model_name=os.getenv("GPT_MODEL_NAME", "gpt-5.4"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
