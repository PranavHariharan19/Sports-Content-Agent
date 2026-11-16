import os
from typing import Optional

from llm.openai_compatible_provider import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    """Groq Cloud API provider (fast Llama 3 models on LPU)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "llama-3.3-70b-versatile",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_seconds: float = 30.0,
    ):
        key = api_key or os.environ.get("GROQ_API_KEY")
        super().__init__(
            name="groq",
            default_model=default_model,
            base_url=base_url,
            api_key=key,
            timeout_seconds=timeout_seconds,
            require_api_key=True,
        )
