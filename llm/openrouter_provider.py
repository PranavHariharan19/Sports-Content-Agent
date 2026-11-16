import os
from typing import Optional

from llm.openai_compatible_provider import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter provider for free and community models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "google/gemini-2.0-flash-exp:free",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 30.0,
    ):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        headers = {
            "HTTP-Referer": "https://github.com/sports-shorts-command-center",
            "X-Title": "Sports Shorts Command Center",
        }
        super().__init__(
            name="openrouter",
            default_model=default_model,
            base_url=base_url,
            api_key=key,
            custom_headers=headers,
            timeout_seconds=timeout_seconds,
            require_api_key=True,
        )
