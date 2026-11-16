import os
from typing import Optional

from llm.openai_compatible_provider import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """Local Ollama provider (runs completely offline with zero API cost)."""

    def __init__(
        self,
        default_model: str = "llama3.2:latest",
        base_url: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ):
        raw_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        # Ollama provides an OpenAI-compatible endpoint at /v1
        endpoint_url = raw_url.rstrip("/")
        if not endpoint_url.endswith("/v1"):
            endpoint_url = f"{endpoint_url}/v1"

        super().__init__(
            name="ollama",
            default_model=default_model,
            base_url=endpoint_url,
            api_key="ollama",  # dummy key for headers
            timeout_seconds=timeout_seconds,
            require_api_key=False,
        )
