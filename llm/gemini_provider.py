import os
import time
from typing import Optional

from google import genai
from google.genai import errors, types

from app.logging_config import get_logger
from llm.base import (
    BaseLLMProvider,
    LLMAuthError,
    LLMProviderError,
    LLMRateLimitError,
)
from schemas.llm import LLMRequest, LLMResponse, TokenUsage

logger = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini Flash provider using the official google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.5-flash",
        timeout_seconds: float = 30.0,
    ):
        super().__init__(name="gemini", default_model=default_model, timeout_seconds=timeout_seconds)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client: Optional[genai.Client] = None
        if self.api_key:
            self._client = genai.Client(api_key=self.api_key)

    def is_available(self) -> bool:
        return bool(self.api_key and self._client is not None)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.is_available():
            raise LLMAuthError("Gemini API key is not configured or missing")

        model_name = request.model or self.default_model

        # Build config
        config_kwargs = {
            "temperature": request.temperature,
        }
        if request.system_prompt:
            config_kwargs["system_instruction"] = request.system_prompt
        if request.max_tokens:
            config_kwargs["max_output_tokens"] = request.max_tokens
        if request.response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = request.response_schema

        config = types.GenerateContentConfig(**config_kwargs)

        start_time = time.perf_counter()
        try:
            assert self._client is not None
            response = await self._client.aio.models.generate_content(
                model=model_name,
                contents=request.prompt,
                config=config,
            )
            elapsed = time.perf_counter() - start_time

            text = response.text or ""
            usage = TokenUsage()
            if response.usage_metadata:
                usage.prompt_tokens = response.usage_metadata.prompt_token_count or 0
                usage.completion_tokens = response.usage_metadata.candidates_token_count or 0
                usage.total_tokens = response.usage_metadata.total_token_count or (usage.prompt_tokens + usage.completion_tokens)

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model_name,
                usage=usage,
                cached=False,
                latency_seconds=round(elapsed, 3),
            )

        except errors.APIError as e:
            code = getattr(e, "code", 500)
            msg = str(e)
            logger.warning("Gemini API error (code=%s): %s", code, msg)
            if code == 429 or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                raise LLMRateLimitError(f"Gemini rate limit exceeded: {msg}") from e
            elif code in (401, 403) or "API_KEY_INVALID" in msg:
                raise LLMAuthError(f"Gemini authentication failed: {msg}") from e
            else:
                raise LLMProviderError(f"Gemini provider error ({code}): {msg}") from e
        except Exception as e:
            logger.error("Unexpected error in GeminiProvider: %s", str(e), exc_info=True)
            raise LLMProviderError(f"Unexpected Gemini failure: {str(e)}") from e
