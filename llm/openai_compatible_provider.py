import json
import time
from typing import Any, Dict, List, Optional
import httpx

from app.logging_config import get_logger
from llm.base import (
    BaseLLMProvider,
    LLMAuthError,
    LLMProviderError,
    LLMRateLimitError,
)
from schemas.llm import LLMRequest, LLMResponse, TokenUsage

logger = get_logger(__name__)


class OpenAICompatibleProvider(BaseLLMProvider):
    """Generic provider for services adhering to the OpenAI Chat Completions API specification."""

    def __init__(
        self,
        name: str,
        default_model: str,
        base_url: str,
        api_key: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 30.0,
        require_api_key: bool = True,
    ):
        super().__init__(name=name, default_model=default_model, timeout_seconds=timeout_seconds)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.custom_headers = custom_headers or {}
        self.require_api_key = require_api_key

    def is_available(self) -> bool:
        if self.require_api_key:
            return bool(self.api_key)
        return True

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.custom_headers)
        return headers

    def _build_payload(self, request: LLMRequest, model_name: str) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.response_schema:
            payload["response_format"] = {"type": "json_object"}

        return payload

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.is_available():
            raise LLMAuthError(f"Provider {self.name} is not configured or missing API credentials")

        model_name = request.model or self.default_model
        payload = self._build_payload(request, model_name)
        endpoint = f"{self.base_url}/chat/completions"

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            try:
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers=self._build_headers(),
                )
            except httpx.TimeoutException as e:
                logger.warning("%s request timed out after %.1f s", self.name, self.timeout_seconds)
                raise LLMProviderError(f"{self.name} request timed out") from e
            except httpx.RequestError as e:
                logger.warning("%s network connection error: %s", self.name, str(e))
                raise LLMProviderError(f"{self.name} connection error: {str(e)}") from e

        elapsed = time.perf_counter() - start_time

        # Handle HTTP error statuses
        if response.status_code == 429:
            logger.warning("%s rate limit reached: %s", self.name, response.text)
            raise LLMRateLimitError(f"{self.name} rate limit reached (HTTP 429)")

        if response.status_code in (401, 403):
            logger.warning("%s auth failure (%d): %s", self.name, response.status_code, response.text)
            raise LLMAuthError(f"{self.name} authentication failure (HTTP {response.status_code})")

        if response.status_code >= 400:
            logger.error("%s API error (%d): %s", self.name, response.status_code, response.text)
            raise LLMProviderError(f"{self.name} error ({response.status_code}): {response.text}")

        try:
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise LLMProviderError(f"{self.name} returned empty choices")

            text = choices[0].get("message", {}).get("content", "")

            raw_usage = data.get("usage", {})
            usage = TokenUsage(
                prompt_tokens=raw_usage.get("prompt_tokens", 0),
                completion_tokens=raw_usage.get("completion_tokens", 0),
                total_tokens=raw_usage.get("total_tokens", 0),
            )

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model_name,
                usage=usage,
                cached=False,
                latency_seconds=round(elapsed, 3),
            )
        except Exception as e:
            if isinstance(e, (LLMProviderError, LLMRateLimitError, LLMAuthError)):
                raise
            logger.error("Failed to parse %s response: %s", self.name, str(e), exc_info=True)
            raise LLMProviderError(f"Failed to parse {self.name} response: {str(e)}") from e
