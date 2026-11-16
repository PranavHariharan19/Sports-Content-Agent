from typing import List, Optional

from llm.base import BaseLLMProvider, LLMProviderError, LLMRateLimitError
from schemas.llm import LLMRequest, LLMResponse, TokenUsage


class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock provider for unit testing without live API keys."""

    def __init__(
        self,
        name: str = "mock",
        default_model: str = "mock-model",
        responses: Optional[List[str]] = None,
        simulate_rate_limit: bool = False,
        simulate_error: bool = False,
    ):
        super().__init__(name=name, default_model=default_model, timeout_seconds=1.0)
        self.responses = list(responses or ["Mock response"])
        self.simulate_rate_limit = simulate_rate_limit
        self.simulate_error = simulate_error
        self.call_history: List[LLMRequest] = []
        self._available = True

    def set_available(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_history.append(request)

        if self.simulate_rate_limit:
            raise LLMRateLimitError(f"Mock rate limit reached for {self.name}")

        if self.simulate_error:
            raise LLMProviderError(f"Mock error in {self.name}")

        text = self.responses.pop(0) if self.responses else "Default mock response"
        return LLMResponse(
            text=text,
            provider=self.name,
            model=request.model or self.default_model,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            cached=False,
            latency_seconds=0.01,
        )
