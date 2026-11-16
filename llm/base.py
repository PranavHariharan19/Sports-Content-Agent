from abc import ABC, abstractmethod
from typing import Optional

from schemas.llm import LLMRequest, LLMResponse


class LLMError(Exception):
    """Base exception for all LLM errors."""
    pass


class LLMProviderError(LLMError):
    """Provider error, e.g. 500 server error, network timeout."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit or quota exceeded (e.g. HTTP 429 or ResourceExhausted)."""
    pass


class LLMAuthError(LLMError):
    """Authentication or authorization failure (e.g. invalid API key)."""
    pass


class LLMQuotaExceededError(LLMError):
    """Internal or external daily quota exceeded."""
    pass


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, name: str, default_model: str, timeout_seconds: float = 30.0):
        self.name = name
        self.default_model = default_model
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available (e.g. API key present)."""
        pass

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response asynchronously."""
        pass

    def generate_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous wrapper for generate."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If already running inside an active event loop, create a new thread or use nest_asyncio
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.generate(request)).result()

        return loop.run_until_complete(self.generate(request))
