import asyncio
from typing import Dict, List, Optional

from app.logging_config import get_logger
from llm.base import (
    BaseLLMProvider,
    LLMAuthError,
    LLMError,
    LLMProviderError,
    LLMQuotaExceededError,
    LLMRateLimitError,
)
from llm.cache import LLMCache
from llm.gemini_provider import GeminiProvider
from llm.groq_provider import GroqProvider
from llm.ollama_provider import OllamaProvider
from llm.openrouter_provider import OpenRouterProvider
from llm.usage import LLMUsageTracker
from pipeline.db import DatabaseManager
from schemas.config import LLMConfig
from schemas.llm import LLMRequest, LLMResponse

logger = get_logger(__name__)


class AllProvidersFailedError(LLMError):
    """Raised when all configured LLM providers fail or have exhausted quotas."""
    pass


class LLMRouter:
    """Orchestrates LLM requests with automatic fallback, response caching,

    usage quotas, and batch processing.
    """

    def __init__(
        self,
        config: LLMConfig,
        db: DatabaseManager,
        custom_providers: Optional[Dict[str, BaseLLMProvider]] = None,
    ):
        self.config = config
        self.db = db
        self.cache = LLMCache(db) if config.cache_enabled else None
        self.usage_tracker = LLMUsageTracker(db)

        # Initialize providers
        self.providers: Dict[str, BaseLLMProvider] = {}
        if custom_providers:
            self.providers.update(custom_providers)
        else:
            self._init_default_providers()

    def _init_default_providers(self) -> None:
        """Instantiate providers based on configuration."""
        providers_cfg = self.config.providers

        # Gemini
        gemini_cfg = providers_cfg.get("gemini")
        gemini_model = gemini_cfg.model if gemini_cfg else "gemini-2.5-flash"
        self.providers["gemini"] = GeminiProvider(
            default_model=gemini_model,
            timeout_seconds=self.config.timeout_seconds,
        )

        # Groq
        groq_cfg = providers_cfg.get("groq")
        groq_model = groq_cfg.model if groq_cfg else "llama-3.3-70b-versatile"
        groq_url = groq_cfg.base_url if groq_cfg and groq_cfg.base_url else "https://api.groq.com/openai/v1"
        self.providers["groq"] = GroqProvider(
            default_model=groq_model,
            base_url=groq_url,
            timeout_seconds=self.config.timeout_seconds,
        )

        # OpenRouter
        or_cfg = providers_cfg.get("openrouter")
        or_model = or_cfg.model if or_cfg else "google/gemini-2.0-flash-exp:free"
        or_url = or_cfg.base_url if or_cfg and or_cfg.base_url else "https://openrouter.ai/api/v1"
        self.providers["openrouter"] = OpenRouterProvider(
            default_model=or_model,
            base_url=or_url,
            timeout_seconds=self.config.timeout_seconds,
        )

        # Ollama
        ollama_cfg = providers_cfg.get("ollama")
        ollama_model = ollama_cfg.model if ollama_cfg else "llama3.2:latest"
        ollama_url = ollama_cfg.base_url if ollama_cfg and ollama_cfg.base_url else "http://localhost:11434"
        self.providers["ollama"] = OllamaProvider(
            default_model=ollama_model,
            base_url=ollama_url,
            timeout_seconds=self.config.timeout_seconds,
        )

    def get_provider(self, name: str) -> Optional[BaseLLMProvider]:
        return self.providers.get(name)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Route request through fallback chain with caching and quota tracking."""
        fallback_chain = self.config.fallback_order
        errors: List[str] = []

        for provider_name in fallback_chain:
            provider = self.providers.get(provider_name)
            if not provider:
                logger.debug("Provider '%s' not registered, skipping", provider_name)
                continue

            if not provider.is_available():
                logger.debug("Provider '%s' is not available (missing credentials), skipping", provider_name)
                continue

            model_name = request.model or provider.default_model

            # Check daily quota
            provider_cfg = self.config.providers.get(provider_name)
            daily_limit = provider_cfg.daily_request_limit if provider_cfg else 1000
            if not self.usage_tracker.can_make_request(provider_name, model_name, daily_limit):
                logger.warning(
                    "Provider '%s' exceeded daily limit (%d), falling back to next provider",
                    provider_name,
                    daily_limit,
                )
                errors.append(f"{provider_name}: daily quota exceeded")
                continue

            # Check cache if enabled
            if self.cache:
                cache_key = request.compute_cache_key(provider=provider_name, effective_model=model_name)
                cached_res = self.cache.get(cache_key)
                if cached_res:
                    return cached_res

            # Try generating with retries
            retries = max(1, self.config.max_retries_per_provider)
            success = False
            last_provider_error: Optional[Exception] = None

            for attempt in range(1, retries + 1):
                try:
                    logger.debug(
                        "Attempting LLM call to %s/%s (attempt %d/%d)",
                        provider_name,
                        model_name,
                        attempt,
                        retries,
                    )
                    response = await provider.generate(request)

                    # Record usage and cache
                    self.usage_tracker.record_usage(
                        provider=provider_name,
                        model=model_name,
                        tokens=response.usage.total_tokens,
                    )
                    if self.cache:
                        self.cache.set(cache_key, response)

                    return response

                except (LLMRateLimitError, LLMQuotaExceededError) as e:
                    logger.warning("Provider %s rate limit / quota hit: %s", provider_name, str(e))
                    last_provider_error = e
                    # Immediately break out of retry loop to fall back to next provider
                    break

                except (LLMAuthError, LLMProviderError, Exception) as e:
                    logger.warning(
                        "Provider %s attempt %d/%d failed: %s",
                        provider_name,
                        attempt,
                        retries,
                        str(e),
                    )
                    last_provider_error = e
                    if attempt < retries:
                        await asyncio.sleep(0.5 * attempt)

            errors.append(f"{provider_name}: {str(last_provider_error)}")
            logger.info("Falling back from provider '%s' to next in chain", provider_name)

        error_summary = "; ".join(errors)
        raise AllProvidersFailedError(f"All configured LLM providers failed: {error_summary}")

    def generate_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous wrapper for generate."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.generate(request)).result()

        return loop.run_until_complete(self.generate(request))

    async def generate_batch(
        self,
        requests: List[LLMRequest],
        max_concurrency: int = 4,
    ) -> List[LLMResponse]:
        """Execute a batch of LLM requests with bounded concurrency."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_one(req: LLMRequest) -> LLMResponse:
            async with semaphore:
                return await self.generate(req)

        tasks = [_run_one(r) for r in requests]
        return await asyncio.gather(*tasks)
