from llm.cache import LLMCache
from pipeline.db import DatabaseManager
from schemas.llm import LLMRequest, LLMResponse, TokenUsage


def test_cache_miss_and_set(db_manager: DatabaseManager):
    """Verify cache miss returns None, and set persists the response."""
    cache = LLMCache(db_manager)

    request = LLMRequest(prompt="Extract facts from article", temperature=0.0)
    cache_key = request.compute_cache_key(provider="gemini", effective_model="gemini-2.5-flash")

    # Miss
    assert cache.get(cache_key) is None

    # Set
    original_resp = LLMResponse(
        text="[Fact 1, Fact 2]",
        provider="gemini",
        model="gemini-2.5-flash",
        usage=TokenUsage(prompt_tokens=25, completion_tokens=15, total_tokens=40),
        cached=False,
        latency_seconds=1.2,
    )
    cache.set(cache_key, original_resp)

    # Hit
    cached_resp = cache.get(cache_key)
    assert cached_resp is not None
    assert cached_resp.text == "[Fact 1, Fact 2]"
    assert cached_resp.provider == "gemini"
    assert cached_resp.model == "gemini-2.5-flash"
    assert cached_resp.cached is True
    assert cached_resp.usage.total_tokens == 40
    assert cached_resp.latency_seconds == 0.0


def test_cache_key_determinism():
    """Verify prompt hashing is deterministic and sensitive to model / schema differences."""
    req1 = LLMRequest(prompt="Story about Virat Kohli", temperature=0.0)
    req2 = LLMRequest(prompt="Story about Virat Kohli", temperature=0.0)
    req3 = LLMRequest(prompt="Story about Virat Kohli", temperature=0.7)
    req4 = LLMRequest(prompt="Story about Rohit Sharma", temperature=0.0)

    key1 = req1.compute_cache_key(provider="gemini", effective_model="gemini-2.5-flash")
    key2 = req2.compute_cache_key(provider="gemini", effective_model="gemini-2.5-flash")
    key3 = req3.compute_cache_key(provider="gemini", effective_model="gemini-2.5-flash")
    key4 = req4.compute_cache_key(provider="gemini", effective_model="gemini-2.5-flash")

    assert key1 == key2
    assert key1 != key3
    assert key1 != key4
