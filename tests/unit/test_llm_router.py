import pytest
from llm.base import LLMProviderError, LLMRateLimitError
from llm.mock_provider import MockLLMProvider
from llm.router import AllProvidersFailedError, LLMRouter
from pipeline.db import DatabaseManager
from schemas.config import LLMConfig, ProviderSettings
from schemas.llm import LLMRequest


@pytest.fixture
def router_setup(db_manager: DatabaseManager):
    config = LLMConfig(
        fallback_order=["primary", "secondary"],
        cache_enabled=True,
        timeout_seconds=2.0,
        max_retries_per_provider=2,
        providers={
            "primary": ProviderSettings(model="p-model", daily_request_limit=2),
            "secondary": ProviderSettings(model="s-model", daily_request_limit=10),
        },
    )
    p_primary = MockLLMProvider(name="primary", default_model="p-model", responses=["Primary Answer"])
    p_secondary = MockLLMProvider(name="secondary", default_model="s-model", responses=["Secondary Answer"])

    router = LLMRouter(
        config=config,
        db=db_manager,
        custom_providers={"primary": p_primary, "secondary": p_secondary},
    )
    return router, p_primary, p_secondary, db_manager


@pytest.mark.asyncio
async def test_router_success_and_caching(router_setup):
    """Verify primary provider is used and subsequent identical calls hit cache."""
    router, p_primary, p_secondary, db = router_setup
    req = LLMRequest(prompt="Who won the 2026 Bahrain GP?")

    # 1st call: uses primary
    res1 = await router.generate(req)
    assert res1.text == "Primary Answer"
    assert res1.provider == "primary"
    assert res1.cached is False
    assert len(p_primary.call_history) == 1

    # 2nd call: hits cache
    res2 = await router.generate(req)
    assert res2.text == "Primary Answer"
    assert res2.cached is True
    assert len(p_primary.call_history) == 1  # No additional network call


@pytest.mark.asyncio
async def test_router_fallback_on_rate_limit(router_setup):
    """Verify automatic fallback to secondary provider when primary hits rate limit."""
    router, p_primary, p_secondary, _ = router_setup
    p_primary.simulate_rate_limit = True

    req = LLMRequest(prompt="Latest Premier League scores")
    res = await router.generate(req)

    assert res.provider == "secondary"
    assert res.text == "Secondary Answer"
    assert len(p_primary.call_history) == 1
    assert len(p_secondary.call_history) == 1


@pytest.mark.asyncio
async def test_router_fallback_on_daily_quota_exhausted(router_setup):
    """Verify automatic fallback when primary provider exhausts daily quota."""
    router, p_primary, p_secondary, db = router_setup

    # Exhaust primary quota (limit is 2)
    router.usage_tracker.record_usage("primary", "p-model", tokens=50)
    router.usage_tracker.record_usage("primary", "p-model", tokens=50)

    req = LLMRequest(prompt="Cricket World Cup schedule")
    res = await router.generate(req)

    # Primary should be skipped because of quota
    assert res.provider == "secondary"
    assert res.text == "Secondary Answer"
    assert len(p_primary.call_history) == 0
    assert len(p_secondary.call_history) == 1


@pytest.mark.asyncio
async def test_router_all_providers_failed(router_setup):
    """Verify AllProvidersFailedError is raised when all providers in the chain fail."""
    router, p_primary, p_secondary, _ = router_setup
    p_primary.simulate_error = True
    p_secondary.simulate_error = True

    req = LLMRequest(prompt="Breaking sports news")
    with pytest.raises(AllProvidersFailedError):
        await router.generate(req)


@pytest.mark.asyncio
async def test_router_batch_generation(router_setup):
    """Verify generate_batch processes multiple requests concurrently."""
    router, p_primary, p_secondary, _ = router_setup
    # Increase primary daily limit so all 3 requests can use primary
    router.config.providers["primary"].daily_request_limit = 10
    p_primary.responses = ["Answer 1", "Answer 2", "Answer 3"]

    reqs = [
        LLMRequest(prompt="Story 1"),
        LLMRequest(prompt="Story 2"),
        LLMRequest(prompt="Story 3"),
    ]
    results = await router.generate_batch(reqs, max_concurrency=2)

    assert len(results) == 3
    assert [r.text for r in results] == ["Answer 1", "Answer 2", "Answer 3"]

