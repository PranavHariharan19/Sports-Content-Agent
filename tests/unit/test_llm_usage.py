from llm.usage import LLMUsageTracker
from pipeline.db import DatabaseManager


def test_usage_tracking_and_daily_limits(db_manager: DatabaseManager):
    """Verify daily request and token counters increment and enforce limits."""
    tracker = LLMUsageTracker(db_manager)
    date_str = "2026-11-16"
    provider = "gemini"
    model = "gemini-2.5-flash"
    daily_limit = 3

    # Initially 0 usage, within limit
    assert tracker.can_make_request(provider, model, daily_limit, date_str=date_str) is True

    # Record 2 requests
    tracker.record_usage(provider, model, tokens=100, date_str=date_str)
    tracker.record_usage(provider, model, tokens=150, date_str=date_str)

    usage = tracker.get_daily_usage(provider, model, date_str=date_str)
    assert usage.requests_count == 2
    assert usage.tokens_count == 250
    assert tracker.can_make_request(provider, model, daily_limit, date_str=date_str) is True

    # 3rd request reaches limit
    tracker.record_usage(provider, model, tokens=50, date_str=date_str)
    usage = tracker.get_daily_usage(provider, model, date_str=date_str)
    assert usage.requests_count == 3
    assert usage.tokens_count == 300

    # 4th request blocked by daily limit
    assert tracker.can_make_request(provider, model, daily_limit, date_str=date_str) is False
