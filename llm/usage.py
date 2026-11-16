from datetime import datetime, timezone
from typing import Optional

from app.logging_config import get_logger
from pipeline.db import DatabaseManager
from schemas.llm import LLMUsageRecord

logger = get_logger(__name__)


class LLMUsageTracker:
    """Tracks daily request and token usage in SQLite to enforce configured limits."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def get_daily_usage(
        self,
        provider: str,
        model: str,
        date_str: Optional[str] = None,
    ) -> LLMUsageRecord:
        d = date_str or self._today()
        with self.db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT usage_date, provider, model, requests_count, tokens_count
                FROM llm_usage
                WHERE usage_date = ? AND provider = ? AND model = ?
                """,
                (d, provider, model),
            ).fetchone()

            if not row:
                return LLMUsageRecord(
                    usage_date=d,
                    provider=provider,
                    model=model,
                    requests_count=0,
                    tokens_count=0,
                )

            return LLMUsageRecord(
                usage_date=row["usage_date"],
                provider=row["provider"],
                model=row["model"],
                requests_count=row["requests_count"],
                tokens_count=row["tokens_count"],
            )

    def can_make_request(
        self,
        provider: str,
        model: str,
        daily_limit: int,
        date_str: Optional[str] = None,
    ) -> bool:
        """Check if provider and model are within their configured daily limit."""
        if daily_limit <= 0:
            return False

        usage = self.get_daily_usage(provider, model, date_str)
        allowed = usage.requests_count < daily_limit
        if not allowed:
            logger.warning(
                "Daily request limit reached for %s/%s on %s: %d / %d",
                provider,
                model,
                usage.usage_date,
                usage.requests_count,
                daily_limit,
            )
        return allowed

    def record_usage(
        self,
        provider: str,
        model: str,
        tokens: int = 0,
        date_str: Optional[str] = None,
    ) -> None:
        """Increment daily request and token counters atomically."""
        d = date_str or self._today()
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO llm_usage (usage_date, provider, model, requests_count, tokens_count)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(usage_date, provider, model) DO UPDATE SET
                    requests_count = requests_count + 1,
                    tokens_count = tokens_count + excluded.tokens_count
                """,
                (d, provider, model, max(tokens, 0)),
            )
        logger.debug("Recorded usage for %s/%s on %s (+1 req, +%d tokens)", provider, model, d, tokens)
