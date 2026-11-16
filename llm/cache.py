from datetime import datetime, timezone
import json
from typing import Optional

from app.logging_config import get_logger
from pipeline.db import DatabaseManager
from schemas.llm import LLMCacheRecord, LLMResponse, TokenUsage

logger = get_logger(__name__)


class LLMCache:
    """SQLite-backed response cache for LLM requests."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def get(self, prompt_hash: str) -> Optional[LLMResponse]:
        """Retrieve a cached response by prompt_hash."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT prompt_hash, provider, model, response_text, metadata_json, created_at FROM llm_cache WHERE prompt_hash = ?",
                (prompt_hash,),
            ).fetchone()

            if not row:
                return None

            try:
                meta = json.loads(row["metadata_json"])
            except Exception:
                meta = {}

            usage_dict = meta.get("usage", {})
            usage = TokenUsage(
                prompt_tokens=usage_dict.get("prompt_tokens", 0),
                completion_tokens=usage_dict.get("completion_tokens", 0),
                total_tokens=usage_dict.get("total_tokens", 0),
            )

            logger.debug("Cache hit for hash %s (%s/%s)", prompt_hash, row["provider"], row["model"])
            return LLMResponse(
                text=row["response_text"],
                provider=row["provider"],
                model=row["model"],
                usage=usage,
                cached=True,
                latency_seconds=0.0,
            )

    def set(self, prompt_hash: str, response: LLMResponse) -> None:
        """Store an LLM response in cache."""
        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "usage": response.usage.model_dump(),
            "original_latency": response.latency_seconds,
        }
        meta_json = json.dumps(metadata)

        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache (prompt_hash, provider, model, response_text, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (prompt_hash, response.provider, response.model, response.text, meta_json, now),
            )
            logger.debug("Cached response for hash %s (%s/%s)", prompt_hash, response.provider, response.model)

    def clear(self) -> int:
        """Clear all cached responses. Returns number of rows deleted."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("DELETE FROM llm_cache")
            return cursor.rowcount
