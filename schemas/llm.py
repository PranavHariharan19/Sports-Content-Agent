from datetime import datetime
import hashlib
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    response_schema: Optional[Dict[str, Any]] = None

    def compute_cache_key(self, provider: str, effective_model: str) -> str:
        """Deterministically compute cache key for a prompt request and model."""
        payload = {
            "provider": provider,
            "model": effective_model,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt or "",
            "temperature": round(self.temperature, 2),
            "max_tokens": self.max_tokens,
            "response_schema": self.response_schema,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class LLMResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cached: bool = False
    latency_seconds: float = 0.0


class LLMCacheRecord(BaseModel):
    prompt_hash: str
    provider: str
    model: str
    response_text: str
    metadata_json: str
    created_at: str


class LLMUsageRecord(BaseModel):
    usage_date: str
    provider: str
    model: str
    requests_count: int = 0
    tokens_count: int = 0
