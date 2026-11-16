from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from schemas.common import SourceTier


class StoredSource(BaseModel):
    source_id: str
    job_id: str
    url: str
    domain: str
    source_tier: SourceTier
    is_allowlisted: bool = True
    title: Optional[str] = None
    cleaned_text: str
    text_hash: str = Field(default_factory=str)
    published_at: Optional[str] = None
    fetched_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    syndication_cluster_id: Optional[str] = None
    is_syndicated: bool = False
    char_count: int = 0
    word_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        if not self.text_hash and self.cleaned_text:
            self.text_hash = hashlib.sha256(self.cleaned_text.encode("utf-8")).hexdigest()
        if not self.char_count:
            self.char_count = len(self.cleaned_text)
        if not self.word_count and self.cleaned_text:
            self.word_count = len(self.cleaned_text.split())


class SourceFetchResult(BaseModel):
    job_id: str
    total_sources: int
    allowlisted_sources: int
    tier_1_count: int
    tier_2_count: int
    tier_3_count: int
    syndicated_groups_count: int
    sources: List[StoredSource]
