from datetime import datetime, timezone
from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from schemas.common import VerificationVerdict


class BeatType(str, Enum):
    HOOK = "hook"
    FACTUAL_BEAT = "factual_beat"
    CLOSING = "closing"


class ScriptBeat(BaseModel):
    beat_id: int
    beat_type: BeatType = BeatType.FACTUAL_BEAT
    spoken_text: str
    is_factual: bool = True
    cited_fact_ids: List[str] = Field(default_factory=list)
    verifier_verdict: VerificationVerdict = VerificationVerdict.SUPPORTED
    verifier_reason: Optional[str] = None
    lint_errors: List[str] = Field(default_factory=list)
    word_count: int = 0
    estimated_duration_seconds: float = 0.0
    human_override: bool = False
    human_override_reason: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if not self.word_count and self.spoken_text:
            self.word_count = len(self.spoken_text.split())
        if not self.estimated_duration_seconds and self.word_count:
            # ~2.5 words per second for fast-paced shorts pacing
            self.estimated_duration_seconds = round(self.word_count / 2.5, 1)


class GeneratedScript(BaseModel):
    script_id: str
    job_id: str
    headline: str
    beats: List[ScriptBeat]
    total_words: int = 0
    estimated_duration_seconds: float = 0.0
    model_used: str = "gemini-2.5-flash"
    verifier_model_used: str = "llama-3.3-70b-versatile"
    lint_passed: bool = True
    all_supported: bool = True
    is_approved: bool = False
    approved_at: Optional[str] = None
    override_logs: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def model_post_init(self, __context: Any) -> None:
        if not self.total_words and self.beats:
            self.total_words = sum(b.word_count for b in self.beats)
        if not self.estimated_duration_seconds and self.beats:
            self.estimated_duration_seconds = round(sum(b.estimated_duration_seconds for b in self.beats), 1)
        self.all_supported = all(
            b.verifier_verdict == VerificationVerdict.SUPPORTED or b.human_override
            for b in self.beats
        )
        self.lint_passed = all(len(b.lint_errors) == 0 or b.human_override for b in self.beats)


class BeatOverrideRequest(BaseModel):
    beat_id: int
    action: str = Field(..., description="'approve', 'edit', or 'reject'")
    edited_text: Optional[str] = None
    override_reason: str = Field(..., min_length=3, description="Logged reason for overriding automated checks")


class ScriptApproveRequest(BaseModel):
    job_id: str
