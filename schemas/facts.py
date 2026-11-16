from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from schemas.common import CorroborationStatus, FactType, SourceTier


class ExtractedFact(BaseModel):
    fact_id: str
    job_id: str
    claim: str
    fact_type: FactType = FactType.OTHER
    normalized_entities: List[str] = Field(default_factory=list)
    normalized_value: Optional[str] = None
    source_id: str
    evidence_quote: str
    source_tier: SourceTier = SourceTier.UNKNOWN
    published_at: Optional[str] = None
    corroborating_source_ids: List[str] = Field(default_factory=list)
    corroboration_status: CorroborationStatus = CorroborationStatus.UNVERIFIED
    requires_attribution: bool = False
    attribution_outlet: Optional[str] = None
    is_hedged: bool = False
    is_temporal: bool = False
    is_conflicted: bool = False
    discard_reason: Optional[str] = None

    @classmethod
    def create(
        cls,
        job_id: str,
        claim: str,
        source_id: str,
        evidence_quote: str,
        source_tier: SourceTier,
        fact_type: FactType = FactType.OTHER,
        normalized_entities: Optional[List[str]] = None,
        normalized_value: Optional[str] = None,
        published_at: Optional[str] = None,
        corroboration_status: CorroborationStatus = CorroborationStatus.UNVERIFIED,
    ) -> "ExtractedFact":
        """Generate a deterministic fact_id from claim and source."""
        raw_key = f"{claim.strip().lower()}|{evidence_quote.strip().lower()}|{source_id}"
        fid = f"fact_{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:12]}"
        return cls(
            fact_id=fid,
            job_id=job_id,
            claim=claim.strip(),
            fact_type=fact_type,
            normalized_entities=normalized_entities or [],
            normalized_value=normalized_value,
            source_id=source_id,
            evidence_quote=evidence_quote.strip(),
            source_tier=source_tier,
            published_at=published_at,
            corroboration_status=corroboration_status,
        )


class ConflictRecord(BaseModel):
    conflict_id: str
    job_id: str
    entity_or_topic: str
    fact_ids: List[str]
    conflicting_values: List[str]
    sources: List[str]
    resolved: bool = False
    chosen_fact_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FactLedger(BaseModel):
    job_id: str
    facts: List[ExtractedFact]
    conflicts: List[ConflictRecord] = Field(default_factory=list)
    verified_count: int = 0
    reported_count: int = 0
    discarded_count: int = 0
    unverified_count: int = 0


class ResolveConflictRequest(BaseModel):
    conflict_id: str
    chosen_fact_id: str
