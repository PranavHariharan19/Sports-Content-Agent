import difflib
import re
from typing import Dict, List, Optional, Set, Tuple
import uuid

from app.logging_config import get_logger
from schemas.common import CorroborationStatus, FactType, SourceTier
from schemas.facts import ConflictRecord, ExtractedFact
from schemas.sources import StoredSource

logger = get_logger(__name__)

HEDGING_KEYWORDS = {
    "rumor", "rumour", "target", "targets", "negotiating", "negotiation", "talks",
    "linked", "considering", "weighing", "injury", "doubt", "expected", "alleged",
    "could", "might", "bid", "interest",
}

TEMPORAL_KEYWORDS = {"currently", "latest", "record", "now", "recent", "all-time"}


class CorroborationEngine:
    """Evaluates fact corroboration, detects conflicts, enforces trust tiers, and handles hedging."""

    def __init__(self, similarity_threshold: float = 0.70):
        self.similarity_threshold = similarity_threshold

    def are_claims_similar(self, claim1: str, claim2: str) -> float:
        """Measure semantic token similarity between two claims."""
        tokens1 = set(re.findall(r"\b[a-zA-Z0-9]+\b", claim1.lower()))
        tokens2 = set(re.findall(r"\b[a-zA-Z0-9]+\b", claim2.lower()))
        if not tokens1 or not tokens2:
            return 0.0
        return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

    def detect_conflicts(
        self,
        facts: List[ExtractedFact],
        sources_by_id: Dict[str, StoredSource],
        job_id: str,
    ) -> List[ConflictRecord]:
        """Detect conflicting numbers, scores, dates, or transfer values across independent sources."""
        conflicts: List[ConflictRecord] = []
        n = len(facts)

        for i in range(n):
            for j in range(i + 1, n):
                f1 = facts[i]
                f2 = facts[j]

                # Only compare facts from different sources
                if f1.source_id == f2.source_id:
                    continue

                # Check if claims are talking about the same metric / event
                claim_sim = self.are_claims_similar(f1.claim, f2.claim)
                set_ent1 = {e.lower() for e in f1.normalized_entities}
                set_ent2 = {e.lower() for e in f2.normalized_entities}
                shared_entities = set_ent1.intersection(set_ent2)

                is_same_score = (f1.fact_type == FactType.SCORE and f2.fact_type == FactType.SCORE and bool(shared_entities))
                if not is_same_score and claim_sim < 0.40:
                    continue

                # Check for numerical / score / value disagreement
                val1 = f1.normalized_value or self._extract_key_metric(f1.claim)
                val2 = f2.normalized_value or self._extract_key_metric(f2.claim)

                if val1 and val2 and val1 != val2 and f1.fact_type == f2.fact_type:
                    # Found a conflict
                    f1.is_conflicted = True
                    f2.is_conflicted = True
                    f1.corroboration_status = CorroborationStatus.CONFLICTED
                    f2.corroboration_status = CorroborationStatus.CONFLICTED

                    s1 = sources_by_id.get(f1.source_id)
                    s2 = sources_by_id.get(f2.source_id)
                    conflict_id = f"conf_{uuid.uuid4().hex[:8]}"

                    conflicts.append(
                        ConflictRecord(
                            conflict_id=conflict_id,
                            job_id=job_id,
                            entity_or_topic=", ".join(shared_entities) or f1.fact_type.value,
                            fact_ids=[f1.fact_id, f2.fact_id],
                            conflicting_values=[val1, val2],
                            sources=[s1.domain if s1 else "Source 1", s2.domain if s2 else "Source 2"],
                            resolved=False,
                        )
                    )
                    logger.warning(
                        "Conflict detected for job %s: '%s' vs '%s' (sources: %s vs %s)",
                        job_id,
                        val1,
                        val2,
                        s1.domain if s1 else "?",
                        s2.domain if s2 else "?",
                    )

        return conflicts

    def _extract_key_metric(self, claim: str) -> Optional[str]:
        """Extract primary score or number from a claim."""
        score_match = re.search(r"\b\d+\s*[-–]\s*\d+\b", claim)
        if score_match:
            return score_match.group(0).replace("–", "-")
        nums = re.findall(r"\b\d+(?:[\.,]\d+)?\b", claim)
        return nums[0] if nums else None

    def evaluate_corroboration(
        self,
        facts: List[ExtractedFact],
        sources_by_id: Dict[str, StoredSource],
    ) -> List[ExtractedFact]:
        """Apply corroboration rules, trust tiers, hedging, and temporal flags."""
        for fact in facts:
            source = sources_by_id.get(fact.source_id)
            if not source:
                fact.corroboration_status = CorroborationStatus.UNVERIFIED
                continue

            # Check for hedging triggers (transfers, injuries, contract speculation)
            claim_lower = fact.claim.lower()
            if fact.fact_type in (FactType.TRANSFER, FactType.INJURY) or any(k in claim_lower for k in HEDGING_KEYWORDS):
                fact.is_hedged = True

            # Check for temporal triggers ("latest", "currently", "record")
            if any(k in claim_lower for k in TEMPORAL_KEYWORDS):
                fact.is_temporal = True

            # If already marked conflicted, keep as conflicted
            if fact.is_conflicted:
                fact.corroboration_status = CorroborationStatus.CONFLICTED
                continue

            # Find corroborating sources for this fact
            corroborating_sources: Set[str] = {fact.source_id}
            seen_syndication_groups: Set[str] = set()
            if source.is_syndicated and source.syndication_cluster_id:
                seen_syndication_groups.add(source.syndication_cluster_id)

            for other_fact in facts:
                if other_fact.fact_id == fact.fact_id or other_fact.source_id == fact.source_id:
                    continue

                # Compare claims
                if self.are_claims_similar(fact.claim, other_fact.claim) >= self.similarity_threshold:
                    other_src = sources_by_id.get(other_fact.source_id)
                    if other_src and other_src.is_allowlisted:
                        # Syndication check: multiple copies of the same wire count as ONE source
                        if other_src.is_syndicated and other_src.syndication_cluster_id:
                            if other_src.syndication_cluster_id not in seen_syndication_groups:
                                seen_syndication_groups.add(other_src.syndication_cluster_id)
                                corroborating_sources.add(other_fact.source_id)
                        else:
                            corroborating_sources.add(other_fact.source_id)

            fact.corroborating_source_ids = list(corroborating_sources)

            # Apply Corroboration Rules (Section 6.4)
            # Rule 1: Backed by 2+ independent allowlisted sources OR 1 Tier 1 source -> VERIFIED
            has_tier_1 = (source.source_tier == SourceTier.TIER_1) or any(
                sources_by_id.get(sid) and sources_by_id[sid].source_tier == SourceTier.TIER_1
                for sid in corroborating_sources
            )

            if len(corroborating_sources) >= 2 or has_tier_1:
                fact.corroboration_status = CorroborationStatus.VERIFIED
                fact.requires_attribution = False
            # Rule 2: Single-source Tier 2 fact -> REPORTED (requires explicit attribution)
            elif source.source_tier == SourceTier.TIER_2:
                fact.corroboration_status = CorroborationStatus.REPORTED
                fact.requires_attribution = True
                fact.attribution_outlet = source.domain
            # Rule 3: Tier 3-only or unknown -> UNVERIFIED (unusable in script)
            else:
                fact.corroboration_status = CorroborationStatus.UNVERIFIED
                fact.requires_attribution = True

        return facts
