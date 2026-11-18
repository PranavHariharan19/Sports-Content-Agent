import asyncio
from typing import Dict, List, Optional

from app.logging_config import get_logger
from llm.router import LLMRouter
from pipeline.corroboration import CorroborationEngine
from pipeline.db import DatabaseManager
from pipeline.evidence_checker import DeterministicEvidenceChecker
from pipeline.facts_extractor import FactsExtractor
from pipeline.state_machine import PipelineStateMachine
from schemas.common import CorroborationStatus, JobStatus, SourceTier, StageName
from schemas.config import PipelineConfig
from schemas.facts import ExtractedFact, FactLedger
from schemas.sources import StoredSource

logger = get_logger(__name__)


class InsufficientVerifiedFactsError(Exception):
    """Raised when fewer than min_verified_facts exist for a topic (Fail Closed rule 6.10)."""
    pass


class FactsPipelineStage:
    """Orchestrates Stage 4: Fact extraction, deterministic evidence validation,

    corroboration, conflict handling, and fail-closed enforcement.
    """

    def __init__(
        self,
        db: DatabaseManager,
        llm_router: LLMRouter,
        pipeline_config: Optional[PipelineConfig] = None,
        state_machine: Optional[PipelineStateMachine] = None,
        evidence_checker: Optional[DeterministicEvidenceChecker] = None,
        corroboration_engine: Optional[CorroborationEngine] = None,
    ):
        self.db = db
        self.router = llm_router
        self.config = pipeline_config or PipelineConfig()
        self.state_machine = state_machine or PipelineStateMachine(db)
        self.extractor = FactsExtractor(llm_router)
        self.evidence_checker = evidence_checker or DeterministicEvidenceChecker(
            fuzzy_threshold=self.config.fuzzy_evidence_threshold
        )
        self.corroboration_engine = corroboration_engine or CorroborationEngine()

    async def process_facts_for_job(self, job_id: str) -> FactLedger:
        """Run the end-to-end fact extraction and anti-hallucination verification pipeline."""
        job = self.db.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        # Retrieve stored sources
        sources = self.db.get_sources_for_job(job_id)
        if not sources:
            raise ValueError(f"No stored sources found for job {job_id}")

        sources_by_id: Dict[str, StoredSource] = {s.source_id: s for s in sources}
        allowlisted_sources = [s for s in sources if s.is_allowlisted]

        # Prioritize substantial sources: Tier 1 & Tier 2 first, preserving stable order within tiers
        tier_order = {SourceTier.TIER_1: 0, SourceTier.TIER_2: 1, SourceTier.TIER_3: 2, SourceTier.UNKNOWN: 3}
        substantial = [s for s in allowlisted_sources if s.word_count >= 40] or allowlisted_sources
        sorted_sources = sorted(
            substantial,
            key=lambda s: tier_order.get(s.source_tier, 3),
        )

        # Select top distinct-domain substantial sources to maximize corroboration
        seen_domains = set()
        sources_to_extract: List[StoredSource] = []
        for s in sorted_sources:
            if s.domain not in seen_domains:
                seen_domains.add(s.domain)
                sources_to_extract.append(s)
            if len(sources_to_extract) >= 3:
                break
        if len(sources_to_extract) < 3:
            for s in sorted_sources:
                if s not in sources_to_extract:
                    sources_to_extract.append(s)
                if len(sources_to_extract) >= 3:
                    break

        logger.info(
            "Extracting facts for job %s from %d sources (%d selected across %d domains)",
            job_id,
            len(sources),
            len(sources_to_extract),
            len(seen_domains),
        )

        # 1. Sequential Fact Extraction (Section 6.1) to avoid free-tier concurrency throttles
        all_candidate_facts: List[ExtractedFact] = []
        for src in sources_to_extract:
            facts = await self.extractor.extract_facts_from_source(src, job_id)
            all_candidate_facts.extend(facts)
            await asyncio.sleep(0.5)

        # 2. Deterministic Evidence Check (Section 6.3)
        valid_facts, discarded_facts = self.evidence_checker.validate_facts(all_candidate_facts, sources_by_id)

        # 3. Detect Conflicts (Section 6.5)
        conflicts = self.corroboration_engine.detect_conflicts(valid_facts, sources_by_id, job_id)
        for conflict in conflicts:
            self.db.save_conflict(conflict)

        # 4. Evaluate Corroboration & Trust Tiers (Section 6.4)
        corroborated_facts = self.corroboration_engine.evaluate_corroboration(valid_facts, sources_by_id)

        # 5. Persist all facts into SQLite
        for fact in corroborated_facts:
            self.db.save_fact(fact)
        for fact in discarded_facts:
            self.db.save_fact(fact)

        # Count statuses
        verified_count = sum(1 for f in corroborated_facts if f.corroboration_status == CorroborationStatus.VERIFIED)
        reported_count = sum(1 for f in corroborated_facts if f.corroboration_status == CorroborationStatus.REPORTED)
        unverified_count = sum(1 for f in corroborated_facts if f.corroboration_status == CorroborationStatus.UNVERIFIED)
        usable_count = verified_count + reported_count

        min_facts = self.config.min_verified_facts

        # 6. Fail-Closed Enforcement (Section 6.10)
        if usable_count < min_facts:
            err_msg = (
                f"Fail-closed: Only {usable_count} usable verified/reported facts extracted "
                f"(minimum required: {min_facts}). Discarded: {len(discarded_facts)}."
            )
            logger.warning(err_msg)
            self.state_machine.record_stage_execution(
                job_id=job_id,
                stage_name=StageName.FACTS,
                input_data={"candidate_facts_count": len(all_candidate_facts)},
                output_data={"error": err_msg, "usable_count": usable_count},
                success=False,
                error_message=err_msg,
                allow_rerun=True,
            )
            raise InsufficientVerifiedFactsError(err_msg)

        # Build ledger
        ledger = FactLedger(
            job_id=job_id,
            facts=corroborated_facts,
            conflicts=conflicts,
            verified_count=verified_count,
            reported_count=reported_count,
            discarded_count=len(discarded_facts),
            unverified_count=unverified_count,
        )

        # Record stage run and advance state to FACTS_VERIFIED
        self.state_machine.record_stage_execution(
            job_id=job_id,
            stage_name=StageName.FACTS,
            input_data={"candidate_facts_count": len(all_candidate_facts)},
            output_data=ledger.model_dump(mode="json"),
            success=True,
            allow_rerun=True,
        )

        logger.info(
            "Fact verification complete for job %s: %d verified, %d reported, %d conflicts, %d discarded",
            job_id,
            verified_count,
            reported_count,
            len(conflicts),
            len(discarded_facts),
        )
        return ledger
