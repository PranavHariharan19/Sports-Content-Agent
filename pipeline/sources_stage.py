import asyncio
from typing import Any, Dict, List, Optional

from app.logging_config import get_logger
from pipeline.db import DatabaseManager
from pipeline.sources_fetcher import SourceFetcher
from pipeline.sources_trust import SourceTrustManager
from pipeline.state_machine import PipelineStateMachine
from pipeline.syndication import SyndicationDeduplicator
from schemas.common import JobStatus, SourceTier, StageName
from schemas.sources import SourceFetchResult, StoredSource

logger = get_logger(__name__)


class SourcesPipelineStage:
    """Orchestrates source fetching, trust tier assignment, prompt-injection sanitization,

    syndication de-duplication, and ground-truth text persistence in SQLite.
    """

    def __init__(
        self,
        db: DatabaseManager,
        state_machine: Optional[PipelineStateMachine] = None,
        trust_manager: Optional[SourceTrustManager] = None,
        fetcher: Optional[SourceFetcher] = None,
        deduplicator: Optional[SyndicationDeduplicator] = None,
    ):
        self.db = db
        self.state_machine = state_machine or PipelineStateMachine(db)
        self.trust_manager = trust_manager or SourceTrustManager()
        self.fetcher = fetcher or SourceFetcher(self.trust_manager)
        self.deduplicator = deduplicator or SyndicationDeduplicator()

    async def fetch_and_store_sources_for_job(
        self,
        job_id: str,
        article_overrides: Optional[List[Dict[str, Any]]] = None,
    ) -> SourceFetchResult:
        """Fetch all articles associated with the chosen topic and persist them as ground truth."""
        job = self.db.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        # Retrieve chosen topic articles
        articles_to_fetch: List[Dict[str, Any]] = []
        if article_overrides:
            articles_to_fetch = article_overrides
        else:
            # Look up chosen topic from ranking stage runs
            stage_runs = self.db.get_stage_runs_for_job(job_id)
            for sr in reversed(stage_runs):
                output = sr.get_output()
                if "chosen_topic" in output:
                    articles_to_fetch = output["chosen_topic"].get("articles", [])
                    break

            if not articles_to_fetch:
                raise ValueError(f"No chosen topic or articles found in job {job_id} history")

        logger.info("Fetching %d source articles for job %s", len(articles_to_fetch), job_id)

        # Concurrently fetch and process article web pages
        tasks = [
            self.fetcher.fetch_and_process_source(
                job_id=job_id,
                url=art.get("url", ""),
                title=art.get("title"),
                published_at=art.get("published_at"),
            )
            for art in articles_to_fetch
            if art.get("url")
        ]

        sources = await asyncio.gather(*tasks)

        # Run syndication de-duplication
        deduped_sources = self.deduplicator.deduplicate(list(sources))

        # Persist all sources in SQLite
        for src in deduped_sources:
            self.db.save_source(src)

        # Count metrics
        tier_1 = sum(1 for s in deduped_sources if s.source_tier == SourceTier.TIER_1)
        tier_2 = sum(1 for s in deduped_sources if s.source_tier == SourceTier.TIER_2)
        tier_3 = sum(1 for s in deduped_sources if s.source_tier == SourceTier.TIER_3)
        allowlisted = sum(1 for s in deduped_sources if s.is_allowlisted)
        syndicated_groups = len({s.syndication_cluster_id for s in deduped_sources if s.is_syndicated and s.syndication_cluster_id})

        result = SourceFetchResult(
            job_id=job_id,
            total_sources=len(deduped_sources),
            allowlisted_sources=allowlisted,
            tier_1_count=tier_1,
            tier_2_count=tier_2,
            tier_3_count=tier_3,
            syndicated_groups_count=syndicated_groups,
            sources=deduped_sources,
        )

        # Record stage run in SQLite
        self.state_machine.record_stage_execution(
            job_id=job_id,
            stage_name=StageName.SOURCES,
            input_data={"articles_to_fetch": len(articles_to_fetch)},
            output_data=result.model_dump(mode="json"),
            success=True,
            allow_rerun=True,
        )

        logger.info(
            "Source stage complete for job %s: %d sources stored (%d allowlisted, %d syndicated groups)",
            job_id,
            len(deduped_sources),
            allowlisted,
            syndicated_groups,
        )
        return result
