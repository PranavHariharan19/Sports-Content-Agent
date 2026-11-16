from datetime import datetime, timezone
import json
from typing import List, Optional

from app.logging_config import get_logger
from llm.router import LLMRouter
from pipeline.pageviews import GoogleTrendsClient, WikipediaPageviewsClient
from pipeline.sources_trust import SourceTrustManager
from schemas.common import SourceTier, SportType
from schemas.config import PipelineConfig
from schemas.discovery import StoryCluster, ViralityBreakdown
from schemas.llm import LLMRequest

logger = get_logger(__name__)


class ViralityRankingEngine:
    """Computes deterministic virality scores and ranks story candidates."""

    def __init__(
        self,
        trust_manager: SourceTrustManager,
        pipeline_config: Optional[PipelineConfig] = None,
        llm_router: Optional[LLMRouter] = None,
        pageviews_client: Optional[WikipediaPageviewsClient] = None,
        trends_client: Optional[GoogleTrendsClient] = None,
    ):
        self.trust_manager = trust_manager
        self.config = pipeline_config or PipelineConfig()
        self.llm_router = llm_router
        self.pageviews_client = pageviews_client or WikipediaPageviewsClient()
        self.trends_client = trends_client or GoogleTrendsClient()

    def compute_recency_score(self, age_hours: float) -> float:
        """Calculate recency score (0-25) using linear decay over max_story_age_hours."""
        max_age = float(self.config.max_story_age_hours)
        if age_hours >= max_age:
            return 0.0
        ratio = max(0.0, 1.0 - (age_hours / max_age))
        return round(25.0 * ratio, 2)

    def compute_sources_score(self, cluster: StoryCluster) -> float:
        """Calculate score (0-25) based on independent domains and trust tiers."""
        domain_count = len(cluster.independent_domains)
        if domain_count == 0:
            return 0.0

        # Base score from number of distinct domains: 1 -> 8, 2 -> 15, 3 -> 20, 4+ -> 24
        base = min(24.0, domain_count * 6.0)

        # Bonus for presence of Tier 1 official body
        has_tier_1 = any(a.source_tier == SourceTier.TIER_1 for a in cluster.articles)
        tier_1_bonus = 2.0 if has_tier_1 else 0.0

        return round(min(25.0, base + tier_1_bonus), 2)

    def compute_reddit_velocity_score(self, cluster: StoryCluster) -> float:
        """Calculate social discussion velocity (0-25) from Reddit upvotes/comments."""
        total_activity = sum(
            a.reddit_score + (a.reddit_comments * 2.0)
            for a in cluster.articles
        )
        if total_activity <= 0:
            return 0.0

        # Activity per hour
        safe_age = max(0.5, cluster.age_hours)
        velocity = total_activity / safe_age

        # Scale velocity to 0-25 (velocity of 50+/hr reaches max)
        score = min(25.0, (velocity / 50.0) * 25.0)
        return round(score, 2)

    def compute_star_weight_score(self, cluster: StoryCluster, sport: SportType) -> float:
        """Calculate star entity relevance weight (0-15)."""
        star_entities = {e.lower() for e in self.trust_manager.get_star_entities(sport)}
        matched = [e for e in cluster.entities if e.lower() in star_entities]

        if not matched:
            return 0.0
        elif len(matched) == 1:
            return 8.0
        else:
            return 15.0

    async def compute_pageview_score(self, cluster: StoryCluster) -> float:
        """Query Wikipedia pageviews for cluster entities (0-10)."""
        if not cluster.entities:
            return 0.0

        # Check the primary entity
        primary_entity = cluster.entities[0]
        views = await self.pageviews_client.get_pageviews(primary_entity, days=7)
        return self.pageviews_client.compute_pageview_score(views)

    async def compute_llm_emotion_subscore(self, cluster: StoryCluster) -> Optional[float]:
        """Optional LLM sub-score (0-10) for emotional impact or controversy."""
        if not self.llm_router:
            return None

        prompt = (
            f"Headline: {cluster.primary_headline}\n"
            f"Entities: {', '.join(cluster.entities)}\n"
            "Score this sports story on a scale of 0.0 to 10.0 for controversy, fan emotional reaction, "
            "and debate potential. Return JSON: {\"emotion_score\": <number between 0.0 and 10.0>}."
        )

        req = LLMRequest(
            prompt=prompt,
            system_prompt="You are a sports engagement analyst. Output valid JSON only.",
            temperature=0.0,
            response_schema={"type": "object", "properties": {"emotion_score": {"type": "number"}}},
        )

        try:
            resp = await self.llm_router.generate(req)
            data = json.loads(resp.text)
            score = float(data.get("emotion_score", 5.0))
            return round(min(10.0, max(0.0, score)), 2)
        except Exception as e:
            logger.debug("LLM emotion subscore failed for '%s': %s", cluster.primary_headline, str(e))
            return None

    async def score_and_rank_clusters(
        self,
        clusters: List[StoryCluster],
        sport: SportType,
        include_llm_emotion: bool = False,
    ) -> List[StoryCluster]:
        """Score each cluster deterministically, filter expired stories, and return top candidates."""
        max_age = float(self.config.max_story_age_hours)
        candidate_count = self.config.candidate_stories_count

        scored_clusters: List[StoryCluster] = []
        for cluster in clusters:
            # Exclude stories older than max_age window
            if cluster.age_hours > max_age:
                continue

            recency = self.compute_recency_score(cluster.age_hours)
            sources = self.compute_sources_score(cluster)
            reddit = self.compute_reddit_velocity_score(cluster)
            star = self.compute_star_weight_score(cluster, sport)
            pageviews = await self.compute_pageview_score(cluster)

            deterministic_total = round(recency + sources + reddit + star + pageviews, 2)
            deterministic_total = min(100.0, max(0.0, deterministic_total))

            emotion_score: Optional[float] = None
            if include_llm_emotion:
                emotion_score = await self.compute_llm_emotion_subscore(cluster)

            if emotion_score is not None:
                # Blend 90% deterministic + 10% LLM subjective
                final = round(min(100.0, (0.90 * deterministic_total) + (1.0 * emotion_score)), 1)
            else:
                final = round(deterministic_total, 1)

            cluster.virality = ViralityBreakdown(
                recency_score=recency,
                sources_score=sources,
                reddit_velocity_score=reddit,
                star_weight_score=star,
                pageview_interest_score=pageviews,
                deterministic_score=deterministic_total,
                llm_emotion_score=emotion_score,
                final_score=final,
            )
            scored_clusters.append(cluster)

        # Sort descending by final virality score
        scored_clusters.sort(key=lambda c: c.virality.final_score, reverse=True)

        top_candidates = scored_clusters[:candidate_count]
        logger.info(
            "Ranked %d clusters for %s, selected top %d candidates (highest score: %.1f)",
            len(scored_clusters),
            sport.value,
            len(top_candidates),
            top_candidates[0].virality.final_score if top_candidates else 0.0,
        )
        return top_candidates
