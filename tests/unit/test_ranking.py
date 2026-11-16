from datetime import datetime, timezone
import pytest
from pipeline.ranking import ViralityRankingEngine
from pipeline.sources_trust import SourceTrustManager
from schemas.common import SourceTier, SportType
from schemas.config import PipelineConfig
from schemas.discovery import DiscoveredArticle, StoryCluster, ViralityBreakdown


@pytest.fixture
def ranking_engine():
    stm = SourceTrustManager()
    cfg = PipelineConfig(max_story_age_hours=24, candidate_stories_count=10)
    return ViralityRankingEngine(trust_manager=stm, pipeline_config=cfg)


def test_recency_score_decay(ranking_engine: ViralityRankingEngine):
    """Verify linear decay of recency score from 25 to 0."""
    score_0h = ranking_engine.compute_recency_score(0.0)
    score_12h = ranking_engine.compute_recency_score(12.0)
    score_24h = ranking_engine.compute_recency_score(24.0)

    assert score_0h == 25.0
    assert 12.0 <= score_12h <= 13.0
    assert score_24h == 0.0


def test_sources_score_and_tier1_bonus(ranking_engine: ViralityRankingEngine):
    """Verify independent sources score scales with domain count and grants tier 1 bonus."""
    now = datetime.now(timezone.utc)
    cluster = StoryCluster(
        cluster_id="c1",
        sport=SportType.F1,
        primary_headline="Ferrari news",
        articles=[
            DiscoveredArticle(
                title="Ferrari news", url="https://formula1.com/1", domain="formula1.com",
                published_at=now, source_tier=SourceTier.TIER_1, sport=SportType.F1,
            ),
            DiscoveredArticle(
                title="Ferrari update", url="https://bbc.com/2", domain="bbc.com",
                published_at=now, source_tier=SourceTier.TIER_2, sport=SportType.F1,
            ),
        ],
        entities=["Ferrari"],
        source_count=2,
        independent_domains=["formula1.com", "bbc.com"],
        age_hours=1.0,
        virality=ViralityBreakdown(),
    )

    score = ranking_engine.compute_sources_score(cluster)
    # 2 domains = 12 + 2 bonus = 14
    assert score == 14.0


@pytest.mark.asyncio
async def test_score_and_rank_clusters_ordering(ranking_engine: ViralityRankingEngine):
    """Verify clusters are sorted in descending order of final virality score."""
    now = datetime.now(timezone.utc)
    # High virality cluster: 2 domains, star entity "Lewis Hamilton", low age
    high_cluster = StoryCluster(
        cluster_id="high",
        sport=SportType.F1,
        primary_headline="Lewis Hamilton claims stunning victory with Ferrari in Monza",
        articles=[
            DiscoveredArticle(
                title="Lewis Hamilton wins", url="https://formula1.com/1", domain="formula1.com",
                published_at=now, source_tier=SourceTier.TIER_1, sport=SportType.F1,
                reddit_score=150, reddit_comments=45,
            ),
            DiscoveredArticle(
                title="Hamilton Ferrari glory", url="https://bbc.com/2", domain="bbc.com",
                published_at=now, source_tier=SourceTier.TIER_2, sport=SportType.F1,
                reddit_score=80, reddit_comments=20,
            ),
        ],
        entities=["Lewis Hamilton", "Ferrari"],
        source_count=2,
        independent_domains=["formula1.com", "bbc.com"],
        age_hours=1.0,
        virality=ViralityBreakdown(),
    )

    # Low virality cluster: single domain, 20 hours old, unknown entities
    low_cluster = StoryCluster(
        cluster_id="low",
        sport=SportType.F1,
        primary_headline="Minor technical regulation clarification announced",
        articles=[
            DiscoveredArticle(
                title="Minor technical update", url="https://autosport.com/1", domain="autosport.com",
                published_at=now, source_tier=SourceTier.TIER_2, sport=SportType.F1,
            )
        ],
        entities=[],
        source_count=1,
        independent_domains=["autosport.com"],
        age_hours=20.0,
        virality=ViralityBreakdown(),
    )

    ranked = await ranking_engine.score_and_rank_clusters([low_cluster, high_cluster], SportType.F1)
    assert len(ranked) == 2
    assert ranked[0].cluster_id == "high"
    assert ranked[0].virality.final_score > ranked[1].virality.final_score
    assert ranked[0].virality.star_weight_score == 15.0
