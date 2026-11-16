from datetime import datetime, timezone
from pipeline.clustering import StoryClusteringEngine
from pipeline.sources_trust import SourceTrustManager
from schemas.common import SourceTier, SportType
from schemas.discovery import DiscoveredArticle


def test_cluster_same_event_across_sources():
    """Verify multiple articles covering the same match are grouped into 1 cluster."""
    stm = SourceTrustManager()
    engine = StoryClusteringEngine(stm, similarity_threshold=0.40)
    now = datetime.now(timezone.utc)

    art1 = DiscoveredArticle(
        title="Arsenal beat Chelsea 3-1 at Emirates Stadium",
        url="https://www.bbc.co.uk/sport/football/1",
        domain="bbc.co.uk",
        published_at=now,
        source_tier=SourceTier.TIER_2,
        sport=SportType.FOOTBALL,
    )
    art2 = DiscoveredArticle(
        title="Arsenal secure 3-1 victory over Chelsea in London derby",
        url="https://www.skysports.com/football/2",
        domain="skysports.com",
        published_at=now,
        source_tier=SourceTier.TIER_2,
        sport=SportType.FOOTBALL,
    )
    art3 = DiscoveredArticle(
        title="Formula 1: Max Verstappen takes pole position in Las Vegas",
        url="https://www.formula1.com/news/3",
        domain="formula1.com",
        published_at=now,
        source_tier=SourceTier.TIER_1,
        sport=SportType.FOOTBALL,  # different topic
    )

    clusters = engine.cluster_articles([art1, art2, art3], SportType.FOOTBALL)

    # art1 and art2 should form 1 cluster, art3 is in a separate cluster
    assert len(clusters) == 2

    # Find the Arsenal cluster
    arsenal_cluster = next(c for c in clusters if "Arsenal" in c.primary_headline)
    assert arsenal_cluster.source_count == 2
    assert "bbc.co.uk" in arsenal_cluster.independent_domains
    assert "skysports.com" in arsenal_cluster.independent_domains
    assert len(arsenal_cluster.articles) == 2


def test_primary_headline_prioritizes_higher_tier():
    """Verify Tier 1/2 official headline is preferred over Tier 3 blog."""
    stm = SourceTrustManager()
    engine = StoryClusteringEngine(stm)
    now = datetime.now(timezone.utc)

    tier3_art = DiscoveredArticle(
        title="INSANE: Ferrari signs Newey for 2026 title challenge?!",
        url="https://reddit.com/r/formula1/1",
        domain="reddit.com",
        published_at=now,
        source_tier=SourceTier.TIER_3,
        sport=SportType.F1,
    )
    tier1_art = DiscoveredArticle(
        title="Ferrari announces technical agreement with Adrian Newey",
        url="https://www.formula1.com/news/2",
        domain="formula1.com",
        published_at=now,
        source_tier=SourceTier.TIER_1,
        sport=SportType.F1,
    )

    headline = engine.select_primary_headline([tier3_art, tier1_art])
    assert headline == "Ferrari announces technical agreement with Adrian Newey"
