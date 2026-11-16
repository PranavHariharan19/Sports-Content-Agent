from pipeline.sources_trust import SourceTrustManager
from schemas.common import SourceTier, SportType


def test_domain_normalization():
    """Verify domains are stripped of protocol, www, and subdomains."""
    assert SourceTrustManager.normalize_domain("https://www.bbc.co.uk/sport/football") == "bbc.co.uk"
    assert SourceTrustManager.normalize_domain("http://feeds.bbci.co.uk/sport/rss.xml") == "bbc.co.uk"
    assert SourceTrustManager.normalize_domain("https://formula1.com/news/123") == "formula1.com"
    assert SourceTrustManager.normalize_domain("m.espn.com/soccer") == "espn.com"


def test_trust_tiers():
    """Verify trust tiers are classified according to sources.yaml."""
    stm = SourceTrustManager()
    assert stm.get_tier("https://formula1.com/latest") == SourceTier.TIER_1
    assert stm.get_tier("https://www.bbc.com/sport") == SourceTier.TIER_2
    assert stm.get_tier("https://www.espncricinfo.com") == SourceTier.TIER_2
    assert stm.get_tier("https://reddit.com/r/soccer") == SourceTier.TIER_3
    assert stm.get_tier("https://unverified-blog.xyz") == SourceTier.UNKNOWN


def test_star_entities_loaded():
    """Verify star entities for each sport exist."""
    stm = SourceTrustManager()
    football_stars = stm.get_star_entities(SportType.FOOTBALL)
    assert any("Messi" in s for s in football_stars)
    assert any("Real Madrid" in s for s in football_stars)

    f1_stars = stm.get_star_entities(SportType.F1)
    assert any("Verstappen" in s for s in f1_stars)
    assert any("Hamilton" in s for s in f1_stars)

    cricket_stars = stm.get_star_entities(SportType.CRICKET)
    assert any("Kohli" in s for s in cricket_stars)
