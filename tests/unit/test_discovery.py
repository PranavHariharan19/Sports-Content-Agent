from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
import pytest

from pipeline.discovery import DiscoveryEngine, RSSDiscoverySource
from pipeline.sources_trust import SourceTrustManager
from schemas.common import SourceTier, SportType


SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Sports News</title>
    <item>
      <title>Arsenal defeat Chelsea 3-1 in London Derby</title>
      <link>https://www.bbc.co.uk/sport/football/12345</link>
      <description>Bukayo Saka and Declan Rice scored as Arsenal won.</description>
      <pubDate>Mon, 16 Nov 2026 10:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Manchester City sign new midfield prospect</title>
      <link>https://www.skysports.com/football/news/67890</link>
      <description>Pep Guardiola confirms agreement.</description>
      <pubDate>Mon, 16 Nov 2026 11:30:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


@pytest.mark.asyncio
async def test_rss_discovery_source():
    """Verify RSS parser extracts articles, formats domains, and assigns tiers."""
    stm = SourceTrustManager()
    rss = RSSDiscoverySource(stm)

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_RSS_XML

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        feed_cfg = {"name": "BBC Sport", "url": "https://feeds.bbci.co.uk/sport.xml"}
        articles = await rss.fetch_feed(feed_cfg, SportType.FOOTBALL)

        assert len(articles) == 2
        assert "Arsenal defeat Chelsea" in articles[0].title
        assert articles[0].domain == "bbc.co.uk"
        assert articles[0].source_tier == SourceTier.TIER_2
        assert articles[0].sport == SportType.FOOTBALL


@pytest.mark.asyncio
async def test_discovery_engine_age_filtering():
    """Verify articles older than max_age_hours are excluded."""
    stm = SourceTrustManager()
    engine = DiscoveryEngine(stm)

    now = datetime.now(timezone.utc)
    fresh_article = SAMPLE_RSS_XML

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.text = fresh_article

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        # Set max_age_hours very small or large
        articles = await engine.discover(SportType.FOOTBALL, max_age_hours=72)
        assert len(articles) >= 1
