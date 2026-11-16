from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.parse import quote
import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)


class WikipediaPageviewsClient:
    """Free, official Wikimedia REST API client for entity popularity metrics."""

    BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents"

    def __init__(self, timeout_seconds: float = 4.0):
        self.timeout_seconds = timeout_seconds
        self.headers = {
            "User-Agent": "SportsShortsCommandCenter/1.0 (sports-shorts-agent@example.com)",
            "Accept": "application/json",
        }

    async def get_pageviews(self, entity_name: str, days: int = 7) -> int:
        """Fetch total pageviews for an entity over the past N days.

        Returns 0 on any error or missing page.
        """
        if not entity_name or not entity_name.strip():
            return 0

        # Format entity name for Wikipedia URL
        formatted = quote(entity_name.strip().replace(" ", "_"))
        now = datetime.now(timezone.utc)
        end_str = (now - timedelta(days=1)).strftime("%Y%m%d")
        start_str = (now - timedelta(days=days)).strftime("%Y%m%d")

        url = f"{self.BASE_URL}/{formatted}/daily/{start_str}/{end_str}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.get(url, headers=self.headers)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        items = data.get("items", [])
                        total_views = sum(item.get("views", 0) for item in items)
                        return total_views
                    except Exception:
                        return 0
                elif resp.status_code != 404:
                    logger.debug("Wikimedia API returned %d for %s", resp.status_code, entity_name)
        except Exception as e:
            logger.debug("Wikipedia pageview lookup failed for %s: %s", entity_name, str(e))

        return 0

    def compute_pageview_score(self, total_views: int) -> float:
        """Map total 7-day pageviews to a normalized 0-10 interest score.

        5,000 views -> ~2.5
        20,000 views -> ~5.0
        100,000+ views -> 10.0
        """
        if total_views <= 0:
            return 0.0
        import math
        # Log scale mapping
        score = min(10.0, max(0.0, math.log10(max(1, total_views) / 500) * 3.5))
        return round(score, 2)


class GoogleTrendsClient:
    """Optional and failure-tolerant Google Trends interest checker."""

    def __init__(self):
        self._available = False
        try:
            import pytrends  # type: ignore # noqa: F401
            self._available = True
        except ImportError:
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def get_interest(self, keyword: str) -> float:
        """Fetch relative interest (0-100) or return 0.0 if unavailable or failed."""
        if not self._available or not keyword:
            return 0.0
        try:
            from pytrends.request import TrendReq  # type: ignore

            pytrend = TrendReq(hl="en-US", tz=0, timeout=(3.0, 5.0))
            pytrend.build_payload([keyword], timeframe="now 1-d")
            df = pytrend.interest_over_time()
            if not df.empty and keyword in df:
                return float(df[keyword].mean())
        except Exception as e:
            logger.debug("Google Trends lookup failed for %s (gracefully ignored): %s", keyword, str(e))
        return 0.0
