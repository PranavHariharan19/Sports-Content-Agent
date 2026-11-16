import asyncio
from datetime import datetime, timedelta, timezone
import html
import re
import time
from typing import Any, Dict, List, Optional
import feedparser
import httpx

from app.logging_config import get_logger
from pipeline.sources_trust import SourceTrustManager
from schemas.common import SourceTier, SportType
from schemas.discovery import DiscoveredArticle

logger = get_logger(__name__)


class RSSDiscoverySource:
    """Fetches trending articles from configured sports RSS feeds."""

    def __init__(self, trust_manager: SourceTrustManager, timeout_seconds: float = 6.0):
        self.trust_manager = trust_manager
        self.timeout_seconds = timeout_seconds
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SportsShorts/1.0",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }

    async def fetch_feed(self, feed_cfg: Dict[str, str], sport: SportType) -> List[DiscoveredArticle]:
        url = feed_cfg.get("url", "")
        name = feed_cfg.get("name", url)
        if not url:
            return []

        articles: List[DiscoveredArticle] = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                resp = await client.get(url, headers=self.headers)
                if resp.status_code != 200:
                    logger.warning("Feed '%s' returned status %d", name, resp.status_code)
                    return []
                content = resp.text

            # Parse with feedparser
            parsed = feedparser.parse(content)
            now = datetime.now(timezone.utc)

            for entry in parsed.entries:
                title = html.unescape(getattr(entry, "title", "")).strip()
                link = getattr(entry, "link", "").strip()
                if not title or not link:
                    continue

                # Parse published datetime
                pub_time = now
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        pub_time = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    except Exception:
                        pub_time = now
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    try:
                        pub_time = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                    except Exception:
                        pub_time = now

                summary = ""
                if hasattr(entry, "summary"):
                    # Strip basic HTML tags
                    raw_summary = html.unescape(entry.summary)
                    summary = re.sub(r"<[^>]+>", "", raw_summary).strip()

                domain = self.trust_manager.normalize_domain(link)
                tier = self.trust_manager.get_tier(domain)

                articles.append(
                    DiscoveredArticle(
                        title=title,
                        url=link,
                        domain=domain,
                        published_at=pub_time,
                        summary=summary[:300] if summary else None,
                        source_tier=tier,
                        sport=sport,
                    )
                )

            logger.info("Fetched %d articles from feed '%s'", len(articles), name)
        except Exception as e:
            logger.warning("Failed to fetch RSS feed '%s' (%s): %s", name, url, str(e))

        return articles


class RedditDiscoverySource:
    """Fetches trending discussion topics from subreddits via RSS / public feed."""

    def __init__(self, trust_manager: SourceTrustManager, timeout_seconds: float = 6.0):
        self.trust_manager = trust_manager
        self.timeout_seconds = timeout_seconds
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; SportsShortsAgent/1.0)",
        }

    async def fetch_subreddit(self, sport: SportType) -> List[DiscoveredArticle]:
        sub = self.trust_manager.get_subreddit(sport)
        if not sub:
            return []

        url = f"https://www.reddit.com/r/{sub}/.rss"
        articles: List[DiscoveredArticle] = []
        now = datetime.now(timezone.utc)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                resp = await client.get(url, headers=self.headers)
                if resp.status_code != 200:
                    logger.debug("Reddit RSS for r/%s returned %d", sub, resp.status_code)
                    return []
                content = resp.text

            parsed = feedparser.parse(content)
            for entry in parsed.entries:
                title = html.unescape(getattr(entry, "title", "")).strip()
                link = getattr(entry, "link", "").strip()
                if not title or not link:
                    continue

                # Filter out generic daily sticky threads
                if "daily discussion" in title.lower() or "free talk" in title.lower():
                    continue

                pub_time = now
                if hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    try:
                        pub_time = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                    except Exception:
                        pub_time = now

                articles.append(
                    DiscoveredArticle(
                        title=title,
                        url=link,
                        domain="reddit.com",
                        published_at=pub_time,
                        summary=None,
                        source_tier=SourceTier.TIER_3,
                        sport=sport,
                        reddit_score=50,  # Baseline score for hot frontpage RSS items
                        reddit_comments=15,
                    )
                )

            logger.info("Fetched %d Reddit posts from r/%s", len(articles), sub)
        except Exception as e:
            logger.debug("Failed to fetch Reddit RSS for r/%s: %s", sub, str(e))

        return articles


class DiscoveryEngine:
    """Orchestrates multi-source discovery across RSS feeds and Reddit."""

    def __init__(self, trust_manager: Optional[SourceTrustManager] = None):
        self.trust_manager = trust_manager or SourceTrustManager()
        self.rss_source = RSSDiscoverySource(self.trust_manager)
        self.reddit_source = RedditDiscoverySource(self.trust_manager)

    async def discover(self, sport: SportType, max_age_hours: int = 24) -> List[DiscoveredArticle]:
        """Fetch articles across all configured feeds for the given sport, filtering by age window."""
        rss_feeds = self.trust_manager.get_rss_feeds(sport)
        tasks = [self.rss_source.fetch_feed(f, sport) for f in rss_feeds]
        tasks.append(self.reddit_source.fetch_subreddit(sport))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles: List[DiscoveredArticle] = []
        for res in results:
            if isinstance(res, list):
                all_articles.extend(res)
            elif isinstance(res, Exception):
                logger.warning("Discovery task error: %s", str(res))

        # Filter by age threshold
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=max_age_hours)
        valid_articles = [a for a in all_articles if a.published_at >= cutoff]

        logger.info(
            "Discovery complete for %s: %d total articles, %d within %dh window",
            sport.value,
            len(all_articles),
            len(valid_articles),
            max_age_hours,
        )
        return valid_articles
