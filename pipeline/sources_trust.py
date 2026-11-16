import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import yaml

from app.logging_config import get_logger
from schemas.common import SourceTier, SportType

logger = get_logger(__name__)


class SourceTrustManager:
    """Manages source trust tiers, domains, and star entities from sources.yaml."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or os.environ.get("SOURCES_CONFIG_PATH", "config/sources.yaml"))
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.config_path.is_file():
            logger.warning("Sources config not found at %s, using empty config", self.config_path)
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def normalize_domain(url_or_domain: str) -> str:
        """Extract base domain from a URL or raw domain string."""
        if not url_or_domain:
            return ""
        raw = url_or_domain.strip().lower()
        if not (raw.startswith("http://") or raw.startswith("https://")):
            raw = f"http://{raw}"
        parsed = urlparse(raw)
        domain = parsed.netloc.split(":")[0]

        # Strip common subdomains
        for prefix in ("www.", "feeds.", "m.", "mobile."):
            if domain.startswith(prefix):
                domain = domain[len(prefix):]

        # Normalize known CDN/feed domains to main publication domain
        if domain in ("bbci.co.uk", "bbci.com") or domain.endswith(".bbci.co.uk"):
            domain = "bbc.co.uk"

        return domain

    def get_tier(self, url_or_domain: str) -> SourceTier:
        """Determine trust tier for a given URL or domain."""
        domain = self.normalize_domain(url_or_domain)
        tiers_cfg = self._data.get("tiers", {})

        # Check Tier 1
        tier_1_domains = tiers_cfg.get("tier_1", {}).get("domains", [])
        if any(domain == d or domain.endswith(f".{d}") for d in tier_1_domains):
            return SourceTier.TIER_1

        # Check Tier 2
        tier_2_domains = tiers_cfg.get("tier_2", {}).get("domains", [])
        if any(domain == d or domain.endswith(f".{d}") for d in tier_2_domains):
            return SourceTier.TIER_2

        # Check Tier 3
        tier_3_domains = tiers_cfg.get("tier_3", {}).get("domains", [])
        if any(domain == d or domain.endswith(f".{d}") for d in tier_3_domains):
            return SourceTier.TIER_3

        return SourceTier.UNKNOWN

    def get_rss_feeds(self, sport: SportType) -> List[Dict[str, str]]:
        """Get configured RSS feeds for a sport."""
        feeds = self._data.get("rss_feeds", {})
        return feeds.get(sport.value, [])

    def get_star_entities(self, sport: SportType) -> List[str]:
        """Get configured high-interest star entities for a sport."""
        stars = self._data.get("star_entities", {})
        return stars.get(sport.value, [])

    def get_subreddit(self, sport: SportType) -> Optional[str]:
        """Get subreddit name for a sport."""
        subs = self._data.get("reddit_subreddits", {})
        return subs.get(sport.value)
