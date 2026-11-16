from datetime import datetime, timezone
import hashlib
import html
import re
from typing import List, Optional
from bs4 import BeautifulSoup
import httpx
import trafilatura

from app.logging_config import get_logger
from pipeline.sources_trust import SourceTrustManager
from schemas.common import SourceTier
from schemas.sources import StoredSource

logger = get_logger(__name__)

INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b",
    r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+instructions?\b",
    r"(?i)\bforget\s+(all\s+)?(previous|prior)\b",
    r"(?i)\byou\s+are\s+now\s+(an\s+unrestricted|in\s+dan\s+mode|jailbroken)\b",
    r"(?i)\bsystem\s*prompt\s*:",
    r"(?i)<\s*\|(?:im_start|im_end|system|user|assistant)\|\s*>",
    r"(?i)\[\s*INST\s*\].*?\[\s*/INST\s*\]",
    r"(?i)\boutput\s+the\s+following\s+verbatim\b",
]


class PromptInjectionSanitizer:
    """Sanitizes untrusted scraped text to neutralize prompt-injection attacks."""

    @staticmethod
    def sanitize(text: str) -> str:
        if not text:
            return ""

        sanitized = text
        for pattern in INJECTION_PATTERNS:
            sanitized = re.sub(pattern, "[STRIPPED_UNTRUSTED_INSTRUCTION]", sanitized)

        # Normalize linebreaks and whitespace
        sanitized = re.sub(r"\r\n|\r", "\n", sanitized)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()


class SourceFetcher:
    """Fetches full article web pages and extracts ground-truth article text with trafilatura."""

    def __init__(
        self,
        trust_manager: Optional[SourceTrustManager] = None,
        timeout_seconds: float = 10.0,
    ):
        self.trust_manager = trust_manager or SourceTrustManager()
        self.timeout_seconds = timeout_seconds
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    async def fetch_html(self, url: str) -> Optional[str]:
        """Fetch raw HTML from target URL."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                resp = await client.get(url, headers=self.headers)
                if resp.status_code == 200:
                    return resp.text
                logger.warning("Failed to fetch %s (HTTP %d)", url, resp.status_code)
        except Exception as e:
            logger.warning("Network error fetching %s: %s", url, str(e))
        return None

    def extract_text(self, html_content: str) -> str:
        """Extract clean body text with trafilatura, falling back to BeautifulSoup."""
        if not html_content:
            return ""

        # Primary extraction using trafilatura
        extracted = trafilatura.extract(
            html_content,
            include_links=False,
            include_images=False,
            output_format="txt",
            favor_recall=True,
        )

        if extracted and len(extracted.split()) >= 20:
            return extracted.strip()

        # Fallback to BeautifulSoup if trafilatura missed main content
        logger.debug("Trafilatura returned empty/short text, attempting BeautifulSoup fallback")
        soup = BeautifulSoup(html_content, "html.parser")

        # Strip scripts, styles, navs, footers
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        body_text = "\n\n".join(p for p in paragraphs if len(p.split()) > 5)
        return body_text.strip()

    async def fetch_and_process_source(
        self,
        job_id: str,
        url: str,
        title: Optional[str] = None,
        published_at: Optional[str] = None,
        raw_html: Optional[str] = None,
    ) -> StoredSource:
        """Fetch, extract, sanitize, and build StoredSource ground-truth model."""
        now = datetime.now(timezone.utc).isoformat()
        domain = self.trust_manager.normalize_domain(url)
        tier = self.trust_manager.get_tier(domain)
        is_allowlisted = (tier != SourceTier.UNKNOWN)

        html_content = raw_html
        if html_content is None:
            html_content = await self.fetch_html(url) or ""

        # Extract text
        raw_text = self.extract_text(html_content) if html_content else ""

        # If still empty (e.g. paywall or JS-only site), use title as fallback text
        if not raw_text and title:
            raw_text = title

        # Sanitize against prompt injections
        cleaned_text = PromptInjectionSanitizer.sanitize(raw_text)

        # Deterministic source ID based on URL
        source_id = f"src_{hashlib.sha256(url.encode('utf-8')).hexdigest()[:12]}"

        source = StoredSource(
            source_id=source_id,
            job_id=job_id,
            url=url,
            domain=domain,
            source_tier=tier,
            is_allowlisted=is_allowlisted,
            title=title,
            cleaned_text=cleaned_text,
            published_at=published_at,
            fetched_at=now,
        )

        logger.info(
            "Processed source %s: domain=%s tier=%s allowlisted=%s words=%d",
            source_id,
            domain,
            tier.value,
            is_allowlisted,
            source.word_count,
        )
        return source
