"""Free media grounding fetcher (Wikimedia Commons, Pexels, Pixabay, Openverse)."""
import os
from pathlib import Path
import re
from typing import Optional, Tuple
import httpx
from PIL import Image

from app.logging_config import get_logger
from pipeline.media.card_generator import MediaCardGenerator
from schemas.common import SportType

logger = get_logger(__name__)

USER_AGENT = "SportsShortsCommandCenter/1.0 (https://github.com/sports-shorts; contact@sportsshorts.local)"


class MediaFetcher:
    """Fetches open-licensed sports media from Wikimedia Commons, Pexels, and Pixabay."""

    def __init__(
        self,
        pexels_api_key: Optional[str] = None,
        pixabay_api_key: Optional[str] = None,
        card_generator: Optional[MediaCardGenerator] = None,
        timeout: float = 12.0,
    ):
        self.pexels_api_key = pexels_api_key or os.getenv("PEXELS_API_KEY")
        self.pixabay_api_key = pixabay_api_key or os.getenv("PIXABAY_API_KEY")
        self.card_generator = card_generator or MediaCardGenerator()
        self.timeout = timeout

    async def search_and_download(
        self,
        query: str,
        output_path: Path,
        sport: SportType,
        prefer_provider: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """Search across providers, download best image, and format to 1080x1920 9:16 vertical canvas.

        Returns (provider_name, attribution) or None if all network providers fail.
        """
        clean_q = re.sub(r"[^\w\s-]", " ", query).strip()
        if not clean_q:
            return None

        temp_download = output_path.with_suffix(".tmp_download")

        # 1. Try Wikimedia Commons (best for specific athletes, teams, venues)
        if not prefer_provider or prefer_provider == "wikimedia":
            res = await self._search_wikimedia(clean_q, temp_download)
            if res:
                self.card_generator.fit_to_vertical_9_16(temp_download, output_path)
                temp_download.unlink(missing_ok=True)
                return res

        # 2. Try Pexels (dynamic sports action photography)
        if self.pexels_api_key and (not prefer_provider or prefer_provider == "pexels"):
            res = await self._search_pexels(clean_q, temp_download, sport)
            if res:
                self.card_generator.fit_to_vertical_9_16(temp_download, output_path)
                temp_download.unlink(missing_ok=True)
                return res

        # 3. Try Pixabay (general sports photography)
        if self.pixabay_api_key and (not prefer_provider or prefer_provider == "pixabay"):
            res = await self._search_pixabay(clean_q, temp_download, sport)
            if res:
                self.card_generator.fit_to_vertical_9_16(temp_download, output_path)
                temp_download.unlink(missing_ok=True)
                return res

        temp_download.unlink(missing_ok=True)
        return None

    async def _search_wikimedia(self, query: str, dest_path: Path) -> Optional[Tuple[str, str]]:
        """Search Wikimedia Commons API for images matching query."""
        url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"file:{query}",
            "gsrlimit": 5,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "format": "json",
        }
        headers = {"User-Agent": USER_AGENT}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code != 200:
                    return None

                data = resp.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_info in pages.items():
                    imageinfo = page_info.get("imageinfo", [])
                    if not imageinfo:
                        continue
                    info = imageinfo[0]
                    mime = info.get("mime", "").lower()
                    if mime not in ("image/jpeg", "image/png", "image/webp"):
                        continue

                    img_url = info.get("url")
                    if not img_url:
                        continue

                    title = page_info.get("title", "").replace("File:", "")
                    if await self._download_and_verify(client, img_url, dest_path, headers):
                        logger.info("Found Wikimedia Commons image for query '%s': %s", query, title)
                        return "wikimedia", f"Wikimedia Commons // {title}"
        except Exception as e:
            logger.debug("Wikimedia search error for '%s': %s", query, str(e))

        return None

    async def _search_pexels(self, query: str, dest_path: Path, sport: SportType) -> Optional[Tuple[str, str]]:
        """Search Pexels API for portrait sports action photography."""
        url = "https://api.pexels.com/v1/search"
        # Augment query with sport context if needed
        search_query = f"{query} {sport.value}"
        params = {
            "query": search_query,
            "per_page": 5,
            "orientation": "portrait",
        }
        headers = {"Authorization": self.pexels_api_key}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code != 200:
                    return None

                data = resp.json()
                photos = data.get("photos", [])
                for photo in photos:
                    img_url = photo.get("src", {}).get("portrait") or photo.get("src", {}).get("large2x")
                    if not img_url:
                        continue

                    photographer = photo.get("photographer", "Pexels Creator")
                    if await self._download_and_verify(client, img_url, dest_path):
                        logger.info("Found Pexels image for query '%s' by %s", search_query, photographer)
                        return "pexels", f"Photo by {photographer} on Pexels"
        except Exception as e:
            logger.debug("Pexels search error for '%s': %s", query, str(e))

        return None

    async def _search_pixabay(self, query: str, dest_path: Path, sport: SportType) -> Optional[Tuple[str, str]]:
        """Search Pixabay API for vertical sports photography."""
        url = "https://pixabay.com/api/"
        search_query = f"{query} {sport.value}"
        params = {
            "key": self.pixabay_api_key,
            "q": search_query,
            "image_type": "photo",
            "orientation": "vertical",
            "per_page": 5,
            "safesearch": "true",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    return None

                data = resp.json()
                hits = data.get("hits", [])
                for hit in hits:
                    img_url = hit.get("largeImageURL") or hit.get("webformatURL")
                    if not img_url:
                        continue

                    user = hit.get("user", "Pixabay Artist")
                    if await self._download_and_verify(client, img_url, dest_path):
                        logger.info("Found Pixabay image for query '%s' by %s", search_query, user)
                        return "pixabay", f"Image by {user} on Pixabay"
        except Exception as e:
            logger.debug("Pixabay search error for '%s': %s", query, str(e))

        return None

    async def _download_and_verify(
        self,
        client: httpx.AsyncClient,
        img_url: str,
        dest_path: Path,
        headers: Optional[dict] = None,
    ) -> bool:
        """Download remote image and verify that PIL can decode it without errors."""
        try:
            resp = await client.get(img_url, headers=headers or {}, follow_redirects=True)
            if resp.status_code != 200:
                return False

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(resp.content)

            # Validate PIL can open and read header
            with Image.open(dest_path) as im:
                im.verify()

            return True
        except Exception as e:
            logger.debug("Download/verify failed for %s: %s", img_url, str(e))
            dest_path.unlink(missing_ok=True)
            return False
