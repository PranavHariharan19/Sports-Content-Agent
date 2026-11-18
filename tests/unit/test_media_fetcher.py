"""Unit tests for MediaFetcher: Wikimedia Commons, Pexels, and Pixabay retrieval."""
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
import pytest

from pipeline.media.card_generator import CANVAS_HEIGHT, CANVAS_WIDTH
from pipeline.media.fetcher import MediaFetcher
from schemas.common import SportType


def _make_dummy_image_bytes(width: int = 400, height: int = 300) -> bytes:
    buf = io.BytesIO()
    with Image.new("RGB", (width, height), (50, 100, 150)) as im:
        im.save(buf, "JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_search_wikimedia_success(tmp_path: Path):
    """Verify Wikimedia Commons search and image download flow."""
    fetcher = MediaFetcher()
    out_file = tmp_path / "wiki_out.jpg"

    dummy_bytes = _make_dummy_image_bytes(800, 600)

    wiki_api_response = {
        "query": {
            "pages": {
                "12345": {
                    "title": "File:Dani Carvajal 2024.jpg",
                    "imageinfo": [
                        {
                            "url": "https://upload.wikimedia.org/dummy.jpg",
                            "mime": "image/jpeg",
                        }
                    ],
                }
            }
        }
    }

    mock_resp_api = MagicMock(status_code=200, json=lambda: wiki_api_response)
    mock_resp_img = MagicMock(status_code=200, content=dummy_bytes)

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = [mock_resp_api, mock_resp_img]
        result = await fetcher.search_and_download("Dani Carvajal", out_file, SportType.FOOTBALL)

    assert result is not None
    provider, attribution = result
    assert provider == "wikimedia"
    assert "Dani Carvajal 2024.jpg" in attribution
    assert out_file.exists()

    with Image.open(out_file) as im:
        assert im.size == (CANVAS_WIDTH, CANVAS_HEIGHT)


@pytest.mark.asyncio
async def test_search_pexels_success(tmp_path: Path):
    """Verify Pexels search and portrait image download flow."""
    fetcher = MediaFetcher(pexels_api_key="mock_key")
    out_file = tmp_path / "pexels_out.jpg"

    dummy_bytes = _make_dummy_image_bytes(600, 900)

    pexels_api_response = {
        "photos": [
            {
                "id": 101,
                "photographer": "Alex Sports",
                "src": {
                    "portrait": "https://images.pexels.com/dummy_portrait.jpg"
                },
            }
        ]
    }

    mock_resp_api = MagicMock(status_code=200, json=lambda: pexels_api_response)
    mock_resp_img = MagicMock(status_code=200, content=dummy_bytes)

    # Force provider to pexels
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = [mock_resp_api, mock_resp_img]
        result = await fetcher.search_and_download(
            "football striker celebration",
            out_file,
            SportType.FOOTBALL,
            prefer_provider="pexels",
        )

    assert result is not None
    provider, attribution = result
    assert provider == "pexels"
    assert "Alex Sports" in attribution
    assert out_file.exists()


@pytest.mark.asyncio
async def test_search_all_fail_returns_none(tmp_path: Path):
    """Verify graceful None return when all network providers fail."""
    fetcher = MediaFetcher(pexels_api_key="mock_key", pixabay_api_key="mock_key")
    out_file = tmp_path / "fail_out.jpg"

    mock_err_resp = MagicMock(status_code=500)

    with patch("httpx.AsyncClient.get", return_value=mock_err_resp):
        result = await fetcher.search_and_download("unknown-entity-123", out_file, SportType.F1)

    assert result is None
    assert not out_file.exists()
