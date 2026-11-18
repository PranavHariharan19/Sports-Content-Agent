"""Unit tests for PIL MediaCardGenerator (1080x1920 vertical canvas & stat cards)."""
from pathlib import Path
from PIL import Image
import pytest

from pipeline.media.card_generator import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    MediaCardGenerator,
    get_font,
    wrap_text,
)
from schemas.common import SportType


def test_get_font_returns_usable_font():
    """Verify font loader returns a valid font."""
    font = get_font(size=40, bold=True)
    assert font is not None


def test_wrap_text_splits_lines():
    """Verify text wrapping respects pixel width boundaries."""
    img = Image.new("RGB", (500, 500))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    font = get_font(size=30, bold=False)

    long_text = "Dani Carvajal scored a dramatic late winner at the Wanda Metropolitano stadium"
    lines = wrap_text(long_text, font=font, max_width=250, draw=draw)
    assert len(lines) >= 3
    # Check that all words are preserved
    reconstructed = " ".join(lines)
    assert reconstructed == long_text


def test_generate_stat_card_dimensions_and_sports(tmp_path: Path):
    """Verify stat cards are generated at 1080x1920 for Football, Cricket, and F1."""
    gen = MediaCardGenerator()

    for sport in [SportType.FOOTBALL, SportType.CRICKET, SportType.F1]:
        card_path = tmp_path / f"stat_card_{sport.value}.jpg"
        out = gen.generate_stat_card(
            headline=f"Historic {sport.value.upper()} Moment",
            stat_value="43 GAMES",
            stat_label="UNBEATEN STREAK",
            detail_text="An unforgettable performance that redefined sports history.",
            sport=sport,
            output_path=card_path,
            evidence_source="BBC SPORT // TIER 1",
        )
        assert out.exists()
        assert out.stat().st_size > 0

        with Image.open(out) as im:
            assert im.size == (CANVAS_WIDTH, CANVAS_HEIGHT)
            assert im.format == "JPEG"


def test_fit_to_vertical_9_16_landscape_image(tmp_path: Path):
    """Verify fitting a 16:9 landscape image into a 1080x1920 vertical canvas."""
    gen = MediaCardGenerator()

    # Create dummy 16:9 landscape image (1920x1080)
    src_landscape = tmp_path / "source_landscape.jpg"
    with Image.new("RGB", (1920, 1080), (30, 80, 150)) as im:
        im.save(src_landscape, "JPEG")

    dest_vertical = tmp_path / "fitted_vertical.jpg"
    gen.fit_to_vertical_9_16(src_landscape, dest_vertical)

    assert dest_vertical.exists()
    with Image.open(dest_vertical) as im:
        assert im.size == (CANVAS_WIDTH, CANVAS_HEIGHT)
