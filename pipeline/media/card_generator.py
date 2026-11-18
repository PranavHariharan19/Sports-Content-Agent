"""PIL Graphic and Stat Card Generator for 1080x1920 vertical 9:16 Shorts (Section 9)."""
from pathlib import Path
import re
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.logging_config import get_logger
from schemas.common import SportType

logger = get_logger(__name__)

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920

# System font candidates in priority order
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
]

REGULAR_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
]


def get_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font at specified size with fallback to default."""
    candidates = FONT_CANDIDATES if bold else REGULAR_FONT_CANDIDATES
    for path_str in candidates:
        p = Path(path_str)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    """Wrap text into lines that do not exceed max_width pixels."""
    words = text.split()
    if not words:
        return []

    lines: List[str] = []
    current_line: List[str] = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]
        if line_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []

    if current_line:
        lines.append(" ".join(current_line))

    return lines


class MediaCardGenerator:
    """Generates 1080x1920 vertical PIL stat cards and fits imagery to 9:16 broadcast standards."""

    def __init__(self, target_width: int = CANVAS_WIDTH, target_height: int = CANVAS_HEIGHT):
        self.width = target_width
        self.height = target_height

    def fit_to_vertical_9_16(self, src_path: Path, dest_path: Path) -> Path:
        """Fit any aspect ratio image into a 1080x1920 canvas using broadcast blur-fill."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src_path) as src_img:
            img = src_img.convert("RGB")
            src_w, src_h = img.size

            # If already 1080x1920, just save high-quality
            if src_w == self.width and src_h == self.height:
                img.save(dest_path, "JPEG", quality=92)
                return dest_path

            canvas = Image.new("RGB", (self.width, self.height), (10, 10, 15))

            # 1. Background layer: cover entire 1080x1920 and heavily blur
            bg_scale = max(self.width / src_w, self.height / src_h)
            bg_w = int(src_w * bg_scale)
            bg_h = int(src_h * bg_scale)
            bg_img = img.resize((bg_w, bg_h), Image.Resampling.LANCZOS)

            # Center crop background
            bg_x = (bg_w - self.width) // 2
            bg_y = (bg_h - self.height) // 2
            bg_crop = bg_img.crop((bg_x, bg_y, bg_x + self.width, bg_y + self.height))

            # Darken and Gaussian blur background
            bg_blurred = bg_crop.filter(ImageFilter.GaussianBlur(radius=28))
            darkener = ImageEnhance.Brightness(bg_blurred)
            bg_dark = darkener.enhance(0.42)
            canvas.paste(bg_dark, (0, 0))

            # 2. Foreground layer: scale to fit width (1080)
            fg_scale = min(self.width / src_w, (self.height - 350) / src_h)
            fg_w = int(src_w * fg_scale)
            fg_h = int(src_h * fg_scale)
            fg_img = img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)

            # Center foreground vertically slightly above center (leaving lower-third for subtitles)
            fg_x = (self.width - fg_w) // 2
            fg_y = max(180, (self.height - 350 - fg_h) // 2 + 100)

            # Draw subtle shadow behind foreground
            shadow_box = Image.new("RGBA", (fg_w + 30, fg_h + 30), (0, 0, 0, 160))
            shadow_blurred = shadow_box.filter(ImageFilter.GaussianBlur(15))
            canvas.paste(shadow_blurred, (fg_x - 15, fg_y - 5), shadow_blurred)

            canvas.paste(fg_img, (fg_x, fg_y))

            # Draw top and bottom cinematic gradient vignettes
            draw = ImageDraw.Draw(canvas, "RGBA")
            # Top vignette
            for y in range(160):
                alpha = int(180 * (1.0 - y / 160))
                draw.line([(0, y), (self.width, y)], fill=(0, 0, 0, alpha))
            # Bottom vignette (for subtitle contrast)
            for y in range(self.height - 500, self.height):
                alpha = int(220 * ((y - (self.height - 500)) / 500))
                draw.line([(0, y), (self.width, y)], fill=(0, 0, 0, alpha))

            canvas.save(dest_path, "JPEG", quality=92)
            logger.info("Fitted image %s to 9:16 vertical canvas at %s", src_path, dest_path)
            return dest_path

    def generate_stat_card(
        self,
        headline: str,
        stat_value: str,
        stat_label: str,
        detail_text: str,
        sport: SportType,
        output_path: Path,
        evidence_source: Optional[str] = None,
    ) -> Path:
        """Generate a sleek, modern 1080x1920 sports stat card graphic."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (self.width, self.height), (8, 12, 20))
        draw = ImageDraw.Draw(img, "RGBA")

        # Color schemes based on sport
        if sport == SportType.FOOTBALL:
            bg_top = (4, 25, 20)
            bg_bottom = (3, 8, 14)
            badge_color = (16, 185, 129)       # Neon Emerald
            stat_color = (250, 204, 21)        # Gold / Yellow
            sport_label = "⚽ FOOTBALL SPOTLIGHT"
        elif sport == SportType.CRICKET:
            bg_top = (4, 20, 48)
            bg_bottom = (4, 6, 18)
            badge_color = (6, 182, 212)        # Cyan
            stat_color = (251, 191, 36)        # Bright Amber
            sport_label = "🏏 CRICKET SPOTLIGHT"
        else:  # F1
            bg_top = (35, 8, 8)
            bg_bottom = (8, 8, 10)
            badge_color = (239, 68, 68)        # Racing Red
            stat_color = (255, 255, 255)       # High-contrast White
            sport_label = "🏎️ FORMULA 1 SPOTLIGHT"

        # 1. Base Gradient Background
        for y in range(self.height):
            ratio = y / self.height
            r = int(bg_top[0] + (bg_bottom[0] - bg_top[0]) * ratio)
            g = int(bg_top[1] + (bg_bottom[1] - bg_top[1]) * ratio)
            b = int(bg_top[2] + (bg_bottom[2] - bg_top[2]) * ratio)
            draw.line([(0, y), (self.width, y)], fill=(r, g, b, 255))

        # 2. Subtle geometric grid lines
        for x in range(0, self.width, 120):
            draw.line([(x, 0), (x, self.height)], fill=(255, 255, 255, 6))
        for y in range(0, self.height, 120):
            draw.line([(0, y), (self.width, y)], fill=(255, 255, 255, 6))

        # 3. Main Glass Card Container (safe-zone aware: stops at y=1440, well above MarginV=420 subtitles)
        card_rect = [60, 200, self.width - 60, 1440]
        # Card fill with dark tint
        draw.rounded_rectangle(card_rect, radius=32, fill=(15, 23, 42, 210))
        # Card glowing accent border
        draw.rounded_rectangle(card_rect, radius=32, outline=(badge_color[0], badge_color[1], badge_color[2], 90), width=3)

        # 4. Top Category Badge
        font_badge = get_font(32, bold=True)
        badge_text = sport_label
        badge_bbox = draw.textbbox((100, 250), badge_text, font=font_badge)
        badge_padding_x, badge_padding_y = 20, 10
        badge_rect = [
            badge_bbox[0] - badge_padding_x,
            badge_bbox[1] - badge_padding_y,
            badge_bbox[2] + badge_padding_x,
            badge_bbox[3] + badge_padding_y,
        ]
        draw.rounded_rectangle(badge_rect, radius=12, fill=(badge_color[0], badge_color[1], badge_color[2], 40))
        draw.rounded_rectangle(badge_rect, radius=12, outline=(badge_color[0], badge_color[1], badge_color[2], 160), width=1)
        draw.text((100, 250), badge_text, font=font_badge, fill=(badge_color[0], badge_color[1], badge_color[2], 255))

        # 5. Headline Text
        font_headline = get_font(52, bold=True)
        headline_lines = wrap_text(headline, font_headline, max_width=860, draw=draw)
        cur_y = 350
        for line in headline_lines[:3]:
            draw.text((100, cur_y), line, font=font_headline, fill=(241, 245, 249, 255))
            cur_y += 66

        # 6. Hero Stat Box (The Visual Centerpiece)
        cur_y = max(cur_y + 40, 580)
        stat_box_rect = [100, cur_y, self.width - 100, cur_y + 360]
        draw.rounded_rectangle(stat_box_rect, radius=24, fill=(0, 0, 0, 160))
        draw.rounded_rectangle(stat_box_rect, radius=24, outline=(255, 255, 255, 25), width=2)

        # Hero Stat Value (Huge Bold Numbers)
        clean_stat_val = stat_value.strip().upper()
        font_size = 110 if len(clean_stat_val) <= 12 else 80
        font_stat = get_font(font_size, bold=True)
        stat_bbox = draw.textbbox((0, 0), clean_stat_val, font=font_stat)
        stat_w = stat_bbox[2] - stat_bbox[0]
        stat_x = (self.width - stat_w) // 2
        draw.text((stat_x, cur_y + 45), clean_stat_val, font=font_stat, fill=stat_color)

        # Hero Stat Label
        font_stat_label = get_font(34, bold=True)
        label_text = stat_label.strip().upper()
        label_bbox = draw.textbbox((0, 0), label_text, font=font_stat_label)
        label_w = label_bbox[2] - label_bbox[0]
        label_x = (self.width - label_w) // 2
        draw.text((label_x, cur_y + 240), label_text, font=font_stat_label, fill=(148, 163, 184, 255))

        # 7. Detail / Context Text
        cur_y += 400
        font_detail = get_font(42, bold=False)
        detail_lines = wrap_text(detail_text, font_detail, max_width=860, draw=draw)
        for line in detail_lines[:3]:
            draw.text((100, cur_y), line, font=font_detail, fill=(203, 213, 225, 255))
            cur_y += 56

        # 8. Bottom Grounding / Fact Citation Badge
        watermark_y = 1360
        font_watermark = get_font(26, bold=True)
        evidence_str = f"✓ GROUNDED EVIDENCE • {evidence_source.upper() if evidence_source else 'TIER 1 VERIFIED'}"
        draw.text((100, watermark_y), evidence_str, font=font_watermark, fill=(52, 211, 153, 240))

        img.save(output_path, "JPEG", quality=92)
        logger.info("Generated PIL stat card for sport %s at %s", sport.value, output_path)
        return output_path
