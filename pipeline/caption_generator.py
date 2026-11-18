"""ASS and SRT subtitle generator with word-level animated highlighting (Section 8).

Formats:
- Advanced SubStation Alpha (.ass) with 9:16 vertical styling (1080x1920)
- Lower-third safe-zone positioning (MarginV=420)
- 2-4 word rhythmic chunking strictly constrained within individual beat boundaries
- Active word accent color highlight (e.g. bright yellow &H0000FFFF&)
- Clean SubRip (.srt) fallback
"""
from pathlib import Path
import re
from typing import List, Optional, Tuple

from app.logging_config import get_logger
from schemas.voice import BeatTimestamp, SubtitleChunk, SubtitleFileRecord, WordTimestamp

logger = get_logger(__name__)


def ms_to_ass_time(ms: int) -> str:
    """Standard ASS timestamp format: H:MM:SS.CC (centiseconds)."""
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    centis = (ms % 1000) // 10
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centis:02d}"


def ms_to_srt_time(ms: int) -> str:
    """Standard SRT timestamp format: HH:MM:SS,mmm."""
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


class CaptionGenerator:
    """Generates stylized vertical .ass and .srt subtitles with word-by-word accent pop."""

    def __init__(
        self,
        font_name: str = "Arial Black",
        font_size: int = 70,
        primary_color: str = "&H00FFFFFF",     # Bright White in BGR
        highlight_color: str = "&H0000FFFF",   # Bright Yellow / Gold in BGR
        outline_color: str = "&H00000000",     # Deep Black
        outline_width: int = 8,
        shadow_depth: int = 4,
        margin_v: int = 420,                   # Safe zone from bottom on 1920 vertical canvas
        max_chunk_words: int = 4,
    ):
        self.font_name = font_name
        self.font_size = font_size
        self.primary_color = primary_color
        self.highlight_color = highlight_color
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.shadow_depth = shadow_depth
        self.margin_v = margin_v
        self.max_chunk_words = max_chunk_words

    def chunk_words(self, words: List[WordTimestamp]) -> List[SubtitleChunk]:
        """Group word timestamps into 2-4 word phrases, never spanning across beat boundaries."""
        if not words:
            return []

        chunks: List[SubtitleChunk] = []
        current_words: List[WordTimestamp] = []
        chunk_id = 1

        for w in words:
            # 1. Hard Beat Boundary Check:
            # If the current word belongs to a different beat than previous words,
            # finalize the current beat's chunk immediately.
            if current_words and w.beat_id is not None and current_words[-1].beat_id is not None:
                if w.beat_id != current_words[-1].beat_id:
                    chunks.append(
                        SubtitleChunk(
                            chunk_id=chunk_id,
                            beat_id=current_words[0].beat_id,
                            words=list(current_words),
                            start_ms=current_words[0].start_ms,
                            end_ms=current_words[-1].end_ms,
                            text=" ".join(item.word for item in current_words),
                        )
                    )
                    chunk_id += 1
                    current_words = []

            current_words.append(w)

            # 2. Break chunk if:
            # - Punctuation ending sentence/phrase
            # - Reached max_chunk_words (e.g. 4)
            ends_with_punct = bool(re.search(r"[\.,!?;:–—]$", w.word))
            if len(current_words) >= self.max_chunk_words or ends_with_punct:
                chunks.append(
                    SubtitleChunk(
                        chunk_id=chunk_id,
                        beat_id=current_words[0].beat_id,
                        words=list(current_words),
                        start_ms=current_words[0].start_ms,
                        end_ms=current_words[-1].end_ms,
                        text=" ".join(item.word for item in current_words),
                    )
                )
                chunk_id += 1
                current_words = []

        if current_words:
            chunks.append(
                SubtitleChunk(
                    chunk_id=chunk_id,
                    beat_id=current_words[0].beat_id,
                    words=list(current_words),
                    start_ms=current_words[0].start_ms,
                    end_ms=current_words[-1].end_ms,
                    text=" ".join(item.word for item in current_words),
                )
            )

        return chunks

    def build_ass_header(self) -> str:
        """Construct the ASS script header with vertical 9:16 styles and safe zone margins."""
        header = f"""[Script Info]
Title: Sports Shorts Command Center Captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{self.font_name},{self.font_size},{self.primary_color},{self.highlight_color},{self.outline_color},&H80000000,-1,0,0,0,100,100,1,0,1,{self.outline_width},{self.shadow_depth},2,60,60,{self.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        return header

    def generate_ass_content(self, words: List[WordTimestamp]) -> str:
        """Generate ASS subtitle document with word-level highlight tags and strict beat boundaries."""
        header = self.build_ass_header()
        chunks = self.chunk_words(words)
        events: List[str] = []

        for chunk in chunks:
            chunk_word_count = len(chunk.words)
            for idx, target_word in enumerate(chunk.words):
                w_start = ms_to_ass_time(target_word.start_ms)
                # Next word's start or end of current word
                if idx + 1 < chunk_word_count:
                    next_word = chunk.words[idx + 1]
                    w_end = ms_to_ass_time(next_word.start_ms)
                else:
                    # Final word in chunk ends exactly at target_word end_ms
                    w_end = ms_to_ass_time(target_word.end_ms)

                # Format chunk line: target word in highlight color with pop scale, other words in white
                formatted_words: List[str] = []
                for j, w in enumerate(chunk.words):
                    if j == idx:
                        clean_word = w.word.upper()
                        formatted_words.append(
                            f"{{\\c{self.highlight_color}\\fscx108\\fscy108}}{clean_word}{{\\c{self.primary_color}\\fscx100\\fscy100}}"
                        )
                    else:
                        formatted_words.append(w.word.upper())

                line_text = " ".join(formatted_words)
                events.append(f"Dialogue: 0,{w_start},{w_end},Default,,0,0,0,,{line_text}")

        return header + "\n".join(events) + "\n"

    def generate_srt_content(self, words: List[WordTimestamp]) -> str:
        """Generate clean SRT subtitles grouped by beat-constrained chunks."""
        chunks = self.chunk_words(words)
        lines: List[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            start = ms_to_srt_time(chunk.start_ms)
            end = ms_to_srt_time(chunk.end_ms)
            lines.append(f"{idx}\n{start} --> {end}\n{chunk.text.upper()}\n")

        return "\n".join(lines)

    def write_subtitles(
        self,
        words: List[WordTimestamp],
        output_dir: Path,
        job_id: str,
        beats: Optional[List[BeatTimestamp]] = None,
    ) -> SubtitleFileRecord:
        """Generate and save both .ass and .srt subtitle files to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ass_path = output_dir / f"captions_{job_id}.ass"
        srt_path = output_dir / f"captions_{job_id}.srt"

        ass_content = self.generate_ass_content(words)
        srt_content = self.generate_srt_content(words)

        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        chunks = self.chunk_words(words)
        logger.info(
            "Wrote subtitles for job %s: ASS at %s, SRT at %s (%d chunks, %d words, %d beats)",
            job_id,
            ass_path,
            srt_path,
            len(chunks),
            len(words),
            len(beats or []),
        )

        return SubtitleFileRecord(
            job_id=job_id,
            ass_path=str(ass_path),
            srt_path=str(srt_path),
            chunk_count=len(chunks),
            word_count=len(words),
            words=words,
            beats=beats or [],
        )
