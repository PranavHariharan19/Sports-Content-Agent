"""Unit tests for ASS and SRT caption generation with word-level highlighting."""
from pathlib import Path
import pytest

from pipeline.caption_generator import (
    CaptionGenerator,
    ms_to_ass_time,
    ms_to_srt_time,
)
from schemas.voice import WordTimestamp


def test_time_conversions():
    """Verify millisecond to ASS and SRT timestamp formatting."""
    assert ms_to_ass_time(0) == "0:00:00.00"
    assert ms_to_ass_time(1540) == "0:00:01.54"
    assert ms_to_ass_time(65120) == "0:01:05.12"

    assert ms_to_srt_time(0) == "00:00:00,000"
    assert ms_to_srt_time(1540) == "00:00:01,540"
    assert ms_to_srt_time(65120) == "00:01:05,120"


def test_chunk_words_grouping():
    """Verify rhythmic 2-4 word chunking and punctuation break behavior."""
    gen = CaptionGenerator(max_chunk_words=4)
    words = [
        WordTimestamp(word="Spain", start_ms=100, end_ms=400),
        WordTimestamp(word="beat", start_ms=450, end_ms=700),
        WordTimestamp(word="England.", start_ms=750, end_ms=1100),
        WordTimestamp(word="A", start_ms=1200, end_ms=1300),
        WordTimestamp(word="stunning", start_ms=1350, end_ms=1700),
        WordTimestamp(word="late", start_ms=1750, end_ms=2000),
        WordTimestamp(word="winner", start_ms=2050, end_ms=2400),
        WordTimestamp(word="sealed", start_ms=2450, end_ms=2800),
    ]

    chunks = gen.chunk_words(words)
    # "Spain beat England." has punctuation on "England.", so chunk 1 must terminate at 3 words
    assert len(chunks) == 3
    assert chunks[0].text == "Spain beat England."
    assert chunks[0].start_ms == 100
    assert chunks[0].end_ms == 1100
    assert len(chunks[0].words) == 3

    # Next chunk takes up to 4 words: "A stunning late winner"
    assert chunks[1].text == "A stunning late winner"
    assert len(chunks[1].words) == 4

    # Last chunk has remaining word "sealed"
    assert chunks[2].text == "sealed"
    assert len(chunks[2].words) == 1


def test_empty_words_chunking():
    """Verify empty words list returns empty chunks."""
    gen = CaptionGenerator()
    assert gen.chunk_words([]) == []


def test_chunk_words_never_spans_multiple_beats():
    """Verify that chunks never mix words from two different beats even without punctuation."""
    gen = CaptionGenerator(max_chunk_words=4)
    # Beat 1 ends with only 2 words and NO punctuation
    # Beat 2 starts immediately with 2 words
    words = [
        WordTimestamp(word="world", start_ms=2000, end_ms=2500, beat_id=1),
        WordTimestamp(word="champions", start_ms=2500, end_ms=3200, beat_id=1),
        WordTimestamp(word="In", start_ms=4000, end_ms=4300, beat_id=2),
        WordTimestamp(word="this", start_ms=4300, end_ms=4600, beat_id=2),
        WordTimestamp(word="thrilling", start_ms=4600, end_ms=5000, beat_id=2),
        WordTimestamp(word="match", start_ms=5000, end_ms=5400, beat_id=2),
    ]

    chunks = gen.chunk_words(words)
    # Must NOT produce "world champions In this"
    # Chunk 1 must contain ONLY Beat 1 words
    assert len(chunks) == 2
    assert chunks[0].beat_id == 1
    assert chunks[0].text == "world champions"
    assert len(chunks[0].words) == 2
    assert chunks[0].end_ms == 3200

    # Chunk 2 must start fresh with Beat 2 words
    assert chunks[1].beat_id == 2
    assert chunks[1].text == "In this thrilling match"
    assert chunks[1].start_ms == 4000
    assert len(chunks[1].words) == 4


def test_ass_header_dimensions_and_safezone():
    """Verify ASS header contains 9:16 vertical resolution and 420px lower safe zone."""
    gen = CaptionGenerator(margin_v=420, font_name="Arial Black", font_size=70)
    header = gen.build_ass_header()

    assert "PlayResX: 1080" in header
    assert "PlayResY: 1920" in header
    assert "Arial Black" in header
    assert ",420," in header  # MarginV safe zone


def test_ass_content_highlighting():
    """Verify word-level pop highlight tags in ASS events."""
    gen = CaptionGenerator()
    words = [
        WordTimestamp(word="Lamine", start_ms=0, end_ms=500),
        WordTimestamp(word="Yamal", start_ms=510, end_ms=1000),
    ]

    ass_text = gen.generate_ass_content(words)
    assert "[Events]" in ass_text
    assert "Dialogue: 0,0:00:00.00,0:00:00.51,Default" in ass_text
    assert r"{\c&H0000FFFF\fscx108\fscy108}LAMINE" in ass_text
    assert r"{\c&H0000FFFF\fscx108\fscy108}YAMAL" in ass_text


def test_srt_content_format():
    """Verify standard SRT format numbering and timestamps."""
    gen = CaptionGenerator()
    words = [
        WordTimestamp(word="Ferrari", start_ms=500, end_ms=900),
        WordTimestamp(word="wins!", start_ms=950, end_ms=1500),
    ]

    srt_text = gen.generate_srt_content(words)
    assert "1\n00:00:00,500 --> 00:00:01,500\nFERRARI WINS!" in srt_text


def test_write_subtitles_to_disk(tmp_path: Path):
    """Verify writing .ass and .srt files to directory."""
    gen = CaptionGenerator()
    words = [
        WordTimestamp(word="India", start_ms=100, end_ms=500),
        WordTimestamp(word="triumphs.", start_ms=550, end_ms=1200),
    ]

    record = gen.write_subtitles(words, output_dir=tmp_path, job_id="test_job_42")
    assert record.job_id == "test_job_42"
    assert Path(record.ass_path).exists()
    assert Path(record.srt_path).exists()
    assert record.word_count == 2
    assert record.chunk_count == 1
    assert "INDIA TRIUMPHS." in Path(record.srt_path).read_text()
