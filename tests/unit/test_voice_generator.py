"""Unit tests for VoiceGenerator: Edge-TTS synthesis, word timestamps, and offline fallback."""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from pipeline.voice_generator import DEFAULT_SPORTS_VOICE, VoiceGenerator
from schemas.voice import WordTimestamp


@pytest.mark.asyncio
async def test_voice_generator_initialization():
    """Verify default and custom parameters."""
    vg = VoiceGenerator()
    assert vg.voice == DEFAULT_SPORTS_VOICE
    assert vg.rate == "+0%"

    vg_custom = VoiceGenerator(voice="en-GB-ThomasNeural", rate="+10%")
    assert vg_custom.voice == "en-GB-ThomasNeural"
    assert vg_custom.rate == "+10%"


@pytest.mark.asyncio
async def test_edge_tts_stream_parsing(tmp_path: Path):
    """Verify that Communicate chunks are parsed into audio file and WordTimestamps."""
    vg = VoiceGenerator()
    out_file = tmp_path / "voice.mp3"

    mock_chunks = [
        {"type": "audio", "data": b"\xff\xfb\x90\x00" * 20},
        {
            "type": "WordBoundary",
            "offset": 1_000_000,   # 100 ms
            "duration": 4_000_000, # 400 ms
            "text": "Ferrari",
        },
        {"type": "audio", "data": b"\xff\xfb\x90\x00" * 20},
        {
            "type": "WordBoundary",
            "offset": 5_500_000,   # 550 ms
            "duration": 3_500_000, # 350 ms
            "text": "dominates",
        },
    ]

    async def fake_stream():
        for c in mock_chunks:
            yield c

    mock_comm = MagicMock()
    mock_comm.stream = fake_stream

    with patch("edge_tts.Communicate", return_value=mock_comm):
        metadata, words = await vg._synthesize_edge_tts("Ferrari dominates", out_file, "job_test")

    assert out_file.exists()
    assert out_file.stat().st_size > 0
    assert len(words) == 2
    assert words[0].word == "Ferrari"
    assert words[0].start_ms == 100
    assert words[0].end_ms == 500
    assert words[1].word == "dominates"
    assert words[1].start_ms == 550
    assert words[1].end_ms == 900
    assert metadata.duration_seconds == 0.9
    assert metadata.word_count == 2


@pytest.mark.asyncio
async def test_offline_fallback_when_edge_tts_fails(tmp_path: Path):
    """Verify graceful fallback synthesis when edge-tts raises network/service error."""
    vg = VoiceGenerator()
    out_file = tmp_path / "fallback.mp3"

    with patch.object(vg, "_synthesize_edge_tts", side_effect=ConnectionError("ReadAloud network timeout")):
        metadata, words = await vg.synthesize("Lamine Yamal with an unbelievable goal!", out_file, "fallback_job")

    # Should fall back cleanly and produce a valid audio file (e.g. .wav)
    assert Path(metadata.audio_path).exists()
    assert Path(metadata.audio_path).stat().st_size > 0
    assert len(words) == 6
    assert words[0].word == "Lamine"
    assert words[-1].word == "goal!"
    assert metadata.voice_used == "offline-fallback"
    assert metadata.duration_seconds > 0.0


@pytest.mark.asyncio
async def test_edge_tts_live_quick_synthesis(tmp_path: Path):
    """Integration test with live Edge-TTS service if network available."""
    vg = VoiceGenerator(voice="en-GB-RyanNeural")
    out_file = tmp_path / "live_test.mp3"

    try:
        metadata, words = await vg.synthesize("Fast corner taken quickly.", out_file, "live_test_job")
        assert Path(metadata.audio_path).exists()
        assert Path(metadata.audio_path).stat().st_size > 0
        assert len(words) >= 4
        assert words[0].word.lower().startswith("fast")
    except Exception as e:
        # If offline or rate limited in CI, skip live test
        pytest.skip(f"Live edge-tts network unreachable: {e}")


def test_align_words_to_beats():
    """Verify that word timestamps are accurately mapped to individual script beats."""
    from pipeline.voice_stage import VoicePipelineStage
    from schemas.script import BeatType, ScriptBeat

    beats = [
        ScriptBeat(
            beat_id=1,
            beat_type=BeatType.HOOK,
            spoken_text="Spain defeated England against the world champions!",
            cited_fact_ids=[],
        ),
        ScriptBeat(
            beat_id=2,
            beat_type=BeatType.FACTUAL_BEAT,
            spoken_text="In this Nations League clash, Tuchel was stunned.",
            cited_fact_ids=[],
        ),
    ]

    words = [
        WordTimestamp(word="Spain", start_ms=100, end_ms=400),
        WordTimestamp(word="defeated", start_ms=450, end_ms=800),
        WordTimestamp(word="England", start_ms=850, end_ms=1200),
        WordTimestamp(word="against", start_ms=1250, end_ms=1500),
        WordTimestamp(word="the", start_ms=1550, end_ms=1700),
        WordTimestamp(word="world", start_ms=1750, end_ms=2100),
        WordTimestamp(word="champions", start_ms=2150, end_ms=2700),  # End of Beat 1 (TTS stripped punctuation)
        WordTimestamp(word="In", start_ms=3500, end_ms=3700),          # Start of Beat 2
        WordTimestamp(word="this", start_ms=3750, end_ms=4000),
        WordTimestamp(word="Nations", start_ms=4050, end_ms=4400),
        WordTimestamp(word="League", start_ms=4450, end_ms=4800),
        WordTimestamp(word="clash", start_ms=4850, end_ms=5200),
        WordTimestamp(word="Tuchel", start_ms=5250, end_ms=5600),
        WordTimestamp(word="was", start_ms=5650, end_ms=5800),
        WordTimestamp(word="stunned", start_ms=5850, end_ms=6300),
    ]

    aligned_words, beat_timings = VoicePipelineStage.align_words_to_beats(beats, words)

    assert len(beat_timings) == 2
    # Beat 1 timings
    assert beat_timings[0].beat_id == 1
    assert beat_timings[0].start_ms == 100
    assert beat_timings[0].end_ms == 2700
    assert beat_timings[0].word_count == 7

    # Beat 2 timings
    assert beat_timings[1].beat_id == 2
    assert beat_timings[1].start_ms == 3500
    assert beat_timings[1].end_ms == 6300
    assert beat_timings[1].word_count == 8

    # Beat 1 words must all have beat_id=1, Beat 2 must have beat_id=2
    for w in aligned_words[:7]:
        assert w.beat_id == 1
    for w in aligned_words[7:]:
        assert w.beat_id == 2

    # Punctuation restored on terminating word of Beat 1
    assert aligned_words[6].word == "champions!"
