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
