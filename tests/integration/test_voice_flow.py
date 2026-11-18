"""Integration tests for Voice & Captions generation flow and FastAPI endpoints."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from pipeline.db import DatabaseManager
from pipeline.voice_stage import VoicePipelineStage
from schemas.common import JobStatus, SportType
from schemas.script import BeatType, GeneratedScript, ScriptBeat


def _seed_approved_script(db: DatabaseManager, job_id: str) -> GeneratedScript:
    """Helper to seed an approved script with beats into the database."""
    script = GeneratedScript(
        script_id=f"script_{job_id}",
        job_id=job_id,
        headline="Spain Claims Nations League Thriller Against England",
        beats=[
            ScriptBeat(
                beat_id=1,
                beat_type=BeatType.HOOK,
                spoken_text="Spain just pulled off a sensational victory over England!",
                cited_fact_ids=[],
                is_factual=False,
            ),
            ScriptBeat(
                beat_id=2,
                beat_type=BeatType.FACTUAL_BEAT,
                spoken_text="Dani Carvajal scored the dramatic 99th minute header at the Wanda Metropolitano.",
                cited_fact_ids=[],
                is_factual=True,
            ),
            ScriptBeat(
                beat_id=3,
                beat_type=BeatType.CLOSING,
                spoken_text="Can Gareth Southgate bounce back? Subscribe for all the latest updates!",
                cited_fact_ids=[],
                is_factual=False,
            ),
        ],
        total_words=32,
        estimated_duration_seconds=12.8,
        model_used="gemini-2.5-flash",
        verifier_model_used="gemini-2.5-flash",
        lint_passed=True,
        all_supported=True,
        is_approved=True,
    )
    db.save_script(script)
    return script


def test_list_voices_endpoint(test_client: TestClient):
    """Verify listing available commentary voices."""
    resp = test_client.get("/api/voices")
    assert resp.status_code == 200
    voices = resp.json()
    assert isinstance(voices, list)
    assert len(voices) >= 3
    ids = [v["id"] for v in voices]
    assert "en-GB-RyanNeural" in ids
    assert "en-US-ChristopherNeural" in ids


def test_voice_generation_and_caption_flow(test_client: TestClient, db_manager: DatabaseManager, tmp_path: Path):
    """End-to-end integration: generate voice and captions, verify audio streaming and ASS/SRT formats."""
    job_id = "job-voice-test-42"
    db_manager.create_job(job_id=job_id, sport=SportType.FOOTBALL)
    db_manager.update_job_status(job_id, JobStatus.SCRIPT_APPROVED)
    _seed_approved_script(db_manager, job_id)

    # Mock edge-tts communicate stream to guarantee fast, deterministic integration test
    mock_chunks = [
        {"type": "audio", "data": b"\xff\xfb\x90\x00" * 50},
        {"type": "WordBoundary", "offset": 1_000_000, "duration": 4_000_000, "text": "Spain"},
        {"type": "WordBoundary", "offset": 5_500_000, "duration": 3_000_000, "text": "just"},
        {"type": "WordBoundary", "offset": 9_000_000, "duration": 4_500_000, "text": "won!"},
    ]

    async def fake_stream():
        for c in mock_chunks:
            yield c

    mock_comm = MagicMock()
    mock_comm.stream = fake_stream

    with patch("edge_tts.Communicate", return_value=mock_comm):
        # 1. POST generate-voice
        resp = test_client.post(
            f"/api/jobs/{job_id}/generate-voice",
            json={"voice_override": "en-GB-RyanNeural"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["audio"]["voice_used"] == "en-GB-RyanNeural"
        assert data["words_total"] == 3
        assert data["subtitles"]["chunk_count"] >= 1

        # 2. GET voice info
        resp_info = test_client.get(f"/api/jobs/{job_id}/voice")
        assert resp_info.status_code == 200
        assert resp_info.json()["words_total"] == 3

        # 3. GET audio stream
        resp_audio = test_client.get(f"/api/jobs/{job_id}/audio")
        assert resp_audio.status_code == 200
        assert resp_audio.headers["content-type"] in ("audio/mpeg", "audio/wav")
        assert len(resp_audio.content) > 0

        # 4. GET captions JSON
        resp_cap_json = test_client.get(f"/api/jobs/{job_id}/captions?format=json")
        assert resp_cap_json.status_code == 200
        cap_data = resp_cap_json.json()
        assert cap_data["word_count"] == 3
        assert len(cap_data["chunks"]) >= 1

        # 5. GET captions ASS
        resp_ass = test_client.get(f"/api/jobs/{job_id}/captions?format=ass")
        assert resp_ass.status_code == 200
        assert "PlayResX: 1080" in resp_ass.text
        assert "PlayResY: 1920" in resp_ass.text
        assert r"\fscx108\fscy108" in resp_ass.text

        # 6. GET captions SRT
        resp_srt = test_client.get(f"/api/jobs/{job_id}/captions?format=srt")
        assert resp_srt.status_code == 200
        assert "-->" in resp_srt.text


def test_voice_generation_unapproved_script_fails(test_client: TestClient, db_manager: DatabaseManager):
    """Verify voice generation fails if job is not in SCRIPT_APPROVED status."""
    job_id = "job-unapproved"
    db_manager.create_job(job_id=job_id, sport=SportType.FOOTBALL)
    # Status is DISCOVERED, not SCRIPT_APPROVED

    resp = test_client.post(f"/api/jobs/{job_id}/generate-voice")
    assert resp.status_code == 400
    assert "must be in SCRIPT_APPROVED status" in resp.json()["detail"]
