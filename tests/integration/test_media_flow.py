"""Integration tests for Media Grounding flow and FastAPI endpoints."""
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image

from pipeline.db import DatabaseManager
from schemas.common import CorroborationStatus, FactType, JobStatus, SourceTier, SportType, StageName
from schemas.facts import ExtractedFact
from schemas.script import BeatType, GeneratedScript, ScriptBeat
from schemas.sources import StoredSource
from schemas.voice import AudioMetadata, BeatTimestamp, SubtitleFileRecord, VoiceStageResult


def _seed_job_with_approved_script_and_facts(db: DatabaseManager, job_id: str):
    """Seed sources, facts, and an approved script with voice stage run."""
    db.create_job(job_id=job_id, sport=SportType.FOOTBALL)
    db.update_job_status(job_id, JobStatus.SCRIPT_APPROVED)

    # Seed source
    src = StoredSource(
        source_id="src_42",
        job_id=job_id,
        url="https://bbc.com/sport/football/1",
        domain="bbc.com",
        source_tier=SourceTier.TIER_1,
        cleaned_text="Dani Carvajal scored in the 99th minute to end Spain's 43-game unbeaten run.",
    )
    db.save_source(src)

    # Seed facts: 1 number fact, 1 entity fact
    f1 = ExtractedFact(
        fact_id="fact_num_1",
        job_id=job_id,
        claim="Spain 43-game unbeaten run ended",
        fact_type=FactType.RECORD,
        normalized_entities=["Spain"],
        normalized_value="43 GAMES",
        source_id="src_42",
        evidence_quote="end Spain's 43-game unbeaten run",
        source_tier=SourceTier.TIER_1,
        corroboration_status=CorroborationStatus.VERIFIED,
    )
    f2 = ExtractedFact(
        fact_id="fact_ent_2",
        job_id=job_id,
        claim="Dani Carvajal scored in the 99th minute",
        fact_type=FactType.OTHER,
        normalized_entities=["Dani Carvajal"],
        normalized_value="99th minute",
        source_id="src_42",
        evidence_quote="Dani Carvajal scored in the 99th minute",
        source_tier=SourceTier.TIER_1,
        corroboration_status=CorroborationStatus.VERIFIED,
    )
    db.save_facts([f1, f2])

    # Seed approved script
    script = GeneratedScript(
        script_id=f"script_{job_id}",
        job_id=job_id,
        headline="England Stuns Spain in Nations League Thriller",
        beats=[
            ScriptBeat(
                beat_id=1,
                beat_type=BeatType.HOOK,
                spoken_text="Did England just pull off the unthinkable against the world champions?",
                cited_fact_ids=[],
                is_factual=False,
            ),
            ScriptBeat(
                beat_id=2,
                beat_type=BeatType.FACTUAL_BEAT,
                spoken_text="Dani Carvajal scored a dramatic header at the Wanda Metropolitano.",
                cited_fact_ids=["fact_ent_2"],
                is_factual=True,
            ),
            ScriptBeat(
                beat_id=3,
                beat_type=BeatType.FACTUAL_BEAT,
                spoken_text="The defeat ended Spain's incredible 43-game unbeaten run!",
                cited_fact_ids=["fact_num_1"],
                is_factual=True,
            ),
        ],
        total_words=35,
        estimated_duration_seconds=14.0,
        model_used="gemini-2.5-flash",
        verifier_model_used="gemini-2.5-flash",
        lint_passed=True,
        all_supported=True,
        is_approved=True,
    )
    db.save_script(script)

    # Seed Voice Stage Run with beat timings
    voice_res = VoiceStageResult(
        job_id=job_id,
        audio=AudioMetadata(
            audio_path="data/renders/test.mp3",
            duration_seconds=14.0,
            voice_used="en-GB-RyanNeural",
            word_count=35,
        ),
        subtitles=SubtitleFileRecord(
            job_id=job_id,
            ass_path="data/renders/test.ass",
            srt_path="data/renders/test.srt",
            chunk_count=7,
            word_count=35,
        ),
        words_total=35,
        total_duration_seconds=14.0,
        beat_timings=[
            BeatTimestamp(beat_id=1, start_ms=0, end_ms=3500, spoken_text="Hook", word_count=10),
            BeatTimestamp(beat_id=2, start_ms=3500, end_ms=8500, spoken_text="Carvajal", word_count=12),
            BeatTimestamp(beat_id=3, start_ms=8500, end_ms=14000, spoken_text="43-game", word_count=13),
        ],
    )
    db.record_stage_run(
        job_id=job_id,
        stage_name=StageName.VOICE,
        input_data={},
        output_data=voice_res.model_dump(mode="json"),
        status="COMPLETED",
    )


def test_media_generation_and_api_flow(test_client: TestClient, db_manager: DatabaseManager):
    """End-to-end integration: generate grounded media assets, verify 1080x1920 visuals and state transition."""
    job_id = "job-media-flow-99"
    _seed_job_with_approved_script_and_facts(db_manager, job_id)

    # 1. POST /api/jobs/{job_id}/generate-media
    resp = test_client.post(f"/api/jobs/{job_id}/generate-media")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["job_id"] == job_id
    assert data["total_assets"] == 3
    assert len(data["assets"]) == 3

    # Verify job status advanced to MEDIA_READY
    job = db_manager.get_job(job_id)
    assert job.status == JobStatus.MEDIA_READY

    # Beat 3 is a record fact -> PIL stat card
    beat_3_asset = data["assets"][2]
    assert beat_3_asset["media_type"] == "stat_card"
    assert beat_3_asset["source_provider"] == "pil_stat_card"
    assert "43" in beat_3_asset["headline"] or "43" in beat_3_asset["caption_text"]

    # Verify physical file on disk is valid 1080x1920 image
    for asset in data["assets"]:
        p = Path(asset["file_path"])
        assert p.exists()
        with Image.open(p) as im:
            assert im.size == (1080, 1920)

    # 2. GET /api/jobs/{job_id}/media
    resp_get = test_client.get(f"/api/jobs/{job_id}/media")
    assert resp_get.status_code == 200
    get_data = resp_get.json()
    assert get_data["total_assets"] == 3

    # 3. GET /api/jobs/{job_id}/media/{beat_id}/image
    resp_img = test_client.get(f"/api/jobs/{job_id}/media/2/image")
    assert resp_img.status_code == 200
    assert resp_img.headers["content-type"] == "image/jpeg"
    assert len(resp_img.content) > 1000


def test_media_generation_unapproved_script_fails(test_client: TestClient, db_manager: DatabaseManager):
    """Verify that attempting to generate media on an unapproved job fails with 400."""
    job_id = "job-unapproved-media"
    db_manager.create_job(job_id=job_id, sport=SportType.F1)
    # Status is DISCOVERED

    resp = test_client.post(f"/api/jobs/{job_id}/generate-media")
    assert resp.status_code == 400
    assert "must be in SCRIPT_APPROVED status" in resp.json()["detail"]
