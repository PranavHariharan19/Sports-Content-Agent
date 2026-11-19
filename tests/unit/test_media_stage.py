"""Unit tests for Media Pipeline Stage query refinement, entity detection, and visual replacement."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pipeline.db import DatabaseManager
from pipeline.media_stage import MediaPipelineStage, STOP_WORDS
from schemas.common import FactType, JobStatus, SportType, StageName, SourceTier, CorroborationStatus
from schemas.facts import ExtractedFact
from schemas.script import BeatType, ScriptBeat, GeneratedScript
from schemas.media import MediaType


def test_clean_entity_strips_stop_words():
    stage = MediaPipelineStage(MagicMock(spec=DatabaseManager))
    assert stage._clean_entity("Did England") == "England"
    assert stage._clean_entity("In this UEFA") == "UEFA"
    assert stage._clean_entity("Tuchel's") == "Tuchel"
    assert stage._clean_entity("Spain's") == "Spain"
    assert stage._clean_entity("What an") is None
    assert stage._clean_entity("65") is None


def test_is_person_name_heuristics():
    stage = MediaPipelineStage(MagicMock(spec=DatabaseManager))
    assert stage._is_person_name("Roy Keane") is True
    assert stage._is_person_name("Thomas Tuchel") is True
    assert stage._is_person_name("Mikel Oyarzabal") is True
    assert stage._is_person_name("Ben White") is True
    # Non-person entities
    assert stage._is_person_name("Nations League") is False
    assert stage._is_person_name("First Half") is False
    assert stage._is_person_name("Red Card") is False
    assert stage._is_person_name("Wanda Metropolitano") is False


def make_fact(fact_id: str, claim: str, normalized_val: str, fact_type: FactType) -> ExtractedFact:
    return ExtractedFact(
        fact_id=fact_id,
        job_id="test_job",
        source_id="test_source",
        claim=claim,
        evidence_quote=claim,
        normalized_value=normalized_val,
        fact_type=fact_type,
        corroboration_status=CorroborationStatus.VERIFIED,
        source_tier=SourceTier.TIER_1,
        source_url="https://skysports.com/1",
        published_at="2026-11-16T22:00:00Z",
    )


def test_is_stat_beat_prefers_photos_for_named_people():
    stage = MediaPipelineStage(MagicMock(spec=DatabaseManager))
    beat = ScriptBeat(
        beat_id=3,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="Roy Keane called England's first half performance perfect, and goals from Ezri Konsa and Ben White followed in the first 12 minutes of the second half!",
        estimated_duration_seconds=7.5,
    )
    quote_fact = make_fact(
        "fact_1",
        "Roy Keane described England's first-half performance as perfect.",
        "perfect",
        FactType.QUOTE
    )
    # Even with fact, because Roy Keane and Ben White are named, it should NOT be classified as stat card
    assert stage._is_stat_beat(beat, [quote_fact]) is False


def test_is_stat_beat_rejects_minute_markers():
    stage = MediaPipelineStage(MagicMock(spec=DatabaseManager))
    beat = ScriptBeat(
        beat_id=4,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="Elliot Anderson got sent off after 65 minutes of intense play.",
        estimated_duration_seconds=6.0,
    )
    minute_fact = make_fact(
        "fact_2",
        "Elliot Anderson was sent off after 65 minutes.",
        "65",
        FactType.OTHER
    )
    assert stage._is_stat_beat(beat, [minute_fact]) is False


def test_is_stat_beat_accepts_genuine_records():
    stage = MediaPipelineStage(MagicMock(spec=DatabaseManager))
    beat = ScriptBeat(
        beat_id=4,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="The defeat finally ended Spain's incredible 43-game unbeaten run.",
        estimated_duration_seconds=6.0,
    )
    record_fact = make_fact(
        "fact_3",
        "Spain's 43-game unbeaten run was ended by the defeat.",
        "43",
        FactType.RECORD
    )
    assert stage._is_stat_beat(beat, [record_fact]) is True


def test_build_queries_prioritizes_athletes_and_sports_context():
    stage = MediaPipelineStage(MagicMock(spec=DatabaseManager))
    beat = ScriptBeat(
        beat_id=2,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="At the Wanda Metropolitano, Thomas Tuchel led England to a massive victory.",
        estimated_duration_seconds=6.0,
    )
    queries = stage._build_queries_for_beat(beat, [], SportType.FOOTBALL)
    assert "Thomas Tuchel football" in queries
    assert "Thomas Tuchel" in queries
    assert any("Wanda Metropolitano" in q for q in queries)
    assert any("England national football team" in q or "England football" in q for q in queries)
    # Ensure no bare question word queries like "Did England"
    assert "Did England" not in queries


def test_create_stat_card_formats_milestones_without_sentiment_words(tmp_path):
    mock_db = MagicMock(spec=DatabaseManager)
    stage = MediaPipelineStage(db=mock_db, assets_dir=str(tmp_path))
    beat = ScriptBeat(
        beat_id=4,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="The defeat ended Spain's 43-game unbeaten run.",
        estimated_duration_seconds=6.0,
    )
    record_fact = make_fact(
        "fact_3",
        "Spain's 43-game unbeaten run was ended by the defeat.",
        "43",
        FactType.RECORD
    )
    timing = {"start_ms": 10000, "end_ms": 16000, "duration_seconds": 6.0}
    out_file = tmp_path / "test_card.jpg"

    asset = stage._create_stat_card_asset(
        job_id="job_test",
        beat=beat,
        cited_facts=[record_fact],
        sport=SportType.FOOTBALL,
        timing=timing,
        output_path=out_file,
    )

    assert asset.media_type == MediaType.STAT_CARD
    assert "43" in asset.headline
    assert "PERFECT" not in asset.headline
    assert out_file.exists()
