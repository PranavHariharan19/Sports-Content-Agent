"""FastAPI endpoints for Voice Synthesis, Word-Level Timestamps, and Subtitle Captions."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.logging_config import get_logger
from pipeline.caption_generator import CaptionGenerator
from pipeline.db import DatabaseManager
from pipeline.state_machine import PipelineStateMachine
from pipeline.voice_generator import DEFAULT_SPORTS_VOICE, VoiceGenerator
from pipeline.voice_stage import VoicePipelineStage
from schemas.common import StageName
from schemas.config import RootConfig
from schemas.voice import AudioMetadata, SubtitleChunk, SubtitleFileRecord, VoiceStageResult

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["Voice & Captions"])

AVAILABLE_VOICES = [
    {
        "id": "en-GB-RyanNeural",
        "name": "Ryan (UK - Commentator)",
        "gender": "Male",
        "locale": "en-GB",
        "description": "Crisp, authoritative British broadcast commentary. Recommended for Football & F1.",
    },
    {
        "id": "en-GB-ThomasNeural",
        "name": "Thomas (UK - Energetic)",
        "gender": "Male",
        "locale": "en-GB",
        "description": "Fast-paced, energetic commentary style.",
    },
    {
        "id": "en-US-ChristopherNeural",
        "name": "Christopher (US - Confident)",
        "gender": "Male",
        "locale": "en-US",
        "description": "Deep, confident American broadcast narrator.",
    },
    {
        "id": "en-US-GuyNeural",
        "name": "Guy (US - Broadcast)",
        "gender": "Male",
        "locale": "en-US",
        "description": "Dynamic, conversational American sports anchor.",
    },
    {
        "id": "en-AU-WilliamNeural",
        "name": "William (AU - Cricket)",
        "gender": "Male",
        "locale": "en-AU",
        "description": "Australian sports commentary voice, excellent for Cricket.",
    },
    {
        "id": "en-GB-SoniaNeural",
        "name": "Sonia (UK - Studio Anchor)",
        "gender": "Female",
        "locale": "en-GB",
        "description": "Engaging, clear British studio anchor voice.",
    },
]


class GenerateVoiceRequest(BaseModel):
    voice_override: Optional[str] = Field(
        default=None,
        description="Optional voice ID override (e.g. en-GB-RyanNeural)",
    )


def get_db(request: Request) -> DatabaseManager:
    return request.app.state.db


def get_state_machine(request: Request) -> PipelineStateMachine:
    return request.app.state.state_machine


def get_config(request: Request) -> RootConfig:
    return request.app.state.config


@router.get("/voices", response_model=List[Dict[str, str]])
async def list_available_voices() -> List[Dict[str, str]]:
    """List curated sports commentary voices available for synthesis."""
    return AVAILABLE_VOICES


@router.post("/jobs/{job_id}/generate-voice", response_model=VoiceStageResult)
async def generate_voice(
    job_id: str,
    payload: Optional[GenerateVoiceRequest] = None,
    db: DatabaseManager = Depends(get_db),
    sm: PipelineStateMachine = Depends(get_state_machine),
) -> VoiceStageResult:
    """Generate audio voiceover track and synchronized ASS/SRT captions for approved script."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    voice_override = payload.voice_override if payload else None
    stage = VoicePipelineStage(db=db, state_machine=sm)

    try:
        result = await stage.process_voice_and_captions(
            job_id=job_id,
            voice_override=voice_override,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Voice generation failed for job %s: %s", job_id, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice generation failed: {str(e)}")


@router.get("/jobs/{job_id}/voice", response_model=VoiceStageResult)
async def get_voice_info(
    job_id: str,
    db: DatabaseManager = Depends(get_db),
) -> VoiceStageResult:
    """Get metadata for synthesized voiceover and caption files."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    latest_run = db.get_latest_stage_run(job_id=job_id, stage_name=StageName.VOICE)
    if not latest_run or latest_run.status != "COMPLETED":
        raise HTTPException(status_code=404, detail="Voice has not been generated for this job yet")

    try:
        data = json.loads(latest_run.output_json)
        return VoiceStageResult(**data)
    except Exception as e:
        logger.error("Failed to parse voice stage run for job %s: %s", job_id, str(e))
        raise HTTPException(status_code=500, detail="Invalid voice stage run record")


@router.get("/jobs/{job_id}/audio")
async def get_audio_file(
    job_id: str,
    db: DatabaseManager = Depends(get_db),
) -> FileResponse:
    """Stream or download the generated voiceover MP3 audio."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Check renders directory
    renders_dir = Path("data/renders") / job_id
    audio_path = renders_dir / "voice.mp3"
    media_type = "audio/mpeg"

    if not audio_path.exists():
        wav_path = renders_dir / "voice.wav"
        if wav_path.exists():
            audio_path = wav_path
            media_type = "audio/wav"
        else:
            raise HTTPException(status_code=404, detail="Audio file not found for this job")

    return FileResponse(
        path=str(audio_path),
        media_type=media_type,
        filename=audio_path.name,
    )


@router.get("/jobs/{job_id}/captions")
async def get_captions(
    job_id: str,
    format: str = Query("json", description="Output format: json, ass, or srt"),
    db: DatabaseManager = Depends(get_db),
) -> Any:
    """Retrieve captions in JSON chunk format, or download .ass / .srt file directly."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    renders_dir = Path("data/renders") / job_id
    ass_path = renders_dir / f"captions_{job_id}.ass"
    srt_path = renders_dir / f"captions_{job_id}.srt"

    if format == "ass":
        if not ass_path.exists():
            raise HTTPException(status_code=404, detail="ASS caption file not found")
        return FileResponse(
            path=str(ass_path),
            media_type="text/plain; charset=utf-8",
            filename=f"captions_{job_id}.ass",
        )
    elif format == "srt":
        if not srt_path.exists():
            raise HTTPException(status_code=404, detail="SRT caption file not found")
        return FileResponse(
            path=str(srt_path),
            media_type="text/plain; charset=utf-8",
            filename=f"captions_{job_id}.srt",
        )
    else:  # json
        latest_run = db.get_latest_stage_run(job_id=job_id, stage_name=StageName.VOICE)
        if not latest_run or latest_run.status != "COMPLETED":
            raise HTTPException(status_code=404, detail="Captions have not been generated yet")
        try:
            data = json.loads(latest_run.output_json)
            stage_result = VoiceStageResult(**data)
            cg = CaptionGenerator()
            chunks = cg.chunk_words(stage_result.subtitles.words)
            return {
                "job_id": job_id,
                "ass_path": stage_result.subtitles.ass_path,
                "srt_path": stage_result.subtitles.srt_path,
                "word_count": stage_result.words_total,
                "chunk_count": len(chunks),
                "chunks": [c.model_dump() for c in chunks],
                "words": [w.model_dump() for w in stage_result.subtitles.words],
            }
        except Exception as e:
            logger.error("Failed to parse captions JSON for job %s: %s", job_id, str(e))
            raise HTTPException(status_code=500, detail="Failed to retrieve captions data")
