"""FastAPI endpoints for Media Grounding (Wikimedia, Pexels, Pixabay, PIL Stat Cards)."""
import json
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from app.logging_config import get_logger
from pipeline.db import DatabaseManager
from pipeline.media_stage import MediaPipelineStage
from pipeline.state_machine import PipelineStateMachine
from schemas.common import StageName
from schemas.media import MediaAsset, MediaStageResult

logger = get_logger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["Media Grounding"])


def get_db(request: Request) -> DatabaseManager:
    return request.app.state.db


def get_state_machine(request: Request) -> PipelineStateMachine:
    return request.app.state.state_machine


@router.post("/{job_id}/generate-media", response_model=MediaStageResult)
async def generate_media(
    job_id: str,
    db: DatabaseManager = Depends(get_db),
    sm: PipelineStateMachine = Depends(get_state_machine),
) -> MediaStageResult:
    """Generate 1080x1920 grounded media assets (photos + PIL stat cards) for an approved script."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    stage = MediaPipelineStage(db=db, state_machine=sm)
    try:
        result = await stage.process_media_for_job(job_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Media generation failed for job %s: %s", job_id, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Media generation failed: {str(e)}")


@router.get("/{job_id}/media", response_model=MediaStageResult)
async def get_media(
    job_id: str,
    db: DatabaseManager = Depends(get_db),
) -> MediaStageResult:
    """Retrieve grounded media assets for a job."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    assets = db.get_media_assets_for_job(job_id)
    if not assets:
        raise HTTPException(status_code=404, detail="Media assets have not been generated for this job yet")

    total_duration = assets[-1].end_sec if assets else 0.0
    providers = sorted(list(set(a.source_provider for a in assets)))

    return MediaStageResult(
        job_id=job_id,
        assets=assets,
        total_assets=len(assets),
        total_duration_seconds=round(total_duration, 2),
        providers_used=providers,
    )


@router.get("/{job_id}/media/{beat_id}/image")
async def get_media_image(
    job_id: str,
    beat_id: int,
    db: DatabaseManager = Depends(get_db),
) -> FileResponse:
    """Stream or download the 1080x1920 image asset for a specific beat."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    asset = db.get_media_asset_for_beat(job_id, beat_id)
    if not asset:
        raise HTTPException(status_code=404, detail=f"No media asset found for beat {beat_id}")

    image_path = Path(asset.file_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Media file on disk not found: {asset.file_path}")

    return FileResponse(
        path=str(image_path),
        media_type="image/jpeg",
        filename=image_path.name,
    )
