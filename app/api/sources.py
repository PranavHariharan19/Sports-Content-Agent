from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.logging_config import get_logger
from pipeline.db import DatabaseManager
from pipeline.sources_stage import SourcesPipelineStage
from pipeline.state_machine import PipelineStateMachine
from schemas.sources import SourceFetchResult, StoredSource

logger = get_logger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["Source Fetching & Trust"])


def get_db(request: Request) -> DatabaseManager:
    return request.app.state.db


def get_state_machine(request: Request) -> PipelineStateMachine:
    return request.app.state.state_machine


@router.post("/{job_id}/fetch-sources", response_model=SourceFetchResult)
async def fetch_sources_for_job(
    job_id: str,
    db: DatabaseManager = Depends(get_db),
    sm: PipelineStateMachine = Depends(get_state_machine),
) -> SourceFetchResult:
    """Fetch full ground-truth texts for the chosen topic, enforce trust tiers,

    sanitize prompt-injections, and de-duplicate syndicated wire copy.
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    stage = SourcesPipelineStage(db=db, state_machine=sm)
    try:
        result = await stage.fetch_and_store_sources_for_job(job_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to fetch sources for job %s: %s", job_id, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Source fetching failed: {str(e)}")


@router.get("/{job_id}/sources", response_model=List[StoredSource])
async def get_job_sources(
    job_id: str,
    allowlisted_only: bool = Query(False, description="Filter only allowlisted domains"),
    db: DatabaseManager = Depends(get_db),
) -> List[StoredSource]:
    """Retrieve all stored ground-truth source texts and syndication metadata for a job."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if allowlisted_only:
        return db.get_allowlisted_sources_for_job(job_id)
    return db.get_sources_for_job(job_id)
