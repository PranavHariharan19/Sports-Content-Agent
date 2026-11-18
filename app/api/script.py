from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.logging_config import get_logger
from llm.router import LLMRouter
from pipeline.db import DatabaseManager
from pipeline.script_stage import ScriptPipelineStage
from pipeline.state_machine import PipelineStateMachine
from schemas.config import RootConfig
from schemas.script import BeatOverrideRequest, GeneratedScript, ScriptApproveRequest

logger = get_logger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["Script Generation & Verification"])


def get_db(request: Request) -> DatabaseManager:
    return request.app.state.db


def get_state_machine(request: Request) -> PipelineStateMachine:
    return request.app.state.state_machine


def get_config(request: Request) -> RootConfig:
    return request.app.state.config


def get_llm_router(request: Request) -> LLMRouter:
    if not hasattr(request.app.state, "llm_router"):
        cfg = request.app.state.config
        db = request.app.state.db
        request.app.state.llm_router = LLMRouter(cfg.llm, db)
    return request.app.state.llm_router


@router.post("/{job_id}/draft-script", response_model=GeneratedScript)
async def draft_script(
    job_id: str,
    db: DatabaseManager = Depends(get_db),
    sm: PipelineStateMachine = Depends(get_state_machine),
    cfg: RootConfig = Depends(get_config),
    router: LLMRouter = Depends(get_llm_router),
) -> GeneratedScript:
    """Generate structured script, execute deterministic linter, and run verifier pass."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    stage = ScriptPipelineStage(
        db=db,
        llm_router=router,
        pipeline_config=cfg.pipeline,
        state_machine=sm,
    )

    try:
        script = await stage.draft_and_verify_script(job_id)
        return script
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Script generation failed for job %s: %s", job_id, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Script generation failed: {str(e)}")


@router.get("/{job_id}/script", response_model=GeneratedScript)
async def get_script(
    job_id: str,
    db: DatabaseManager = Depends(get_db),
) -> GeneratedScript:
    """Retrieve the generated script with beat-by-beat provenance and verifier verdicts."""
    script = db.get_script_for_job(job_id)
    if not script:
        raise HTTPException(status_code=404, detail="No script found for this job")
    return script


@router.post("/{job_id}/override-beat", response_model=GeneratedScript)
async def override_beat(
    job_id: str,
    payload: BeatOverrideRequest,
    db: DatabaseManager = Depends(get_db),
) -> GeneratedScript:
    """Record an explicit human override on a script beat with full provenance logging."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    updated_script = db.record_beat_override(
        job_id=job_id,
        beat_id=payload.beat_id,
        action=payload.action,
        reason=payload.override_reason,
        edited_text=payload.edited_text,
    )
    if not updated_script:
        raise HTTPException(status_code=404, detail=f"Failed to override beat {payload.beat_id}")
    return updated_script


@router.post("/{job_id}/approve-script", response_model=GeneratedScript)
async def approve_script(
    job_id: str,
    force: bool = Query(True, description="Human approval overrides individual line states"),
    db: DatabaseManager = Depends(get_db),
    sm: PipelineStateMachine = Depends(get_state_machine),
    cfg: RootConfig = Depends(get_config),
    router: LLMRouter = Depends(get_llm_router),
) -> GeneratedScript:
    """Human Checkpoint 2: Approve script to advance pipeline to SCRIPT_APPROVED."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    stage = ScriptPipelineStage(
        db=db,
        llm_router=router,
        pipeline_config=cfg.pipeline,
        state_machine=sm,
    )

    try:
        approved_script = stage.approve_script_checkpoint(job_id, force=force)
        return approved_script
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Script approval failed for job %s: %s", job_id, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Script approval failed: {str(e)}")
