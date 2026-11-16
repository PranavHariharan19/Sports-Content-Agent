from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.logging_config import get_logger
from llm.router import LLMRouter
from pipeline.db import DatabaseManager
from pipeline.facts_stage import FactsPipelineStage, InsufficientVerifiedFactsError
from pipeline.state_machine import PipelineStateMachine
from schemas.config import RootConfig
from schemas.facts import ConflictRecord, ExtractedFact, FactLedger, ResolveConflictRequest

logger = get_logger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["Fact Extraction & Verification"])


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


@router.post("/{job_id}/extract-facts", response_model=FactLedger)
async def extract_facts(
    job_id: str,
    db: DatabaseManager = Depends(get_db),
    sm: PipelineStateMachine = Depends(get_state_machine),
    cfg: RootConfig = Depends(get_config),
    router: LLMRouter = Depends(get_llm_router),
) -> FactLedger:
    """Run atomic fact extraction, deterministic evidence check, corroboration,

    conflict detection, and fail-closed validation.
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    stage = FactsPipelineStage(
        db=db,
        llm_router=router,
        pipeline_config=cfg.pipeline,
        state_machine=sm,
    )

    try:
        ledger = await stage.process_facts_for_job(job_id)
        return ledger
    except InsufficientVerifiedFactsError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Fact extraction stage failed for job %s: %s", job_id, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fact extraction failed: {str(e)}")


@router.get("/{job_id}/facts", response_model=List[ExtractedFact])
async def get_job_facts(
    job_id: str,
    include_discarded: bool = Query(False, description="Whether to include discarded facts"),
    db: DatabaseManager = Depends(get_db),
) -> List[ExtractedFact]:
    """Retrieve the fact ledger for a job."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return db.get_facts_for_job(job_id, include_discarded=include_discarded)


@router.get("/{job_id}/conflicts", response_model=List[ConflictRecord])
async def get_job_conflicts(
    job_id: str,
    db: DatabaseManager = Depends(get_db),
) -> List[ConflictRecord]:
    """Retrieve all detected conflict records for a job."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return db.get_conflicts_for_job(job_id)


@router.post("/{job_id}/resolve-conflict", response_model=ConflictRecord)
async def resolve_conflict(
    job_id: str,
    payload: ResolveConflictRequest,
    db: DatabaseManager = Depends(get_db),
) -> ConflictRecord:
    """Human decision to resolve a factual conflict by choosing the winning fact."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    resolved = db.resolve_conflict(payload.conflict_id, payload.chosen_fact_id)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Conflict {payload.conflict_id} not found")

    conflicts = db.get_conflicts_for_job(job_id)
    chosen = next((c for c in conflicts if c.conflict_id == payload.conflict_id), None)
    if not chosen:
        raise HTTPException(status_code=404, detail="Conflict record not found after update")
    return chosen
