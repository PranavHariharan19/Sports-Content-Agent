from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.logging_config import get_logger, setup_logging
from pipeline.db import DatabaseManager
from pipeline.state_machine import PipelineStateMachine
from schemas.common import JobStatus, SportType
from schemas.config import RootConfig, load_config
from schemas.job import JobRecord, StageRunRecord

logger = get_logger(__name__)


def create_app(config: Optional[RootConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = config or load_config()
    setup_logging(level=cfg.app.log_level)

    db = DatabaseManager(cfg.app.database_path)
    state_machine = PipelineStateMachine(db)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("Starting Sports Shorts Command Center API")
        yield
        logger.info("Shutting down Sports Shorts Command Center API")

    app = FastAPI(
        title=cfg.app.name,
        version="0.1.0",
        description="Sports Shorts Command Center - Zero Cost & Human in the Loop",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach shared instances to app state
    app.state.config = cfg
    app.state.db = db
    app.state.state_machine = state_machine

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        return {
            "status": "healthy",
            "app": cfg.app.name,
            "env": cfg.app.env,
        }

    @app.get("/api/jobs", response_model=List[JobRecord])
    async def list_jobs(
        sport: Optional[SportType] = None,
        limit: int = Query(50, ge=1, le=200),
    ) -> List[JobRecord]:
        return db.list_jobs(limit=limit, sport=sport)

    @app.get("/api/jobs/{job_id}", response_model=JobRecord)
    async def get_job(job_id: str) -> JobRecord:
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/api/jobs/{job_id}/stages", response_model=List[StageRunRecord])
    async def get_job_stages(job_id: str) -> List[StageRunRecord]:
        stages = db.get_stage_runs_for_job(job_id)
        return stages

    return app


app = create_app()
