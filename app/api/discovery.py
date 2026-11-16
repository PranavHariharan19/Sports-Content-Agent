import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.logging_config import get_logger
from pipeline.clustering import StoryClusteringEngine
from pipeline.db import DatabaseManager
from pipeline.discovery import DiscoveryEngine
from pipeline.ranking import ViralityRankingEngine
from pipeline.sources_trust import SourceTrustManager
from pipeline.state_machine import PipelineStateMachine
from schemas.common import JobStatus, SportType, StageName
from schemas.config import RootConfig
from schemas.discovery import ChooseTopicRequest, DiscoveryRunResult, StoryCluster
from schemas.job import JobRecord

logger = get_logger(__name__)

router = APIRouter(prefix="/api/discovery", tags=["Discovery & Ranking"])


def get_db(request: Request) -> DatabaseManager:
    return request.app.state.db


def get_state_machine(request: Request) -> PipelineStateMachine:
    return request.app.state.state_machine


def get_config(request: Request) -> RootConfig:
    return request.app.state.config


@router.post("/run", response_model=DiscoveryRunResult)
async def run_discovery(
    sport: SportType = Query(..., description="Sport category to discover"),
    job_id: Optional[str] = Query(None, description="Existing job ID or create new"),
    include_llm_emotion: bool = Query(False, description="Whether to include LLM emotion subscore"),
    db: DatabaseManager = Depends(get_db),
    sm: PipelineStateMachine = Depends(get_state_machine),
    cfg: RootConfig = Depends(get_config),
) -> DiscoveryRunResult:
    """Run discovery, clustering, and deterministic ranking for a sport."""
    # 1. Ensure Job Record exists
    if job_id:
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    else:
        job = sm.create_new_job(sport=sport)
        job_id = job.job_id

    trust_manager = SourceTrustManager()
    discovery_engine = DiscoveryEngine(trust_manager)
    clustering_engine = StoryClusteringEngine(trust_manager)
    ranking_engine = ViralityRankingEngine(
        trust_manager=trust_manager,
        pipeline_config=cfg.pipeline,
    )

    try:
        # 2. Fetch raw articles
        articles = await discovery_engine.discover(sport, max_age_hours=cfg.pipeline.max_story_age_hours)

        # 3. Cluster articles
        clusters = clustering_engine.cluster_articles(articles, sport)

        # 4. Score and rank clusters
        top_candidates = await ranking_engine.score_and_rank_clusters(
            clusters=clusters,
            sport=sport,
            include_llm_emotion=include_llm_emotion,
        )

        # 5. Record discovery and ranking stage runs in DB
        discovery_input = {"sport": sport.value, "max_age_hours": cfg.pipeline.max_story_age_hours}
        discovery_output = {
            "articles_count": len(articles),
            "sources": list({a.domain for a in articles}),
        }
        sm.record_stage_execution(
            job_id=job_id,
            stage_name=StageName.DISCOVERY,
            input_data=discovery_input,
            output_data=discovery_output,
            success=True,
            allow_rerun=True,
        )

        ranking_output = {
            "candidates_count": len(top_candidates),
            "candidates": [c.model_dump(mode="json") for c in top_candidates],
        }
        sm.record_stage_execution(
            job_id=job_id,
            stage_name=StageName.RANKING,
            input_data={"clusters_count": len(clusters)},
            output_data=ranking_output,
            success=True,
            allow_rerun=True,
        )

        return DiscoveryRunResult(
            job_id=job_id,
            sport=sport,
            candidates=top_candidates,
            total_articles_discovered=len(articles),
            total_clusters_formed=len(clusters),
        )

    except Exception as e:
        logger.error("Discovery run failed: %s", str(e), exc_info=True)
        sm.transition_job(job_id=job_id, target_status=JobStatus.FAILED, error_message=str(e), allow_rerun=True)
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


@router.get("/candidates", response_model=List[StoryCluster])
async def get_candidates(
    job_id: str = Query(..., description="Job ID to fetch ranked candidates for"),
    db: DatabaseManager = Depends(get_db),
) -> List[StoryCluster]:
    """Retrieve ranked story candidates from the latest ranking stage of a job."""
    stage_runs = db.get_stage_runs_for_job(job_id)
    for sr in reversed(stage_runs):
        if sr.stage_name == StageName.RANKING:
            output = sr.get_output()
            raw_candidates = output.get("candidates", [])
            if raw_candidates:
                return [StoryCluster.model_validate(c) for c in raw_candidates]

    raise HTTPException(status_code=404, detail="No ranking run found for this job")


@router.post("/choose-topic", response_model=JobRecord)
async def choose_topic(
    job_id: str = Query(..., description="Job ID"),
    payload: ChooseTopicRequest = ...,
    db: DatabaseManager = Depends(get_db),
    sm: PipelineStateMachine = Depends(get_state_machine),
) -> JobRecord:
    """Human Checkpoint 1: Select a candidate topic to advance pipeline to TOPIC_CHOSEN."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in (JobStatus.RANKED, JobStatus.TOPIC_CHOSEN):
        raise HTTPException(
            status_code=400,
            detail=f"Job must be in RANKED status to choose topic (current: {job.status.value})",
        )

    target_cluster_id = payload.cluster_id.strip()

    # Find candidate clusters from all ranking stage runs
    stage_runs = db.get_stage_runs_for_job(job_id)
    candidates: List[Dict[str, Any]] = []
    already_chosen: Optional[Dict[str, Any]] = None

    for sr in reversed(stage_runs):
        if sr.stage_name == StageName.RANKING:
            out = sr.get_output()
            if not candidates and out.get("candidates"):
                candidates = out["candidates"]
            if not already_chosen and out.get("chosen_topic"):
                already_chosen = out["chosen_topic"]

    chosen = next((c for c in candidates if c.get("cluster_id") == target_cluster_id), None)
    if not chosen and already_chosen and already_chosen.get("cluster_id") == target_cluster_id:
        chosen = already_chosen

    if not chosen:
        raise HTTPException(status_code=404, detail=f"Candidate cluster {target_cluster_id} not found in job ranking")

    # Record checkpoint action, preserving candidates list for re-inspection and reruns
    output_data: Dict[str, Any] = {"chosen_topic": chosen}
    if candidates:
        output_data["candidates"] = candidates

    sm.record_stage_execution(
        job_id=job_id,
        stage_name=StageName.RANKING,  # updates with selection
        input_data={"action": "choose_topic", "cluster_id": target_cluster_id},
        output_data=output_data,
        success=True,
        allow_rerun=True,
    )

    # Advance state to TOPIC_CHOSEN
    updated_job = sm.transition_job(job_id, JobStatus.TOPIC_CHOSEN, allow_rerun=True)
    logger.info("Human Checkpoint 1 completed: Job %s topic chosen '%s'", job_id, chosen.get("primary_headline"))
    return updated_job
