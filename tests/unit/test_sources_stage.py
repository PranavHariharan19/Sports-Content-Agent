import pytest
from pipeline.db import DatabaseManager
from pipeline.sources_stage import SourcesPipelineStage
from pipeline.state_machine import PipelineStateMachine
from schemas.common import JobStatus, SportType, StageName


@pytest.mark.asyncio
async def test_sources_pipeline_stage_orchestration(db_manager: DatabaseManager, state_machine: PipelineStateMachine):
    """Verify SourcesPipelineStage stores sources, runs syndication, and records stage execution."""
    job = state_machine.create_new_job(SportType.FOOTBALL, job_id="stage-sources-job")
    state_machine.transition_job(job.job_id, JobStatus.RANKED)
    state_machine.transition_job(job.job_id, JobStatus.TOPIC_CHOSEN)

    stage = SourcesPipelineStage(db=db_manager, state_machine=state_machine)

    articles = [
        {
            "url": "https://www.bbc.co.uk/sport/football/story1",
            "title": "Arsenal win London derby",
            "published_at": "2026-11-16T12:00:00Z",
        },
        {
            "url": "https://www.skysports.com/football/story2",
            "title": "Arsenal defeat Chelsea 3-1",
            "published_at": "2026-11-16T12:15:00Z",
        },
    ]

    result = await stage.fetch_and_store_sources_for_job(
        job_id="stage-sources-job",
        article_overrides=articles,
    )

    assert result.total_sources == 2
    assert result.allowlisted_sources == 2
    assert len(result.sources) == 2

    # Check persistence in DB
    db_sources = db_manager.get_sources_for_job("stage-sources-job")
    assert len(db_sources) == 2
    assert db_sources[0].is_allowlisted is True

    # Check stage run recorded
    stage_run = db_manager.get_latest_stage_run("stage-sources-job", StageName.SOURCES)
    assert stage_run is not None
    assert stage_run.status == "COMPLETED"
