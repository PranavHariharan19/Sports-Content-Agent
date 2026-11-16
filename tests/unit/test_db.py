import json
from pipeline.db import DatabaseManager
from schemas.common import JobStatus, SportType, StageName


def test_init_db_and_pragmas(db_manager: DatabaseManager):
    """Verify SQLite WAL mode and foreign keys are active."""
    with db_manager.get_connection() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]

    assert journal_mode.lower() == "wal"
    assert foreign_keys == 1


def test_job_crud(db_manager: DatabaseManager):
    """Test job creation, retrieval, listing, and status update."""
    job = db_manager.create_job(job_id="test-job-1", sport=SportType.FOOTBALL)
    assert job.job_id == "test-job-1"
    assert job.sport == SportType.FOOTBALL
    assert job.status == JobStatus.DISCOVERED

    retrieved = db_manager.get_job("test-job-1")
    assert retrieved is not None
    assert retrieved.job_id == "test-job-1"
    assert retrieved.status == JobStatus.DISCOVERED

    # Update status
    updated = db_manager.update_job_status("test-job-1", JobStatus.RANKED)
    assert updated is True

    refreshed = db_manager.get_job("test-job-1")
    assert refreshed is not None
    assert refreshed.status == JobStatus.RANKED

    # List jobs
    jobs = db_manager.list_jobs(sport=SportType.FOOTBALL)
    assert len(jobs) >= 1
    assert jobs[0].job_id == "test-job-1"


def test_stage_run_recording(db_manager: DatabaseManager):
    """Test recording and retrieving stage runs with JSON payloads."""
    db_manager.create_job(job_id="job-stage-test", sport=SportType.F1)

    input_payload = {"topics_discovered": 5, "source": "rss"}
    output_payload = {"ranked_topics": ["f1-story-1", "f1-story-2"]}

    run = db_manager.record_stage_run(
        job_id="job-stage-test",
        stage_name=StageName.DISCOVERY,
        input_data=input_payload,
        output_data=output_payload,
        status="COMPLETED",
    )

    assert run.id is not None
    assert run.job_id == "job-stage-test"
    assert run.stage_name == StageName.DISCOVERY
    assert run.get_input() == input_payload
    assert run.get_output() == output_payload

    history = db_manager.get_stage_runs_for_job("job-stage-test")
    assert len(history) == 1
    assert history[0].id == run.id

    latest = db_manager.get_latest_stage_run("job-stage-test", StageName.DISCOVERY)
    assert latest is not None
    assert latest.id == run.id
