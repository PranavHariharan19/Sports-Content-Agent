from fastapi.testclient import TestClient
from pipeline.db import DatabaseManager
from schemas.common import JobStatus, SportType, StageName


def test_health_check(test_client: TestClient):
    """Verify /health endpoint returns healthy status."""
    res = test_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"


def test_job_api_lifecycle(test_client: TestClient, db_manager: DatabaseManager):
    """Verify job listing, detail retrieval, and stage tracking via API."""
    # Create a job in DB
    job = db_manager.create_job(job_id="api-job-123", sport=SportType.FOOTBALL)
    assert job.status == JobStatus.DISCOVERED

    # Record discovery stage run
    db_manager.record_stage_run(
        job_id="api-job-123",
        stage_name=StageName.DISCOVERY,
        input_data={"sources": ["bbc", "sky"]},
        output_data={"articles_found": 15},
    )

    # API: List jobs
    list_res = test_client.get("/api/jobs")
    assert list_res.status_code == 200
    jobs_data = list_res.json()
    assert len(jobs_data) >= 1
    assert any(j["job_id"] == "api-job-123" for j in jobs_data)

    # API: Get job detail
    detail_res = test_client.get("/api/jobs/api-job-123")
    assert detail_res.status_code == 200
    assert detail_res.json()["job_id"] == "api-job-123"
    assert detail_res.json()["sport"] == "football"

    # API: Get job stages
    stages_res = test_client.get("/api/jobs/api-job-123/stages")
    assert stages_res.status_code == 200
    stages_data = stages_res.json()
    assert len(stages_data) == 1
    assert stages_data[0]["stage_name"] == "discovery"
    assert stages_data[0]["status"] == "COMPLETED"

    # API: 404 for nonexistent job
    not_found = test_client.get("/api/jobs/nonexistent-id")
    assert not_found.status_code == 404
