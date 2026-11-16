from fastapi.testclient import TestClient
from pipeline.db import DatabaseManager
from schemas.common import JobStatus, SportType, StageName


def test_sources_api_lifecycle(test_client: TestClient, db_manager: DatabaseManager):
    """Verify fetching sources and querying sources endpoints via API."""
    # Create job in TOPIC_CHOSEN status
    job = db_manager.create_job(job_id="api-sources-job", sport=SportType.F1)
    db_manager.update_job_status("api-sources-job", JobStatus.TOPIC_CHOSEN)

    # Record fake chosen topic in ranking stage
    chosen_topic = {
        "cluster_id": "clust_f1_test",
        "primary_headline": "Max Verstappen takes pole in Las Vegas",
        "articles": [
            {
                "url": "https://formula1.com/news/verstappen-pole",
                "title": "Verstappen grabs pole",
                "published_at": "2026-11-16T14:00:00Z",
            },
            {
                "url": "https://unverified-blog.xyz/rumors",
                "title": "F1 news gossip",
                "published_at": "2026-11-16T14:30:00Z",
            },
        ],
    }
    db_manager.record_stage_run(
        job_id="api-sources-job",
        stage_name=StageName.RANKING,
        input_data={},
        output_data={"chosen_topic": chosen_topic},
    )

    # API: Trigger source fetching
    fetch_res = test_client.post("/api/jobs/api-sources-job/fetch-sources")
    assert fetch_res.status_code == 200
    fetch_data = fetch_res.json()

    assert fetch_data["total_sources"] == 2
    assert fetch_data["tier_1_count"] == 1  # formula1.com
    assert fetch_data["allowlisted_sources"] == 1  # only formula1.com, blog is untrusted

    # API: Get all sources
    all_sources_res = test_client.get("/api/jobs/api-sources-job/sources")
    assert all_sources_res.status_code == 200
    assert len(all_sources_res.json()) == 2

    # API: Get allowlisted only
    allowlisted_res = test_client.get("/api/jobs/api-sources-job/sources?allowlisted_only=true")
    assert allowlisted_res.status_code == 200
    allowlisted_data = allowlisted_res.json()
    assert len(allowlisted_data) == 1
    assert allowlisted_data[0]["domain"] == "formula1.com"
    assert allowlisted_data[0]["is_allowlisted"] is True
