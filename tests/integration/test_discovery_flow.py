from datetime import datetime, timezone
import email.utils
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from pipeline.db import DatabaseManager
from schemas.common import JobStatus, SportType, StageName


def get_sample_rss_data() -> str:
    now_rfc822 = email.utils.format_datetime(datetime.now(timezone.utc))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Sports Feed</title>
    <item>
      <title>Virat Kohli smashes record-breaking century for India</title>
      <link>https://www.espncricinfo.com/story/1001</link>
      <description>Masterclass innings in Chennai.</description>
      <pubDate>{now_rfc822}</pubDate>
    </item>
    <item>
      <title>Kohli century powers India to dominant victory</title>
      <link>https://www.cricbuzz.com/cricket-news/2002</link>
      <description>Sensational batting display against Australia.</description>
      <pubDate>{now_rfc822}</pubDate>
    </item>
  </channel>
</rss>
"""


def test_dashboard_ui_renders(test_client: TestClient):
    """Verify that the dashboard HTML page is served correctly."""
    res = test_client.get("/")
    assert res.status_code == 200
    assert "Sports Shorts Command Center" in res.text
    assert "Discover Trending Stories" in res.text


def test_discovery_and_topic_choice_lifecycle(test_client: TestClient, db_manager: DatabaseManager):
    """Verify discovery run, ranking stage persistence, and Checkpoint 1 topic selection."""
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.text = get_sample_rss_data()
    mock_resp.json = MagicMock(return_value={"items": []})

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        # 1. Run discovery
        run_res = test_client.post("/api/discovery/run?sport=cricket")
        assert run_res.status_code == 200
        run_data = run_res.json()

        job_id = run_data["job_id"]
        candidates = run_data["candidates"]
        assert len(candidates) >= 1
        assert run_data["sport"] == "cricket"

        chosen_candidate = candidates[0]
        cluster_id = chosen_candidate["cluster_id"]
        assert "Virat Kohli" in chosen_candidate["primary_headline"]
        assert chosen_candidate["virality"]["final_score"] > 0

        # Check job status is now RANKED
        job_record = db_manager.get_job(job_id)
        assert job_record is not None
        assert job_record.status == JobStatus.RANKED

        # Verify stage runs in DB
        ranking_run = db_manager.get_latest_stage_run(job_id, StageName.RANKING)
        assert ranking_run is not None
        assert ranking_run.status == "COMPLETED"

        # 2. Query candidates endpoint
        cand_res = test_client.get(f"/api/discovery/candidates?job_id={job_id}")
        assert cand_res.status_code == 200
        cand_data = cand_res.json()
        assert len(cand_data) >= 1

        # 3. Checkpoint 1: Human selects the topic
        choose_res = test_client.post(
            f"/api/discovery/choose-topic?job_id={job_id}",
            json={"cluster_id": cluster_id},
        )
        assert choose_res.status_code == 200
        updated_job = choose_res.json()
        assert updated_job["status"] == "TOPIC_CHOSEN"

        # Verify DB reflects TOPIC_CHOSEN
        refreshed_job = db_manager.get_job(job_id)
        assert refreshed_job is not None
        assert refreshed_job.status == JobStatus.TOPIC_CHOSEN

        # 4. Candidates list remains queryable after topic choice (resilience check)
        cand_res_after = test_client.get(f"/api/discovery/candidates?job_id={job_id}")
        assert cand_res_after.status_code == 200
        assert len(cand_res_after.json()) == len(cand_data)

        # 5. Re-selecting topic or confirming choice does not fail with "not found in job ranking"
        rechoose_res = test_client.post(
            f"/api/discovery/choose-topic?job_id={job_id}",
            json={"cluster_id": cluster_id},
        )
        assert rechoose_res.status_code == 200
        assert rechoose_res.json()["status"] == "TOPIC_CHOSEN"
