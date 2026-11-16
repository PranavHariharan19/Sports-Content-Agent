import json
from unittest.mock import patch
from fastapi.testclient import TestClient

from pipeline.db import DatabaseManager
from schemas.common import JobStatus, SourceTier, SportType
from schemas.sources import StoredSource


SAMPLE_EXTRACTION_JSON = json.dumps([
    {
        "claim": "Lewis Hamilton won at Silverstone",
        "fact_type": "other",
        "normalized_entities": ["Lewis Hamilton", "Silverstone"],
        "normalized_value": None,
        "evidence_quote": "Lewis Hamilton won at Silverstone in front of 160000 fans.",
    },
    {
        "claim": "Race attendance was 160000 fans",
        "fact_type": "number",
        "normalized_entities": ["Silverstone"],
        "normalized_value": "160000",
        "evidence_quote": "Lewis Hamilton won at Silverstone in front of 160000 fans.",
    },
    {
        "claim": "Hamilton won his 104th Grand Prix",
        "fact_type": "record",
        "normalized_entities": ["Hamilton"],
        "normalized_value": "104",
        "evidence_quote": "Hamilton won his 104th Grand Prix after 52 laps.",
    },
])


def test_facts_api_lifecycle(test_client: TestClient, db_manager: DatabaseManager):
    """Verify fact extraction API, fact retrieval, conflict tracking, and resolution."""
    # 1. Setup job in TOPIC_CHOSEN status with stored sources
    job = db_manager.create_job(job_id="api-facts-job", sport=SportType.F1)
    db_manager.update_job_status("api-facts-job", JobStatus.TOPIC_CHOSEN)

    s1 = StoredSource(
        source_id="src_api_1",
        job_id="api-facts-job",
        url="https://formula1.com/news/silverstone",
        domain="formula1.com",
        source_tier=SourceTier.TIER_1,
        cleaned_text="Lewis Hamilton won at Silverstone in front of 160000 fans. Hamilton won his 104th Grand Prix after 52 laps.",
    )
    db_manager.save_source(s1)

    # 2. Call POST /api/jobs/{job_id}/extract-facts (with mock LLM router response)
    with patch("llm.router.LLMRouter.generate") as mock_generate:
        from schemas.llm import LLMResponse, TokenUsage
        mock_generate.return_value = LLMResponse(
            text=SAMPLE_EXTRACTION_JSON,
            provider="gemini",
            model="gemini-2.5-flash",
            usage=TokenUsage(prompt_tokens=50, completion_tokens=50, total_tokens=100),
        )

        res = test_client.post("/api/jobs/api-facts-job/extract-facts")
        assert res.status_code == 200
        ledger_data = res.json()
        assert ledger_data["verified_count"] >= 3
        assert ledger_data["discarded_count"] == 0

    # Verify job status advanced to FACTS_VERIFIED
    refreshed_job = db_manager.get_job("api-facts-job")
    assert refreshed_job.status == JobStatus.FACTS_VERIFIED

    # 3. GET /api/jobs/{job_id}/facts
    facts_res = test_client.get("/api/jobs/api-facts-job/facts")
    assert facts_res.status_code == 200
    facts_list = facts_res.json()
    assert len(facts_list) >= 3

    # 4. Create and resolve a conflict via API
    from schemas.facts import ConflictRecord
    test_conflict = ConflictRecord(
        conflict_id="conf_test_123",
        job_id="api-facts-job",
        entity_or_topic="Attendance",
        fact_ids=[facts_list[0]["fact_id"], facts_list[1]["fact_id"]],
        conflicting_values=["160000", "150000"],
        sources=["formula1.com", "autosport.com"],
        resolved=False,
    )
    db_manager.save_conflict(test_conflict)

    conf_list_res = test_client.get("/api/jobs/api-facts-job/conflicts")
    assert conf_list_res.status_code == 200
    assert len(conf_list_res.json()) >= 1

    # Resolve conflict
    resolve_res = test_client.post(
        "/api/jobs/api-facts-job/resolve-conflict",
        json={"conflict_id": "conf_test_123", "chosen_fact_id": facts_list[0]["fact_id"]},
    )
    assert resolve_res.status_code == 200
    resolved_data = resolve_res.json()
    assert resolved_data["resolved"] is True
    assert resolved_data["chosen_fact_id"] == facts_list[0]["fact_id"]
