import json
from unittest.mock import patch
from fastapi.testclient import TestClient

from pipeline.db import DatabaseManager
from schemas.common import CorroborationStatus, FactType, JobStatus, SourceTier, SportType, VerificationVerdict
from schemas.facts import ExtractedFact
from schemas.llm import LLMResponse, TokenUsage
from schemas.sources import StoredSource


SAMPLE_SCRIPT_JSON = json.dumps({
    "beats": [
        {
            "beat_id": 1,
            "beat_type": "hook",
            "spoken_text": "Did Lewis Hamilton just pull off the comeback of the year?",
            "cited_fact_ids": [],
            "is_factual": False,
        },
        {
            "beat_id": 2,
            "beat_type": "factual_beat",
            "spoken_text": "Hamilton held off Verstappen by 1.465 seconds at Silverstone.",
            "cited_fact_ids": ["fact_101", "fact_102"],
            "is_factual": True,
        },
        {
            "beat_id": 3,
            "beat_type": "factual_beat",
            "spoken_text": "This was his 104th Grand Prix victory in front of 160000 fans.",
            "cited_fact_ids": ["fact_103", "fact_104"],
            "is_factual": True,
        },
        {
            "beat_id": 4,
            "beat_type": "closing",
            "spoken_text": "Can anyone catch him now? Subscribe for more updates!",
            "cited_fact_ids": [],
            "is_factual": False,
        },
    ]
})

VERIFIER_SUPPORTED_JSON = json.dumps({
    "verdict": "SUPPORTED",
    "reason": "Directly confirmed by cited evidence quotes.",
})

VERIFIER_UNSUPPORTED_JSON = json.dumps({
    "verdict": "UNSUPPORTED",
    "reason": "Evidence does not mention record-breaking pit stop duration.",
})


def _seed_facts(db: DatabaseManager, job_id: str):
    # Foreign key requires sources to exist in DB first
    s1 = StoredSource(
        source_id="src_1",
        job_id=job_id,
        url="https://bbc.com/sport/formula1/101",
        domain="bbc.com",
        source_tier=SourceTier.TIER_2,
        cleaned_text="Hamilton held off Verstappen in a thrilling finish. Hamilton beat Verstappen by 1.465 seconds at Silverstone.",
    )
    s2 = StoredSource(
        source_id="src_2",
        job_id=job_id,
        url="https://formula1.com/news/102",
        domain="formula1.com",
        source_tier=SourceTier.TIER_1,
        cleaned_text="This was Hamilton's 104th Grand Prix victory. Over 160000 fans packed Silverstone circuit.",
    )
    db.save_source(s1)
    db.save_source(s2)

    f1 = ExtractedFact(
        fact_id="fact_101",
        job_id=job_id,
        claim="Hamilton held off Verstappen",
        fact_type=FactType.OTHER,
        normalized_entities=["Hamilton", "Verstappen", "Silverstone"],
        source_id="src_1",
        evidence_quote="Hamilton held off Verstappen in a thrilling finish.",
        source_tier=SourceTier.TIER_2,
        corroboration_status=CorroborationStatus.VERIFIED,
    )
    f2 = ExtractedFact(
        fact_id="fact_102",
        job_id=job_id,
        claim="Winning margin was 1.465 seconds",
        fact_type=FactType.NUMBER,
        normalized_entities=["Hamilton", "Verstappen", "Silverstone"],
        normalized_value="1.465",
        source_id="src_1",
        evidence_quote="Hamilton beat Verstappen by 1.465 seconds at Silverstone.",
        source_tier=SourceTier.TIER_2,
        corroboration_status=CorroborationStatus.VERIFIED,
    )
    f3 = ExtractedFact(
        fact_id="fact_103",
        job_id=job_id,
        claim="Hamilton won his 104th Grand Prix",
        fact_type=FactType.RECORD,
        normalized_entities=["Hamilton"],
        normalized_value="104",
        source_id="src_2",
        evidence_quote="This was Hamilton's 104th Grand Prix victory.",
        source_tier=SourceTier.TIER_1,
        corroboration_status=CorroborationStatus.VERIFIED,
    )
    f4 = ExtractedFact(
        fact_id="fact_104",
        job_id=job_id,
        claim="Silverstone attendance was 160000",
        fact_type=FactType.NUMBER,
        normalized_entities=["Silverstone"],
        normalized_value="160000",
        source_id="src_2",
        evidence_quote="Over 160000 fans packed Silverstone circuit.",
        source_tier=SourceTier.TIER_1,
        corroboration_status=CorroborationStatus.VERIFIED,
    )
    db.save_facts([f1, f2, f3, f4])


def test_script_generation_and_approval_flow(test_client: TestClient, db_manager: DatabaseManager):
    """End-to-end integration: draft script, run linter and verifier, and approve via Checkpoint 2."""
    job_id = "job-script-flow"
    db_manager.create_job(job_id=job_id, sport=SportType.F1)
    db_manager.update_job_status(job_id, JobStatus.FACTS_VERIFIED)
    _seed_facts(db_manager, job_id)

    # 1. Draft script
    with patch("llm.router.LLMRouter.generate") as mock_gen:
        # First call is generator, subsequent calls are verifier passes for beats
        mock_gen.side_effect = [
            LLMResponse(
                text=SAMPLE_SCRIPT_JSON,
                provider="gemini",
                model="gemini-2.5-flash",
                usage=TokenUsage(prompt_tokens=100, completion_tokens=100, total_tokens=200),
            ),
            LLMResponse(
                text=VERIFIER_SUPPORTED_JSON,
                provider="gemini",
                model="gemini-2.5-flash",
                usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
            ),
            LLMResponse(
                text=VERIFIER_SUPPORTED_JSON,
                provider="gemini",
                model="gemini-2.5-flash",
                usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
            ),
        ]

        resp = test_client.post(f"/api/jobs/{job_id}/draft-script")
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["job_id"] == job_id
        assert len(data["beats"]) == 4  # Hook + 2 factual + closing
        assert data["total_words"] > 0
        assert data["all_supported"] is True
        assert data["lint_passed"] is True

    # 2. Check DB job status transitioned to SCRIPT_DRAFTED
    job = db_manager.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.SCRIPT_DRAFTED

    # 3. Retrieve script via GET endpoint
    get_resp = test_client.get(f"/api/jobs/{job_id}/script")
    assert get_resp.status_code == 200
    assert get_resp.json()["script_id"] == data["script_id"]

    # 4. Human Checkpoint 2: Approve script
    approve_resp = test_client.post(f"/api/jobs/{job_id}/approve-script")
    assert approve_resp.status_code == 200
    appr_data = approve_resp.json()
    assert appr_data["approved_at"] is not None

    # Verify DB status transitioned to SCRIPT_APPROVED
    job_after = db_manager.get_job(job_id)
    assert job_after is not None
    assert job_after.status == JobStatus.SCRIPT_APPROVED


def test_script_approval_blocked_on_unsupported_beat_and_override(test_client: TestClient, db_manager: DatabaseManager):
    """Verify that unsupported beats block approval until overridden by human editor."""
    job_id = "job-script-blocked"
    db_manager.create_job(job_id=job_id, sport=SportType.F1)
    db_manager.update_job_status(job_id, JobStatus.FACTS_VERIFIED)
    _seed_facts(db_manager, job_id)

    # Verifier marks beat 1 as UNSUPPORTED
    with patch("llm.router.LLMRouter.generate") as mock_gen:
        mock_gen.side_effect = [
            LLMResponse(
                text=SAMPLE_SCRIPT_JSON,
                provider="gemini",
                model="gemini-2.5-flash",
                usage=TokenUsage(prompt_tokens=100, completion_tokens=100, total_tokens=200),
            ),
            LLMResponse(
                text=VERIFIER_UNSUPPORTED_JSON,
                provider="gemini",
                model="gemini-2.5-flash",
                usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
            ),
            LLMResponse(
                text=VERIFIER_SUPPORTED_JSON,
                provider="gemini",
                model="gemini-2.5-flash",
                usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
            ),
        ]

        draft_resp = test_client.post(f"/api/jobs/{job_id}/draft-script")
        assert draft_resp.status_code == 200
        draft_data = draft_resp.json()
        assert draft_data["all_supported"] is False

    # Checkpoint 2 must fail-close: approval should be rejected with 400
    block_resp = test_client.post(f"/api/jobs/{job_id}/approve-script")
    assert block_resp.status_code == 400
    assert "not verified as SUPPORTED" in block_resp.json()["detail"]

    # Human reviews provenance and issues an override for beat 2
    override_resp = test_client.post(
        f"/api/jobs/{job_id}/override-beat",
        json={
            "beat_id": 2,
            "action": "override",
            "override_reason": "Verified manually via race telemetry footage",
        },
    )
    assert override_resp.status_code == 200
    overridden_script = override_resp.json()
    beat2 = [b for b in overridden_script["beats"] if b["beat_id"] == 2][0]
    assert beat2["human_override"] is True
    assert beat2["human_override_reason"] == "Verified manually via race telemetry footage"

    # Now approval succeeds
    approve_resp = test_client.post(f"/api/jobs/{job_id}/approve-script")
    assert approve_resp.status_code == 200

    job_final = db_manager.get_job(job_id)
    assert job_final is not None
    assert job_final.status == JobStatus.SCRIPT_APPROVED


def test_script_draft_fails_when_no_verified_facts(test_client: TestClient, db_manager: DatabaseManager):
    """Verify that drafting fails closed when no verified facts exist for the job."""
    job_id = "job-script-empty"
    db_manager.create_job(job_id=job_id, sport=SportType.FOOTBALL)
    db_manager.update_job_status(job_id, JobStatus.FACTS_VERIFIED)
    # No facts saved

    resp = test_client.post(f"/api/jobs/{job_id}/draft-script")
    assert resp.status_code == 400
    assert "No usable verified facts" in resp.json()["detail"]
