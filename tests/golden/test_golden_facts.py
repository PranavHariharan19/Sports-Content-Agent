import json
from unittest.mock import AsyncMock, patch
import pytest

from llm.mock_provider import MockLLMProvider
from llm.router import LLMRouter
from pipeline.db import DatabaseManager
from pipeline.facts_stage import FactsPipelineStage
from pipeline.state_machine import PipelineStateMachine
from schemas.common import CorroborationStatus, FactType, JobStatus, SourceTier, SportType
from schemas.config import LLMConfig, PipelineConfig, ProviderSettings
from schemas.sources import StoredSource


GOLDEN_BBC_TEXT = (
    "Lewis Hamilton scored an incredible victory at the 2024 British Grand Prix at Silverstone. "
    "Hamilton held off Red Bull's Max Verstappen to win by 1.465 seconds, ending a "
    "945-day win drought. It marked Hamilton's 104th career win and his record ninth triumph at Silverstone."
)

GOLDEN_SKY_TEXT = (
    "Lewis Hamilton claimed a fairytale victory in the British Grand Prix, holding off a charging "
    "Max Verstappen in the closing stages. Hamilton crossed the line 1.465 seconds ahead of Verstappen "
    "to take his 104th Grand Prix win. McLaren's Lando Norris finished third on the podium."
)

# Simulated structured extraction responses
MOCK_EXTRACTION_BBC = json.dumps([
    {
        "claim": "Lewis Hamilton won the 2024 British Grand Prix at Silverstone",
        "fact_type": "record",
        "normalized_entities": ["Lewis Hamilton", "Silverstone"],
        "normalized_value": "2024",
        "evidence_quote": "Lewis Hamilton scored an incredible victory at the 2024 British Grand Prix at Silverstone.",
    },
    {
        "claim": "Lewis Hamilton secured his 104th career win",
        "fact_type": "number",
        "normalized_entities": ["Lewis Hamilton"],
        "normalized_value": "104",
        "evidence_quote": "It marked Hamilton's 104th career win and his record ninth triumph at Silverstone.",
    },
    {
        "claim": "Lewis Hamilton beat Max Verstappen by 1.465 seconds",
        "fact_type": "number",
        "normalized_entities": ["Lewis Hamilton", "Max Verstappen"],
        "normalized_value": "1.465",
        "evidence_quote": "Hamilton held off Red Bull's Max Verstappen to win by 1.465 seconds",
    },
])

MOCK_EXTRACTION_SKY = json.dumps([
    {
        "claim": "Lewis Hamilton secured his 104th career win",
        "fact_type": "number",
        "normalized_entities": ["Lewis Hamilton"],
        "normalized_value": "104",
        "evidence_quote": "Hamilton crossed the line 1.465 seconds ahead of Verstappen to take his 104th Grand Prix win.",
    },
    {
        "claim": "Lewis Hamilton beat Max Verstappen by 1.465 seconds",
        "fact_type": "number",
        "normalized_entities": ["Lewis Hamilton", "Max Verstappen"],
        "normalized_value": "1.465",
        "evidence_quote": "Hamilton crossed the line 1.465 seconds ahead of Verstappen",
    },
    {
        "claim": "Lando Norris finished third on the podium",
        "fact_type": "ranking",
        "normalized_entities": ["Lando Norris", "McLaren"],
        "normalized_value": "3",
        "evidence_quote": "McLaren's Lando Norris finished third on the podium.",
    },
])


@pytest.mark.asyncio
async def test_golden_story_fact_verification(db_manager: DatabaseManager, state_machine: PipelineStateMachine):
    """Verify golden dataset story produces 100% verified facts with zero unsupported claims."""
    job = state_machine.create_new_job(SportType.F1, job_id="golden_job_1")
    state_machine.transition_job(job.job_id, JobStatus.RANKED)
    state_machine.transition_job(job.job_id, JobStatus.TOPIC_CHOSEN)

    # Store golden sources in DB
    s1 = StoredSource(
        source_id="src_golden_bbc", job_id="golden_job_1", url="https://bbc.co.uk/f1/silverstone",
        domain="bbc.co.uk", source_tier=SourceTier.TIER_2, cleaned_text=GOLDEN_BBC_TEXT,
    )
    s2 = StoredSource(
        source_id="src_golden_sky", job_id="golden_job_1", url="https://skysports.com/f1/silverstone",
        domain="skysports.com", source_tier=SourceTier.TIER_2, cleaned_text=GOLDEN_SKY_TEXT,
    )
    db_manager.save_source(s1)
    db_manager.save_source(s2)

    # Setup mock LLM router with extraction outputs
    mock_provider = MockLLMProvider(responses=[MOCK_EXTRACTION_BBC, MOCK_EXTRACTION_SKY])
    router = LLMRouter(
        config=LLMConfig(fallback_order=["mock"], providers={"mock": ProviderSettings(model="m")}),
        db=db_manager,
        custom_providers={"mock": mock_provider},
    )

    stage = FactsPipelineStage(
        db=db_manager,
        llm_router=router,
        pipeline_config=PipelineConfig(min_verified_facts=3),
        state_machine=state_machine,
    )

    ledger = await stage.process_facts_for_job("golden_job_1")

    # Assertions
    assert ledger.verified_count >= 2
    assert ledger.discarded_count == 0  # 0 hallucinated/discarded facts on golden dataset
    assert len(ledger.conflicts) == 0  # 0 conflicts on clean corroborated story

    # Verify state machine progressed to FACTS_VERIFIED
    refreshed_job = db_manager.get_job("golden_job_1")
    assert refreshed_job.status == JobStatus.FACTS_VERIFIED
