import json
import pytest

from llm.mock_provider import MockLLMProvider
from llm.router import LLMRouter
from pipeline.db import DatabaseManager
from pipeline.script_generator import ScriptGenerator
from schemas.common import CorroborationStatus, FactType, SourceTier
from schemas.config import LLMConfig, ProviderSettings
from schemas.facts import ExtractedFact
from schemas.script import BeatType


MOCK_SCRIPT_JSON = json.dumps({
    "beats": [
        {
            "beat_id": 1,
            "beat_type": "hook",
            "spoken_text": "Did Lewis Hamilton just pull off the comeback of the century?",
            "is_factual": False,
            "cited_fact_ids": [],
        },
        {
            "beat_id": 2,
            "beat_type": "factual_beat",
            "spoken_text": "At Silverstone, Hamilton held off Max Verstappen by 1.465 seconds to win his 104th Grand Prix.",
            "is_factual": True,
            "cited_fact_ids": ["fact_104", "fact_margin"],
        },
        {
            "beat_id": 3,
            "beat_type": "closing",
            "spoken_text": "Can anyone stop Mercedes now? Subscribe for more F1 updates!",
            "is_factual": False,
            "cited_fact_ids": [],
        },
    ]
})


@pytest.mark.asyncio
async def test_generate_script_success(db_manager: DatabaseManager):
    """Verify script generator produces structured beats strictly citing facts."""
    mock_provider = MockLLMProvider(responses=[MOCK_SCRIPT_JSON])
    router = LLMRouter(
        config=LLMConfig(fallback_order=["mock"], providers={"mock": ProviderSettings(model="m")}),
        db=db_manager,
        custom_providers={"mock": mock_provider},
    )
    generator = ScriptGenerator(router)

    facts = [
        ExtractedFact.create(
            job_id="j1", claim="Hamilton won 104th race", source_id="s1",
            evidence_quote="won 104th race", source_tier=SourceTier.TIER_2,
            corroboration_status=CorroborationStatus.VERIFIED,
        ),
        ExtractedFact.create(
            job_id="j1", claim="Hamilton beat Verstappen by 1.465 seconds", source_id="s1",
            evidence_quote="beat Verstappen by 1.465 seconds", source_tier=SourceTier.TIER_2,
            corroboration_status=CorroborationStatus.VERIFIED,
        ),
    ]

    script = await generator.generate_script(
        job_id="j1",
        headline="Hamilton wins Silverstone",
        facts=facts,
    )

    assert len(script.beats) == 3
    assert script.beats[0].beat_type == BeatType.HOOK
    assert script.beats[1].is_factual is True
    assert script.total_words > 0
    assert script.estimated_duration_seconds > 0.0


@pytest.mark.asyncio
async def test_generate_script_insufficient_facts(db_manager: DatabaseManager):
    """Verify generator refuses to invent content when insufficient facts exist."""
    router = LLMRouter(
        config=LLMConfig(fallback_order=["mock"], providers={"mock": ProviderSettings(model="m")}),
        db=db_manager,
        custom_providers={"mock": MockLLMProvider()},
    )
    generator = ScriptGenerator(router)

    with pytest.raises(ValueError, match="Insufficient usable facts"):
        await generator.generate_script(job_id="j1", headline="Empty topic", facts=[])
