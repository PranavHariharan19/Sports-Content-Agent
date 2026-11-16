import pytest
from llm.mock_provider import MockLLMProvider
from llm.router import LLMRouter
from pipeline.db import DatabaseManager
from pipeline.script_verifier import ScriptVerifier
from schemas.common import CorroborationStatus, SourceTier, VerificationVerdict
from schemas.config import LLMConfig, ProviderSettings
from schemas.facts import ExtractedFact
from schemas.script import BeatType, ScriptBeat


@pytest.mark.asyncio
async def test_verifier_supported(db_manager: DatabaseManager):
    """Verify that a claim directly entailed by evidence returns SUPPORTED."""
    mock_response = '{"verdict": "SUPPORTED", "reason": "Hamilton won 104th race is directly stated in premise."}'
    mock_provider = MockLLMProvider(responses=[mock_response])
    router = LLMRouter(
        config=LLMConfig(fallback_order=["mock"], providers={"mock": ProviderSettings(model="m")}),
        db=db_manager,
        custom_providers={"mock": mock_provider},
    )
    verifier = ScriptVerifier(router)

    fact = ExtractedFact.create(
        job_id="job_v1",
        claim="Lewis Hamilton won his 104th career race at Silverstone.",
        source_id="s1",
        evidence_quote="Lewis Hamilton won his 104th career race at Silverstone on Sunday.",
        source_tier=SourceTier.TIER_2,
        corroboration_status=CorroborationStatus.VERIFIED,
    )

    beat = ScriptBeat(
        beat_id=1,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="Lewis Hamilton claimed his 104th victory at Silverstone.",
        cited_fact_ids=[fact.fact_id],
        is_factual=True,
    )

    result = await verifier.verify_beat(beat, [fact])
    assert result.verifier_verdict == VerificationVerdict.SUPPORTED
    assert "directly stated" in result.verifier_reason.lower()


@pytest.mark.asyncio
async def test_verifier_partially_supported(db_manager: DatabaseManager):
    """Verify that exaggeration or extra unproven details return PARTIALLY_SUPPORTED."""
    mock_response = '{"verdict": "PARTIALLY_SUPPORTED", "reason": "Premise says he won, but not that it was dominant or easy."}'
    mock_provider = MockLLMProvider(responses=[mock_response])
    router = LLMRouter(
        config=LLMConfig(fallback_order=["mock"], providers={"mock": ProviderSettings(model="m")}),
        db=db_manager,
        custom_providers={"mock": mock_provider},
    )
    verifier = ScriptVerifier(router)

    fact = ExtractedFact.create(
        job_id="job_v2",
        claim="Hamilton won by 1.465 seconds.",
        source_id="s1",
        evidence_quote="Hamilton held off Verstappen by 1.465 seconds.",
        source_tier=SourceTier.TIER_2,
        corroboration_status=CorroborationStatus.VERIFIED,
    )

    beat = ScriptBeat(
        beat_id=2,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="Hamilton secured the most dominant victory of his entire career.",
        cited_fact_ids=[fact.fact_id],
        is_factual=True,
    )

    result = await verifier.verify_beat(beat, [fact])
    assert result.verifier_verdict == VerificationVerdict.PARTIALLY_SUPPORTED
    assert "dominant" in result.verifier_reason.lower()


@pytest.mark.asyncio
async def test_verifier_unsupported(db_manager: DatabaseManager):
    """Verify that claims with no evidence support return UNSUPPORTED."""
    mock_response = '{"verdict": "UNSUPPORTED", "reason": "No evidence about Ferrari engine upgrades in premise."}'
    mock_provider = MockLLMProvider(responses=[mock_response])
    router = LLMRouter(
        config=LLMConfig(fallback_order=["mock"], providers={"mock": ProviderSettings(model="m")}),
        db=db_manager,
        custom_providers={"mock": mock_provider},
    )
    verifier = ScriptVerifier(router)

    fact = ExtractedFact.create(
        job_id="job_v3",
        claim="Hamilton won race.",
        source_id="s1",
        evidence_quote="Hamilton won race.",
        source_tier=SourceTier.TIER_2,
        corroboration_status=CorroborationStatus.VERIFIED,
    )

    beat = ScriptBeat(
        beat_id=3,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="Ferrari secretly supplied upgraded engines overnight.",
        cited_fact_ids=[fact.fact_id],
        is_factual=True,
    )

    result = await verifier.verify_beat(beat, [fact])
    assert result.verifier_verdict == VerificationVerdict.UNSUPPORTED


@pytest.mark.asyncio
async def test_verifier_contradicted(db_manager: DatabaseManager):
    """Verify that directly contradictory claims return CONTRADICTED."""
    mock_response = '{"verdict": "CONTRADICTED", "reason": "Premise says Hamilton won, but hypothesis says Verstappen won."}'
    mock_provider = MockLLMProvider(responses=[mock_response])
    router = LLMRouter(
        config=LLMConfig(fallback_order=["mock"], providers={"mock": ProviderSettings(model="m")}),
        db=db_manager,
        custom_providers={"mock": mock_provider},
    )
    verifier = ScriptVerifier(router)

    fact = ExtractedFact.create(
        job_id="job_v4",
        claim="Hamilton won.",
        source_id="s1",
        evidence_quote="Hamilton held off Verstappen to win.",
        source_tier=SourceTier.TIER_2,
        corroboration_status=CorroborationStatus.VERIFIED,
    )

    beat = ScriptBeat(
        beat_id=4,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="Verstappen crushed Hamilton to take first place.",
        cited_fact_ids=[fact.fact_id],
        is_factual=True,
    )

    result = await verifier.verify_beat(beat, [fact])
    assert result.verifier_verdict == VerificationVerdict.CONTRADICTED


@pytest.mark.asyncio
async def test_verifier_non_factual_beat(db_manager: DatabaseManager):
    """Rhetorical beats with is_factual=False automatically pass without LLM call."""
    mock_provider = MockLLMProvider(responses=[])
    router = LLMRouter(
        config=LLMConfig(fallback_order=["mock"], providers={"mock": ProviderSettings(model="m")}),
        db=db_manager,
        custom_providers={"mock": mock_provider},
    )
    verifier = ScriptVerifier(router)

    beat = ScriptBeat(
        beat_id=1,
        beat_type=BeatType.HOOK,
        spoken_text="Could this be the biggest comeback in sporting history?",
        cited_fact_ids=[],
        is_factual=False,
    )

    result = await verifier.verify_beat(beat, [])
    assert result.verifier_verdict == VerificationVerdict.SUPPORTED
    assert "Rhetorical line" in result.verifier_reason
    assert len(mock_provider.call_history) == 0


@pytest.mark.asyncio
async def test_verifier_factual_beat_missing_facts(db_manager: DatabaseManager):
    """Factual beat citing no facts should immediately return UNSUPPORTED."""
    mock_provider = MockLLMProvider(responses=[])
    router = LLMRouter(
        config=LLMConfig(fallback_order=["mock"], providers={"mock": ProviderSettings(model="m")}),
        db=db_manager,
        custom_providers={"mock": mock_provider},
    )
    verifier = ScriptVerifier(router)

    beat = ScriptBeat(
        beat_id=2,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="Hamilton drove at 300 km/h.",
        cited_fact_ids=[],
        is_factual=True,
    )

    result = await verifier.verify_beat(beat, [])
    assert result.verifier_verdict == VerificationVerdict.UNSUPPORTED
    assert "No cited facts" in result.verifier_reason
    assert len(mock_provider.call_history) == 0
