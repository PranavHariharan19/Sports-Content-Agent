from pipeline.corroboration import CorroborationEngine
from schemas.common import CorroborationStatus, FactType, SourceTier
from schemas.facts import ExtractedFact
from schemas.sources import StoredSource


def test_two_independent_sources_verify_fact():
    """Verify 2 independent sources elevate a fact to VERIFIED."""
    engine = CorroborationEngine()

    src1 = StoredSource(
        source_id="src_bbc", job_id="j1", url="https://bbc.co.uk/1", domain="bbc.co.uk",
        source_tier=SourceTier.TIER_2, cleaned_text="Arsenal beat Chelsea 3-1.",
    )
    src2 = StoredSource(
        source_id="src_sky", job_id="j1", url="https://skysports.com/2", domain="skysports.com",
        source_tier=SourceTier.TIER_2, cleaned_text="Arsenal defeat Chelsea 3-1.",
    )
    sources = {src1.source_id: src1, src2.source_id: src2}

    f1 = ExtractedFact.create(
        job_id="j1", claim="Arsenal defeated Chelsea 3-1", source_id="src_bbc",
        evidence_quote="Arsenal beat Chelsea 3-1.", source_tier=SourceTier.TIER_2,
    )
    f2 = ExtractedFact.create(
        job_id="j1", claim="Arsenal defeated Chelsea 3-1", source_id="src_sky",
        evidence_quote="Arsenal defeat Chelsea 3-1.", source_tier=SourceTier.TIER_2,
    )

    facts = engine.evaluate_corroboration([f1, f2], sources)
    assert facts[0].corroboration_status == CorroborationStatus.VERIFIED
    assert len(facts[0].corroborating_source_ids) == 2


def test_single_tier1_source_is_verified():
    """Verify single Tier 1 official body fact is directly VERIFIED without needing second source."""
    engine = CorroborationEngine()

    src_fia = StoredSource(
        source_id="src_fia", job_id="j1", url="https://fia.com/decision", domain="fia.com",
        source_tier=SourceTier.TIER_1, cleaned_text="Penalty awarded to car 4.",
    )
    sources = {src_fia.source_id: src_fia}

    fact = ExtractedFact.create(
        job_id="j1", claim="Car 4 received a ten-second time penalty", source_id="src_fia",
        evidence_quote="Penalty awarded to car 4.", source_tier=SourceTier.TIER_1,
    )

    facts = engine.evaluate_corroboration([fact], sources)
    assert facts[0].corroboration_status == CorroborationStatus.VERIFIED


def test_single_tier2_source_is_reported_with_attribution():
    """Verify single Tier 2 fact is marked REPORTED and requires explicit attribution."""
    engine = CorroborationEngine()

    src_espn = StoredSource(
        source_id="src_espn", job_id="j1", url="https://espn.com/rumor", domain="espn.com",
        source_tier=SourceTier.TIER_2, cleaned_text="Arsenal in discussions for new winger.",
    )
    sources = {src_espn.source_id: src_espn}

    fact = ExtractedFact.create(
        job_id="j1", claim="Arsenal are in discussions for a new winger", source_id="src_espn",
        evidence_quote="Arsenal in discussions for new winger.", source_tier=SourceTier.TIER_2,
        fact_type=FactType.TRANSFER,
    )

    facts = engine.evaluate_corroboration([fact], sources)
    assert facts[0].corroboration_status == CorroborationStatus.REPORTED
    assert facts[0].requires_attribution is True
    assert facts[0].attribution_outlet == "espn.com"
    assert facts[0].is_hedged is True


def test_syndicated_copies_count_as_one_source():
    """Verify syndicated copies sharing syndication_cluster_id do NOT count as 2 independent sources."""
    engine = CorroborationEngine()

    src1 = StoredSource(
        source_id="src_1", job_id="j1", url="https://outletA.com/wire", domain="outletA.com",
        source_tier=SourceTier.TIER_2, cleaned_text="Wire copy story text",
        is_syndicated=True, syndication_cluster_id="syn_wire_123",
    )
    src2 = StoredSource(
        source_id="src_2", job_id="j1", url="https://outletB.com/wire", domain="outletB.com",
        source_tier=SourceTier.TIER_2, cleaned_text="Wire copy story text",
        is_syndicated=True, syndication_cluster_id="syn_wire_123",
    )
    sources = {src1.source_id: src1, src2.source_id: src2}

    f1 = ExtractedFact.create(
        job_id="j1", claim="Wire copy claim statement", source_id="src_1",
        evidence_quote="Wire copy story text", source_tier=SourceTier.TIER_2,
    )
    f2 = ExtractedFact.create(
        job_id="j1", claim="Wire copy claim statement", source_id="src_2",
        evidence_quote="Wire copy story text", source_tier=SourceTier.TIER_2,
    )

    facts = engine.evaluate_corroboration([f1, f2], sources)
    # Because both are the same syndication cluster, they do NOT reach 2 independent sources
    assert facts[0].corroboration_status == CorroborationStatus.REPORTED


def test_conflict_detection_on_disagreeing_scores():
    """Verify conflicting scores across sources are flagged and recorded."""
    engine = CorroborationEngine()

    src1 = StoredSource(source_id="s1", job_id="j1", url="https://bbc.com", domain="bbc.com", source_tier=SourceTier.TIER_2, cleaned_text="Arsenal won 3-1.")
    src2 = StoredSource(source_id="s2", job_id="j1", url="https://sky.com", domain="sky.com", source_tier=SourceTier.TIER_2, cleaned_text="Arsenal won 2-1.")
    sources = {"s1": src1, "s2": src2}

    f1 = ExtractedFact.create(
        job_id="j1", claim="Arsenal won 3-1 against Chelsea", source_id="s1",
        evidence_quote="Arsenal won 3-1.", source_tier=SourceTier.TIER_2,
        fact_type=FactType.SCORE, normalized_entities=["Arsenal", "Chelsea"], normalized_value="3-1",
    )
    f2 = ExtractedFact.create(
        job_id="j1", claim="Arsenal won 2-1 against Chelsea", source_id="s2",
        evidence_quote="Arsenal won 2-1.", source_tier=SourceTier.TIER_2,
        fact_type=FactType.SCORE, normalized_entities=["Arsenal", "Chelsea"], normalized_value="2-1",
    )

    conflicts = engine.detect_conflicts([f1, f2], sources, job_id="j1")
    assert len(conflicts) == 1
    assert "3-1" in conflicts[0].conflicting_values
    assert "2-1" in conflicts[0].conflicting_values
    assert f1.is_conflicted is True
    assert f2.is_conflicted is True
