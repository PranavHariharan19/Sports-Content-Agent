from pipeline.evidence_checker import DeterministicEvidenceChecker
from schemas.common import FactType, SourceTier
from schemas.facts import ExtractedFact
from schemas.sources import StoredSource


SAMPLE_SOURCE_TEXT = (
    "Silverstone, UK - Lewis Hamilton delivered a masterclass at the British Grand Prix, "
    "securing his 104th career Formula 1 victory on Sunday. The 39-year-old British driver "
    "held off Max Verstappen by 1.465 seconds in changing weather conditions. "
    "Lando Norris finished in third place after a late pit stop delay cost McLaren."
)

SAMPLE_SOURCE = StoredSource(
    source_id="src_silverstone",
    job_id="job_silverstone",
    url="https://www.bbc.co.uk/sport/formula1/100",
    domain="bbc.co.uk",
    source_tier=SourceTier.TIER_2,
    cleaned_text=SAMPLE_SOURCE_TEXT,
)


def test_verbatim_quote_passes():
    """Verify an authentic verbatim quote and grounded claim pass validation."""
    checker = DeterministicEvidenceChecker()
    fact = ExtractedFact.create(
        job_id="job_silverstone",
        claim="Lewis Hamilton secured his 104th career victory",
        source_id="src_silverstone",
        evidence_quote="Lewis Hamilton delivered a masterclass at the British Grand Prix, securing his 104th career Formula 1 victory on Sunday",
        source_tier=SourceTier.TIER_2,
        fact_type=FactType.RECORD,
        normalized_entities=["Lewis Hamilton"],
        normalized_value="104",
    )

    passed, discarded = checker.validate_facts([fact], {SAMPLE_SOURCE.source_id: SAMPLE_SOURCE})
    assert len(passed) == 1
    assert len(discarded) == 0


def test_fabricated_quote_is_discarded():
    """Verify a hallucinated/fabricated quote not in source text is discarded."""
    checker = DeterministicEvidenceChecker()
    fact = ExtractedFact.create(
        job_id="job_silverstone",
        claim="Lewis Hamilton called it the greatest race of his life",
        source_id="src_silverstone",
        evidence_quote="This was unequivocally the greatest race of my entire life, Hamilton beamed",
        source_tier=SourceTier.TIER_2,
    )

    passed, discarded = checker.validate_facts([fact], {SAMPLE_SOURCE.source_id: SAMPLE_SOURCE})
    assert len(passed) == 0
    assert len(discarded) == 1
    assert "not found in stored source text" in discarded[0].discard_reason


def test_invented_number_in_claim_is_discarded():
    """Verify a claim containing a number absent in the evidence quote is discarded."""
    checker = DeterministicEvidenceChecker()
    fact = ExtractedFact.create(
        job_id="job_silverstone",
        claim="Lewis Hamilton held off Verstappen by 5.2 seconds",  # Source says 1.465 seconds
        source_id="src_silverstone",
        evidence_quote="held off Max Verstappen by 1.465 seconds in changing weather conditions",
        source_tier=SourceTier.TIER_2,
    )

    passed, discarded = checker.validate_facts([fact], {SAMPLE_SOURCE.source_id: SAMPLE_SOURCE})
    assert len(passed) == 0
    assert len(discarded) == 1
    assert "Number '5.2' in claim does not appear in evidence quote" in discarded[0].discard_reason


def test_invented_entity_in_claim_is_discarded():
    """Verify a claim introducing an entity not in the quote is discarded."""
    checker = DeterministicEvidenceChecker()
    fact = ExtractedFact.create(
        job_id="job_silverstone",
        claim="Charles Leclerc congratulated Lewis Hamilton",
        source_id="src_silverstone",
        evidence_quote="Lewis Hamilton delivered a masterclass at the British Grand Prix",
        source_tier=SourceTier.TIER_2,
    )

    passed, discarded = checker.validate_facts([fact], {SAMPLE_SOURCE.source_id: SAMPLE_SOURCE})
    assert len(passed) == 0
    assert len(discarded) == 1
    assert "Proper noun 'Charles' in claim does not appear in evidence quote" in discarded[0].discard_reason
