from pipeline.script_linter import DeterministicScriptLinter
from schemas.common import CorroborationStatus, FactType, SourceTier
from schemas.facts import ExtractedFact
from schemas.script import BeatType, ScriptBeat


def test_grounded_beat_passes_lint():
    """Verify clean, properly cited beat passes lint with 0 errors."""
    linter = DeterministicScriptLinter()

    f1 = ExtractedFact.create(
        job_id="j1", claim="Lewis Hamilton won 104 races", source_id="s1",
        evidence_quote="Lewis Hamilton won his 104th race", source_tier=SourceTier.TIER_2,
        normalized_entities=["Lewis Hamilton"], normalized_value="104",
    )
    facts_by_id = {f1.fact_id: f1}
    ledger_entities = {"Lewis Hamilton"}

    beat = ScriptBeat(
        beat_id=1,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="Lewis Hamilton sealed his 104th career win.",
        is_factual=True,
        cited_fact_ids=[f1.fact_id],
    )

    errors = linter.lint_beat(beat, facts_by_id, ledger_entities)
    assert len(errors) == 0


def test_ungrounded_number_fails_lint():
    """Verify beat with an invented number is rejected."""
    linter = DeterministicScriptLinter()

    f1 = ExtractedFact.create(
        job_id="j1", claim="Arsenal won 3-1", source_id="s1",
        evidence_quote="Arsenal won 3-1.", source_tier=SourceTier.TIER_2,
        normalized_entities=["Arsenal"], normalized_value="3-1",
    )
    facts_by_id = {f1.fact_id: f1}
    ledger_entities = {"Arsenal"}

    # Beat claims 7 goals, which is absent in cited fact
    beat = ScriptBeat(
        beat_id=1,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="Arsenal won 7-1 in a complete blowout.",
        is_factual=True,
        cited_fact_ids=[f1.fact_id],
    )

    errors = linter.lint_beat(beat, facts_by_id, ledger_entities)
    assert len(errors) >= 1
    assert any("Number '7' in beat is not present" in e for e in errors)


def test_ungrounded_entity_fails_lint():
    """Verify beat with an invented entity absent from ledger is rejected."""
    linter = DeterministicScriptLinter()

    f1 = ExtractedFact.create(
        job_id="j1", claim="Max Verstappen won in Bahrain", source_id="s1",
        evidence_quote="Max Verstappen won in Bahrain", source_tier=SourceTier.TIER_1,
        normalized_entities=["Max Verstappen", "Bahrain"],
    )
    facts_by_id = {f1.fact_id: f1}
    ledger_entities = {"Max Verstappen", "Bahrain"}

    # Beat invents Charles Leclerc
    beat = ScriptBeat(
        beat_id=1,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="Max Verstappen passed Charles Leclerc to take victory.",
        is_factual=True,
        cited_fact_ids=[f1.fact_id],
    )

    errors = linter.lint_beat(beat, facts_by_id, ledger_entities)
    assert len(errors) >= 1
    assert any("Proper noun 'Charles' is ungrounded" in e for e in errors)


def test_missing_required_attribution_fails_lint():
    """Verify single-source reported fact fails lint if outlet attribution is omitted."""
    linter = DeterministicScriptLinter()

    f1 = ExtractedFact.create(
        job_id="j1", claim="Arsenal in talks for winger", source_id="s1",
        evidence_quote="Arsenal in talks for winger", source_tier=SourceTier.TIER_2,
    )
    f1.requires_attribution = True
    f1.attribution_outlet = "bbc.co.uk"
    facts_by_id = {f1.fact_id: f1}
    ledger_entities = {"Arsenal"}

    # Beat mentions claim but forgets to say "according to BBC"
    beat = ScriptBeat(
        beat_id=1,
        beat_type=BeatType.FACTUAL_BEAT,
        spoken_text="Arsenal are actively in talks for a winger.",
        is_factual=True,
        cited_fact_ids=[f1.fact_id],
    )

    errors = linter.lint_beat(beat, facts_by_id, ledger_entities)
    assert len(errors) >= 1
    assert any("omits required attribution to 'bbc.co.uk'" in e for e in errors)


def test_rhetorical_line_with_numbers_fails_lint():
    """Verify rhetorical line sneaking in factual numbers fails lint."""
    linter = DeterministicScriptLinter()

    beat = ScriptBeat(
        beat_id=1,
        beat_type=BeatType.HOOK,
        spoken_text="Can you believe they scored 5 goals in 10 minutes?",
        is_factual=False,
        cited_fact_ids=[],
    )

    errors = linter.lint_beat(beat, {}, set())
    assert len(errors) >= 1
    assert any("introduces unauthorized numbers" in e for e in errors)
