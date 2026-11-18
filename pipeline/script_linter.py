import re
from typing import Dict, List, Optional, Set

from app.logging_config import get_logger
from schemas.facts import ExtractedFact
from schemas.script import ScriptBeat

logger = get_logger(__name__)

BANNED_SUPERLATIVES = [
    r"\bgreatest\s+ever\b",
    r"\bbest\s+(?:player|driver|striker|match|race)\s+in\s+history\b",
    r"\bfirst\s+time\s+in\s+history\b",
    r"\bnever\s+before\s+seen\b",
    r"\bunprecedented\s+in\s+all\s+sports\b",
]

COMMON_WORDS = {
    "The", "A", "An", "In", "On", "At", "To", "For", "With", "After", "Before",
    "And", "Or", "He", "She", "They", "It", "Three", "Two", "One", "Match",
    "Race", "Game", "Victory", "Defeat", "Today", "Yesterday", "Sunday",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "This", "That", "Here", "Now", "Watch", "What", "Who", "How", "Why",
}


class DeterministicScriptLinter:
    """Non-LLM deterministic linter enforcing strict grounding, entity checks,
    attribution, and banning hallucinated superlatives.
    """

    def __init__(self, enforce_outlet_attribution: bool = False):
        self.enforce_outlet_attribution = enforce_outlet_attribution

    def lint_beat(
        self,
        beat: ScriptBeat,
        facts_by_id: Dict[str, ExtractedFact],
        ledger_entities: Set[str],
    ) -> List[str]:
        """Validate a single script beat against cited facts and fact ledger."""
        errors: List[str] = []
        text = beat.spoken_text
        text_lower = text.lower()

        # 1. Rhetorical beats rule (Section 6.7)
        if not beat.is_factual:
            # Rhetorical beats may not introduce numbers or dates
            numbers = re.findall(r"\b\d+(?:[\.,]\d+)?\b", text)
            if numbers:
                errors.append(f"Rhetorical line introduces unauthorized numbers: {numbers}")
            return errors

        # 2. Citation existence rule
        if not beat.cited_fact_ids:
            errors.append("Factual beat does not cite any fact_ids from the ledger")
            return errors

        cited_facts: List[ExtractedFact] = []
        for fid in beat.cited_fact_ids:
            fact = facts_by_id.get(fid)
            if not fact:
                errors.append(f"Cited fact_id '{fid}' does not exist in fact ledger")
            else:
                cited_facts.append(fact)

        if not cited_facts:
            return errors

        # Aggregate allowed numbers and text from cited facts
        cited_text = " ".join(f"{f.claim} {f.evidence_quote}" for f in cited_facts).lower()
        allowed_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", cited_text))

        # 3. Numbers grounding check
        beat_numbers = re.findall(r"\b\d+(?:[\.,]\d+)?\b", text)
        for num in beat_numbers:
            if num not in allowed_numbers:
                errors.append(f"Number '{num}' in beat is not present in cited facts {beat.cited_fact_ids}")

        # 4. Proper noun / entity presence check
        proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        first_word = proper_nouns[0] if proper_nouns and text.startswith(proper_nouns[0]) else None
        clean_ledger_entities = {e.lower() for e in ledger_entities}

        for pn in proper_nouns:
            if pn in COMMON_WORDS or pn == first_word:
                continue
            pn_lower = pn.lower()
            # Allowed if in cited text or known in ledger
            if pn_lower not in cited_text and pn_lower not in clean_ledger_entities:
                errors.append(f"Proper noun '{pn}' is ungrounded and does not appear in ledger")

        # 5. Banned superlatives check
        for pattern in BANNED_SUPERLATIVES:
            if re.search(pattern, text_lower):
                # Only allowed if a cited fact explicitly supports a record or history claim
                has_record_fact = any(
                    "record" in f.claim.lower() or "history" in f.claim.lower()
                    for f in cited_facts
                )
                if not has_record_fact:
                    errors.append(f"Banned unsupported superlative pattern detected in beat")
                    break

        # 6. Optional attribution check (only if enforce_outlet_attribution is enabled)
        if self.enforce_outlet_attribution:
            for f in cited_facts:
                if f.requires_attribution and f.attribution_outlet:
                    clean_outlet = f.attribution_outlet.split(".")[0].lower()
                    if clean_outlet not in text_lower:
                        errors.append(f"Beat cites single-source fact '{f.fact_id}' but omits required attribution to '{f.attribution_outlet}'")

        return errors

    def lint_script(
        self,
        beats: List[ScriptBeat],
        facts_by_id: Dict[str, ExtractedFact],
        ledger_entities: Set[str],
    ) -> List[ScriptBeat]:
        """Lint all beats in a script and attach errors."""
        for beat in beats:
            beat.lint_errors = self.lint_beat(beat, facts_by_id, ledger_entities)
        return beats
