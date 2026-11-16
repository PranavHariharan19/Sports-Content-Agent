import difflib
import re
from typing import Dict, List, Optional, Set, Tuple

from app.logging_config import get_logger
from schemas.facts import ExtractedFact
from schemas.sources import StoredSource

logger = get_logger(__name__)

COMMON_WORDS = {
    "The", "A", "An", "In", "On", "At", "To", "For", "With", "After", "Before",
    "And", "Or", "He", "She", "They", "It", "Three", "Two", "One", "Match",
    "Race", "Game", "Victory", "Defeat", "Today", "Yesterday", "Sunday",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
}


class DeterministicEvidenceChecker:
    """Deterministic, non-LLM validator that catches fabricated evidence and ungrounded claims."""

    def __init__(self, fuzzy_threshold: float = 0.85):
        self.fuzzy_threshold = fuzzy_threshold

    def normalize_text_for_search(self, text: str) -> str:
        """Strip punctuation and extra whitespace for fuzzy match."""
        return " ".join(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))

    def verify_quote_in_source(self, quote: str, source_text: str) -> bool:
        """Verify evidence quote actually appears in the source text."""
        if not quote or not source_text:
            return False

        # 1. Exact substring match
        if quote in source_text:
            return True

        # 2. Normalized whitespace match
        clean_quote = " ".join(quote.split()).lower()
        clean_source = " ".join(source_text.split()).lower()
        if clean_quote in clean_source:
            return True

        # 3. Fuzzy search for short quotes or punctuation variations
        tokens_quote = self.normalize_text_for_search(quote)
        tokens_source = self.normalize_text_for_search(source_text)
        if tokens_quote in tokens_source:
            return True

        # Check best match ratio across source chunks
        q_len = len(tokens_quote)
        if q_len > 15:
            words = tokens_source.split()
            q_words = tokens_quote.split()
            k = len(q_words)
            for i in range(0, max(1, len(words) - k + 1), max(1, k // 2)):
                window = " ".join(words[i : i + k])
                ratio = difflib.SequenceMatcher(None, tokens_quote, window).ratio()
                if ratio >= self.fuzzy_threshold:
                    return True

        return False

    def verify_claim_grounded_in_quote(
        self,
        claim: str,
        quote: str,
        source_text: str = "",
    ) -> Optional[str]:
        """Verify numbers, dates, and proper nouns in claim are strictly present in the quote.

        Allows coreference resolution if full name is present in source text and surname is in quote.
        """
        clean_quote_lower = quote.lower()
        clean_source_lower = source_text.lower() if source_text else clean_quote_lower

        # 1. Check numbers (e.g. scores, lap numbers, ages, prices)
        claim_numbers = re.findall(r"\b\d+(?:[\.,]\d+)?\b", claim)
        quote_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", quote))
        for num in claim_numbers:
            if num not in quote_numbers:
                return f"Number '{num}' in claim does not appear in evidence quote"

        # 2. Check proper nouns / capitalized names
        words_in_claim = re.findall(r"\b[A-Z][a-z]{2,}\b", claim)
        # Exclude first word if it is a common sentence starter or grammatical word
        first_word = words_in_claim[0] if words_in_claim and claim.startswith(words_in_claim[0]) else None

        for pn in words_in_claim:
            if pn in COMMON_WORDS:
                continue
            if pn == first_word and pn.lower() in ("race", "match", "game", "first", "second", "third", "three", "two", "one"):
                continue

            pn_lower = pn.lower()
            if pn_lower in clean_quote_lower:
                continue

            # Coreference check: if proper noun is first name (e.g. "Lewis") and surname ("Hamilton") is in quote
            # and full name is in parent source text, allow coreference
            has_coreference = False
            for other_pn in words_in_claim:
                if other_pn != pn and other_pn.lower() in clean_quote_lower:
                    full_name = f"{pn_lower} {other_pn.lower()}"
                    if full_name in clean_source_lower:
                        has_coreference = True
                        break

            if not has_coreference:
                return f"Proper noun '{pn}' in claim does not appear in evidence quote"

        return None

    def validate_facts(
        self,
        facts: List[ExtractedFact],
        sources_by_id: Dict[str, StoredSource],
    ) -> Tuple[List[ExtractedFact], List[ExtractedFact]]:
        """Validate all candidate facts against their backing source text."""
        passed: List[ExtractedFact] = []
        discarded: List[ExtractedFact] = []

        for fact in facts:
            source = sources_by_id.get(fact.source_id)
            if not source:
                fact.discard_reason = f"Backing source {fact.source_id} not found in database"
                discarded.append(fact)
                continue

            # Check 1: Quote in source
            if not self.verify_quote_in_source(fact.evidence_quote, source.cleaned_text):
                fact.discard_reason = "Evidence quote was not found in stored source text (fabricated quote)"
                logger.warning("Discarded fact %s: quote not in source", fact.fact_id)
                discarded.append(fact)
                continue

            # Check 2: Claim grounded in quote (with source text for coreference)
            grounding_error = self.verify_claim_grounded_in_quote(
                fact.claim,
                fact.evidence_quote,
                source.cleaned_text,
            )
            if grounding_error:
                fact.discard_reason = grounding_error
                logger.warning("Discarded fact %s: %s", fact.fact_id, grounding_error)
                discarded.append(fact)
                continue

            passed.append(fact)

        logger.info("Deterministic evidence check: %d passed, %d discarded", len(passed), len(discarded))
        return passed, discarded
