import json
from typing import Dict, List, Optional

from app.logging_config import get_logger
from llm.router import LLMRouter
from schemas.common import VerificationVerdict
from schemas.facts import ExtractedFact
from schemas.llm import LLMRequest
from schemas.script import ScriptBeat

logger = get_logger(__name__)

VERIFIER_SYSTEM_PROMPT = """You are an adversarial Natural Language Inference (NLI) Entailment Judge for news broadcast verification.
Your sole responsibility is to evaluate whether the Premise (evidence quotes) strictly and logically ENTAILS the Hypothesis (the spoken sentence).
You must be uncompromising and vigilant against extrapolation, exaggeration, and invented details.
"""

VERIFIER_PROMPT_TEMPLATE = """Evaluate whether the Hypothesis is strictly supported by the Premise.

PREMISE (Verbatim Evidence Quotes):
\"\"\"
{premise_text}
\"\"\"

HYPOTHESIS (Spoken Sentence):
\"{hypothesis}\"

Rules:
- SUPPORTED: Every fact, number, entity, and event stated in the Hypothesis is directly proven by the Premise.
- PARTIALLY_SUPPORTED: The core idea is present, but details, numbers, or superlatives are exaggerated or unproven.
- UNSUPPORTED: The Premise does not provide evidence for the statement.
- CONTRADICTED: The Hypothesis directly disagrees with the Premise.

Output a JSON object with this exact schema:
{{
  "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED",
  "reason": "Detailed explanation"
}}
"""


class ScriptVerifier:
    """Independent LLM entailment verifier (Section 6.9)."""

    def __init__(self, llm_router: LLMRouter, verifier_model: Optional[str] = None):
        self.router = llm_router
        self.verifier_model = verifier_model

    async def verify_beat(
        self,
        beat: ScriptBeat,
        cited_facts: List[ExtractedFact],
    ) -> ScriptBeat:
        """Verify a single script beat against its cited evidence quotes."""
        # Non-factual rhetorical beats (e.g. "Can you believe what happened?") with no claims pass by default
        if not beat.is_factual:
            beat.verifier_verdict = VerificationVerdict.SUPPORTED
            beat.verifier_reason = "Rhetorical line (no factual claims to verify)"
            return beat

        if not cited_facts:
            beat.verifier_verdict = VerificationVerdict.UNSUPPORTED
            beat.verifier_reason = "No cited facts provided for factual beat"
            return beat

        premise_lines = [f"- \"{f.evidence_quote}\" (Source: {f.source_tier.value})" for f in cited_facts]
        premise_text = "\n".join(premise_lines)

        prompt = VERIFIER_PROMPT_TEMPLATE.format(
            premise_text=premise_text,
            hypothesis=beat.spoken_text,
        )

        request = LLMRequest(
            prompt=prompt,
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            model=self.verifier_model,
            temperature=0.0,
            response_schema={
                "type": "object",
                "properties": {
                    "verdict": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["verdict", "reason"],
            },
        )

        try:
            response = await self.router.generate(request)
            raw = response.text.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            data = json.loads(raw)
            raw_verdict = data.get("verdict", "UNSUPPORTED").upper()
            try:
                verdict = VerificationVerdict(raw_verdict)
            except ValueError:
                verdict = VerificationVerdict.UNSUPPORTED

            beat.verifier_verdict = verdict
            beat.verifier_reason = data.get("reason", "No reason provided")
            logger.info("Beat %d verification verdict: %s (%s)", beat.beat_id, verdict.value, beat.verifier_reason)

        except Exception as e:
            logger.error("Verification failed for beat %d: %s", beat.beat_id, str(e), exc_info=True)
            beat.verifier_verdict = VerificationVerdict.UNSUPPORTED
            beat.verifier_reason = f"Verifier error: {str(e)}"

        return beat

    async def verify_script(
        self,
        beats: List[ScriptBeat],
        facts_by_id: Dict[str, ExtractedFact],
    ) -> List[ScriptBeat]:
        """Verify all beats in a script."""
        verified_beats: List[ScriptBeat] = []
        for beat in beats:
            cited = [facts_by_id[fid] for fid in beat.cited_fact_ids if fid in facts_by_id]
            v_beat = await self.verify_beat(beat, cited)
            verified_beats.append(v_beat)
        return verified_beats
