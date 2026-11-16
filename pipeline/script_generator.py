from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional
import uuid

from app.logging_config import get_logger
from llm.router import LLMRouter
from schemas.common import CorroborationStatus
from schemas.facts import ExtractedFact
from schemas.llm import LLMRequest
from schemas.script import BeatType, GeneratedScript, ScriptBeat

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a professional sports video scriptwriter producing punchy, factual YouTube Shorts scripts.
CRITICAL ANTI-HALLUCINATION RULES:
1. Use ONLY the provided verified facts below. NEVER use outside memory or invent any entity, number, date, score, or rumor.
2. Structure: Hook (first 2 seconds) -> 3-5 factual beats -> Closing line.
3. Word count: 80 to 120 words total (target: 30-45 seconds at ~2.5 words/sec).
4. Every beat making a factual claim MUST set is_factual=true and cite its backing fact_ids in cited_fact_ids.
5. If a fact requires attribution, you MUST explicitly name the reporting outlet in the spoken text (e.g. 'according to BBC Sport...').
6. Any beat with is_factual=false (hooks, transitions, closing lines) MUST NOT introduce any new entity, number, date, or claim.
7. Tone: Punchy, high-energy, neutral, strictly truthful.
"""

SCRIPT_PROMPT_TEMPLATE = """Write a 30-45 second YouTube Short script about: "{headline}".

Target word count: 80 - 120 words total.

AVAILABLE GROUND-TRUTH FACTS (Cite using fact_id):
{facts_list}

Output a JSON object with this exact schema:
{{
  "beats": [
    {{
      "beat_id": 1,
      "beat_type": "hook" | "factual_beat" | "closing",
      "spoken_text": "Spoken sentence here.",
      "is_factual": true | false,
      "cited_fact_ids": ["fact_id_1"]
    }}
  ]
}}
"""


class ScriptGenerator:
    """Generates structured, constrained video scripts strictly citing ledger facts."""

    def __init__(self, llm_router: LLMRouter):
        self.router = llm_router

    def _format_facts_for_prompt(self, facts: List[ExtractedFact]) -> str:
        lines: List[str] = []
        for f in facts:
            # Exclude unverified or conflicted facts
            if f.corroboration_status in (CorroborationStatus.UNVERIFIED, CorroborationStatus.CONFLICTED):
                continue

            attr_note = ""
            if f.requires_attribution and f.attribution_outlet:
                attr_note = f" (REQUIRES ATTRIBUTION: cite '{f.attribution_outlet}')"
            elif f.is_hedged:
                attr_note = " (HEDGED: state as reported/target, not definite)"

            lines.append(
                f"- [{f.fact_id}] {f.claim}{attr_note}\n"
                f"  Evidence quote: \"{f.evidence_quote}\""
            )
        return "\n".join(lines)

    async def generate_script(
        self,
        job_id: str,
        headline: str,
        facts: List[ExtractedFact],
        model: Optional[str] = None,
    ) -> GeneratedScript:
        """Draft a constrained structured script based exclusively on verified facts."""
        usable_facts = [
            f for f in facts
            if f.corroboration_status in (CorroborationStatus.VERIFIED, CorroborationStatus.REPORTED)
            and not f.is_conflicted
        ]

        if len(usable_facts) < 2:
            raise ValueError(f"Insufficient usable facts ({len(usable_facts)}) to generate script for job {job_id}")

        facts_text = self._format_facts_for_prompt(usable_facts)
        prompt = SCRIPT_PROMPT_TEMPLATE.format(headline=headline, facts_list=facts_text)

        request = LLMRequest(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            model=model,
            temperature=0.0,
            response_schema={
                "type": "object",
                "properties": {
                    "beats": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "beat_id": {"type": "integer"},
                                "beat_type": {"type": "string"},
                                "spoken_text": {"type": "string"},
                                "is_factual": {"type": "boolean"},
                                "cited_fact_ids": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["beat_id", "beat_type", "spoken_text", "is_factual", "cited_fact_ids"],
                        },
                    }
                },
                "required": ["beats"],
            },
        )

        response = await self.router.generate(request)
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        data = json.loads(raw_text)
        beats_data = data.get("beats", [])

        beats: List[ScriptBeat] = []
        for b in beats_data:
            raw_type = b.get("beat_type", "factual_beat").lower()
            try:
                b_type = BeatType(raw_type)
            except ValueError:
                b_type = BeatType.FACTUAL_BEAT

            beat = ScriptBeat(
                beat_id=int(b.get("beat_id", len(beats) + 1)),
                beat_type=b_type,
                spoken_text=b.get("spoken_text", "").strip(),
                is_factual=bool(b.get("is_factual", True)),
                cited_fact_ids=b.get("cited_fact_ids", []),
            )
            beats.append(beat)

        script_id = f"scr_{uuid.uuid4().hex[:10]}"
        return GeneratedScript(
            script_id=script_id,
            job_id=job_id,
            headline=headline,
            beats=beats,
            model_used=response.model,
        )
