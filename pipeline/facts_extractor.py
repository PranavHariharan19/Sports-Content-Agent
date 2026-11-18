import json
from typing import Any, Dict, List, Optional

from app.logging_config import get_logger
from llm.router import LLMRouter
from schemas.common import FactType, SourceTier
from schemas.facts import ExtractedFact
from schemas.llm import LLMRequest
from schemas.sources import StoredSource

logger = get_logger(__name__)

SYSTEM_INSTRUCTION = (
    "You are a forensic sports fact extractor. "
    "CRITICAL RULE: Use only the provided text. If the answer is not in it, output nothing for it. "
    "Do not use outside knowledge. Extract atomic facts into a JSON array of objects."
)

EXTRACTION_PROMPT_TEMPLATE = """Use only the provided text. If the answer is not in it, output nothing for it. Do not use outside knowledge.

Extract atomic facts from the provided text into a JSON array of objects with the exact schema:
[
  {{
    "claim": "one atomic factual statement",
    "fact_type": "score" | "date" | "name" | "number" | "quote" | "transfer" | "injury" | "record" | "ranking" | "other",
    "normalized_entities": ["Entity1", "Entity2"],
    "normalized_value": "3-1" or "58" or "2026-11-16" or null,
    "evidence_quote": "VERBATIM EXACT substring from the text supporting this claim"
  }}
]

TEXT:
\"\"\"
{text}
\"\"\"
"""


class FactsExtractor:
    """Extracts atomic facts with verbatim evidence quotes from stored source texts."""

    def __init__(self, llm_router: LLMRouter):
        self.router = llm_router

    async def extract_facts_from_source(self, source: StoredSource, job_id: str) -> List[ExtractedFact]:
        """Extract atomic facts from a single stored source."""
        if not source.cleaned_text or len(source.cleaned_text.strip()) < 10:
            logger.warning("Source %s has empty or very short text, skipping extraction", source.source_id)
            return []

        # Enforce prompt injection boundary
        text = source.cleaned_text[:4000]  # Cap at 4000 characters for focused extraction
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(text=text)

        request = LLMRequest(
            prompt=prompt,
            system_prompt=SYSTEM_INSTRUCTION,
            temperature=0.0,
            response_schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "fact_type": {"type": "string"},
                        "normalized_entities": {"type": "array", "items": {"type": "string"}},
                        "normalized_value": {"type": "string"},
                        "evidence_quote": {"type": "string"},
                    },
                    "required": ["claim", "fact_type", "evidence_quote"],
                },
            },
        )

        try:
            response = await self.router.generate(request)
            raw_text = response.text.strip()
            # If wrapped in markdown json block, clean it
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                for key in ("facts", "claims", "extracted_facts", "atomic_facts", "items", "data"):
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
                else:
                    for val in parsed.values():
                        if isinstance(val, list) and (not val or isinstance(val[0], dict)):
                            parsed = val
                            break

            if not isinstance(parsed, list):
                logger.warning("Extraction response for source %s was not a list: %s", source.source_id, raw_text[:200])
                return []

            facts: List[ExtractedFact] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                claim = item.get("claim", "").strip()
                quote = item.get("evidence_quote", "").strip()
                if not claim or not quote:
                    continue

                raw_ftype = item.get("fact_type", "other").lower()
                try:
                    ftype = FactType(raw_ftype)
                except ValueError:
                    ftype = FactType.OTHER

                entities = item.get("normalized_entities", [])
                if not isinstance(entities, list):
                    entities = []

                norm_val = item.get("normalized_value")
                if norm_val is not None:
                    norm_val = str(norm_val).strip()

                fact = ExtractedFact.create(
                    job_id=job_id,
                    claim=claim,
                    source_id=source.source_id,
                    evidence_quote=quote,
                    source_tier=source.source_tier,
                    fact_type=ftype,
                    normalized_entities=[str(e).strip() for e in entities if e],
                    normalized_value=norm_val,
                    published_at=source.published_at,
                )
                facts.append(fact)

            logger.info("Extracted %d candidate facts from source %s (%s)", len(facts), source.source_id, source.domain)
            return facts

        except Exception as e:
            logger.error("Fact extraction failed for source %s: %s", source.source_id, str(e), exc_info=True)
            return []
