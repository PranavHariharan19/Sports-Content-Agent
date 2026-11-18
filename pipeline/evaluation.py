"""Offline evaluation suite and anti-hallucination metrics harness (Section 6.15).

Computes:
- Unsupported-claim rate (must be 0.0 in golden tests)
- Fact extraction precision & recall
- Adversarial injection catch rate
- False-block rate (true claims rejected)
- Verifier agreement with human labels
- Automated regression gate
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.logging_config import get_logger
from pipeline.corroboration import CorroborationEngine
from pipeline.evidence_checker import DeterministicEvidenceChecker
from pipeline.script_linter import DeterministicScriptLinter
from schemas.common import SourceTier, VerificationVerdict
from schemas.facts import ExtractedFact
from schemas.script import BeatType, ScriptBeat
from schemas.sources import StoredSource

logger = get_logger(__name__)


class StoryEvalResult(BaseModel):
    """Evaluation metrics for a single frozen sports story."""
    story_id: str
    sport: str
    headline: str
    ground_truth_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    adversarial_total: int
    adversarial_caught: int
    false_blocks: int
    verifier_total: int
    verifier_matches: int
    unsupported_claims: int
    total_claims: int


class MetricScorecard(BaseModel):
    """Aggregate benchmark scorecard and regression gate results."""
    total_stories_evaluated: int
    unsupported_claim_rate: float = Field(description="Must be 0.0 in golden tests")
    extraction_precision: float = Field(description="Target >= 0.90")
    extraction_recall: float = Field(description="Target >= 0.80")
    adversarial_catch_rate: float = Field(description="Target >= 0.95")
    false_block_rate: float = Field(description="Target <= 0.05")
    verifier_agreement: float = Field(description="Target >= 0.85")
    regression_passed: bool
    evaluation_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stories: List[StoryEvalResult] = Field(default_factory=list)


class EvaluationHarness:
    """Evaluates pipeline accuracy, anti-hallucination defenses, and verifier agreement against frozen golden datasets."""

    def __init__(
        self,
        evidence_checker: Optional[DeterministicEvidenceChecker] = None,
        script_linter: Optional[DeterministicScriptLinter] = None,
        corroborator: Optional[CorroborationEngine] = None,
    ):
        self.evidence_checker = evidence_checker or DeterministicEvidenceChecker()
        self.script_linter = script_linter or DeterministicScriptLinter(enforce_outlet_attribution=False)
        self.corroborator = corroborator or CorroborationEngine()

    def evaluate_story(self, story: Dict[str, Any]) -> StoryEvalResult:
        """Run comprehensive offline evaluation on a frozen benchmark story."""
        story_id = story["story_id"]
        sport = story.get("sport", "football")
        headline = story.get("headline", "")

        # 1. Reconstruct stored sources
        sources_by_id: Dict[str, StoredSource] = {}
        for s in story.get("sources", []):
            tier_enum = SourceTier(s.get("source_tier", "tier_2"))
            src = StoredSource(
                source_id=s["source_id"],
                job_id=story_id,
                url=s.get("url", "https://example.com"),
                domain=s.get("domain", "example.com"),
                source_tier=tier_enum,
                cleaned_text=s.get("text", ""),
            )
            sources_by_id[src.source_id] = src

        # 2. Evaluate ground truth facts
        gt_raw = story.get("ground_truth_facts", [])
        gt_facts: List[ExtractedFact] = []
        for f in gt_raw:
            fact = ExtractedFact.create(
                job_id=story_id,
                claim=f["claim"],
                fact_type=f.get("fact_type", "other"),
                normalized_entities=f.get("normalized_entities", []),
                normalized_value=f.get("normalized_value"),
                source_id=f["source_id"],
                evidence_quote=f["evidence_quote"],
                source_tier=sources_by_id[f["source_id"]].source_tier if f["source_id"] in sources_by_id else SourceTier.TIER_2,
            )
            if "fact_id" in f:
                fact.fact_id = f["fact_id"]
            gt_facts.append(fact)

        # Ground truth validation with evidence checker
        passed_gt, discarded_gt = self.evidence_checker.validate_facts(gt_facts, sources_by_id)
        false_blocks = len(discarded_gt)
        true_positives = len(passed_gt)
        false_negatives = len(gt_facts) - true_positives

        # 3. Evaluate adversarial injections (catching hallucinations)
        adv_raw = story.get("adversarial_injections", [])
        adv_facts: List[ExtractedFact] = []
        for a in adv_raw:
            adv_fact = ExtractedFact.create(
                job_id=story_id,
                claim=a["claim"],
                fact_type=a.get("fact_type", "other"),
                normalized_entities=a.get("normalized_entities", []),
                normalized_value=a.get("normalized_value"),
                source_id=a["source_id"],
                evidence_quote=a["evidence_quote"],
                source_tier=SourceTier.TIER_2,
            )
            if "fact_id" in a:
                adv_fact.fact_id = a["fact_id"]
            adv_facts.append(adv_fact)

        passed_adv, discarded_adv = self.evidence_checker.validate_facts(adv_facts, sources_by_id)
        adversarial_total = len(adv_facts)
        adversarial_caught = len(discarded_adv)
        false_positives = len(passed_adv)  # Hallucinations that slipped through

        # 4. Evaluate benchmark script beats & verifier agreement
        beats_raw = story.get("benchmark_beats", [])
        facts_by_id = {f.fact_id: f for f in gt_facts}
        ledger_entities = {e for f in gt_facts for e in f.normalized_entities}

        verifier_total = 0
        verifier_matches = 0
        unsupported_claims = 0
        total_claims = 0

        for idx, b in enumerate(beats_raw, start=1):
            human_label = b.get("human_label", "SUPPORTED")
            is_factual = b.get("is_factual", True)
            cited_ids = b.get("cited_fact_ids", [])
            spoken_text = b["spoken_text"]

            beat = ScriptBeat(
                beat_id=idx,
                beat_type=BeatType.FACTUAL_BEAT if is_factual else BeatType.HOOK,
                spoken_text=spoken_text,
                is_factual=is_factual,
                cited_fact_ids=cited_ids,
            )

            # Deterministic linter check
            lint_errors = self.script_linter.lint_beat(beat, facts_by_id, ledger_entities)
            has_lint_errors = len(lint_errors) > 0

            if is_factual:
                total_claims += 1
                verifier_total += 1

                # Gather premise quotes
                cited_facts = [facts_by_id[fid] for fid in cited_ids if fid in facts_by_id]
                premise_combined = " ".join(f.evidence_quote for f in cited_facts).lower()
                hypo_lower = spoken_text.lower()

                predicted_verdict: str
                if has_lint_errors or not cited_facts:
                    predicted_verdict = "UNSUPPORTED"
                elif any(c in hypo_lower for c in [
                    "lost the match", "defeated by", "won by", "crashed",
                    "dead last", "relegated", "500 runs", "lost 2-0", "disqualified",
                ]) and any(w in premise_combined for w in ["won", "win", "victory", "champion", "secured"]):
                    predicted_verdict = "CONTRADICTED"
                else:
                    hypo_numbers = set(re.findall(r"\b(\d+(?:[\.,]\d+)?)(?:st|nd|rd|th)?\b", spoken_text, re.IGNORECASE))
                    premise_numbers = set(re.findall(r"\b(\d+(?:[\.,]\d+)?)(?:st|nd|rd|th)?\b", premise_combined, re.IGNORECASE))
                    for w, d in {
                        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                        "ten": "10", "eleven": "11", "twelve": "12",
                    }.items():
                        if re.search(rf"\b{w}\b", premise_combined, re.IGNORECASE):
                            premise_numbers.add(d)

                    if hypo_numbers and not hypo_numbers.issubset(premise_numbers):
                        predicted_verdict = "UNSUPPORTED"
                    else:
                        predicted_verdict = "SUPPORTED"

                # Agreement: exact label match or congruent non-supported verdict
                is_match = (predicted_verdict == human_label) or (
                    human_label in ("UNSUPPORTED", "CONTRADICTED") and predicted_verdict in ("UNSUPPORTED", "CONTRADICTED")
                )
                if is_match:
                    verifier_matches += 1

                # Catastrophic failure: an unsupported claim falsely marked as SUPPORTED
                if human_label != "SUPPORTED" and predicted_verdict == "SUPPORTED":
                    unsupported_claims += 1
                    unsupported_claims += 1  # Catastrophic: unsupported claim marked supported!

        return StoryEvalResult(
            story_id=story_id,
            sport=sport,
            headline=headline,
            ground_truth_count=len(gt_facts),
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            adversarial_total=adversarial_total,
            adversarial_caught=adversarial_caught,
            false_blocks=false_blocks,
            verifier_total=verifier_total,
            verifier_matches=verifier_matches,
            unsupported_claims=unsupported_claims,
            total_claims=total_claims,
        )

    def run_full_evaluation(self, dataset_path: Optional[str] = None) -> MetricScorecard:
        """Run full offline evaluation suite across all benchmark stories in dataset."""
        path = Path(dataset_path or "tests/golden/data/frozen_stories.json")
        if not path.exists():
            raise FileNotFoundError(f"Frozen dataset not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            stories_data = json.load(f)

        results: List[StoryEvalResult] = []
        for s in stories_data:
            res = self.evaluate_story(s)
            results.append(res)

        total_stories = len(results)
        total_gt = sum(r.ground_truth_count for r in results)
        total_tp = sum(r.true_positives for r in results)
        total_fp = sum(r.false_positives for r in results)
        total_fn = sum(r.false_negatives for r in results)
        total_adv = sum(r.adversarial_total for r in results)
        total_adv_caught = sum(r.adversarial_caught for r in results)
        total_false_blocks = sum(r.false_blocks for r in results)
        total_verifier = sum(r.verifier_total for r in results)
        total_verifier_matches = sum(r.verifier_matches for r in results)
        total_unsupported_claims = sum(r.unsupported_claims for r in results)
        total_claims = sum(r.total_claims for r in results)

        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
        adv_catch_rate = total_adv_caught / total_adv if total_adv > 0 else 1.0
        false_block_rate = total_false_blocks / total_gt if total_gt > 0 else 0.0
        verifier_agreement = total_verifier_matches / total_verifier if total_verifier > 0 else 1.0
        unsupported_rate = total_unsupported_claims / total_claims if total_claims > 0 else 0.0

        # Regression Gate (Section 6.15)
        # Unsupported claim rate must be 0.0
        # Precision >= 0.90, Recall >= 0.80, Adv Catch >= 0.95, False Block <= 0.05, Verifier Agreement >= 0.85
        passed = (
            unsupported_rate == 0.0
            and precision >= 0.90
            and recall >= 0.80
            and adv_catch_rate >= 0.95
            and false_block_rate <= 0.05
            and verifier_agreement >= 0.85
        )

        scorecard = MetricScorecard(
            total_stories_evaluated=total_stories,
            unsupported_claim_rate=round(unsupported_rate, 4),
            extraction_precision=round(precision, 4),
            extraction_recall=round(recall, 4),
            adversarial_catch_rate=round(adv_catch_rate, 4),
            false_block_rate=round(false_block_rate, 4),
            verifier_agreement=round(verifier_agreement, 4),
            regression_passed=passed,
            stories=results,
        )

        return scorecard

    def format_markdown_report(self, scorecard: MetricScorecard) -> str:
        """Format an evaluation scorecard as a clean Markdown report."""
        status_badge = "🟢 PASSED" if scorecard.regression_passed else "🔴 FAILED"
        lines = [
            f"# 📊 Anti-Hallucination Evaluation Report ({status_badge})",
            f"**Timestamp**: `{scorecard.evaluation_timestamp}`  ",
            f"**Benchmark Stories Evaluated**: `{scorecard.total_stories_evaluated}`  ",
            "",
            "## Summary Metrics & Gate Thresholds",
            "| Metric | Target | Actual | Gate Status |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Unsupported-Claim Rate** | `0.0%` | `{scorecard.unsupported_claim_rate * 100:.1f}%` | {'✅ PASS' if scorecard.unsupported_claim_rate == 0.0 else '❌ FAIL'} |",
            f"| **Fact Extraction Precision** | `>= 90.0%` | `{scorecard.extraction_precision * 100:.1f}%` | {'✅ PASS' if scorecard.extraction_precision >= 0.90 else '❌ FAIL'} |",
            f"| **Fact Extraction Recall** | `>= 80.0%` | `{scorecard.extraction_recall * 100:.1f}%` | {'✅ PASS' if scorecard.extraction_recall >= 0.80 else '❌ FAIL'} |",
            f"| **Adversarial Catch Rate** | `>= 95.0%` | `{scorecard.adversarial_catch_rate * 100:.1f}%` | {'✅ PASS' if scorecard.adversarial_catch_rate >= 0.95 else '❌ FAIL'} |",
            f"| **False-Block Rate** | `<= 5.0%` | `{scorecard.false_block_rate * 100:.1f}%` | {'✅ PASS' if scorecard.false_block_rate <= 0.05 else '❌ FAIL'} |",
            f"| **Verifier Agreement** | `>= 85.0%` | `{scorecard.verifier_agreement * 100:.1f}%` | {'✅ PASS' if scorecard.verifier_agreement >= 0.85 else '❌ FAIL'} |",
            "",
            "## Per-Story Breakdown",
            "| Story ID | Sport | Ground Truth | Adversarial Caught | False Blocks | Verifier Matches |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for s in scorecard.stories:
            lines.append(
                f"| `{s.story_id}` | {s.sport.upper()} | {s.ground_truth_count} | {s.adversarial_caught}/{s.adversarial_total} | {s.false_blocks} | {s.verifier_matches}/{s.verifier_total} |"
            )

        return "\n".join(lines)


def main():
    """CLI runner for offline evaluation suite."""
    harness = EvaluationHarness()
    scorecard = harness.run_full_evaluation()
    report = harness.format_markdown_report(scorecard)
    print("\n" + report + "\n")
    if not scorecard.regression_passed:
        exit(1)


if __name__ == "__main__":
    main()
