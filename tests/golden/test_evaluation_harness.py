"""Automated pytest test suite and regression gate for Milestone 6 evaluation harness (Section 6.15)."""
from pathlib import Path
import pytest

from pipeline.evaluation import EvaluationHarness, MetricScorecard, StoryEvalResult


def test_evaluation_harness_regression_gate():
    """Verify that the full frozen dataset passes all anti-hallucination regression gates."""
    harness = EvaluationHarness()
    scorecard: MetricScorecard = harness.run_full_evaluation()

    assert scorecard.total_stories_evaluated >= 10
    # Mandatory Gate 1: Unsupported-claim rate must be exactly 0.0%
    assert scorecard.unsupported_claim_rate == 0.0, f"Unsupported-claim rate regressed: {scorecard.unsupported_claim_rate}"
    # Mandatory Gate 2: Fact extraction precision >= 90%
    assert scorecard.extraction_precision >= 0.90, f"Extraction precision regressed: {scorecard.extraction_precision}"
    # Mandatory Gate 3: Fact extraction recall >= 80%
    assert scorecard.extraction_recall >= 0.80, f"Extraction recall regressed: {scorecard.extraction_recall}"
    # Mandatory Gate 4: Adversarial injection catch rate >= 95%
    assert scorecard.adversarial_catch_rate >= 0.95, f"Adversarial catch rate regressed: {scorecard.adversarial_catch_rate}"
    # Mandatory Gate 5: False-block rate <= 5%
    assert scorecard.false_block_rate <= 0.05, f"False-block rate regressed: {scorecard.false_block_rate}"
    # Mandatory Gate 6: Verifier agreement >= 85%
    assert scorecard.verifier_agreement >= 0.85, f"Verifier agreement regressed: {scorecard.verifier_agreement}"

    assert scorecard.regression_passed is True


def test_evaluation_markdown_report_formatting():
    """Verify markdown report generation produces properly structured tables and badges."""
    harness = EvaluationHarness()
    scorecard = harness.run_full_evaluation()
    report = harness.format_markdown_report(scorecard)

    assert "# 📊 Anti-Hallucination Evaluation Report" in report
    assert "🟢 PASSED" in report
    assert "| **Unsupported-Claim Rate** | `0.0%` |" in report
    assert "| **Fact Extraction Precision** | `>= 90.0%` |" in report
    assert "| **Adversarial Catch Rate** | `>= 95.0%` |" in report
    assert "## Per-Story Breakdown" in report
    assert "`fb_1`" in report
    assert "`cr_1`" in report
    assert "`f1_1`" in report


def test_evaluation_single_story_eval():
    """Verify evaluate_story executes correctly on an isolated frozen story."""
    harness = EvaluationHarness()
    sample_story = {
        "story_id": "test_sample_1",
        "sport": "cricket",
        "headline": "India win thrilling match",
        "sources": [
            {
                "source_id": "s1",
                "domain": "espncricinfo.com",
                "source_tier": "tier_2",
                "text": "India defeated England by 15 runs in a thriller at Edgbaston. Jasprit Bumrah took 3 wickets for 20 runs.",
            }
        ],
        "ground_truth_facts": [
            {
                "fact_id": "f1",
                "claim": "India defeated England by 15 runs at Edgbaston",
                "source_id": "s1",
                "evidence_quote": "India defeated England by 15 runs in a thriller at Edgbaston.",
                "normalized_entities": ["India", "England", "Edgbaston"],
                "normalized_value": "15 runs",
            }
        ],
        "adversarial_injections": [
            {
                "fact_id": "adv_1",
                "claim": "England won by 500 runs",
                "source_id": "s1",
                "evidence_quote": "India defeated England by 15 runs in a thriller at Edgbaston.",
            }
        ],
        "benchmark_beats": [
            {
                "spoken_text": "India defeated England by 15 runs at Edgbaston.",
                "is_factual": True,
                "cited_fact_ids": ["f1"],
                "human_label": "SUPPORTED",
            }
        ],
    }

    result: StoryEvalResult = harness.evaluate_story(sample_story)
    assert result.story_id == "test_sample_1"
    assert result.true_positives == 1
    assert result.false_blocks == 0
    assert result.adversarial_caught == 1
    assert result.unsupported_claims == 0
    assert result.verifier_matches == 1
