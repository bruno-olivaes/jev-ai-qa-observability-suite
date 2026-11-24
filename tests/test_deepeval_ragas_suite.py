import json
import sys
from pathlib import Path

import pytest
from deepeval.test_case import LLMTestCase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_framework import (
    DeterministicAnswerRelevancyMetric,
    DeterministicContextualPrecisionMetric,
    DeterministicFaithfulnessMetric,
    DeterministicHallucinationMetric,
    evaluate_dataset,
    run_eval_assert,
)


def test_golden_and_adversarial_eval_dataset():
    raw_cases = json.loads((ROOT / "data" / "golden_eval_cases.json").read_text(encoding="utf-8"))
    cases = [
        LLMTestCase(
            input=r["input"],
            actual_output=r["actual_output"],
            expected_output=r["expected_output"],
            retrieval_context=r["retrieval_context"],
        )
        for r in raw_cases
    ]

    report = evaluate_dataset(cases, velith_cutoff=7.0)
    assert report.total_cases == 6
    for idx, row in enumerate(raw_cases):
        assert report.case_details[idx]["passed"] is row["expect_pass"], (
            f"Mismatch on {row['id']}: {report.case_details[idx]}"
        )


def test_assert_test_contract_passes_golden_and_fails_hallucination():
    metrics = [
        DeterministicFaithfulnessMetric(threshold=0.70),
        DeterministicAnswerRelevancyMetric(threshold=0.65),
        DeterministicHallucinationMetric(threshold=0.30),
        DeterministicContextualPrecisionMetric(threshold=0.50),
    ]

    good_case = LLMTestCase(
        input="What latency SLA does JEV enforce?",
        expected_output="JEV enforces P95 latency under 500 ms.",
        actual_output="JEV enforces P95 latency under 500 ms for typed decisions.",
        retrieval_context=["JEV enforces P95 latency under 500 ms for structured classification."],
    )
    res = run_eval_assert(good_case, metrics)
    assert res["Faithfulness"]["score"] >= 0.70

    bad_case = LLMTestCase(
        input="What latency SLA does JEV enforce?",
        expected_output="JEV enforces P95 latency under 500 ms.",
        actual_output="JEV guarantees 15 ms latency with 9999 TPS throughput.",
        retrieval_context=["JEV enforces P95 latency under 500 ms for structured classification."],
    )
    with pytest.raises(AssertionError):
        run_eval_assert(bad_case, metrics)
