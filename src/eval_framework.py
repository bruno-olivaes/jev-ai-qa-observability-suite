"""
DeepEval & RAGAS-Compatible Deterministic Evaluation Framework (Velith Quality Gate).

Implements `FaithfulnessMetric`, `AnswerRelevancyMetric`, `HallucinationMetric`,
`ContextualPrecisionMetric`, and `assert_test(test_case, metrics)` compatible with
`deepeval.test_case.LLMTestCase` so automated pytest & CI gates verify grounding,
relevance, and factual precision without requiring external paid API keys at test time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

_STOPWORDS: Set[str] = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "have", "has", "had", "into", "onto", "about", "what", "when", "where",
    "which", "who", "how", "can", "will", "would", "should", "could", "your",
    "our", "their", "its", "para", "com", "sem", "que", "uma", "uns", "das",
    "dos", "por", "como", "mais", "mas", "sobre", "entre", "esta", "este",
    "isso", "pelo", "pela", "nao", "não", "sim", "all", "only", "under",
    "does", "did", "doing", "been", "being", "may", "might", "within", "after",
    "maximum", "minimum", "give", "tell",
}
_NEGATIONS: Set[str] = {"not", "never", "no", "cannot", "non", "without", "não", "nunca", "jamais", "sem"}


def _stem_token(tok: str) -> str:
    if len(tok) > 5 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 4 and tok.endswith(("ses", "xes", "zes", "ches", "shes")):
        return tok[:-2]
    if len(tok) > 4 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _tokenize_content(text: str) -> Set[str]:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())
    return {_stem_token(t) for t in tokens if len(t) >= 3 and t not in _STOPWORDS and not t.isdigit()}


def _extract_numbers(text: str) -> Set[str]:
    nums = re.findall(r"\b\d+(?:[.,]\d+)?\b", text)
    return {n.replace(",", ".") for n in nums}


def _split_claims(text: str) -> List[str]:
    # Preserve decimal numbers like 0.72 or 25.5 when splitting sentences
    parts = [p.strip() for p in re.split(r"(?<!\d)\.(?!\d)|[!?\n;]+", text) if p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def _has_negation(text: str) -> bool:
    tokens = set(re.findall(r"[a-zA-ZÀ-ÿ0-9']+", text.lower()))
    return bool(tokens & _NEGATIONS)


class DeterministicFaithfulnessMetric(BaseMetric):
    """Checks whether every claim & numeric figure in `actual_output` is grounded in `retrieval_context`."""

    def __init__(self, threshold: float = 0.70, include_reason: bool = True) -> None:
        self.threshold = threshold
        self.include_reason = include_reason
        self.score: Optional[float] = None
        self.reason: Optional[str] = None
        self.success: Optional[bool] = None
        self.unsupported_claims: List[str] = []

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        claims = _split_claims(test_case.actual_output or "")
        contexts = list(test_case.retrieval_context or test_case.context or [])
        if not claims or not contexts:
            self.score = 0.0
            self.success = False
            self.reason = "Missing claims or retrieval context."
            return 0.0

        ctx_tokens_per_chunk = [_tokenize_content(c) for c in contexts]
        ctx_union_tokens: Set[str] = set().union(*ctx_tokens_per_chunk)
        ctx_numbers: Set[str] = set().union(*(_extract_numbers(c) for c in contexts))

        supported = 0
        self.unsupported_claims = []
        for claim in claims:
            c_tokens = _tokenize_content(claim)
            c_nums = _extract_numbers(claim)
            ungrounded_nums = c_nums - ctx_numbers

            if not c_tokens and not c_nums:
                supported += 1
                continue

            max_chunk_overlap = max(
                (len(c_tokens & ch) / max(len(c_tokens), 1) for ch in ctx_tokens_per_chunk),
                default=0.0,
            )
            union_overlap = len(c_tokens & ctx_union_tokens) / max(len(c_tokens), 1)

            # Check polarity contradiction against best matching context sentence
            claim_neg = _has_negation(claim)
            polarity_conflict = False
            for ctx in contexts:
                for ctx_sent in _split_claims(ctx):
                    s_tokens = _tokenize_content(ctx_sent)
                    if c_tokens and len(c_tokens & s_tokens) / max(len(c_tokens), 1) >= 0.45:
                        if _has_negation(ctx_sent) != claim_neg:
                            polarity_conflict = True
                            break

            if not ungrounded_nums and not polarity_conflict and (max_chunk_overlap >= 0.45 or union_overlap >= 0.55):
                supported += 1
            else:
                reasons = []
                if ungrounded_nums:
                    reasons.append(f"ungrounded numbers {sorted(ungrounded_nums)}")
                if polarity_conflict:
                    reasons.append("negation/polarity mismatch")
                if max_chunk_overlap < 0.45 and union_overlap < 0.55:
                    reasons.append("low lexical grounding")
                self.unsupported_claims.append(f"{claim} ({', '.join(reasons)})")

        self.score = round(supported / len(claims), 4)
        self.success = self.score >= self.threshold
        if self.include_reason:
            self.reason = (
                f"{supported}/{len(claims)} claims grounded."
                if self.success
                else f"Unsupported claims detected: {'; '.join(self.unsupported_claims)}"
            )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Faithfulness"


class DeterministicAnswerRelevancyMetric(BaseMetric):
    """Measures how directly `actual_output` covers the core concepts asked in `input`."""

    def __init__(self, threshold: float = 0.65, include_reason: bool = True) -> None:
        self.threshold = threshold
        self.include_reason = include_reason
        self.score: Optional[float] = None
        self.reason: Optional[str] = None
        self.success: Optional[bool] = None

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        q_tokens = _tokenize_content(test_case.input or "")
        a_tokens = _tokenize_content(test_case.actual_output or "")
        if not q_tokens or not a_tokens:
            self.score = 0.0
            self.success = False
            self.reason = "Empty question or answer tokens."
            return 0.0

        q_cov = len(q_tokens & a_tokens) / len(q_tokens)
        e_cov = 0.0
        if test_case.expected_output:
            e_tokens = _tokenize_content(test_case.expected_output)
            if e_tokens:
                e_cov = len(e_tokens & a_tokens) / len(e_tokens)

        concept_cov = q_cov if q_cov == 0.0 else max(q_cov, 0.30 * q_cov + 0.70 * e_cov)
        ans_density = len((q_tokens | _tokenize_content(test_case.expected_output or "")) & a_tokens) / max(len(a_tokens), 1)
        score = min(1.0, 0.85 * concept_cov + 0.15 * min(1.0, ans_density * 2.5))
        self.score = round(float(score), 4)
        self.success = self.score >= self.threshold
        self.reason = f"Query coverage={q_cov:.2f}; concept relevancy={self.score:.2f}."
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Answer Relevancy"


class DeterministicHallucinationMetric(BaseMetric):
    """Inverse faithfulness rate: fraction of ungrounded/contradicted claims (lower is better)."""

    def __init__(self, threshold: float = 0.30, include_reason: bool = True) -> None:
        self.threshold = threshold
        self.include_reason = include_reason
        self.score: Optional[float] = None
        self.reason: Optional[str] = None
        self.success: Optional[bool] = None

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        faith = DeterministicFaithfulnessMetric(threshold=1.0 - self.threshold)
        f_score = faith.measure(test_case)
        self.score = round(1.0 - f_score, 4)
        self.success = self.score <= self.threshold
        self.reason = (
            f"Hallucination rate={self.score:.2f} <= {self.threshold:.2f}."
            if self.success
            else f"Hallucination rate={self.score:.2f} > {self.threshold:.2f}: {faith.reason}"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Hallucination"


class DeterministicContextualPrecisionMetric(BaseMetric):
    """RAGAS-style ranked retrieval precision: checks whether relevant context chunks appear early."""

    def __init__(self, threshold: float = 0.60, include_reason: bool = True) -> None:
        self.threshold = threshold
        self.include_reason = include_reason
        self.score: Optional[float] = None
        self.reason: Optional[str] = None
        self.success: Optional[bool] = None

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        target_text = (test_case.expected_output or test_case.input or "").strip()
        contexts = list(test_case.retrieval_context or test_case.context or [])
        target_tokens = _tokenize_content(target_text)
        if not target_tokens or not contexts:
            self.score = 0.0
            self.success = False
            self.reason = "Missing target text or retrieval contexts."
            return 0.0

        hits = 0
        precision_sum = 0.0
        for rank_idx, chunk in enumerate(contexts, start=1):
            overlap = len(target_tokens & _tokenize_content(chunk)) / max(len(target_tokens), 1)
            if overlap >= 0.30:
                hits += 1
                precision_sum += hits / rank_idx

        self.score = round(precision_sum / hits, 4) if hits > 0 else 0.0
        self.success = self.score >= self.threshold
        self.reason = f"Ranked contextual precision={self.score:.2f} ({hits}/{len(contexts)} relevant chunks)."
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Contextual Precision"


def compute_velith_score(faithfulness: float, relevancy: float, hallucination: float) -> float:
    """Computes the Velith 0-10 composite quality score (passing threshold >= 7.0)."""
    return round(10.0 * (0.50 * faithfulness + 0.30 * relevancy + 0.20 * (1.0 - hallucination)), 4)


def run_eval_assert(test_case: LLMTestCase, metrics: Sequence[BaseMetric]) -> Dict[str, Any]:
    """DeepEval `assert_test`-style assertion returning structured per-metric scores."""
    results: Dict[str, Any] = {}
    failures: List[str] = []
    for m in metrics:
        val = float(m.measure(test_case))
        ok = m.is_successful()
        results[m.__name__] = {"score": val, "success": ok, "reason": getattr(m, "reason", "")}
        if not ok:
            failures.append(f"{m.__name__}={val:.2f} ({getattr(m, 'reason', '')})")
    if failures:
        raise AssertionError("Eval metrics failed: " + " | ".join(failures))
    return results


@dataclass
class BatchEvalReport:
    total_cases: int
    passed_cases: int
    mean_faithfulness: float
    mean_relevancy: float
    mean_hallucination: float
    mean_contextual_precision: float
    mean_velith_score: float
    case_details: List[Dict[str, Any]]


def evaluate_dataset(cases: Sequence[LLMTestCase], velith_cutoff: float = 7.0) -> BatchEvalReport:
    """Runs full RAGAS/DeepEval-style suite over a golden+adversarial evaluation dataset."""
    faith_m = DeterministicFaithfulnessMetric(threshold=0.70)
    rel_m = DeterministicAnswerRelevancyMetric(threshold=0.65)
    hal_m = DeterministicHallucinationMetric(threshold=0.30)
    prec_m = DeterministicContextualPrecisionMetric(threshold=0.60)

    details: List[Dict[str, Any]] = []
    for idx, tc in enumerate(cases):
        f_s = faith_m.measure(tc)
        r_s = rel_m.measure(tc)
        h_s = hal_m.measure(tc)
        p_s = prec_m.measure(tc)
        velith = compute_velith_score(f_s, r_s, h_s)
        passed = bool(faith_m.is_successful() and rel_m.is_successful() and hal_m.is_successful() and velith >= velith_cutoff)
        details.append(
            {
                "case_index": idx,
                "input": tc.input,
                "faithfulness": f_s,
                "answer_relevancy": r_s,
                "hallucination": h_s,
                "contextual_precision": p_s,
                "velith_score": velith,
                "passed": passed,
                "unsupported_claims": list(faith_m.unsupported_claims),
            }
        )

    n = max(len(details), 1)
    return BatchEvalReport(
        total_cases=len(details),
        passed_cases=sum(1 for d in details if d["passed"]),
        mean_faithfulness=round(sum(d["faithfulness"] for d in details) / n, 4),
        mean_relevancy=round(sum(d["answer_relevancy"] for d in details) / n, 4),
        mean_hallucination=round(sum(d["hallucination"] for d in details) / n, 4),
        mean_contextual_precision=round(sum(d["contextual_precision"] for d in details) / n, 4),
        mean_velith_score=round(sum(d["velith_score"] for d in details) / n, 4),
        case_details=details,
    )
