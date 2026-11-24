"""
LangGraph Stateful QA Workflow: JEV System-1 Triage -> Generator -> Evaluator -> Conditional Retry Loop.

Demonstrates an end-to-end stateful multi-node LangGraph pipeline combining:
1. JEV System-1 typed pre-triage (`jev_triage_node`)
2. Pluggable / mockable LLM generation (`generator_node`)
3. DeepEval/Velith automated quality gate evaluation (`evaluator_node`, score >= 7.0)
4. Conditional retry routing (`route_after_evaluation`) sending critique back on failures
5. OpenTelemetry/OpenInference & Datadog span correlation on every node transition
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

from deepeval.test_case import LLMTestCase
from langgraph.graph import END, START, StateGraph

from src.eval_framework import (
    DeterministicAnswerRelevancyMetric,
    DeterministicFaithfulnessMetric,
    DeterministicHallucinationMetric,
    compute_velith_score,
)
from src.jev_engine import JEVClassifier, JEVDecisionRequest
from src.observability_phoenix_datadog import DEFAULT_TELEMETRY, TelemetryManager


class WorkflowState(TypedDict, total=False):
    request_id: str
    query: str
    amount_brl: float
    account_age_days: int
    context: List[str]
    jev_decision: Dict[str, Any]
    draft_answer: str
    eval_score: float
    eval_details: Dict[str, Any]
    critique: str
    attempt_count: int
    retry_count: int
    max_retries: int
    status: str
    history: List[Dict[str, Any]]


GeneratorCallable = Callable[[WorkflowState], str]


def default_grounded_generator(state: WorkflowState) -> str:
    """Synthesizes a strictly context-grounded answer without unsupported claims."""
    ctx_joined = ". ".join(s.strip().rstrip(".") for s in state.get("context", []) if s.strip())
    return f"Based on verified policy context: {ctx_joined}." if ctx_joined else "No context provided."


def build_qa_langgraph(
    generator_fn: Optional[GeneratorCallable] = None,
    classifier: Optional[JEVClassifier] = None,
    telemetry: Optional[TelemetryManager] = None,
    velith_threshold: float = 7.0,
):
    """Builds and compiles the stateful LangGraph QA verification workflow."""
    tel = telemetry or DEFAULT_TELEMETRY
    jev = classifier or JEVClassifier(telemetry=tel)
    gen = generator_fn or default_grounded_generator

    def jev_triage_node(state: WorkflowState) -> Dict[str, Any]:
        req = JEVDecisionRequest(
            request_id=state["request_id"],
            text=state["query"],
            amount_brl=float(state.get("amount_brl", 0.0)),
            account_age_days=int(state.get("account_age_days", 365)),
        )
        decision = jev.classify(req).model_dump()
        hist = list(state.get("history", []))
        hist.append({"node": "jev_triage_node", "route": decision["action_route"], "class": decision["predicted_class"]})
        return {"jev_decision": decision, "history": hist, "attempt_count": 0, "retry_count": 0}

    def route_after_jev(state: WorkflowState) -> Literal["blocked", "generate"]:
        if state["jev_decision"]["action_route"] == "REJECT":
            return "blocked"
        return "generate"

    def blocked_node(state: WorkflowState) -> Dict[str, Any]:
        hist = list(state.get("history", []))
        hist.append({"node": "blocked_node", "reason": state["jev_decision"]["predicted_class"]})
        return {
            "draft_answer": f"Request blocked by System-1 JEV guardrail ({state['jev_decision']['predicted_class']}).",
            "eval_score": 10.0,
            "status": "BLOCKED_BY_JEV",
            "history": hist,
        }

    def generator_node(state: WorkflowState) -> Dict[str, Any]:
        attempt = int(state.get("attempt_count", 0)) + 1
        retries = attempt - 1
        with tel.start_span(
            "langgraph.generator_node",
            span_kind="LLM",
            attributes={
                "llm.model_name": "mock-or-pluggable-llm",
                "input.value": state["query"][:200],
                "langgraph.attempt": attempt,
            },
            dd_logger="langgraph.generator",
        ) as span:
            answer = gen(state)
            span.set_attribute("output.value", answer[:250])
        hist = list(state.get("history", []))
        hist.append({"node": "generator_node", "attempt": attempt, "retry_count": retries})
        return {
            "draft_answer": answer,
            "attempt_count": attempt,
            "retry_count": retries,
            "history": hist,
        }

    def evaluator_node(state: WorkflowState) -> Dict[str, Any]:
        with tel.start_span(
            "langgraph.evaluator_node",
            span_kind="EVALUATOR",
            attributes={"eval.threshold": velith_threshold, "jev.request_id": state["request_id"]},
            dd_logger="langgraph.evaluator",
        ) as span:
            tc = LLMTestCase(
                input=state["query"],
                actual_output=state.get("draft_answer", ""),
                retrieval_context=list(state.get("context", [])),
            )
            faith = DeterministicFaithfulnessMetric(threshold=0.70)
            rel = DeterministicAnswerRelevancyMetric(threshold=0.65)
            hal = DeterministicHallucinationMetric(threshold=0.30)

            f_score = faith.measure(tc)
            r_score = rel.measure(tc)
            h_score = hal.measure(tc)
            velith_score = compute_velith_score(f_score, r_score, h_score)

            critique = ""
            if velith_score < velith_threshold:
                critique = f"Score {velith_score:.2f} < {velith_threshold:.2f}. {faith.reason} {rel.reason}"

            span.set_attribute("eval.faithfulness", f_score)
            span.set_attribute("eval.relevancy", r_score)
            span.set_attribute("eval.hallucination", h_score)
            span.set_attribute("eval.score", velith_score)

        hist = list(state.get("history", []))
        hist.append({"node": "evaluator_node", "velith_score": velith_score, "passed": velith_score >= velith_threshold})
        return {
            "eval_score": velith_score,
            "eval_details": {
                "faithfulness": f_score,
                "relevancy": r_score,
                "hallucination": h_score,
                "unsupported_claims": list(faith.unsupported_claims),
            },
            "critique": critique,
            "history": hist,
        }

    def route_after_evaluation(state: WorkflowState) -> Literal["approved", "retry", "escalated"]:
        if float(state.get("eval_score", 0.0)) >= velith_threshold:
            return "approved"
        if int(state.get("retry_count", 0)) < int(state.get("max_retries", 2)):
            return "retry"
        return "escalated"

    def finalize_approved_node(state: WorkflowState) -> Dict[str, Any]:
        hist = list(state.get("history", []))
        hist.append({"node": "finalize_approved_node"})
        return {"status": "APPROVED", "history": hist}

    def finalize_escalated_node(state: WorkflowState) -> Dict[str, Any]:
        hist = list(state.get("history", []))
        hist.append({"node": "finalize_escalated_node"})
        return {"status": "ESCALATED_LOW_SCORE", "history": hist}

    graph = StateGraph(WorkflowState)
    graph.add_node("jev_triage_node", jev_triage_node)
    graph.add_node("blocked_node", blocked_node)
    graph.add_node("generator_node", generator_node)
    graph.add_node("evaluator_node", evaluator_node)
    graph.add_node("finalize_approved_node", finalize_approved_node)
    graph.add_node("finalize_escalated_node", finalize_escalated_node)

    graph.add_edge(START, "jev_triage_node")
    graph.add_conditional_edges(
        "jev_triage_node",
        route_after_jev,
        {"blocked": "blocked_node", "generate": "generator_node"},
    )
    graph.add_edge("blocked_node", END)
    graph.add_edge("generator_node", "evaluator_node")
    graph.add_conditional_edges(
        "evaluator_node",
        route_after_evaluation,
        {
            "approved": "finalize_approved_node",
            "retry": "generator_node",
            "escalated": "finalize_escalated_node",
        },
    )
    graph.add_edge("finalize_approved_node", END)
    graph.add_edge("finalize_escalated_node", END)

    return graph.compile()
