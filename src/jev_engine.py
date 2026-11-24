"""
JEV (System-1) Fast Structured Classification & Decision Engine.

Models the architectural discipline of TypeSafe AI's JEV "System-1" concept:
- Sub-500ms typed decision classification without free-text generation (zero JSON schema hallucination)
- Strict Pydantic v2 `extra="forbid"` request/response contracts
- Temperature-scaled softmax confidence calibration and automatic escalation below threshold
- OpenTelemetry/OpenInference + Datadog log emission on every call
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.observability_phoenix_datadog import DEFAULT_TELEMETRY, TelemetryManager

JEVClass = Literal["LOW_RISK_AUTO", "POLICY_CHECK_NEEDED", "HIGH_RISK_FRAUD", "PROMPT_INJECTION"]
ActionRoute = Literal["APPROVE", "HUMAN_REVIEW", "REJECT"]

CLASSES: List[JEVClass] = [
    "LOW_RISK_AUTO",
    "POLICY_CHECK_NEEDED",
    "HIGH_RISK_FRAUD",
    "PROMPT_INJECTION",
]


class JEVDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., min_length=1, max_length=64)
    text: str = Field(..., min_length=3, max_length=4000)
    amount_brl: float = Field(default=0.0, ge=0.0, le=10_000_000.0)
    account_age_days: int = Field(default=365, ge=0, le=50_000)
    channel: Literal["api", "web", "whatsapp", "erp"] = "api"


class JEVDecisionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    predicted_class: JEVClass
    action_route: ActionRoute
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[JEVClass, float]
    latency_ms: float = Field(..., ge=0.0)
    schema_version: Literal["jev-s1-v1"] = "jev-s1-v1"
    policy_flags: List[str] = Field(default_factory=list)


class JEVClassifier:
    """Fast System-1 typed classifier with confidence-gated routing and temperature scaling."""

    _INJECTION_TOKENS = (
        "ignore previous",
        "ignore instructions",
        "system prompt",
        "bypass",
        "jailbreak",
        "reveal secret",
        "drop table",
        "sudo ",
    )
    _FRAUD_TOKENS = (
        "urgent wire",
        "laranja",
        "chargeback",
        "stolen card",
        "cripto saque imediato",
        "unauthorized transfer",
        "fraud",
        "lavagem",
    )
    _POLICY_TOKENS = (
        "refund",
        "reembolso",
        "cancelamento",
        "dispute",
        "disputa",
        "manual review",
        "estorno",
        "sla breach",
        "procon",
    )

    def __init__(
        self,
        temperature: float = 1.15,
        approve_confidence_threshold: float = 0.72,
        telemetry: Optional[TelemetryManager] = None,
    ) -> None:
        self.temperature = temperature
        self.approve_confidence_threshold = approve_confidence_threshold
        self.telemetry = telemetry or DEFAULT_TELEMETRY

    def _compute_logits(self, req: JEVDecisionRequest) -> tuple[Dict[JEVClass, float], List[str]]:
        lowered = req.text.lower()
        flags: List[str] = []

        inj_hits = sum(1 for tok in self._INJECTION_TOKENS if tok in lowered)
        fraud_hits = sum(1 for tok in self._FRAUD_TOKENS if tok in lowered)
        policy_hits = sum(1 for tok in self._POLICY_TOKENS if tok in lowered)

        if inj_hits:
            flags.append("injection_signature")
        if fraud_hits:
            flags.append("fraud_keyword_hit")
        if req.amount_brl >= 5000.0 and req.account_age_days < 14:
            fraud_hits += 2
            flags.append("high_value_new_account")
        elif req.amount_brl >= 10000.0:
            policy_hits += 1
            flags.append("high_value_threshold")
        if policy_hits:
            flags.append("policy_escalation_trigger")

        logits: Dict[JEVClass, float] = {
            "LOW_RISK_AUTO": 3.05 - 2.1 * policy_hits - 2.9 * fraud_hits - 3.6 * inj_hits,
            "POLICY_CHECK_NEEDED": 0.15 + 2.65 * policy_hits - 0.8 * fraud_hits - 1.5 * inj_hits,
            "HIGH_RISK_FRAUD": -0.05 + 2.9 * fraud_hits - 0.4 * policy_hits,
            "PROMPT_INJECTION": -0.35 + 3.8 * inj_hits,
        }
        return logits, flags

    def _softmax(self, logits: Dict[JEVClass, float]) -> Dict[JEVClass, float]:
        scaled = {k: v / self.temperature for k, v in logits.items()}
        max_l = max(scaled.values())
        exps = {k: math.exp(v - max_l) for k, v in scaled.items()}
        denom = sum(exps.values())
        probs = {k: round(exps[k] / denom, 6) for k in CLASSES}
        # Normalize rounding residual on max element so exact sum matches 1.0
        residual = round(1.0 - sum(probs.values()), 6)
        top_key = max(CLASSES, key=lambda c: probs[c])
        probs[top_key] = round(probs[top_key] + residual, 6)
        return probs

    def classify(self, req: JEVDecisionRequest) -> JEVDecisionResponse:
        t0 = time.perf_counter()
        with self.telemetry.start_span(
            "jev.system1.classify",
            span_kind="GUARDRAIL",
            attributes={
                "llm.model_name": "jev-system1-classifier-v1",
                "input.value": req.text[:250],
                "jev.channel": req.channel,
                "jev.amount_brl": req.amount_brl,
            },
            dd_logger="jev.system1",
        ) as span:
            logits, flags = self._compute_logits(req)
            probs = self._softmax(logits)
            predicted_class = max(CLASSES, key=lambda c: probs[c])
            confidence = float(probs[predicted_class])

            if predicted_class in ("HIGH_RISK_FRAUD", "PROMPT_INJECTION"):
                action_route: ActionRoute = "REJECT"
            elif predicted_class == "POLICY_CHECK_NEEDED" or confidence < self.approve_confidence_threshold:
                action_route = "HUMAN_REVIEW"
                if confidence < self.approve_confidence_threshold and "low_confidence_fallback" not in flags:
                    flags.append("low_confidence_fallback")
            else:
                action_route = "APPROVE"

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            span.set_attribute("jev.decision", action_route)
            span.set_attribute("jev.predicted_class", predicted_class)
            span.set_attribute("jev.confidence", confidence)
            span.set_attribute("output.value", f"{predicted_class}:{action_route}")

            return JEVDecisionResponse(
                request_id=req.request_id,
                predicted_class=predicted_class,
                action_route=action_route,
                confidence=confidence,
                probabilities=probs,
                latency_ms=elapsed_ms,
                policy_flags=flags,
            )
