"""
FastAPI Service exposing JEV System-1 Classifier & Telemetry SLA Metrics for Load Testing (k6).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from fastapi import FastAPI

from src.jev_engine import JEVClassifier, JEVDecisionRequest, JEVDecisionResponse
from src.jev_qa_metrics import verify_latency_sla
from src.observability_phoenix_datadog import DEFAULT_TELEMETRY

app = FastAPI(
    title="JEV System-1 & AI QA Telemetry API",
    version="1.0.0",
    description="Sub-500ms typed decision endpoint instrumented with OpenTelemetry & Datadog logs.",
)

_classifier = JEVClassifier(telemetry=DEFAULT_TELEMETRY)
_latencies_ms: List[float] = []
_confidences: List[float] = []
_routes: Counter = Counter()


@app.get("/v1/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": "jev-s1-v1"}


@app.post("/v1/jev/decide", response_model=JEVDecisionResponse)
def decide(req: JEVDecisionRequest) -> JEVDecisionResponse:
    res = _classifier.classify(req)
    _latencies_ms.append(res.latency_ms)
    _confidences.append(res.confidence)
    _routes[res.action_route] += 1
    return res


@app.get("/v1/telemetry/metrics")
def telemetry_metrics() -> Dict[str, Any]:
    lat_stats = verify_latency_sla(_latencies_ms) if _latencies_ms else {"count": 0, "sla_passed": True}
    mean_conf = round(sum(_confidences) / len(_confidences), 4) if _confidences else 0.0
    return {
        "service": DEFAULT_TELEMETRY.service_name,
        "requests_total": len(_latencies_ms),
        "mean_confidence": mean_conf,
        "route_distribution": dict(_routes),
        "latency_sla": lat_stats,
        "otel_spans_captured": len(DEFAULT_TELEMETRY.span_exporter.spans),
        "datadog_logs_captured": len(DEFAULT_TELEMETRY.datadog_logs),
    }
