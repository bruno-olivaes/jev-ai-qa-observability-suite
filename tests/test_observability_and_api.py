import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api import app
from src.jev_engine import JEVClassifier, JEVDecisionRequest
from src.observability_phoenix_datadog import TelemetryManager


def test_phoenix_otel_and_datadog_log_export(tmp_path: Path):
    tel = TelemetryManager(service_name="jev-test-svc", env="ci")
    clf = JEVClassifier(telemetry=tel)
    clf.classify(JEVDecisionRequest(request_id="obs-1", text="Standard invoice receipt #99", amount_brl=50.0, account_age_days=200))

    traces_file = tmp_path / "traces.json"
    dd_file = tmp_path / "dd.jsonl"
    counts = tel.flush_to_disk(traces_file, dd_file)
    assert counts["spans_exported"] == 1
    assert counts["datadog_logs_exported"] == 1

    spans = json.loads(traces_file.read_text(encoding="utf-8"))
    assert spans[0]["attributes"]["openinference.span.kind"] == "GUARDRAIL"
    assert spans[0]["attributes"]["jev.decision"] == "APPROVE"

    dd_row = json.loads(dd_file.read_text(encoding="utf-8").strip())
    assert dd_row["service"] == "jev-test-svc"
    assert "dd.trace_id" in dd_row
    assert "dd.span_id" in dd_row
    assert dd_row["latency_ms"] < 500.0


def test_fastapi_jev_endpoints_and_strict_422():
    client = TestClient(app)
    r_health = client.get("/v1/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_ok = client.post(
        "/v1/jev/decide",
        json={
            "request_id": "api-1",
            "text": "Customer asking for manual refund review #10",
            "amount_brl": 300.0,
            "account_age_days": 90,
            "channel": "api",
        },
    )
    assert r_ok.status_code == 200
    body = r_ok.json()
    assert body["action_route"] == "HUMAN_REVIEW"
    assert body["latency_ms"] < 500.0

    r_extra = client.post(
        "/v1/jev/decide",
        json={
            "request_id": "api-bad",
            "text": "Test unknown schema key",
            "rogue_field": 123,
        },
    )
    assert r_extra.status_code == 422

    r_metrics = client.get("/v1/telemetry/metrics")
    assert r_metrics.status_code == 200
    assert r_metrics.json()["requests_total"] >= 1
