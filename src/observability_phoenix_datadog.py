"""
OpenTelemetry (Arize Phoenix / OpenInference-compatible) tracing and Datadog structured log/APM exporter.

Captures in-process spans with OpenInference attributes (`openinference.span.kind`, `llm.model_name`,
`input.value`, `output.value`, `jev.*`, `eval.*`) and Datadog-correlated JSON logs (`dd.trace_id`,
`dd.span_id`, `service`, `env`, `latency_ms`) without requiring an external collector daemon in CI.
If `PHOENIX_COLLECTOR_ENDPOINT` is configured, spans can additionally be forwarded via OTLP HTTP.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span, Status, StatusCode


class InMemoryJsonSpanExporter(SpanExporter):
    """Stores exported OpenTelemetry spans as OpenInference/Phoenix-friendly dicts."""

    def __init__(self) -> None:
        self.spans: List[Dict[str, Any]] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            ctx = span.get_span_context()
            start_ns = span.start_time or 0
            end_ns = span.end_time or start_ns
            duration_ms = round((end_ns - start_ns) / 1_000_000.0, 4)
            self.spans.append(
                {
                    "name": span.name,
                    "trace_id": format(ctx.trace_id, "032x"),
                    "span_id": format(ctx.span_id, "016x"),
                    "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
                    "status": span.status.status_code.name,
                    "duration_ms": duration_ms,
                    "start_time_utc": datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat(),
                    "attributes": dict(span.attributes or {}),
                }
            )
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


class TelemetryManager:
    """Isolated OpenTelemetry + Datadog log manager suitable for production and deterministic pytest CI."""

    def __init__(self, service_name: str = "jev-ai-qa-service", env: str = "production") -> None:
        self.service_name = service_name
        self.env = env
        self.span_exporter = InMemoryJsonSpanExporter()
        resource = Resource.create({"service.name": service_name, "deployment.environment": env})
        self.provider = TracerProvider(resource=resource)
        self.provider.add_span_processor(SimpleSpanProcessor(self.span_exporter))

        # Optional live Arize Phoenix OTLP forwarding when explicitly configured
        otlp_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore

                self.provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
            except Exception:
                pass

        self.tracer = self.provider.get_tracer("jev.qa.observability", "1.0.0")
        self.datadog_logs: List[Dict[str, Any]] = []

    @contextmanager
    def start_span(
        self,
        name: str,
        span_kind: str = "CHAIN",
        attributes: Optional[Dict[str, Any]] = None,
        dd_logger: str = "jev.pipeline",
    ) -> Iterator[Span]:
        merged: Dict[str, Any] = {"openinference.span.kind": span_kind}
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    merged[k] = v if isinstance(v, (bool, int, float, str)) else json.dumps(v, ensure_ascii=False)

        with self.tracer.start_as_current_span(name, attributes=merged) as span:
            t0 = datetime.now(timezone.utc)
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
                level = "INFO"
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                level = "ERROR"
                raise
            finally:
                dt_ms = round((datetime.now(timezone.utc) - t0).total_seconds() * 1000.0, 4)
                span.set_attribute("latency_ms", dt_ms)
                ctx = span.get_span_context()
                dd_record: Dict[str, Any] = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": level,
                    "service": self.service_name,
                    "env": self.env,
                    "logger.name": dd_logger,
                    "dd.trace_id": str(ctx.trace_id & ((1 << 64) - 1)),
                    "dd.span_id": str(ctx.span_id),
                    "dd.trace_id_hex": format(ctx.trace_id, "032x"),
                    "span_name": name,
                    "openinference.span.kind": span_kind,
                    "latency_ms": dt_ms,
                }
                for key, val in (getattr(span, "attributes", None) or {}).items():
                    if key.startswith(("jev.", "eval.", "llm.")):
                        dd_record[key] = val
                self.datadog_logs.append(dd_record)

    def flush_to_disk(
        self,
        traces_path: Path,
        datadog_logs_path: Path,
    ) -> Dict[str, int]:
        traces_path.parent.mkdir(parents=True, exist_ok=True)
        datadog_logs_path.parent.mkdir(parents=True, exist_ok=True)

        traces_path.write_text(json.dumps(self.span_exporter.spans, indent=2, ensure_ascii=False), encoding="utf-8")
        with datadog_logs_path.open("w", encoding="utf-8") as fh:
            for row in self.datadog_logs:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        return {
            "spans_exported": len(self.span_exporter.spans),
            "datadog_logs_exported": len(self.datadog_logs),
        }

    def clear(self) -> None:
        self.span_exporter.spans.clear()
        self.datadog_logs.clear()


DEFAULT_TELEMETRY = TelemetryManager()
