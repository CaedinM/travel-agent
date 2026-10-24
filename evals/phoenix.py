"""Optional Phoenix tracing for evaluation runs; production tracing remains separate."""

from __future__ import annotations

import json
import os

from evals.contracts import EvaluationReport


def configure(project_name: str = "travel-agent-evals") -> None:
    """Configure Phoenix only for an explicit manual evaluation run."""
    try:
        from phoenix.otel import register
    except ImportError as exc:  # pragma: no cover - optional integration
        raise RuntimeError("Install arize-phoenix-otel to enable Phoenix tracing.") from exc
    # Pass endpoint explicitly: recent Phoenix OTEL releases otherwise default to
    # gRPC localhost:4317 even when a local HTTP collector URL is configured.
    collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/")
    if not collector_endpoint.endswith("/v1/traces"):
        collector_endpoint = f"{collector_endpoint}/v1/traces"
    register(
        project_name=os.getenv("PHOENIX_EVAL_PROJECT", project_name),
        endpoint=collector_endpoint,
        protocol="http/protobuf",
        auto_instrument=False,
        batch=True,
        api_key=os.getenv("PHOENIX_API_KEY") or None,
    )


def record_report(report: EvaluationReport, session_id: str) -> None:
    """Emit one trace per turn and nested tool spans, grouped by an opaque session id."""
    try:
        from opentelemetry import trace
        from phoenix.otel import using_metadata, using_session, using_tags
    except ImportError as exc:  # pragma: no cover - optional integration
        raise RuntimeError("Install Phoenix OpenTelemetry dependencies to record reports.") from exc
    tracer = trace.get_tracer("travel_agent.evals")
    with using_session(session_id), using_metadata({"trace_source": "evaluation"}), using_tags(["evaluation"]):
        for index, turn in enumerate(report.turns):
            with tracer.start_as_current_span("evaluation.turn") as span:
                span.set_attribute("openinference.span.kind", "AGENT")
                span.set_attribute("input.value", turn.user_message)
                span.set_attribute("output.value", turn.assistant_message)
                span.set_attribute("metadata.trace_source", "evaluation")
                span.set_attribute("metadata.trace_population", "evaluation")
                span.set_attribute("metadata.scenario_id", report.scenario_id)
                span.set_attribute("metadata.turn_index", index)
                for call in turn.tool_calls:
                    with tracer.start_as_current_span(f"tool.{call.name}") as tool_span:
                        tool_span.set_attribute("openinference.span.kind", "TOOL")
                        tool_span.set_attribute("tool.name", call.name)
                        tool_span.set_attribute(
                            "input.value",
                            json.dumps(
                                {"arguments": call.arguments, "state_before": call.state_before},
                                default=str,
                            ),
                        )
                        tool_span.set_attribute(
                            "output.value",
                            json.dumps(
                                {"result": call.result, "state_after": call.state_after},
                                default=str,
                            ),
                        )
                for result in (item for item in report.results if item.turn_index == index):
                    with tracer.start_as_current_span(f"evaluator.{result.criterion}") as evaluator_span:
                        evaluator_span.set_attribute("openinference.span.kind", "EVALUATOR")
                        evaluator_span.set_attribute("input.value", json.dumps(result.as_dict()))
                        evaluator_span.set_attribute("output.value", result.verdict)
                        evaluator_span.set_attribute("metadata.evaluation_source", result.source)
        session_results = [item.as_dict() for item in report.results if item.turn_index is None]
        with tracer.start_as_current_span("evaluation.session_results") as span:
            span.set_attribute("openinference.span.kind", "EVALUATOR")
            span.set_attribute("input.value", report.scenario_id)
            span.set_attribute("output.value", json.dumps(session_results))
