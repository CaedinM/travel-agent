"""Optional Phoenix tracing for runtime conversations, kept separate from eval traces."""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Iterator


_configured = False


def _enabled() -> bool:
    return os.getenv("PHOENIX_TRACING_ENABLED", "false").lower() in {"1", "true", "yes"}


def _collector_endpoint() -> str:
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/")
    return endpoint if endpoint.endswith("/v1/traces") else f"{endpoint}/v1/traces"


def _trace_population() -> str:
    """Return the configured trace population without recording user data as metadata."""
    population = os.getenv("PHOENIX_TRACE_POPULATION", "manual").lower()
    return population if population in {"manual", "production"} else "manual"


def _runtime_trace_metadata(turn_index: int | None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "trace_source": "runtime",
        "trace_population": _trace_population(),
        "agent_model_version": os.getenv("PHOENIX_AGENT_MODEL_VERSION", "gpt-5"),
        "agent_prompt_version": os.getenv("PHOENIX_AGENT_PROMPT_VERSION", "v1"),
    }
    if turn_index is not None:
        metadata["turn_index"] = turn_index
    return metadata


def configure_runtime_tracing() -> bool:
    """Enable LangChain tracing only when explicitly configured for this process."""
    global _configured
    if not _enabled() or _configured:
        return _configured
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError as exc:
        raise RuntimeError(
            "Phoenix tracing is enabled but its optional tracing dependencies are unavailable."
        ) from exc

    capture_content = os.getenv("PHOENIX_CAPTURE_CONTENT", "false").lower() in {"1", "true", "yes"}
    if not capture_content:
        for setting in (
            "OPENINFERENCE_HIDE_INPUTS",
            "OPENINFERENCE_HIDE_OUTPUTS",
            "OPENINFERENCE_HIDE_INPUT_MESSAGES",
            "OPENINFERENCE_HIDE_OUTPUT_MESSAGES",
            "OPENINFERENCE_HIDE_LLM_TOOLS",
        ):
            os.environ.setdefault(setting, "true")
    provider = register(
        project_name=os.getenv("PHOENIX_RUNTIME_PROJECT", "travel-agent-runtime"),
        endpoint=_collector_endpoint(),
        protocol="http/protobuf",
        batch=True,
        auto_instrument=False,
        api_key=os.getenv("PHOENIX_API_KEY") or None,
    )
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _configured = True
    return True


@contextmanager
def runtime_chat_turn(
    thread_id: str, message: str, *, turn_index: int | None = None
) -> Iterator[object | None]:
    """Group one browser turn and every child LangChain/tool span by opaque thread id."""
    if not _enabled():
        with nullcontext(None) as span:
            yield span
        return

    from opentelemetry import trace
    from phoenix.otel import using_metadata, using_session, using_tags

    with (
        using_session(thread_id),
        using_metadata(_runtime_trace_metadata(turn_index)),
        using_tags(["runtime"]),
        trace.get_tracer("travel_agent.runtime").start_as_current_span("travel_agent.chat_turn") as span,
    ):
        span.set_attribute("openinference.span.kind", "AGENT")
        for name, value in _runtime_trace_metadata(turn_index).items():
            span.set_attribute(f"metadata.{name}", value)
        span.set_attribute("input.value", message)
        yield span
