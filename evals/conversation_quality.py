"""Advisory Phoenix-backed LLM evaluation for conversation quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from time import perf_counter
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Protocol

from evals.judges import (
    DECISION_SUPPORT_RUBRIC,
    JUDGE_MODEL,
    RUBRIC_VERSION,
    SESSION_RUBRICS,
    TURN_RUBRICS,
    CalibrationMetrics,
    calibrate,
    phoenix_judges,
)


@dataclass(frozen=True)
class ToolEvidence:
    name: str
    input_value: str | None
    output_value: str | None


@dataclass(frozen=True)
class ConversationTurn:
    project_name: str
    trace_id: str
    span_id: str
    session_id: str
    turn_index: int
    started_at: str
    user_message: str
    assistant_message: str
    tools: tuple[ToolEvidence, ...]


@dataclass(frozen=True)
class Judgment:
    criterion: str
    verdict: str
    score: float | None
    explanation: str
    scope: str
    span_id: str
    identifier: str
    session_id: str
    error: str | None = None
    latency_ms: float | None = None
    token_usage: int | None = None
    cost_usd: float | None = None


class Judge(Protocol):
    def evaluate(self, eval_input: dict[str, Any]) -> list[Any]: ...


def _value(item: object, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _attributes(span: object) -> Mapping[str, Any]:
    attributes = _value(span, "attributes", {})
    return attributes if isinstance(attributes, Mapping) else {}


def _attribute(span: object, name: str, default: Any = None) -> Any:
    return _attributes(span).get(name, default)


def _span_id(span: object) -> str:
    context = _value(span, "context", {})
    return str(_value(context, "span_id", _value(span, "id", "")))


def _trace_id(trace: object, span: object) -> str:
    context = _value(span, "context", {})
    return str(_value(context, "trace_id", _value(trace, "trace_id", "")))


def _root_span(trace: object) -> object | None:
    spans = tuple(_value(trace, "spans", ()) or ())
    for span in spans:
        if _value(span, "name") in {"travel_agent.chat_turn", "evaluation.turn"}:
            return span
    return next((span for span in spans if _value(span, "parent_id") is None), None)


def _tool_evidence(trace: object) -> tuple[ToolEvidence, ...]:
    tools: list[ToolEvidence] = []
    for span in _value(trace, "spans", ()) or ():
        if _value(span, "span_kind") != "TOOL" and not str(_value(span, "name", "")).startswith("tool."):
            continue
        tools.append(
            ToolEvidence(
                name=str(_attribute(span, "tool.name", _value(span, "name", "tool"))),
                input_value=_attribute(span, "input.value"),
                output_value=_attribute(span, "output.value"),
            )
        )
    return tuple(tools)


def _include_trace(root_span: object) -> bool:
    source = _attribute(root_span, "metadata.trace_source")
    if source == "evaluation":
        return True
    return source == "runtime" and _attribute(root_span, "metadata.trace_population") == "manual"


def turns_from_traces(project_name: str, traces: Iterable[object]) -> tuple[ConversationTurn, ...]:
    """Extract only synthetic-evaluation and manually tagged root turn spans."""
    turns: list[ConversationTurn] = []
    for trace in traces:
        root_span = _root_span(trace)
        if root_span is None or not _include_trace(root_span):
            continue
        trace_id = _trace_id(trace, root_span)
        session_id = str(_attribute(root_span, "session.id", trace_id))
        turn_index = _attribute(root_span, "metadata.turn_index")
        turns.append(
            ConversationTurn(
                project_name=project_name,
                trace_id=trace_id,
                span_id=_span_id(root_span),
                session_id=session_id,
                turn_index=int(turn_index) if isinstance(turn_index, int) else 0,
                started_at=str(_value(root_span, "start_time", _value(trace, "start_time", ""))),
                user_message=str(_attribute(root_span, "input.value", "")),
                assistant_message=str(_attribute(root_span, "output.value", "")),
                tools=_tool_evidence(trace),
            )
        )
    return tuple(sorted(turns, key=lambda turn: (turn.session_id, turn.started_at, turn.turn_index)))


def _tool_payload(turn: ConversationTurn) -> list[dict[str, str | None]]:
    return [asdict(tool) for tool in turn.tools]


def evidence_packet(turns: Iterable[ConversationTurn], current_index: int) -> dict[str, Any]:
    """Build judge evidence without putting conversation content in annotation metadata."""
    transcript = []
    for turn in tuple(turns)[: current_index + 1]:
        transcript.append(
            {
                "turn": turn.turn_index,
                "user": turn.user_message,
                "assistant": turn.assistant_message,
                "tools": _tool_payload(turn),
            }
        )
    return {"current_turn": transcript[-1], "transcript": transcript}


def _decision_support_applies(turn: ConversationTurn) -> bool:
    tools = {tool.name for tool in turn.tools}
    if tools & {"choose_prefetched_flights", "search_and_select_flights", "get_saved_flights_info"}:
        return True
    message = turn.assistant_message.lower()
    return any(phrase in message for phrase in ("recommend", "option", "tradeoff", "assum"))


def _identifier(turn: ConversationTurn, criterion: str, scope: str, packet: Mapping[str, Any], model: str) -> str:
    digest = hashlib.sha256(json.dumps(packet, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"conversation-quality:{RUBRIC_VERSION}:{model}:{scope}:{criterion}:{turn.trace_id}:{turn.span_id}:{digest}"


def _score_value(score: object, name: str, default: Any = None) -> Any:
    return _value(score, name, default)


class ConversationQualityRunner:
    """Run advisory judges over already-completed Phoenix conversation turns."""

    def __init__(self, client: Any, *, model: str = JUDGE_MODEL, judges: Mapping[str, Judge] | None = None):
        self.client = client
        self.model = model
        self.judges = dict(judges or {judge.name: judge for judge in phoenix_judges(model)})

    def collect(self, project_names: Iterable[str], *, limit: int = 100) -> tuple[ConversationTurn, ...]:
        turns: list[ConversationTurn] = []
        for project_name in project_names:
            traces = self.client.traces.get_traces(
                project_identifier=project_name, include_spans=True, limit=limit
            )
            turns.extend(turns_from_traces(project_name, traces))
        return tuple(sorted(turns, key=lambda turn: (turn.session_id, turn.started_at, turn.turn_index)))

    def _annotate(self, turn: ConversationTurn, criterion: str, scope: str, packet: Mapping[str, Any]) -> Judgment:
        identifier = _identifier(turn, criterion, scope, packet, self.model)
        started = perf_counter()
        try:
            scores = self.judges[criterion].evaluate(
                {"input": json.dumps(packet["transcript"]), "context": json.dumps(packet["current_turn"])}
            )
            score = scores[0] if scores else None
            label = _score_value(score, "label")
            numeric_score = _score_value(score, "score")
            explanation = _score_value(score, "explanation", "")
            if label not in {"pass", "fail"} or numeric_score not in {0, 0.0, 1, 1.0}:
                raise ValueError("judge returned neither a binary pass/fail label nor a binary score")
            verdict = str(label)
            score_metadata = _score_value(score, "metadata", {}) or {}
            token_usage = score_metadata.get("token_usage", score_metadata.get("total_tokens"))
            cost_usd = score_metadata.get("cost_usd")
            result = Judgment(
                criterion,
                verdict,
                float(numeric_score),
                str(explanation),
                scope,
                turn.span_id,
                identifier,
                turn.session_id,
                latency_ms=round((perf_counter() - started) * 1000, 2),
                token_usage=token_usage if isinstance(token_usage, int) else None,
                cost_usd=float(cost_usd) if isinstance(cost_usd, (int, float)) else None,
            )
            metadata = {
                "rubric_version": RUBRIC_VERSION,
                "judge_model": self.model,
                "scope": scope,
                "judge_latency_ms": result.latency_ms,
                "judge_run_id": identifier,
            }
            if result.token_usage is not None:
                metadata["judge_token_usage"] = result.token_usage
            if result.cost_usd is not None:
                metadata["judge_cost_usd"] = result.cost_usd
            self.client.spans.add_span_annotation(
                span_id=turn.span_id,
                annotation_name=criterion,
                annotator_kind="LLM",
                label=result.verdict,
                score=result.score,
                explanation=result.explanation,
                metadata=metadata,
                identifier=identifier,
                sync=True,
            )
            return result
        except Exception as exc:
            error = Judgment(
                criterion,
                "error",
                None,
                "",
                scope,
                turn.span_id,
                identifier,
                turn.session_id,
                error=str(exc),
                latency_ms=round((perf_counter() - started) * 1000, 2),
            )
            self.client.spans.add_span_annotation(
                span_id=turn.span_id,
                annotation_name=criterion,
                annotator_kind="LLM",
                label="error",
                explanation=str(exc),
                metadata={
                    "rubric_version": RUBRIC_VERSION,
                    "judge_model": self.model,
                    "scope": scope,
                    "judge_latency_ms": error.latency_ms,
                    "judge_run_id": identifier,
                },
                identifier=identifier,
                sync=True,
            )
            return error

    def run(self, turns: Iterable[ConversationTurn]) -> tuple[Judgment, ...]:
        """Judge each eligible turn and one repetition result per completed session."""
        sessions: dict[str, list[ConversationTurn]] = defaultdict(list)
        for turn in turns:
            sessions[turn.session_id].append(turn)
        results: list[Judgment] = []
        for session_turns in sessions.values():
            session_turns.sort(key=lambda turn: (turn.started_at, turn.turn_index))
            for index, turn in enumerate(session_turns):
                packet = evidence_packet(session_turns, index)
                for criterion in TURN_RUBRICS:
                    results.append(self._annotate(turn, criterion, "turn", packet))
                if _decision_support_applies(turn):
                    results.append(self._annotate(turn, "decision_support", "turn", packet))
            final_turn = session_turns[-1]
            packet = evidence_packet(session_turns, len(session_turns) - 1)
            results.append(self._annotate(final_turn, "repetition", "session", packet))
        return tuple(results)


def quality_summary(judgments: Iterable[Judgment]) -> dict[str, Any]:
    values = tuple(judgments)
    grouped: dict[str, list[Judgment]] = defaultdict(list)
    for judgment in values:
        grouped[judgment.criterion].append(judgment)
    return {
        "sessions": len({item.session_id for item in values}),
        "judgments": len(values),
        "judge_latency_ms": round(sum(item.latency_ms or 0 for item in values), 2),
        "judge_token_usage": sum(item.token_usage or 0 for item in values),
        "judge_cost_usd": sum(item.cost_usd or 0 for item in values),
        "criteria": {
            criterion: {
                "count": len(items),
                "pass_rate": sum(item.score or 0 for item in items if item.score is not None)
                / sum(item.score is not None for item in items)
                if any(item.score is not None for item in items)
                else None,
                "errors": sum(item.verdict == "error" for item in items),
            }
            for criterion, items in sorted(grouped.items())
        },
    }


def _annotation_value(annotation: object, name: str, default: Any = None) -> Any:
    value = _value(annotation, name, None)
    if value is not None:
        return value
    result = _value(annotation, "result", {})
    return _value(result, name, default)


def calibration_metrics_from_annotations(annotations: Iterable[object]) -> dict[str, CalibrationMetrics]:
    """Compare matching human and LLM Phoenix labels by rubric and span."""
    labels: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for annotation in annotations:
        criterion = str(_annotation_value(annotation, "annotation_name", _annotation_value(annotation, "name", "")))
        if criterion not in TURN_RUBRICS | DECISION_SUPPORT_RUBRIC | SESSION_RUBRICS:
            continue
        span_id = str(_annotation_value(annotation, "span_id", ""))
        kind = str(_annotation_value(annotation, "annotator_kind", "")).lower()
        label = _annotation_value(annotation, "label")
        if kind in {"human", "llm"} and label in {"pass", "fail"}:
            labels[(criterion, span_id)][kind] = label == "pass"
    pairs: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for (criterion, _), values in labels.items():
        if "human" in values and "llm" in values:
            pairs[criterion].append((values["human"], values["llm"]))
    return {
        criterion: calibrate(
            [human for human, _ in criterion_pairs], [judge for _, judge in criterion_pairs]
        )
        for criterion, criterion_pairs in pairs.items()
    }


def calibration_disagreements(annotations: Iterable[object]) -> list[dict[str, str]]:
    """Return evidence locations where a human label and judge label differ."""
    labels: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for annotation in annotations:
        criterion = str(_annotation_value(annotation, "annotation_name", _annotation_value(annotation, "name", "")))
        span_id = str(_annotation_value(annotation, "span_id", ""))
        kind = str(_annotation_value(annotation, "annotator_kind", "")).lower()
        label = _annotation_value(annotation, "label")
        if criterion in TURN_RUBRICS | DECISION_SUPPORT_RUBRIC | SESSION_RUBRICS and kind in {"human", "llm"} and label in {"pass", "fail"}:
            labels[(criterion, span_id)][kind] = label
    return [
        {"criterion": criterion, "span_id": span_id, "human": values["human"], "llm": values["llm"]}
        for (criterion, span_id), values in labels.items()
        if values.get("human") != values.get("llm") and {"human", "llm"} <= values.keys()
    ]


def phoenix_annotations(client: Any, turns: Iterable[ConversationTurn]) -> list[object]:
    """Fetch annotations for the evaluated spans without retrieving unrelated traces."""
    by_project: dict[str, list[str]] = defaultdict(list)
    for turn in turns:
        by_project[turn.project_name].append(turn.span_id)
    annotations: list[object] = []
    for project_name, span_ids in by_project.items():
        annotations.extend(
            client.spans.get_span_annotations(
                span_ids=span_ids,
                project_identifier=project_name,
                limit=len(span_ids) * 10,
            )
        )
    return annotations


def phoenix_calibration_metrics(client: Any, turns: Iterable[ConversationTurn]) -> dict[str, CalibrationMetrics]:
    """Fetch human and LLM annotations for collected turns and calculate agreement."""
    return calibration_metrics_from_annotations(phoenix_annotations(client, turns))


def _client() -> Any:
    from phoenix.client import Client

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").removesuffix("/v1/traces")
    return Client(base_url=endpoint, api_key=os.getenv("PHOENIX_API_KEY") or None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run advisory Phoenix conversation-quality judges")
    parser.add_argument("--project", action="append", dest="projects")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--calibrate", action="store_true", help="compare Phoenix human and LLM annotations")
    args = parser.parse_args()
    projects = args.projects or list(
        dict.fromkeys(
            [
                os.getenv("PHOENIX_EVAL_PROJECT", "travel-agent-evals"),
                os.getenv("PHOENIX_RUNTIME_PROJECT", "travel-agent-runtime"),
            ]
        )
    )
    runner = ConversationQualityRunner(_client())
    turns = runner.collect(projects, limit=args.limit)
    if args.calibrate:
        annotations = phoenix_annotations(runner.client, turns)
        metrics = calibration_metrics_from_annotations(annotations)
        print(json.dumps({"projects": projects, "calibration": {name: asdict(value) | {"ready_for_release_signal": value.ready_for_release_signal} for name, value in metrics.items()}, "disagreements": calibration_disagreements(annotations)}, indent=2))
        return
    results = runner.run(turns)
    print(json.dumps({"projects": projects, "results": quality_summary(results)}, indent=2))


if __name__ == "__main__":
    main()
