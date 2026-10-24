from types import SimpleNamespace

from evals.conversation_quality import (
    ConversationQualityRunner,
    ConversationTurn,
    ToolEvidence,
    calibration_disagreements,
    calibration_metrics_from_annotations,
    evidence_packet,
    quality_summary,
    turns_from_traces,
)


def span(
    span_id, name, *, parent_id=None, attributes=None, span_kind="AGENT", start_time="2027-01-01T00:00:00Z"
):
    return {
        "name": name,
        "parent_id": parent_id,
        "context": {"span_id": span_id, "trace_id": f"trace-{span_id}"},
        "attributes": attributes or {},
        "span_kind": span_kind,
        "start_time": start_time,
    }


def trace(root, *children):
    return {"trace_id": root["context"]["trace_id"], "spans": [root, *children]}


def manual_trace(span_id="root", *, time="2027-01-01T00:00:00Z"):
    root = span(
        span_id,
        "travel_agent.chat_turn",
        start_time=time,
        attributes={
            "metadata.trace_source": "runtime",
            "metadata.trace_population": "manual",
            "metadata.turn_index": 1,
            "session.id": "opaque-session",
            "input.value": "Can I take the Paris flight?",
            "output.value": "Yes. It leaves at 10:00.",
        },
    )
    tool = span(
        f"tool-{span_id}",
        "tool.get_saved_flights_info",
        parent_id=span_id,
        span_kind="TOOL",
        attributes={"tool.name": "get_saved_flights_info", "input.value": "{}", "output.value": '{"offer":"saved"}'},
    )
    return trace(root, tool)


def test_turn_extraction_orders_sessions_and_keeps_tool_state_evidence():
    later = manual_trace("later", time="2027-01-02T00:00:00Z")
    earlier = manual_trace("earlier", time="2027-01-01T00:00:00Z")

    turns = turns_from_traces("runtime", [later, earlier])

    assert [turn.span_id for turn in turns] == ["earlier", "later"]
    assert turns[0].tools == (ToolEvidence("get_saved_flights_info", "{}", '{"offer":"saved"}'),)
    packet = evidence_packet(turns, 1)
    assert [item["user"] for item in packet["transcript"]] == [
        "Can I take the Paris flight?", "Can I take the Paris flight?"
    ]


def test_turn_extraction_excludes_production_traces():
    production = manual_trace()
    production["spans"][0]["attributes"]["metadata.trace_population"] = "production"
    evaluation = manual_trace("eval")
    evaluation["spans"][0]["attributes"].update(
        {"metadata.trace_source": "evaluation", "metadata.trace_population": "production"}
    )

    turns = turns_from_traces("project", [production, evaluation])

    assert [turn.span_id for turn in turns] == ["eval"]


class PassingJudge:
    def evaluate(self, value):
        assert "Can I take the Paris flight?" in value["input"]
        return [SimpleNamespace(label="pass", score=1.0, explanation="Turn 1 uses saved-flight tool evidence.")]


class InvalidJudge:
    def evaluate(self, value):
        return [SimpleNamespace(label="unclear", score=0.5, explanation="")]


class FakeSpans:
    def __init__(self):
        self.annotations = []

    def add_span_annotation(self, **kwargs):
        self.annotations.append(kwargs)


class FakeClient:
    def __init__(self):
        self.spans = FakeSpans()


def test_runner_records_idempotent_binary_annotations_and_session_repetition():
    client = FakeClient()
    judges = {name: PassingJudge() for name in (
        "grounded_communication", "context_continuity", "progress_and_recovery", "conversational_naturalness", "repetition", "decision_support"
    )}
    runner = ConversationQualityRunner(client, judges=judges)
    turn = turns_from_traces("project", [manual_trace()])

    results = runner.run(turn)

    assert {result.criterion for result in results} == {
        "grounded_communication", "context_continuity", "progress_and_recovery", "conversational_naturalness", "decision_support", "repetition"
    }
    assert all(result.verdict == "pass" for result in results)
    assert len({annotation["identifier"] for annotation in client.spans.annotations}) == len(client.spans.annotations)
    assert all(annotation["metadata"].keys() <= {"rubric_version", "judge_model", "scope", "judge_latency_ms", "judge_token_usage", "judge_cost_usd", "judge_run_id"} for annotation in client.spans.annotations)
    assert quality_summary(results)["criteria"]["repetition"] == {"count": 1, "pass_rate": 1.0, "errors": 0}


def test_runner_records_judge_errors_without_a_score():
    client = FakeClient()
    judges = {name: InvalidJudge() for name in (
        "grounded_communication", "context_continuity", "progress_and_recovery", "conversational_naturalness", "repetition", "decision_support"
    )}
    results = ConversationQualityRunner(client, judges=judges).run(turns_from_traces("project", [manual_trace()]))

    assert all(result.verdict == "error" and result.score is None for result in results)
    assert all(annotation["label"] == "error" for annotation in client.spans.annotations)


def test_calibration_matches_human_and_llm_annotations_by_span_and_rubric():
    annotations = [
            {"span_id": "one", "annotation_name": "grounded_communication", "annotator_kind": "HUMAN", "label": "fail"},
            {"span_id": "one", "annotation_name": "grounded_communication", "annotator_kind": "LLM", "label": "fail"},
            {"span_id": "two", "annotation_name": "grounded_communication", "annotator_kind": "HUMAN", "label": "pass"},
            {"span_id": "two", "annotation_name": "grounded_communication", "annotator_kind": "LLM", "label": "fail"},
        ]
    metrics = calibration_metrics_from_annotations(annotations)

    assert metrics["grounded_communication"].accuracy == 0.5
    assert metrics["grounded_communication"].total == 2
    assert calibration_disagreements(annotations) == [
        {"criterion": "grounded_communication", "span_id": "two", "human": "pass", "llm": "fail"}
    ]
