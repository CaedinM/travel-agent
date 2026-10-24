"""Narrow advisory LLM rubrics and calibration metrics for conversational quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


RUBRIC_VERSION = "v2"
JUDGE_MODEL = "gpt-5-mini"
TURN_RUBRICS = {
    "grounded_communication": "Pass only when claims about flights, tools, failures, and accommodation scope are supported by the supplied evidence.",
    "context_continuity": "Pass only when the agent remembers supplied facts and corrections and does not ask for known information.",
    "progress_and_recovery": "Pass only when the response advances the goal or clearly explains a real blocker and a useful next step.",
    "conversational_naturalness": "Pass only when the response is direct, concise, and asks at most one focused question.",
}
DECISION_SUPPORT_RUBRIC = {
    "decision_support": "Pass only when relevant tradeoffs and any normalized preference assumptions are clearly disclosed."
}
SESSION_RUBRICS = {
    "repetition": "Pass only when there is no redundant question, recommendation, or tool retry without new evidence."
}
RUBRICS = TURN_RUBRICS | DECISION_SUPPORT_RUBRIC | SESSION_RUBRICS


def phoenix_judges(model: str):
    """Build advisory Phoenix classifiers lazily so deterministic tests need no API key."""
    try:
        from phoenix.evals import LLM, ClassificationEvaluator
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install arize-phoenix-evals to run LLM judges.") from exc
    llm = LLM(provider="openai", model=model)
    return [
        ClassificationEvaluator(
            name=name,
            llm=llm,
            choices={"fail": 0.0, "pass": 1.0},
            prompt_template=(
                f"You are grading travel-agent conversation quality. {instruction}\n"
                "Return pass or fail and a concise explanation citing specific turn or tool evidence. "
                "Do not infer facts missing from the evidence.\n"
                "<conversation>{{input}}</conversation>\n<evidence>{{context}}</evidence>"
            ),
        )
        for name, instruction in RUBRICS.items()
    ]


@dataclass(frozen=True)
class CalibrationMetrics:
    accuracy: float
    failure_recall: float
    specificity: float
    total: int

    @property
    def ready_for_release_signal(self) -> bool:
        return self.total >= 100 and self.failure_recall >= 0.8 and self.specificity >= 0.8


def calibrate(human_labels: Iterable[bool], judge_labels: Iterable[bool]) -> CalibrationMetrics:
    """Calculate pass/fail judge agreement without adding a heavy ML dependency."""
    pairs = list(zip(human_labels, judge_labels, strict=True))
    if not pairs:
        raise ValueError("Calibration requires at least one human-labelled example.")
    # False is failure: recall measures a judge's ability to find human failures.
    true_positive = sum(not human and not judge for human, judge in pairs)
    false_negative = sum(not human and judge for human, judge in pairs)
    true_negative = sum(human and judge for human, judge in pairs)
    false_positive = sum(human and not judge for human, judge in pairs)
    return CalibrationMetrics(
        accuracy=(true_positive + true_negative) / len(pairs),
        failure_recall=true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0,
        specificity=true_negative / (true_negative + false_positive) if true_negative + false_positive else 1.0,
        total=len(pairs),
    )
