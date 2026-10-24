"""Portable contracts shared by evaluation harnesses, scorers, and reporters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Verdict = Literal["pass", "fail", "not_applicable", "error"]
Source = Literal["code", "llm", "human"]


@dataclass(frozen=True)
class ToolCall:
    """One tool call, with the exact state visible when the tool was chosen."""

    name: str
    arguments: dict[str, Any]
    state_before: dict[str, Any]
    state_after: dict[str, Any] = field(default_factory=dict)
    turn_index: int = 0
    result: str | None = None


@dataclass(frozen=True)
class Turn:
    """A user/assistant exchange and the tools chosen while handling it."""

    user_message: str
    assistant_message: str
    tool_calls: tuple[ToolCall, ...] = ()
    latency_ms: float | None = None
    token_usage: int | None = None


@dataclass(frozen=True)
class Scenario:
    """A deterministic evaluation case; expectations describe behavior, not wording."""

    id: str
    category: str
    initial_state: dict[str, Any]
    traveller_facts: dict[str, Any]
    provider_fixtures: dict[str, Any]
    permitted_assumptions: tuple[str, ...]
    expected: dict[str, Any]
    user_turns: tuple[str, ...]
    turn_limit: int = 20
    requires_return_home: bool = True


@dataclass(frozen=True)
class EvaluationResult:
    """A single auditable evaluation result suitable for Phoenix annotations."""

    criterion: str
    version: str
    verdict: Verdict
    source: Source
    explanation: str
    score: float | None = None
    turn_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    scenario_id: str
    results: tuple[EvaluationResult, ...]
    turns: tuple[Turn, ...]
    completion_reason: str

    @property
    def has_critical_failure(self) -> bool:
        return any(
            result.verdict == "fail" and result.criterion in {
                "flight_selection_prerequisites",
                "itinerary_validity",
                "confirmation_integrity",
            }
            for result in self.results
        )

    @property
    def accepted_flights_completed(self) -> bool:
        return any(
            result.criterion == "accepted_flight_completion" and result.verdict == "pass"
            for result in self.results
        )

    @property
    def summary(self) -> dict[str, Any]:
        scored = [result for result in self.results if result.score is not None]
        if self.accepted_flights_completed and not self.has_critical_failure:
            outcome = "completed"
        elif self.completion_reason.startswith("blocked"):
            outcome = "blocked"
        else:
            outcome = "incomplete"
        return {
            "scenario_id": self.scenario_id,
            "completion_reason": self.completion_reason,
            "outcome": outcome,
            "critical_invariant_violations": sum(
                1 for result in self.results
                if result.verdict == "fail" and result.criterion in {
                    "flight_selection_prerequisites", "itinerary_validity", "confirmation_integrity"
                }
            ),
            "accepted_flights_completed": self.accepted_flights_completed,
            "pass_rate": (sum(result.score for result in scored) / len(scored)) if scored else None,
            "turns_to_completion": len(self.turns),
            "redundant_tool_calls": sum(
                1 for result in self.results if result.criterion == "redundant_tool_call" and result.verdict == "fail"
            ),
            "latency_ms": sum(turn.latency_ms or 0 for turn in self.turns),
            "token_usage": sum(turn.token_usage or 0 for turn in self.turns),
        }
