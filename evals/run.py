"""CLI support for manual, reproducible baseline and candidate comparisons."""

from __future__ import annotations

import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from evals.contracts import EvaluationReport, Scenario
from evals.state import validate_state


def load_scenarios(path: Path = Path("evals/scenarios/v1.json")) -> tuple[Scenario, ...]:
    raw = json.loads(path.read_text())
    scenarios = tuple(
        Scenario(
            **{**item, "permitted_assumptions": tuple(item["permitted_assumptions"]), "user_turns": tuple(item["user_turns"])}
        )
        for item in raw
    )
    for scenario in scenarios:
        validate_state(scenario.initial_state, requires_return_home=scenario.requires_return_home)
    return scenarios


def load_turn_cases(path: Path = Path("evals/scenarios/turns/v1.json")) -> tuple[Scenario, ...]:
    """Load turn-level fixtures whose custom state is validated before execution."""
    return load_scenarios(path)


def aggregate(reports: Iterable[EvaluationReport]) -> dict:
    all_results = [result for report in reports for result in report.results]
    by_criterion: dict[str, list[float]] = defaultdict(list)
    for result in all_results:
        if result.score is not None:
            by_criterion[result.criterion].append(result.score)
    return {
        "scenario_count": len({report.scenario_id for report in reports}),
        "criteria": {
            name: {"pass_rate": sum(scores) / len(scores), "count": len(scores)}
            for name, scores in sorted(by_criterion.items())
        },
        "critical_invariant_violations": sum(report.summary["critical_invariant_violations"] for report in reports),
        "accepted_flight_completion_rate": (
            sum(report.accepted_flights_completed for report in reports) / len(reports) if reports else None
        ),
    }


def main() -> None:
    """Validate and inspect the versioned synthetic dataset without provider calls."""
    parser = argparse.ArgumentParser(description="Travel-agent evaluation dataset tools")
    parser.add_argument("--list", action="store_true", help="print scenario ids and categories")
    args = parser.parse_args()
    scenarios = load_scenarios()
    if args.list:
        for scenario in scenarios:
            print(f"{scenario.id}\t{scenario.category}")
    else:
        print(json.dumps({"dataset": "travel-agent-scenarios-v1", "scenario_count": len(scenarios)}, indent=2))


if __name__ == "__main__":
    main()
