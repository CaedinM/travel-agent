"""Deterministic workflow and state evaluators for recorded agent trajectories."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

from evals.contracts import EvaluationReport, EvaluationResult, Scenario, ToolCall, Turn


EVALUATOR_VERSION = "v1"


def _result(criterion: str, passed: bool, explanation: str, *, turn: int | None = None) -> EvaluationResult:
    return EvaluationResult(
        criterion=criterion,
        version=EVALUATOR_VERSION,
        verdict="pass" if passed else "fail",
        source="code",
        score=1.0 if passed else 0.0,
        explanation=explanation,
        turn_index=turn,
    )


def _value(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)


def _has_research_prerequisites(state: Mapping[str, Any]) -> bool:
    exact_dates = all(state.get(field) for field in ("origin", "departure_date", "return_date"))
    flexible_dates = all(state.get(field) for field in ("origin", "season", "trip_length"))
    return exact_dates or flexible_dates


def _has_flight_prerequisites(state: Mapping[str, Any]) -> bool:
    legs = state.get("legs") or []
    return bool(legs) and isinstance(state.get("travelers"), int) and state["travelers"] > 0


def _has_offer_for_every_leg(state: Mapping[str, Any]) -> bool:
    legs = state.get("legs") or []
    return bool(legs) and all(_value(leg, "offer") for leg in legs)


def _validate_legs(legs: Iterable[Any], origin: str | None, requires_return_home: bool) -> str | None:
    values = list(legs)
    if not values:
        return "itinerary is empty"
    prior_date: date | None = None
    prior_destination: str | None = None
    for index, leg in enumerate(values):
        leg_origin = _value(leg, "origin")
        destination = _value(leg, "destination")
        departure_date = _value(leg, "departure_date")
        if not isinstance(leg_origin, str) or len(leg_origin) != 3 or not leg_origin.isalpha():
            return f"leg {index} has an invalid origin"
        if not isinstance(destination, str) or len(destination) != 3 or not destination.isalpha():
            return f"leg {index} has an invalid destination"
        if index and leg_origin.upper() != prior_destination:
            return f"leg {index} does not start where leg {index - 1} ends"
        try:
            parsed_date = date.fromisoformat(departure_date)
        except (TypeError, ValueError):
            return f"leg {index} has no concrete ISO departure date"
        if parsed_date <= date.today():
            return f"leg {index} is not in the future"
        if prior_date and parsed_date <= prior_date:
            return f"leg {index} is not after leg {index - 1}"
        prior_date, prior_destination = parsed_date, destination.upper()
    if requires_return_home and origin and prior_destination != origin.upper():
        return "itinerary does not end at the traveller's origin"
    return None


def _tool_results(scenario: Scenario, calls: Iterable[ToolCall]) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    prior_signatures: set[tuple[str, str]] = set()
    for call in calls:
        signature = (call.name, repr(sorted(call.arguments.items())))
        if signature in prior_signatures:
            results.append(_result("redundant_tool_call", False, f"{call.name} repeated with unchanged arguments", turn=call.turn_index))
        prior_signatures.add(signature)

        if call.name == "web_search":
            allowed = _has_research_prerequisites(call.state_before)
            results.append(_result("research_prerequisites", allowed, "research has origin and a usable date plan" if allowed else "research ran before origin and dates/timeframe were known", turn=call.turn_index))
        elif call.name in {"choose_prefetched_flights", "search_and_select_flights"}:
            allowed = _has_flight_prerequisites(call.state_before)
            results.append(_result("flight_selection_prerequisites", allowed, "flight selection has itinerary and traveller count" if allowed else "flight selection ran before itinerary or traveller count was saved", turn=call.turn_index))
        elif call.name == "set_trip_legs":
            legs = call.arguments.get("legs", [])
            issue = _validate_legs(legs, call.state_before.get("origin"), scenario.requires_return_home)
            results.append(_result("itinerary_validity", issue is None, "itinerary is complete, connected, chronological, and valid" if issue is None else issue, turn=call.turn_index))
        elif call.name == "confirm_flights":
            offers_exist = _has_offer_for_every_leg(call.state_before)
            explicit_acceptance = bool(call.arguments.get("explicit_acceptance"))
            passed = offers_exist and explicit_acceptance
            explanation = "every leg had an offer and the scenario marked explicit acceptance" if passed else "confirmation requires offers for every leg and scenario-marked explicit acceptance"
            results.append(_result("confirmation_integrity", passed, explanation, turn=call.turn_index))
        elif call.name == "get_saved_flights_info":
            offers_exist = _has_offer_for_every_leg(call.state_before)
            results.append(_result("saved_offer_query_eligibility", offers_exist, "saved offer query had selected offers" if offers_exist else "saved offer query ran without selected offers", turn=call.turn_index))
    return results


def evaluate_session(
    scenario: Scenario,
    turns: Iterable[Turn],
    *,
    completion_reason: str,
    final_state: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    """Score concrete invariants without prescribing an exact valid tool sequence."""
    recorded_turns = tuple(turns)
    calls = tuple(call for turn in recorded_turns for call in turn.tool_calls)
    results = _tool_results(scenario, calls)
    final_state = dict(final_state or (calls[-1].state_after if calls and calls[-1].state_after else scenario.initial_state))

    expected_state = scenario.expected.get("state_contains", {})
    if expected_state:
        matches = all(final_state.get(key) == value for key, value in expected_state.items())
        results.append(_result("expected_state", matches, "final state contains scenario expectations" if matches else "final state misses one or more scenario expectations"))
    unchanged_fields = scenario.expected.get("state_unchanged", [])
    if unchanged_fields:
        unchanged = all(final_state.get(field) == scenario.initial_state.get(field) for field in unchanged_fields)
        results.append(_result("preserved_state", unchanged, "fields that should remain unchanged were preserved" if unchanged else "a field changed despite the scenario requiring preservation"))

    required_tools = set(scenario.expected.get("required_tools", []))
    if required_tools:
        called_tools = {call.name for call in calls}
        results.append(_result("required_tool_use", required_tools <= called_tools, "all required tools were called" if required_tools <= called_tools else f"missing required tools: {', '.join(sorted(required_tools - called_tools))}"))
    forbidden_tools = set(scenario.expected.get("forbidden_tools", []))
    if forbidden_tools:
        called_tools = {call.name for call in calls}
        unexpected_tools = forbidden_tools & called_tools
        results.append(_result("forbidden_tool_use", not unexpected_tools, "no forbidden tools were called" if not unexpected_tools else f"called forbidden tools: {', '.join(sorted(unexpected_tools))}"))

    accepted = _has_offer_for_every_leg(final_state) and bool(final_state.get("flights_confirmed"))
    critical_failure = any(
        result.verdict == "fail" and result.criterion in {
            "flight_selection_prerequisites", "itinerary_validity", "confirmation_integrity"
        }
        for result in results
    )
    expected_completion = scenario.expected.get("accepted_flights", False)
    if expected_completion:
        results.append(_result("accepted_flight_completion", accepted and not critical_failure, "selected offers were explicitly accepted" if accepted and not critical_failure else "scenario did not finish with valid accepted offers"))
    else:
        results.append(EvaluationResult("accepted_flight_completion", EVALUATOR_VERSION, "not_applicable", "code", "scenario does not require accepted flights"))
    return EvaluationReport(scenario.id, tuple(results), recorded_turns, completion_reason)
