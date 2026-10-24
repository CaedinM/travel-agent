from evals.contracts import Scenario, ToolCall, Turn
from evals.deterministic import evaluate_session
from evals.judges import calibrate
from evals.harness import run_decision
from evals.run import load_scenarios, load_turn_cases
from evals.state import state_error


def scenario(**overrides):
    defaults = dict(
        id="test", category="test", initial_state={}, traveller_facts={}, provider_fixtures={},
        permitted_assumptions=(), expected={"accepted_flights": True}, user_turns=(),
    )
    return Scenario(**(defaults | overrides))


def offers_state():
    return {
        "origin": "SFO", "travelers": 1, "flights_confirmed": True,
        "legs": [
            {"origin": "SFO", "destination": "CDG", "departure_date": "2027-05-10", "offer": {"offer_id": "1"}},
            {"origin": "CDG", "destination": "SFO", "departure_date": "2027-05-17", "offer": {"offer_id": "2"}},
        ],
    }


def test_flight_selection_requires_traveler_count():
    call = ToolCall("choose_prefetched_flights", {}, {"legs": [{}]}, turn_index=2)
    report = evaluate_session(scenario(), [Turn("find flights", "", (call,))], completion_reason="complete")
    assert any(result.criterion == "flight_selection_prerequisites" and result.verdict == "fail" for result in report.results)


def test_confirmation_needs_explicit_acceptance_and_offers():
    call = ToolCall("confirm_flights", {"explicit_acceptance": False}, offers_state(), offers_state(), 3)
    report = evaluate_session(scenario(), [Turn("sounds good", "", (call,))], completion_reason="complete")
    assert any(result.criterion == "confirmation_integrity" and result.verdict == "fail" for result in report.results)


def test_valid_accepted_flights_complete_without_critical_failure():
    state = offers_state()
    call = ToolCall("confirm_flights", {"explicit_acceptance": True}, state, state, 3)
    report = evaluate_session(scenario(), [Turn("confirm", "", (call,))], completion_reason="complete")
    assert report.accepted_flights_completed
    assert not report.has_critical_failure


def test_invalid_itinerary_is_caught():
    call = ToolCall("set_trip_legs", {"legs": [{"origin": "SFO", "destination": "CDG", "departure_date": "2027-05-10"}]}, {"origin": "SFO"}, turn_index=1)
    report = evaluate_session(scenario(), [Turn("save", "", (call,))], completion_reason="complete")
    assert any(result.criterion == "itinerary_validity" and result.verdict == "fail" for result in report.results)


def test_scenario_fixture_has_thirty_versioned_cases():
    scenarios = load_scenarios()
    assert len(scenarios) == 30
    assert {scenario.category for scenario in scenarios} == {"standard", "incomplete", "correction", "provider", "boundary"}


def test_turn_cases_use_reachable_custom_states():
    cases = load_turn_cases()
    assert len(cases) == 11
    assert {case.initial_state["phase"] for case in cases} == {"intake", "research", "flights", "accommodations"}


def test_flights_phase_does_not_require_a_traveller_count():
    assert state_error(
        {
            "phase": "flights",
            "origin": "SFO",
            "departure_date": "2027-05-10",
            "return_date": "2027-05-17",
            "legs": [
                {"origin": "SFO", "destination": "CDG", "departure_date": "2027-05-10"},
                {"origin": "CDG", "destination": "SFO", "departure_date": "2027-05-17"},
            ],
        },
        requires_return_home=True,
    ) is None


def test_decision_harness_scores_the_state_returned_by_agent():
    case = next(case for case in load_turn_cases() if case.id == "turn-intake-completes-date-plan")

    def agent(message, state, fixtures):
        final_state = state | {"phase": "research", "departure_date": "2027-05-10", "return_date": "2027-05-17"}
        call = ToolCall("update_trip_info", {}, state, final_state)
        return Turn(message, "Saved.", (call,)), final_state

    report = run_decision(case, case.user_turns[0], agent)
    assert all(result.verdict != "fail" for result in report.results)


def test_decision_harness_fails_when_returned_state_violates_expectation():
    case = next(case for case in load_turn_cases() if case.id == "turn-intake-completes-date-plan")

    def agent(message, state, fixtures):
        call = ToolCall("update_trip_info", {}, state, state)
        return Turn(message, "Saved.", (call,)), state

    report = run_decision(case, case.user_turns[0], agent)
    assert any(result.criterion == "expected_state" and result.verdict == "fail" for result in report.results)


def test_calibration_needs_one_hundred_examples_and_quality_thresholds():
    metrics = calibrate([True] * 50 + [False] * 50, [True] * 50 + [False] * 50)
    assert metrics.ready_for_release_signal
    assert calibrate([True], [True]).ready_for_release_signal is False


def test_unaccepted_provider_case_is_reported_as_incomplete_not_success():
    report = evaluate_session(
        scenario(expected={"accepted_flights": False}), [], completion_reason="blocked_provider_timeout"
    )
    assert report.summary["outcome"] == "blocked"
