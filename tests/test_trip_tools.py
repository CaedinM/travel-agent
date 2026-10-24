from copy import deepcopy
from datetime import date
from types import SimpleNamespace

import pytest
from langchain.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import ValidationError

from travel_agent.core.models import BestOffer, Leg
from travel_agent.core.state import TripSnapshot, TripState
from travel_agent.tools.trip import set_trip_legs, update_trip_info


INVALID_DATES = [
    "Spring", "February", "", "2027-02-29", "2027-13-01", "2027-04-31",
    "2027-05-10T00:00:00Z", "20270510", "2027-W19-1", "2027-5-1",
    " 2027-05-10", "2027-05-10 ", "2027-05-10\n", "05/10/2027",
]


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 23)

    monkeypatch.setattr("travel_agent.core.state.date", FixedDate)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


def assert_date_error(result, detail):
    assert isinstance(result, Command)
    assert set(result.update) == {"messages"}
    assert len(result.update["messages"]) == 1
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert message.tool_call_id == "test-call"
    assert detail in message.content
    assert "formatted exactly as YYYY-MM-DD" in message.content
    assert "2027-05-10" in message.content
    assert "`season`" in message.content
    assert "retry once concrete dates are chosen" in message.content


@pytest.mark.parametrize("field", ["departure_date", "return_date"])
@pytest.mark.parametrize("value", INVALID_DATES)
def test_update_trip_info_rejects_invalid_dates(field, value):
    runtime = SimpleNamespace(state={}, tool_call_id="test-call")
    result = update_trip_info.func(runtime, origin="LAX", travelers=2, **{field: value})
    assert_date_error(result, field)
    assert runtime.state == {}


@pytest.mark.parametrize("field", ["departure_date", "return_date"])
@pytest.mark.parametrize("value", INVALID_DATES)
def test_trip_snapshot_rejects_invalid_date_formats(field, value):
    with pytest.raises(ValidationError, match="YYYY-MM-DD"):
        TripSnapshot(**{field: value})


@pytest.mark.parametrize("value", INVALID_DATES + [None])
def test_set_trip_legs_rejects_invalid_dates(value):
    runtime = SimpleNamespace(state={}, tool_call_id="test-call")
    result = set_trip_legs.func(
        runtime, [Leg(origin="SFO", destination="BKK", departure_date=value)]
    )
    assert_date_error(result, "[0] SFO -> BKK: departure_date")
    assert runtime.state == {}


@pytest.mark.parametrize("value", ["2000-02-12", "2026-10-23"])
def test_tools_reject_past_and_today_dates(value):
    runtime = SimpleNamespace(state={}, tool_call_id="test-call")
    assert_date_error(
        update_trip_info.func(runtime, departure_date=value),
        "departure_date must be after 2026-10-23",
    )
    assert_date_error(
        set_trip_legs.func(runtime, [Leg(origin="SFO", destination="BKK", departure_date=value)]),
        "[0] SFO -> BKK: departure_date must be after 2026-10-23",
    )


@pytest.mark.parametrize("return_date", ["2027-05-09", "2027-05-10"])
def test_update_trip_info_checks_return_against_existing_departure(return_date):
    runtime = SimpleNamespace(state={"departure_date": "2027-05-10"}, tool_call_id="test-call")
    assert_date_error(
        update_trip_info.func(runtime, return_date=return_date),
        "return_date must be after departure_date",
    )


def test_update_trip_info_checks_departure_against_existing_return():
    runtime = SimpleNamespace(state={"return_date": "2027-05-10"}, tool_call_id="test-call")
    assert_date_error(
        update_trip_info.func(runtime, departure_date="2027-05-11"),
        "return_date must be after departure_date",
    )


@pytest.mark.parametrize("last_date", ["2027-03-18", "2027-03-19"])
def test_set_trip_legs_rejects_non_chronological_dates(last_date):
    runtime = SimpleNamespace(state={}, tool_call_id="test-call")
    result = set_trip_legs.func(runtime, [
        Leg(origin="SFO", destination="LIS", departure_date="2027-03-05"),
        Leg(origin="LIS", destination="SVQ", departure_date="2027-03-19"),
        Leg(origin="SVQ", destination="SFO", departure_date=last_date),
    ])
    assert_date_error(result, "each later leg must depart after the prior leg")
    assert "Leg [2]" in result.update["messages"][0].content


@pytest.mark.parametrize("departure_date,return_date", [
    ("2027-05-10", "2027-05-17"), ("2028-02-29", "2028-03-01"),
])
def test_tools_accept_exact_dates_and_preserve_existing_offers(departure_date, return_date):
    offer = BestOffer(offer_id="offer-1", reasoning="Best match", price="500", currency="USD")
    leg = Leg(origin="SFO", destination="BKK", departure_date=departure_date, offer=offer)
    runtime = SimpleNamespace(state={"legs": [leg]}, tool_call_id="test-call")
    result = update_trip_info.func(runtime, departure_date=departure_date, return_date=return_date)
    assert result.update["departure_date"] == departure_date
    assert result.update["return_date"] == return_date
    assert result.update["messages"][0].status == "success"

    result = set_trip_legs.func(runtime, [
        Leg(origin="SFO", destination="BKK", departure_date=departure_date),
        Leg(origin="BKK", destination="SFO", departure_date=return_date),
    ])
    assert result.update["legs"][0].offer == offer
    assert result.update["legs"][1].departure_date == return_date
    assert result.update["messages"][0].status == "success"


def test_update_trip_info_allows_omitted_dates_and_flexible_season():
    runtime = SimpleNamespace(state={}, tool_call_id="test-call")
    result = update_trip_info.func(runtime, season="Spring", travelers=2)
    assert result.update["season"] == "Spring"
    assert result.update["travelers"] == 2
    assert "departure_date" not in result.update
    assert "return_date" not in result.update
    assert result.update["messages"][0].status == "success"


@pytest.mark.parametrize("tool_name", ["update_trip_info", "set_trip_legs"])
@pytest.mark.parametrize("value", ["Spring", "20270510", "2027-02-29", "2000-02-12", "2026-10-23"])
def test_date_errors_reach_agent_without_changing_trip_state(tool_name, value):
    existing_leg = Leg(
        origin="SFO", destination="BKK", departure_date="2027-05-10",
        offer=BestOffer(offer_id="offer-1", reasoning="Best match", price="500", currency="USD"),
    )
    state = {
        "origin": "SFO", "travelers": 1, "phase": "accommodations",
        "departure_date": "2027-05-10", "return_date": "2027-05-17",
        "legs": [existing_leg], "flights_confirmed": True,
    }
    original_state = deepcopy(state)
    args = (
        {"departure_date": value, "origin": "LAX", "travelers": 3}
        if tool_name == "update_trip_info" else
        {"legs": [
            {"origin": "SFO", "destination": "BKK", "departure_date": "2027-05-10"},
            {"origin": "BKK", "destination": "SFO", "departure_date": value},
        ]}
    )
    state["messages"] = [AIMessage(content="", tool_calls=[{
        "name": tool_name, "args": args, "id": "test-call", "type": "tool_call",
    }])]
    graph = StateGraph(TripState)
    graph.add_node("tools", ToolNode([update_trip_info, set_trip_legs]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    result = graph.compile().invoke(state)

    assert {key: result[key] for key in original_state} == original_state
    assert result["messages"][-1].status == "error"
    assert result["messages"][-1].tool_call_id == "test-call"
    assert "YYYY-MM-DD" in result["messages"][-1].content
    assert "`season`" in result["messages"][-1].content


@pytest.mark.parametrize("field", ["departure_date", "return_date"])
@pytest.mark.parametrize("value", ["2000-02-12", "2026-10-22", "2026-10-23"])
def test_update_trip_info_rejects_non_future_dates_for_either_field(field, value):
    runtime = SimpleNamespace(state={}, tool_call_id="test-call")
    result = update_trip_info.func(runtime, travelers=2, **{field: value})
    assert_date_error(result, f"{field} must be after 2026-10-23")
    assert runtime.state == {}


def test_update_trip_info_accepts_tomorrow_and_later_return():
    runtime = SimpleNamespace(state={}, tool_call_id="test-call")
    result = update_trip_info.func(
        runtime, departure_date="2026-10-24", return_date="2026-10-25",
    )
    assert result.update["departure_date"] == "2026-10-24"
    assert result.update["return_date"] == "2026-10-25"
    assert result.update["messages"][0].status == "success"
