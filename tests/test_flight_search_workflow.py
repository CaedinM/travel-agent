import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from travel_agent.core.models import BestOffer, FlightSearchPreferences, Leg
from travel_agent.flights.duffel import DuffelClient
from travel_agent.flights.flight_selector import FlightOption
from travel_agent.tools import trip


def flight_option():
    return FlightOption(
        offer_id="offer-1", offer_request_id="request-1", price=100,
        price_text="100.00", currency="USD",
        departure="2027-05-10T10:00:00", arrival="2027-05-10T12:00:00",
        duration="PT2H", duration_minutes=120, stops=0,
        carriers=("Test Air",), expires_at=None,
    )


def test_search_flights_returns_normalized_options():
    client = SimpleNamespace(search_one_way=AsyncMock(return_value={
        "id": "request-1",
        "offers": [{
            "id": "offer-1", "total_amount": "100.00", "total_currency": "USD",
            "slices": [{
                "duration": "PT2H",
                "segments": [{
                    "departing_at": "2027-05-10T10:00:00",
                    "arriving_at": "2027-05-10T12:00:00",
                    "operating_carrier": {"name": "Test Air"},
                }],
            }],
        }],
    }))
    leg = Leg(origin="SFO", destination="LAX", departure_date="2027-05-10")
    preferences = FlightSearchPreferences(cabin_class="business", max_connections=0)

    options = asyncio.run(trip.search_flights(
        leg, client=client, travelers=2, preferences=preferences,
    ))

    assert options == [flight_option()]
    client.search_one_way.assert_awaited_once_with(
        origin="SFO", destination="LAX", departure_date="2027-05-10",
        travelers=2, preferences=preferences,
    )


@pytest.mark.parametrize("with_baggage", [False, True])
def test_select_flight_uses_options_and_preferences(monkeypatch, with_baggage):
    preferences = FlightSearchPreferences()
    options = [flight_option()]
    selected = BestOffer(offer_id="offer-1", reasoning="Best match", price="100.00", currency="USD")
    selector_model = object()
    client = object()
    choose = AsyncMock(return_value=(selected, 12, 4, "selector-model"))
    enrich = AsyncMock(return_value=options)
    monkeypatch.setattr(trip, "select_model_offer", choose)
    monkeypatch.setattr(trip, "baggage_requested", lambda _: with_baggage)
    monkeypatch.setattr(trip, "enrich_baggage_details", enrich)

    offer, metric = asyncio.run(trip.select_flight(
        client=client, selector_model=selector_model, options=options,
        leg_index=1, preferences=preferences,
    ))

    assert offer == selected
    choose.assert_awaited_once_with(
        selector_model=selector_model, candidates=options, preferences=preferences,
    )
    if with_baggage:
        enrich.assert_awaited_once_with(client, options)
        assert metric.baggage_enrichment_ms is not None
    else:
        enrich.assert_not_awaited()
    assert metric.leg_index == 1
    assert metric.candidates == 1
    assert metric.selected_offer_id == selected.offer_id
    assert (metric.input_tokens, metric.output_tokens, metric.model) == (12, 4, "selector-model")


@pytest.mark.parametrize("failure", ["search", "selection", "empty"])
def test_search_and_select_flights_preserves_partial_success(monkeypatch, failure):
    options = [flight_option()]
    selected = BestOffer(offer_id="offer-1", reasoning="Best match", price="100.00", currency="USD")
    existing = BestOffer(offer_id="saved", reasoning="Already selected", price="200.00", currency="USD")
    preferences = FlightSearchPreferences(max_connections=0)
    client = object.__new__(DuffelClient)
    selector_model = object()
    legs = [
        Leg(origin="SFO", destination="LAX", departure_date="2027-05-10"),
        Leg(origin="LAX", destination="JFK", departure_date="2027-05-12"),
        Leg(origin="JFK", destination="SFO", departure_date="2027-05-15", offer=existing),
    ]

    async def search(leg, **kwargs):
        assert kwargs == dict(client=client, travelers=2, preferences=preferences)
        if leg.destination == "JFK":
            if failure == "search":
                raise RuntimeError("Search failed")
            if failure == "empty":
                return []
            return [replace(flight_option(), offer_id="failed")]
        return options

    async def choose(*, selector_model, candidates, preferences):
        if candidates[0].offer_id == "failed":
            raise RuntimeError("Selection failed")
        return selected, 10, 5, "selector-model"

    search_mock = AsyncMock(side_effect=search)
    monkeypatch.setattr(trip, "search_flights", search_mock)
    monkeypatch.setattr(trip, "select_model_offer", choose)
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            pipeline="deterministic", duffel_client=client,
            flight_selector_model=selector_model,
        ),
        state={"legs": legs, "travelers": 2, "flight_preferences": preferences},
        tool_call_id="test-call",
    )

    result = asyncio.run(trip.search_and_select_flights.coroutine(runtime))

    assert search_mock.await_count == 2
    assert [leg.offer for leg in result.update["legs"]] == [selected, None, existing]
    metrics = result.update["flight_pipeline_metrics"]
    assert metrics[0].selected_offer_id == selected.offer_id
    assert metrics[0].duffel_ms is not None
    assert metrics[1].error
    assert "Problems:" in result.update["messages"][0].content
    assert result.update["flight_preferences"] == preferences
