import asyncio

import pytest

from travel_agent.core.models import FlightSearchPreferences, Leg
from travel_agent.flights.flight_selector import (
    FlightOption,
    FlightSelectionDecision,
    NoEligibleFlightOfferError,
    prepare_baggage_candidates,
    select_deterministic_offer,
    select_model_offer,
)
from travel_agent.flights.prefetch import FlightPrefetchCache


def option(
    identifier: str,
    *,
    price: float,
    duration: int,
    stops: int,
    departure: str,
    carrier_codes: tuple[str, ...] = (),
    included_checked_baggage: str = "Unknown (not returned by airline)",
) -> FlightOption:
    return FlightOption(
        offer_id=identifier,
        offer_request_id="orq_test",
        price=price,
        price_text=str(price),
        currency="USD",
        departure=departure,
        arrival="2027-05-01T20:00:00",
        duration="PT1H",
        duration_minutes=duration,
        stops=stops,
        carriers=("Test Air",),
        expires_at=None,
        carrier_iata_codes=carrier_codes,
        included_checked_baggage=included_checked_baggage,
    )


def test_baggage_shortlist_keeps_pareto_extremes_and_is_bounded():
    options = [
        option("cheap", price=100, duration=600, stops=2, departure="2027-05-01T05:00:00"),
        option("fast", price=400, duration=90, stops=1, departure="2027-05-01T12:00:00"),
        option("direct", price=300, duration=180, stops=0, departure="2027-05-01T20:00:00"),
        *[
            option(
                f"dominated-{index}",
                price=500 + index,
                duration=700 + index,
                stops=3,
                departure=f"2027-05-02T{index:02d}:00:00",
            )
            for index in range(20)
        ],
    ]

    shortlist = prepare_baggage_candidates(options, limit=12)

    assert len(shortlist) <= 12
    assert {"cheap", "fast", "direct"}.issubset(
        {candidate.offer_id for candidate in shortlist}
    )
    assert not any(candidate.offer_id.startswith("dominated-") for candidate in shortlist)


def test_prefetch_is_reused_and_then_consumed():
    class FakeDuffel:
        def __init__(self):
            self.calls = 0

        async def search_one_way(self, **kwargs):
            self.calls += 1
            return {"offers": [], "request": kwargs}

    async def exercise():
        client = FakeDuffel()
        cache = FlightPrefetchCache()
        legs = [Leg(origin="SFO", destination="LAX", departure_date="2027-05-01")]
        cache.start("thread", client=client, legs=legs, travelers=1)
        cache.start("thread", client=client, legs=legs, travelers=1)
        result = await cache.take("thread", legs=legs, travelers=1)
        assert client.calls == 1
        assert isinstance(result, dict) and result[0]["offers"] == []
        assert await cache.take("thread", legs=legs, travelers=1) is None

    asyncio.run(exercise())


def test_deterministic_selection_applies_hard_constraints_before_price():
    candidates = [
        option(
            "cheap-connection", price=100, duration=180, stops=1,
            departure="2027-05-01T09:00:00", carrier_codes=("UA",),
        ),
        option(
            "wrong-airline", price=120, duration=90, stops=0,
            departure="2027-05-01T09:30:00", carrier_codes=("AS",),
        ),
        option(
            "eligible", price=150, duration=120, stops=0,
            departure="2027-05-01T10:00:00", carrier_codes=("UA",),
            included_checked_baggage="1 checked bag included per passenger",
        ),
    ]
    preferences = FlightSearchPreferences(
        cabin_class="economy",
        max_connections=0,
        departure_time_from="09:00",
        required_airline_iata_codes=["UA"],
        checked_bags_per_traveler=1,
        require_included_checked_bags=True,
    )

    selected = select_deterministic_offer(
        candidates=candidates, preferences=preferences
    )

    assert selected.offer_id == "eligible"


def test_deterministic_selection_uses_stable_default_tie_breakers():
    candidates = [
        option("late", price=100, duration=100, stops=0, departure="2027-05-01T12:00:00"),
        option("connection", price=100, duration=80, stops=1, departure="2027-05-01T08:00:00"),
        option("early", price=100, duration=100, stops=0, departure="2027-05-01T08:00:00"),
    ]

    selected = select_deterministic_offer(
        candidates=candidates, preferences=FlightSearchPreferences()
    )

    assert selected.offer_id == "early"


def test_deterministic_selection_never_relaxes_hard_constraints():
    candidates = [
        option("one-stop", price=100, duration=90, stops=1, departure="2027-05-01T09:00:00"),
    ]

    with pytest.raises(NoEligibleFlightOfferError, match="no constraints were relaxed"):
        select_deterministic_offer(
            candidates=candidates,
            preferences=FlightSearchPreferences(max_connections=0),
        )


def test_typed_preferences_reject_invalid_duffel_values():
    with pytest.raises(ValueError, match="cabin_class"):
        FlightSearchPreferences(cabin_class="premium economy")
    with pytest.raises(ValueError, match="departure_time_from"):
        FlightSearchPreferences(
            departure_time_from="18:00", departure_time_to="09:00"
        )
    with pytest.raises(ValueError, match="IATA"):
        FlightSearchPreferences(required_airline_iata_codes=["United"])


def test_midnight_latest_arrival_is_normalized_to_end_of_day():
    preferences = FlightSearchPreferences(arrival_time_to="00:00")

    assert preferences.arrival_time_to == "23:59"


def test_applied_filter_summary_discloses_fuzzy_timing_assumption():
    preferences = FlightSearchPreferences(
        departure_time_from="08:00",
        arrival_time_to="20:00",
        summary=(
            "Interpreted 'not too early or late' as departing after 08:00 "
            "and arriving by 20:00."
        ),
    )

    assert preferences.applied_filter_summary() == (
        "departure 08:00–end of day; arrival start of day–20:00. "
        "Assumption/summary: Interpreted 'not too early or late' as departing "
        "after 08:00 and arriving by 20:00."
    )


def test_model_selection_is_constrained_to_eligible_offer_ids():
    class RawResponse:
        usage_metadata = {"input_tokens": 321, "output_tokens": 27}
        response_metadata = {"model_name": "gpt-5-mini"}

    class StructuredSelector:
        async def ainvoke(self, messages):
            assert len(messages) == 2
            return {
                "parsed": FlightSelectionDecision(
                    offer_id="preferred",
                    tradeoff_explanation="United costs a little more but avoids a connection and includes the needed bag.",
                ),
                "raw": RawResponse(),
            }

    class SelectorModel:
        def with_structured_output(self, schema, **kwargs):
            assert schema is FlightSelectionDecision
            assert kwargs["strict"] is True
            return StructuredSelector()

    async def exercise():
        return await select_model_offer(
            selector_model=SelectorModel(),
            candidates=[
                option("cheap", price=100, duration=180, stops=1, departure="2027-05-01T09:00:00"),
                option("preferred", price=130, duration=120, stops=0, departure="2027-05-01T10:00:00"),
            ],
            preferences=FlightSearchPreferences(),
        )

    selected, input_tokens, output_tokens, model = asyncio.run(exercise())

    assert selected.offer_id == "preferred"
    assert input_tokens == 321
    assert output_tokens == 27
    assert model == "gpt-5-mini"
