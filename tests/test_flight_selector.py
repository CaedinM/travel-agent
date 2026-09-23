import asyncio

from travel_agent.core.models import Leg
from travel_agent.flights.flight_selector import FlightOption, prepare_baggage_candidates
from travel_agent.flights.prefetch import FlightPrefetchCache


def option(
    identifier: str, *, price: float, duration: int, stops: int, departure: str
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
