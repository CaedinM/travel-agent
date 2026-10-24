import asyncio
import json

import httpx

from travel_agent.core.models import FlightSearchPreferences
from travel_agent.flights.duffel import DuffelClient


def test_search_one_way_maps_typed_preferences_to_duffel_request():
    captured: dict = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": {"id": "orq_test", "offers": []}})

    async def exercise():
        client = DuffelClient("test-token")
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url="https://api.duffel.com",
            transport=httpx.MockTransport(respond),
        )
        try:
            await client.search_one_way(
                origin="sfo",
                destination="lax",
                departure_date="2027-05-01",
                travelers=2,
                preferences=FlightSearchPreferences(
                    cabin_class="premium_economy",
                    max_connections=0,
                    departure_time_from="09:00",
                    departure_time_to="18:30",
                    arrival_time_to="22:00",
                ),
            )
        finally:
            await client.aclose()

    asyncio.run(exercise())

    assert captured["body"] == {
        "data": {
            "passengers": [{"type": "adult"}, {"type": "adult"}],
            "slices": [{
                "origin": "SFO",
                "destination": "LAX",
                "departure_date": "2027-05-01",
                "departure_time": {"from": "09:00", "to": "18:30"},
                "arrival_time": {"to": "22:00"},
            }],
            "cabin_class": "premium_economy",
            "max_connections": 0,
        }
    }
