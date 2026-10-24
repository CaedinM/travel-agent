"""Small direct async client for the Duffel Air API.

The application needs two fixed operations, so this deliberately avoids an agent-tool
or MCP boundary. It is also easy to replace with a fake transport in tests.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from travel_agent.core.models import FlightSearchPreferences


class DuffelAPIError(RuntimeError):
    """A provider response that cannot safely be treated as a flight search result."""


class DuffelClient:
    def __init__(self, access_token: str, *, timeout: float = 25.0) -> None:
        if not access_token.strip():
            raise ValueError("Duffel API key is required.")
        self._client = httpx.AsyncClient(
            base_url="https://api.duffel.com",
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "Duffel-Version": "v2",
                "Authorization": f"Bearer {access_token}",
            },
        )

    @classmethod
    def from_environment(cls) -> "DuffelClient":
        key_name = (
            "DUFFEL_LIVE_API_KEY"
            if os.environ.get("LIVE_MODE", "false").strip().lower() == "true"
            else "DUFFEL_TEST_API_KEY"
        )
        return cls(os.environ[key_name])

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search_one_way(
        self,
        *,
        origin: str,
        destination: str,
        departure_date: str,
        travelers: int,
        preferences: FlightSearchPreferences | None = None,
    ) -> dict[str, Any]:
        """Create a one-way Duffel offer request and return its API resource."""
        preferences = preferences or FlightSearchPreferences()
        slice_data: dict[str, Any] = {
            "origin": origin.upper(),
            "destination": destination.upper(),
            "departure_date": departure_date,
        }
        departure_time = {
            key: value
            for key, value in {
                "from": preferences.departure_time_from,
                "to": preferences.departure_time_to,
            }.items()
            if value is not None
        }
        if departure_time:
            slice_data["departure_time"] = departure_time
        arrival_time = {
            key: value
            for key, value in {
                "from": preferences.arrival_time_from,
                "to": preferences.arrival_time_to,
            }.items()
            if value is not None
        }
        if arrival_time:
            slice_data["arrival_time"] = arrival_time
        request_data: dict[str, Any] = {
            "passengers": [{"type": "adult"} for _ in range(travelers)],
            "slices": [slice_data],
        }
        if preferences.cabin_class is not None:
            request_data["cabin_class"] = preferences.cabin_class
        if preferences.max_connections is not None:
            request_data["max_connections"] = preferences.max_connections
        response = await self._client.post(
            "/air/offer_requests",
            params={"return_offers": "true", "supplier_timeout": "10000"},
            json={"data": request_data},
        )
        return self._data(response)

    async def get_offer(
        self, offer_id: str, *, return_available_services: bool = False
    ) -> dict[str, Any]:
        """Return current offer details, optionally including purchasable ancillaries."""
        response = await self._client.get(
            f"/air/offers/{offer_id}",
            params={"return_available_services": "true"}
            if return_available_services
            else None,
        )
        return self._data(response)

    @staticmethod
    def _data(response: httpx.Response) -> dict[str, Any]:
        try:
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPStatusError, ValueError) as exc:
            detail = response.text[:500] if response.content else str(exc)
            raise DuffelAPIError(f"Duffel request failed: {detail}") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise DuffelAPIError("Duffel response did not contain an object in data.")
        return data
