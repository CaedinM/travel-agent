"""Short-lived Duffel offer cache used while the traveller states preferences."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from travel_agent.core.models import Leg
from travel_agent.flights.duffel import DuffelClient


@dataclass
class PrefetchedOffers:
    signature: tuple[tuple[str, str, str | None], ...]
    travelers: int
    task: asyncio.Task[dict[int, dict[str, Any] | Exception]]


class FlightPrefetchCache:
    """Keeps one in-flight or completed search per conversation itinerary."""

    def __init__(self) -> None:
        self._entries: dict[str, PrefetchedOffers] = {}

    @staticmethod
    def _signature(legs: list[Leg]) -> tuple[tuple[str, str, str | None], ...]:
        return tuple((leg.origin, leg.destination, leg.departure_date) for leg in legs)

    def start(
        self, key: str, *, client: DuffelClient, legs: list[Leg], travelers: int,
        include_selected: bool = False,
    ) -> None:
        targets = [
            (index, leg)
            for index, leg in enumerate(legs)
            if include_selected or leg.offer is None
        ]
        if not targets:
            return
        signature = self._signature(legs)
        existing = self._entries.get(key)
        if existing and existing.signature == signature and existing.travelers == travelers:
            return

        async def fetch_all() -> dict[int, dict[str, Any] | Exception]:
            semaphore = asyncio.Semaphore(3)

            async def fetch(index: int, leg: Leg):
                try:
                    async with semaphore:
                        return index, await client.search_one_way(
                            origin=leg.origin,
                            destination=leg.destination,
                            departure_date=leg.departure_date or "",
                            travelers=travelers,
                        )
                except Exception as exc:
                    return index, exc

            return dict(await asyncio.gather(*(fetch(index, leg) for index, leg in targets)))

        self._entries[key] = PrefetchedOffers(
            signature=signature, travelers=travelers, task=asyncio.create_task(fetch_all())
        )

    async def take(
        self, key: str, *, legs: list[Leg], travelers: int
    ) -> dict[int, dict[str, Any] | Exception] | None:
        entry = self._entries.get(key)
        if not entry or entry.signature != self._signature(legs) or entry.travelers != travelers:
            return None
        self._entries.pop(key, None)
        return await entry.task

    def close(self) -> None:
        for entry in self._entries.values():
            if not entry.task.done():
                entry.task.cancel()
        self._entries.clear()
