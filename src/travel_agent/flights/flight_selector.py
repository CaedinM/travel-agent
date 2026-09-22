"""Prepare direct Duffel offers and select one with TypeSafe Jev."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, replace
from typing import Any

from langchain_typesafe import Choice, TypeSafeClassifier

from travel_agent.core.models import BestOffer


@dataclass(frozen=True)
class FlightOption:
    offer_id: str
    offer_request_id: str
    price: float
    price_text: str
    currency: str
    departure: str
    arrival: str
    duration: str
    duration_minutes: int
    stops: int
    carriers: tuple[str, ...]
    expires_at: str | None
    included_checked_baggage: str = "Unknown (not returned by airline)"
    additional_checked_baggage: tuple[str, ...] = ()

    def state(self) -> dict[str, Any]:
        return {
            "offer_id": self.offer_id,
            "total_price": self.price_text,
            "currency": self.currency,
            "departure": self.departure,
            "arrival": self.arrival,
            "duration_minutes": self.duration_minutes,
            "stops": self.stops,
            "carriers": list(self.carriers),
            "expires_at": self.expires_at,
            "included_checked_baggage": self.included_checked_baggage,
            "additional_checked_baggage_options": list(self.additional_checked_baggage),
        }


def _baggage_details(offer: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Return conservative, readable baggage facts from a detailed Duffel offer."""
    checked_counts: list[int] = []
    passenger_records_found = False
    for slice_ in offer.get("slices") or []:
        for segment in slice_.get("segments") or []:
            passengers = segment.get("passengers")
            if not isinstance(passengers, list):
                continue
            passenger_records_found = True
            for passenger in passengers:
                baggages = passenger.get("baggages") if isinstance(passenger, dict) else None
                if not isinstance(baggages, list):
                    continue
                checked_counts.append(
                    sum(
                        int(bag.get("quantity") or 0)
                        for bag in baggages
                        if isinstance(bag, dict) and bag.get("type") == "checked"
                    )
                )

    if not passenger_records_found or not checked_counts:
        included = "Unknown (not returned by airline)"
    elif len(set(checked_counts)) == 1:
        count = checked_counts[0]
        included = (
            f"{count} checked bag{'s' if count != 1 else ''} included per passenger"
        )
    else:
        included = "Varies by passenger or segment: " + ", ".join(
            str(count) for count in sorted(set(checked_counts))
        ) + " checked bags included"

    services: list[str] = []
    for service in offer.get("available_services") or []:
        if not isinstance(service, dict) or service.get("type") != "baggage":
            continue
        amount = service.get("total_amount")
        currency = service.get("total_currency")
        if amount is None or not currency:
            continue
        passengers = len(service.get("passenger_ids") or [])
        segments = len(service.get("segment_ids") or [])
        maximum = service.get("maximum_quantity")
        scope = f"for {passengers or 'applicable'} passenger(s), {segments or 'applicable'} segment(s)"
        if maximum is not None:
            scope += f", max quantity {maximum}"
        services.append(f"{currency} {amount} ({scope})")
    return included, tuple(services)


def _duration_minutes(value: str | None) -> int:
    """Convert Duffel's ISO-8601 duration to minutes, sorting unknowns last."""
    import re

    match = re.fullmatch(r"P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?", value or "")
    if not match:
        return 10**6
    days, hours, minutes = (int(part) if part else 0 for part in match.groups())
    return days * 1440 + hours * 60 + minutes


def parse_duffel_offers(request: dict[str, Any]) -> list[FlightOption]:
    """Flatten direct Duffel offers without losing the provider's exact IDs."""
    out: list[FlightOption] = []
    for offer in request.get("offers", []):
        slices = offer.get("slices") or []
        segments = [segment for slice_ in slices for segment in slice_.get("segments", [])]
        if not segments or not isinstance(offer.get("id"), str):
            continue
        try:
            price_text = str(offer["total_amount"])
            price = float(price_text)
        except (KeyError, TypeError, ValueError):
            continue
        first, last = segments[0], segments[-1]
        carrier_names = tuple(
            dict.fromkeys(
                str((segment.get("operating_carrier") or segment.get("marketing_carrier") or {}).get("name") or "Unknown")
                for segment in segments
            )
        )
        included_checked_baggage, additional_checked_baggage = _baggage_details(offer)
        out.append(
            FlightOption(
                offer_id=offer["id"],
                offer_request_id=str(offer.get("offer_request_id") or request.get("id") or ""),
                price=price,
                price_text=price_text,
                currency=str(offer.get("total_currency") or ""),
                departure=str(first.get("departing_at") or ""),
                arrival=str(last.get("arriving_at") or ""),
                duration=str(slices[0].get("duration") or "") if slices else "",
                duration_minutes=sum(_duration_minutes(slice_.get("duration")) for slice_ in slices),
                stops=max(0, len(segments) - len(slices)),
                carriers=carrier_names,
                expires_at=offer.get("expires_at"),
                included_checked_baggage=included_checked_baggage,
                additional_checked_baggage=additional_checked_baggage,
            )
        )
    return out


def prepare_candidates(options: list[FlightOption], *, limit: int = 40) -> list[FlightOption]:
    """Remove duplicate itineraries while retaining real traveller tradeoffs."""
    cheapest: dict[tuple[str, str, int], FlightOption] = {}
    for option in sorted(options, key=lambda item: item.price):
        cheapest.setdefault((option.departure, option.arrival, option.stops), option)
    return sorted(
        cheapest.values(), key=lambda item: (item.price, item.duration_minutes, item.stops)
    )[:limit]


def baggage_requested(preferences: str | None) -> bool:
    """Use a cheap, explicit gate before making per-offer ancillary requests."""
    return bool(re.search(r"\b(bag|bags|baggage|checked|luggage|suitcase)\b", preferences or "", re.I))


async def enrich_baggage_details(client: Any, options: list[FlightOption]) -> list[FlightOption]:
    """Refresh candidates with the provider's current allowance and bag-service prices.

    Duffel returns these fields only from its single-offer endpoint. Failures preserve
    the original candidate with an explicit unknown allowance rather than guessing.
    """
    semaphore = asyncio.Semaphore(6)

    async def enrich(option: FlightOption) -> FlightOption:
        try:
            async with semaphore:
                detailed_offer = await client.get_offer(
                    option.offer_id, return_available_services=True
                )
            included, services = _baggage_details(detailed_offer)
            return replace(
                option,
                included_checked_baggage=included,
                additional_checked_baggage=services,
            )
        except Exception:
            return option

    return list(await asyncio.gather(*(enrich(option) for option in options)))


async def select_best_offer(
    classifier: TypeSafeClassifier,
    *,
    origin: str,
    destination: str,
    departure_date: str,
    preferences: str | None,
    candidates: list[FlightOption],
) -> tuple[BestOffer, float, int | None, int | None, str]:
    """Use one constrained Choice decision and validate it against known offer IDs."""
    if not candidates:
        raise ValueError("Cannot select a flight from an empty candidate list.")
    by_id = {option.offer_id: option for option in candidates}
    response = await classifier.ainvoke(
        {
            "state": {
                "leg": {
                    "origin": origin,
                    "destination": destination,
                    "departure_date": departure_date,
                },
                "traveler_preferences_raw": preferences or "No additional preferences stated.",
                "candidates": [option.state() for option in candidates],
            },
            "questions": {
                "best_offer": Choice(
                    instructions=(
                        "Choose the best flight offer for this one-way leg. Honor explicit "
                        "traveler preferences first. Otherwise favor lower total price, but "
                        "do not choose a materially longer or more-stop itinerary for a trivial "
                        "saving. Penalize departures before 06:00 and arrivals after 23:00 "
                        "unless the saving is substantial. If the traveler mentions checked bags "
                        "or baggage, compare the included allowance and purchasable bag prices; "
                        "do not treat unknown baggage details as included. Choose only an offered ID."
                    ),
                    criteria={offer_id: option.state() for offer_id, option in by_id.items()},
                )
            },
        }
    )
    answer = response.choices["best_offer"]
    option = by_id.get(answer.choice)
    if option is None:
        raise ValueError("Jev selected an offer that was not in the supplied candidate set.")
    return (
        BestOffer(
            offer_id=option.offer_id,
            price=option.price_text,
            currency=option.currency,
            offer_request_id=option.offer_request_id or None,
            expires_at=option.expires_at,
            reasoning=(
                f"Selected by Jev from {len(candidates)} viable options "
                f"with {answer.confidence:.0%} decision confidence."
            ),
        ),
        answer.confidence,
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.model,
    )
