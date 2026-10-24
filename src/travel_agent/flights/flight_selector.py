"""Prepare direct Duffel offers and select one deterministically."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from travel_agent.core.models import BestOffer, FlightSearchPreferences


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
    carrier_iata_codes: tuple[str, ...] = ()
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
            "carrier_iata_codes": list(self.carrier_iata_codes),
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
        carrier_iata_codes = tuple(
            dict.fromkeys(
                str((segment.get("operating_carrier") or segment.get("marketing_carrier") or {}).get("iata_code") or "").upper()
                for segment in segments
                if (segment.get("operating_carrier") or segment.get("marketing_carrier") or {}).get("iata_code")
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
                carrier_iata_codes=carrier_iata_codes,
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


def prepare_baggage_candidates(
    options: list[FlightOption], *, limit: int = 12
) -> list[FlightOption]:
    """Keep a compact but meaningful set before paid baggage lookups.

    An option is discarded only when another is no more expensive, no longer, and
    has no more stops. From the remaining Pareto frontier, retain the extreme
    choices and fill the rest by departure-time diversity. This avoids making a
    detailed Duffel call for every search result while preserving real tradeoffs.
    """
    if len(options) <= limit:
        return options

    frontier = [
        option
        for option in options
        if not any(
            other is not option
            and other.price <= option.price
            and other.duration_minutes <= option.duration_minutes
            and other.stops <= option.stops
            and (
                other.price < option.price
                or other.duration_minutes < option.duration_minutes
                or other.stops < option.stops
            )
            for other in options
        )
    ]
    pool = frontier or options
    ranked = sorted(pool, key=lambda item: (item.price, item.duration_minutes, item.stops))
    if len(ranked) <= limit:
        return ranked

    selected: list[FlightOption] = []
    for extreme in (
        min(ranked, key=lambda item: item.price),
        min(ranked, key=lambda item: item.duration_minutes),
        min(ranked, key=lambda item: item.stops),
    ):
        if extreme not in selected:
            selected.append(extreme)

    # Time-of-day can be a user preference, so select evenly across departure order
    # instead of allowing the cheapest end of the frontier to crowd it out.
    by_departure = sorted(ranked, key=lambda item: item.departure)
    slots = limit - len(selected)
    for index in range(slots):
        candidate = by_departure[round(index * (len(by_departure) - 1) / max(slots - 1, 1))]
        if candidate not in selected:
            selected.append(candidate)
    for candidate in ranked:
        if len(selected) == limit:
            break
        if candidate not in selected:
            selected.append(candidate)
    return sorted(selected, key=lambda item: (item.price, item.duration_minutes, item.stops))


def baggage_requested(preferences: FlightSearchPreferences | None) -> bool:
    """Use a cheap, explicit gate before making per-offer ancillary requests."""
    return bool(preferences and preferences.checked_bags_per_traveler > 0)


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


class NoEligibleFlightOfferError(ValueError):
    """No offer met the traveller's explicit, non-relaxable constraints."""


class FlightSelectionDecision(BaseModel):
    """The one constrained decision returned by the flight-selector model."""

    offer_id: str = Field(
        description="Exact offer_id copied from one eligible candidate. Never invent an ID."
    )
    tradeoff_explanation: str = Field(
        min_length=1,
        max_length=320,
        description="Concise explanation of the decisive price, itinerary, airline, and baggage tradeoff."
    )


def _time_of_day(value: str) -> int | None:
    try:
        value_as_datetime = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value_as_datetime.hour * 60 + value_as_datetime.minute
    except ValueError:
        return None


def _clock_minutes(value: str) -> int:
    hours, minutes = value.split(":", maxsplit=1)
    return int(hours) * 60 + int(minutes)


def _included_bag_count(option: FlightOption) -> int:
    match = re.search(r"\b(\d+) checked bags? included", option.included_checked_baggage, re.I)
    return int(match.group(1)) if match else 0


def _matches_hard_constraints(
    option: FlightOption, preferences: FlightSearchPreferences
) -> bool:
    carrier_codes = set(option.carrier_iata_codes)
    departure_minutes = _time_of_day(option.departure)
    arrival_minutes = _time_of_day(option.arrival)
    if preferences.max_connections is not None and option.stops > preferences.max_connections:
        return False
    if (
        preferences.required_airline_iata_codes
        and not carrier_codes.intersection(preferences.required_airline_iata_codes)
    ):
        return False
    if carrier_codes.intersection(preferences.excluded_airline_iata_codes):
        return False
    if preferences.departure_time_from is not None and (
        departure_minutes is None
        or departure_minutes < _clock_minutes(preferences.departure_time_from)
    ):
        return False
    if preferences.departure_time_to is not None and (
        departure_minutes is None
        or departure_minutes > _clock_minutes(preferences.departure_time_to)
    ):
        return False
    if preferences.arrival_time_from is not None and (
        arrival_minutes is None or arrival_minutes < _clock_minutes(preferences.arrival_time_from)
    ):
        return False
    if preferences.arrival_time_to is not None and (
        arrival_minutes is None or arrival_minutes > _clock_minutes(preferences.arrival_time_to)
    ):
        return False
    return not preferences.require_included_checked_bags or (
        _included_bag_count(option) >= preferences.checked_bags_per_traveler
    )


def eligible_flight_options(
    candidates: list[FlightOption], preferences: FlightSearchPreferences
) -> list[FlightOption]:
    """Return only offers that satisfy explicit, non-relaxable preferences."""
    if not candidates:
        raise ValueError("Cannot select a flight from an empty candidate list.")
    eligible = [
        option for option in candidates if _matches_hard_constraints(option, preferences)
    ]
    if not eligible:
        raise NoEligibleFlightOfferError(
            "No flight offers met the explicit hard preferences; no constraints were relaxed."
        )
    return eligible


def _best_offer(option: FlightOption, reasoning: str) -> BestOffer:
    return BestOffer(
        offer_id=option.offer_id,
        price=option.price_text,
        currency=option.currency,
        offer_request_id=option.offer_request_id or None,
        expires_at=option.expires_at,
        departure=option.departure or None,
        arrival=option.arrival or None,
        duration_minutes=option.duration_minutes if option.duration_minutes < 10**6 else None,
        stops=option.stops,
        carriers=list(option.carriers),
        included_checked_baggage=option.included_checked_baggage,
        additional_checked_baggage=list(option.additional_checked_baggage),
        reasoning=reasoning,
    )


def select_deterministic_offer(
    *, candidates: list[FlightOption], preferences: FlightSearchPreferences
) -> BestOffer:
    """Select the cheapest eligible offer using stable, documented tie-breakers."""
    eligible = eligible_flight_options(candidates, preferences)

    def rank(option: FlightOption) -> tuple[float, int, int, int, str, str]:
        carrier_codes = set(option.carrier_iata_codes)
        preferred_carrier_penalty = int(
            bool(preferences.preferred_airline_iata_codes)
            and not carrier_codes.intersection(preferences.preferred_airline_iata_codes)
        )
        return (
            option.price,
            option.stops,
            option.duration_minutes,
            preferred_carrier_penalty,
            option.departure,
            option.offer_id,
        )

    option = min(eligible, key=rank)
    return _best_offer(
        option,
        (
            f"Selected deterministically from {len(eligible)} eligible offer(s): "
            "lowest price, then fewest stops, shortest duration, preferred airline, "
            "earliest departure, and offer ID."
        ),
    )


async def select_model_offer(
    *, selector_model: Any, candidates: list[FlightOption], preferences: FlightSearchPreferences
) -> tuple[BestOffer, int | None, int | None, str]:
    """Use one schema-constrained model call to weigh soft travel preferences."""
    eligible = eligible_flight_options(candidates, preferences)
    eligible_by_id = {option.offer_id: option for option in eligible}
    decision_model = selector_model.with_structured_output(
        FlightSelectionDecision,
        method="json_schema",
        strict=True,
        include_raw=True,
    )
    result = await decision_model.ainvoke([
        SystemMessage(content=(
            "Choose the best flight from the supplied eligible offers. Hard constraints have "
            "already been enforced. Weigh soft preferences and explain the decisive tradeoff. "
            "Treat words such as 'prefer', 'usually', and 'if reasonable' as flexible. Compare "
            "total fare, stops, duration, schedule, carrier, and checked-baggage facts. If an "
            "alternative is materially better, choose it even when it is not the preferred airline. "
            "Select only an offer_id provided in eligible_candidates. Keep the explanation under 45 words."
        )),
        HumanMessage(content=json.dumps({
            "traveller_preferences": preferences.model_dump(mode="json"),
            "eligible_candidates": [option.state() for option in eligible],
        })),
    ])
    decision = result.get("parsed") if isinstance(result, dict) else result
    if not isinstance(decision, FlightSelectionDecision):
        raise ValueError("Flight selector did not return a valid structured decision.")
    option = eligible_by_id.get(decision.offer_id)
    if option is None:
        raise ValueError("Flight selector chose an offer outside the eligible candidate set.")
    raw = result.get("raw") if isinstance(result, dict) else None
    usage = getattr(raw, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
    response_model = getattr(raw, "response_metadata", {}).get("model_name") if raw else None
    return _best_offer(option, decision.tradeoff_explanation), input_tokens, output_tokens, response_model or "gpt-5-mini"
