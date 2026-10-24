"""Validation for reachable trip-state snapshots used by evaluation fixtures."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any


_PHASES = {"intake", "research", "flights", "accommodations"}


def _leg_value(leg: Any, name: str) -> Any:
    return leg.get(name) if isinstance(leg, Mapping) else None


def _is_iata(value: object) -> bool:
    return isinstance(value, str) and len(value) == 3 and value.isalpha()


def _has_date_plan(state: Mapping[str, Any]) -> bool:
    exact_dates = bool(state.get("origin") and state.get("departure_date") and state.get("return_date"))
    flexible_dates = bool(state.get("origin") and state.get("season") and state.get("trip_length"))
    return exact_dates or flexible_dates


def _itinerary_error(state: Mapping[str, Any], requires_return_home: bool) -> str | None:
    legs = state.get("legs") or []
    if not legs:
        return "requires at least one itinerary leg"
    origin = state.get("origin")
    if not _is_iata(origin):
        return "requires a three-letter origin when legs exist"
    previous_destination: str | None = None
    previous_date: date | None = None
    for index, leg in enumerate(legs):
        leg_origin = _leg_value(leg, "origin")
        destination = _leg_value(leg, "destination")
        departure_date = _leg_value(leg, "departure_date")
        if not _is_iata(leg_origin) or not _is_iata(destination):
            return f"leg {index} must have three-letter origin and destination codes"
        if index == 0 and leg_origin.upper() != origin.upper():
            return "first leg must leave from the saved origin"
        if previous_destination and leg_origin.upper() != previous_destination:
            return f"leg {index} must start where leg {index - 1} ends"
        try:
            parsed_date = date.fromisoformat(departure_date)
        except (TypeError, ValueError):
            return f"leg {index} must have a concrete ISO departure date"
        if parsed_date <= date.today():
            return f"leg {index} must be in the future"
        if previous_date and parsed_date <= previous_date:
            return f"leg {index} must depart after leg {index - 1}"
        previous_destination, previous_date = destination.upper(), parsed_date
    if requires_return_home and previous_destination != origin.upper():
        return "return-trip itinerary must end at the saved origin"
    return None


def state_error(state: Mapping[str, Any], *, requires_return_home: bool) -> str | None:
    """Return an explanation when a fixture state cannot arise in this workflow."""
    phase = state.get("phase", "intake")
    if phase not in _PHASES:
        return f"has unknown phase {phase!r}"
    origin = state.get("origin")
    if origin is not None and not _is_iata(origin):
        return "origin must be a three-letter airport or metro code"
    travelers = state.get("travelers")
    if travelers is not None and (not isinstance(travelers, int) or travelers < 1):
        return "travelers must be a positive integer"

    departure_date = state.get("departure_date")
    return_date = state.get("return_date")
    if bool(departure_date) != bool(return_date):
        return "concrete dates must include both departure_date and return_date"
    if departure_date:
        try:
            departure = date.fromisoformat(departure_date)
            returning = date.fromisoformat(return_date)
        except (TypeError, ValueError):
            return "concrete dates must be future YYYY-MM-DD values"
        if departure <= date.today() or returning <= departure:
            return "concrete dates must be future and ordered"

    legs = state.get("legs") or []
    if phase == "intake":
        if legs:
            return "intake cannot already contain an itinerary"
        if _has_date_plan(state):
            return "intake with a complete date plan should already be in research"
    elif phase == "research":
        if not _has_date_plan(state):
            return "research requires origin and exact dates or a flexible date plan"
        if legs:
            return "research cannot retain an itinerary that would advance it to flights"
    else:
        if not _has_date_plan(state):
            return f"{phase} requires a completed research date plan"
        itinerary_error = _itinerary_error(state, requires_return_home)
        if itinerary_error:
            return itinerary_error

    has_offers = bool(legs) and all(_leg_value(leg, "offer") for leg in legs)
    confirmed = bool(state.get("flights_confirmed"))
    if phase == "accommodations":
        if not (has_offers and confirmed):
            return "accommodations requires confirmed offers for every leg"
    elif confirmed:
        return f"{phase} cannot have confirmed flights"
    return None


def validate_state(state: Mapping[str, Any], *, requires_return_home: bool) -> None:
    """Raise when an evaluation fixture contains a state production cannot reach."""
    error = state_error(state, requires_return_home=requires_return_home)
    if error:
        raise ValueError(f"Unreachable evaluation state: {error}.")
