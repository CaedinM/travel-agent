"""Backend-owned trip workflow transitions."""

from collections.abc import Mapping
from typing import Any

from state import INITIAL_TRIP_PHASE, TripPhase


def _field(value: object, name: str) -> Any:
    """Read a field from either a Pydantic state model or a serialized mapping."""
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def next_phase(state: Mapping[str, Any]) -> TripPhase:
    """Return the next valid phase for a fully merged TripState.

    This only advances the workflow. Handling changes that invalidate already-planned
    work (for example, new travel dates) will be added as explicit regression rules.
    """
    try:
        phase = TripPhase(state.get("phase", INITIAL_TRIP_PHASE))
    except (TypeError, ValueError):
        phase = INITIAL_TRIP_PHASE

    if phase is TripPhase.INTAKE:
        has_exact_dates = all(
            state.get(field)
            for field in ("origin", "departure_date", "return_date")
        )
        has_flexible_dates = all(
            state.get(field)
            for field in ("origin", "season", "trip_length")
        )
        if has_exact_dates or has_flexible_dates:
            return TripPhase.RESEARCH

    if phase is TripPhase.RESEARCH:
        legs = state.get("legs") or []
        if legs and all(_field(leg, "departure_date") for leg in legs):
            return TripPhase.FLIGHTS

    if phase is TripPhase.FLIGHTS:
        legs = state.get("legs") or []
        every_leg_has_offer = bool(legs) and all(_field(leg, "offer") for leg in legs)
        if state.get("flights_confirmed") and every_leg_has_offer:
            return TripPhase.ACCOMMODATIONS

    return phase


def apply_phase_transition(
    current_state: Mapping[str, Any], updates: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge a tool update with state and append the backend-computed phase."""
    updated = dict(updates)
    updated["phase"] = next_phase({**current_state, **updated})
    return updated
