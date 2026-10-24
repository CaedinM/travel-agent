"""Typed, non-persistent response data returned to chat clients."""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from travel_agent.core.state import TripPhase


class DestinationSuggestion(BaseModel):
    """One destination proposed while the traveller is still choosing a trip."""

    country: str = Field(description="Country containing the suggested city.")
    city: str = Field(description="Suggested city or destination area.")
    description: str = Field(
        max_length=280,
        description="Concise description of the destination and its travel practicality.",
    )
    reasoning: str | None = Field(
        default=None,
        max_length=280,
        description="Optional tie to the traveller's stated preferences.",
    )


class PlannerResponse(BaseModel):
    """A final planner reply with optional, bounded destination choices."""

    response: str = Field(
        max_length=800,
        description="Concise conversational reply. Do not repeat destination details in this text.",
    )
    destination_suggestions: list[DestinationSuggestion] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Zero to three destination suggestions, only in the research phase after "
            "origin, exact departure and return dates, and traveller count are saved. "
            "Include cards only when this reply presents destination options and asks "
            "the traveller to choose or refine them. Return an empty list when asking "
            "for a departure airport, dates, traveller count, or other trip details."
        ),
    )

    def for_trip_state(self, state: Mapping[str, Any]) -> "PlannerResponse":
        """Withhold destination cards until the saved trip is ready for research."""
        ready_for_destinations = state.get("phase") == TripPhase.RESEARCH and all(
            state.get(field)
            for field in ("origin", "departure_date", "return_date", "travelers")
        )
        if ready_for_destinations:
            return self
        return self.model_copy(update={"destination_suggestions": []})
