from enum import StrEnum
from typing import Annotated

from langchain.agents import AgentState
from pydantic import Field

from travel_agent.core.models import FlightPipelineMetric, HotelOption, Leg


class TripPhase(StrEnum):
    """The deterministic stages of the trip-planning workflow."""

    INTAKE = "intake"
    RESEARCH = "research"
    FLIGHTS = "flights"
    ACCOMMODATIONS = "accommodations"


INITIAL_TRIP_PHASE = TripPhase.INTAKE


class TripState(AgentState):
    """Persistent trip-planning details for one conversation."""
    phase: Annotated[
        TripPhase,
        Field(
            default=INITIAL_TRIP_PHASE,
            description="The current deterministic stage of the trip-planning workflow.",
        ),
    ]
    origin: Annotated[
        str | None,
        Field(description="Traveller's home or departure airport/metro IATA code, e.g. LON or JFK."),
    ]
    legs: Annotated[
        list[Leg],
        Field(
            default_factory=list,
            description="Complete ordered itinerary of one-way legs, including a return leg home when applicable.",
        ),
    ]
    hotels: Annotated[
        dict[str, list[HotelOption]],
        Field(
            default_factory=dict,
            description=(
                "Bookable hotel options grouped by a stable itinerary-leg key in the "
                "form 'ORIGIN:DESTINATION:DEPARTURE_DATE', for example "
                "'LHR:CDG:2026-05-14'. Each option is a distinct room/rate, even "
                "when multiple options belong to the same hotel."
            ),
        ),
    ]
    departure_date: Annotated[
        str | None,
        Field(description="Overall outbound date in YYYY-MM-DD format, if known."),
    ]
    return_date: Annotated[
        str | None,
        Field(description="Overall return date in YYYY-MM-DD format, if known."),
    ]
    travelers: Annotated[
        int | None,
        Field(default=None, ge=1, description="Number of people travelling, if known."),
    ]
    season: Annotated[
        str | None,
        Field(description="Preferred travel season when exact dates are not settled, e.g. 'spring 2027'."),
    ]
    trip_length: Annotated[
        str | None,
        Field(description="Desired trip duration in natural language, e.g. 'one week'."),
    ]
    budget: Annotated[
        str | None,
        Field(description="Flight budget including scope and currency, e.g. 'under $900 total for flights'."),
    ]
    flight_preferences: Annotated[
        str | None,
        Field(description="Flight-selection constraints in the traveller's own words."),
    ]
    flight_pipeline_metrics: Annotated[
        list[FlightPipelineMetric],
        Field(
            default_factory=list,
            description="Recent per-leg timings and token usage for the active flight-selection pipeline.",
        ),
    ]
    flights_confirmed: Annotated[
        bool,
        Field(
            default=False,
            description="Whether the traveller has accepted the saved flight offers.",
        ),
    ]
