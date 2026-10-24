from datetime import date
from enum import StrEnum
from typing import Annotated

from langchain.agents import AgentState
from pydantic import BaseModel, Field, field_validator, model_validator

from travel_agent.core.models import FlightPipelineMetric, FlightSearchPreferences, HotelOption, Leg


class TripPhase(StrEnum):
    """The deterministic stages of the trip-planning workflow."""

    INTAKE = "intake"
    RESEARCH = "research"
    FLIGHTS = "flights"
    ACCOMMODATIONS = "accommodations"


INITIAL_TRIP_PHASE = TripPhase.INTAKE


def validate_future_date(value: str | None, field_name: str) -> date:
    """Parse an exact YYYY-MM-DD calendar date and require it to be future."""
    try:
        parsed_date = date.fromisoformat(value or "")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a real calendar date formatted exactly as YYYY-MM-DD.") from exc
    if parsed_date.isoformat() != value:
        raise ValueError(f"{field_name} must be formatted exactly as YYYY-MM-DD.")
    today = date.today()
    if parsed_date <= today:
        raise ValueError(f"{field_name} must be after {today.isoformat()}.")
    return parsed_date


def validate_trip_dates(departure_date: str | None, return_date: str | None) -> None:
    """Require concrete future Duffel-compatible dates when dates are supplied."""
    parsed: dict[str, date] = {}
    for field_name, value in (("departure_date", departure_date), ("return_date", return_date)):
        if value is None:
            continue
        parsed[field_name] = validate_future_date(value, field_name)
    if parsed.get("departure_date") and parsed.get("return_date"):
        if parsed["return_date"] <= parsed["departure_date"]:
            raise ValueError("return_date must be after departure_date.")


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
        FlightSearchPreferences | None,
        Field(description="Typed flight-search constraints collected from the traveller."),
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

    @field_validator("flight_preferences", mode="before")
    @classmethod
    def migrate_legacy_preference_text(cls, value):
        if isinstance(value, str):
            return FlightSearchPreferences(summary=value)
        return value

    @model_validator(mode="after")
    def validate_dates(self):
        validate_trip_dates(self.departure_date, self.return_date)
        return self


class TripSnapshot(BaseModel):
    """Validated, message-free planning state stored as a durable trip record."""

    phase: TripPhase = INITIAL_TRIP_PHASE
    origin: str | None = None
    legs: list[Leg] = Field(default_factory=list)
    hotels: dict[str, list[HotelOption]] = Field(default_factory=dict)
    departure_date: str | None = None
    return_date: str | None = None
    travelers: int | None = Field(default=None, ge=1)
    season: str | None = None
    trip_length: str | None = None
    budget: str | None = None
    flight_preferences: FlightSearchPreferences | None = None
    flight_pipeline_metrics: list[FlightPipelineMetric] = Field(default_factory=list)
    flights_confirmed: bool = False

    @field_validator("flight_preferences", mode="before")
    @classmethod
    def migrate_legacy_preference_text(cls, value):
        if isinstance(value, str):
            return FlightSearchPreferences(summary=value)
        return value

    @model_validator(mode="after")
    def validate_dates(self):
        validate_trip_dates(self.departure_date, self.return_date)
        return self
