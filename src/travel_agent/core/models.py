from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


DuffelCabinClass = Literal["economy", "premium_economy", "business", "first"]


class FlightSearchPreferences(BaseModel):
    """Typed traveller preferences used in Duffel searches and deterministic ranking."""

    cabin_class: DuffelCabinClass | None = Field(
        default=None,
        description=(
            "Required cabin for Duffel's `cabin_class` filter. Use exactly one of: "
            "economy, premium_economy, business, first. Leave null when unspecified."
        ),
    )
    max_connections: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Hard maximum connections per flight slice for Duffel's `max_connections` "
            "filter. Use 0 only for nonstop-only. Leave null when connections are acceptable."
        ),
    )
    departure_time_from: str | None = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description=(
            "Earliest departure time for Duffel's slice `departure_time.from`, exactly "
            "HH:MM in 24-hour time; otherwise null. For a fuzzy request such as 'not "
            "too early', use 08:00 unless the traveller gives a clearer constraint."
        ),
    )
    departure_time_to: str | None = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description="Latest departure time for Duffel's slice `departure_time.to`, exactly HH:MM in 24-hour time; otherwise null.",
    )
    arrival_time_from: str | None = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description="Earliest arrival time for Duffel's slice `arrival_time.from`, exactly HH:MM in 24-hour time; otherwise null.",
    )
    arrival_time_to: str | None = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description=(
            "Latest arrival time for Duffel's slice `arrival_time.to`, exactly HH:MM "
            "in 24-hour time; otherwise null. For a fuzzy request such as 'do not land "
            "late', use 20:00 unless the traveller gives a clearer constraint."
        ),
    )
    required_airline_iata_codes: list[str] = Field(
        default_factory=list,
        description="Hard airline requirement. Use uppercase two-character IATA airline codes, e.g. ['UA', 'AS']; empty means no requirement.",
    )
    excluded_airline_iata_codes: list[str] = Field(
        default_factory=list,
        description="Airlines to exclude after search. Use uppercase two-character IATA airline codes, e.g. ['NK']; empty means none.",
    )
    preferred_airline_iata_codes: list[str] = Field(
        default_factory=list,
        description="Soft airline tie-breaker. Use uppercase two-character IATA airline codes; empty means no preference.",
    )
    checked_bags_per_traveler: int = Field(
        default=0,
        ge=0,
        description="Number of checked bags each traveller expects to bring; 0 when none or unknown.",
    )
    require_included_checked_bags: bool = Field(
        default=False,
        description="True only when the checked-bag allowance is a hard requirement; false means bag cost/allowance is a preference, not a filter.",
    )
    summary: str = Field(
        default="No additional preferences stated.",
        max_length=500,
        description="Brief factual explanation of the structured values, for display only. Do not put unstructured requirements here.",
    )

    @field_validator(
        "required_airline_iata_codes",
        "excluded_airline_iata_codes",
        "preferred_airline_iata_codes",
    )
    @classmethod
    def validate_airline_codes(cls, values: list[str]) -> list[str]:
        import re

        codes = [value.strip().upper() for value in values]
        if any(not re.fullmatch(r"[A-Z0-9]{2}", code) for code in codes):
            raise ValueError("Airline codes must be uppercase two-character IATA codes.")
        return list(dict.fromkeys(codes))

    @field_validator("arrival_time_to")
    @classmethod
    def normalize_end_of_day_arrival(cls, value: str | None) -> str | None:
        """Keep a conversational 'by midnight' preference usable by Duffel."""
        return "23:59" if value == "00:00" else value

    @model_validator(mode="after")
    def validate_time_windows(self) -> "FlightSearchPreferences":
        for lower, upper, label in (
            (self.departure_time_from, self.departure_time_to, "departure"),
            (self.arrival_time_from, self.arrival_time_to, "arrival"),
        ):
            if lower and upper and lower > upper:
                raise ValueError(f"{label}_time_from must be at or before {label}_time_to.")
        if self.require_included_checked_bags and self.checked_bags_per_traveler == 0:
            raise ValueError(
                "checked_bags_per_traveler must be at least 1 when included bags are required."
            )
        return self

    def applied_filter_summary(self) -> str:
        """Return the exact deterministic filters applied to a flight search."""
        filters: list[str] = []
        if self.cabin_class:
            filters.append(f"cabin: {self.cabin_class.replace('_', ' ')}")
        if self.max_connections is not None:
            filters.append(
                "nonstop only" if self.max_connections == 0
                else f"at most {self.max_connections} connection(s)"
            )
        if self.departure_time_from or self.departure_time_to:
            filters.append(
                "departure " + "–".join(
                    [self.departure_time_from or "start of day", self.departure_time_to or "end of day"]
                )
            )
        if self.arrival_time_from or self.arrival_time_to:
            filters.append(
                "arrival " + "–".join(
                    [self.arrival_time_from or "start of day", self.arrival_time_to or "end of day"]
                )
            )
        if self.required_airline_iata_codes:
            filters.append("required airline: " + ", ".join(self.required_airline_iata_codes))
        if self.excluded_airline_iata_codes:
            filters.append("excluded airlines: " + ", ".join(self.excluded_airline_iata_codes))
        if self.preferred_airline_iata_codes:
            filters.append("preferred airline: " + ", ".join(self.preferred_airline_iata_codes))
        if self.checked_bags_per_traveler:
            bag_filter = f"{self.checked_bags_per_traveler} checked bag(s) per traveller"
            if self.require_included_checked_bags:
                bag_filter += ", included"
            filters.append(bag_filter)
        detail = "; ".join(filters) if filters else "no additional flight filters"
        if self.summary != "No additional preferences stated.":
            return f"{detail}. Assumption/summary: {self.summary}"
        return detail


@dataclass
class Resources:
    """Runtime services for the production direct-Duffel flight workflow."""

    pipeline: str
    duffel_client: object | None = None
    flight_selector_model: object | None = None
    flight_prefetch: object | None = None


class BestOffer(BaseModel):
    """The subagent's answer: which offer it chose."""
    offer_id: str = Field(description="The exact offer_id string copied from a search_flights result. Never invent or reformat one.")
    reasoning: str = Field(description="One sentence on why this offer beats the alternatives.")
    price: str = Field(description="Required. The amount alone, copied from the chosen row's price column, e.g. '41.59'. No currency code.")
    currency: str = Field(description="Required. The chosen row's currency column value alone, e.g. 'USD'.")
    offer_request_id: str | None = Field(
        default=None,
        description="Duffel offer-request ID that produced this offer, when using the direct Duffel pipeline.",
    )
    expires_at: str | None = Field(
        default=None,
        description="Duffel's ISO-8601 offer expiry time, when supplied by the provider.",
    )
    departure: str | None = Field(default=None, description="Scheduled departure timestamp.")
    arrival: str | None = Field(default=None, description="Scheduled final-arrival timestamp.")
    duration_minutes: int | None = Field(default=None, ge=0)
    stops: int | None = Field(default=None, ge=0)
    carriers: list[str] = Field(default_factory=list)
    included_checked_baggage: str | None = None
    additional_checked_baggage: list[str] = Field(default_factory=list)


class FlightPipelineMetric(BaseModel):
    """One per-leg measurement for the bounded flight-selection workflow."""

    pipeline: str
    leg_index: int
    duffel_ms: int | None = None
    baggage_enrichment_ms: int | None = None
    selection_ms: int | None = None
    total_ms: int
    candidates: int = 0
    selected_offer_id: str | None = None
    selection_confidence: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    error: str | None = None


class HotelOption(BaseModel):
    """One bookable room/rate returned for an itinerary destination.

    A property can have several options with different room types, prices, or
    cancellation terms. Each option is stored separately so the traveller can choose
    the exact rate they want.
    """

    offer_id: str = Field(
        description="Provider's exact identifier for this bookable room/rate option."
    )
    hotel_id: str = Field(description="Provider's exact identifier for the property.")
    hotel_name: str = Field(description="Name of the hotel or accommodation property.")
    room_type: str = Field(description="Exact room type included in this option.")
    check_in: str = Field(description="Stay check-in date in YYYY-MM-DD format.")
    check_out: str = Field(description="Stay check-out date in YYYY-MM-DD format.")
    total_price: str = Field(
        description="Total price for this room/rate, excluding the currency code."
    )
    currency: str = Field(description="ISO 4217 currency code for total_price, e.g. USD.")
    booking_url: str = Field(description="Provider URL for this exact bookable option.")
    address: str | None = Field(
        default=None, description="Property street address or neighborhood, when available."
    )
    city: str | None = Field(default=None, description="Property city, when available.")
    country: str | None = Field(default=None, description="Property country, when available.")
    stars: float | None = Field(
        default=None, ge=0, description="Official property star rating, when available."
    )
    amenities: list[str] = Field(
        default_factory=list, description="Amenities confirmed for the property or room."
    )
    meal_plan: str | None = Field(
        default=None, description="Included meal plan, when the provider supplies one."
    )
    cancellation_policy: str | None = Field(
        default=None, description="Cancellation or refund terms for this exact rate."
    )
    image_url: str | None = Field(
        default=None, description="Primary property image URL, when available."
    )
    hotel_photo_urls: list[str] = Field(
        default_factory=list, description="Additional property-level photo URLs."
    )
    room_photo_urls: list[str] = Field(
        default_factory=list, description="Photo URLs for this option's specific room."
    )


class Leg(BaseModel):
    """One one-way hop of the itinerary."""
    origin: str = Field(description="Origin as a 3-letter IATA code, e.g. LHR or LON. Never a city name.")
    destination: str = Field(description="Destination as a 3-letter IATA code, e.g. CDG or PAR. Never a city name.")
    departure_date: str | None = Field(None, description="Future calendar date formatted exactly as YYYY-MM-DD. Required when saving itinerary legs; never a month, season, or timestamp.")
    offer: BestOffer | None = Field(None, description="The flight offer found for this leg. Leave unset; the flights search fills it in.")
