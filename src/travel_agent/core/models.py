from dataclasses import dataclass

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


@dataclass
class Resources:
    """Runtime services for one selected flight pipeline.

    The legacy MCP/subagent resources remain optional so a benchmark run can opt into
    them, but normal production runs use the direct Duffel and Jev clients.
    """

    pipeline: str
    duffel_client: object | None = None
    classifier: object | None = None
    flight_prefetch: object | None = None
    flights_agent: object | None = None
    flights_tools_by_name: dict[str, BaseTool] | None = None


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
    """One per-leg measurement for comparing the Jev and legacy pipelines."""

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
    departure_date: str | None = Field(None, description="Departure date (YYYY-MM-DD), if the user has settled on one.")
    offer: BestOffer | None = Field(None, description="The flight offer found for this leg. Leave unset; the flights search fills it in.")
