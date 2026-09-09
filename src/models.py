from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from dataclasses import dataclass

@dataclass
class Resources:
    flights_agent: object
    flights_tools_by_name: dict[str, BaseTool]


class BestOffer(BaseModel):
    """The subagent's answer: which offer it chose."""
    offer_id: str = Field(description="The exact offer_id string copied from a search_flights result. Never invent or reformat one.")
    reasoning: str = Field(description="One sentence on why this offer beats the alternatives.")
    price: str = Field(description="Required. The amount alone, copied from the chosen row's price column, e.g. '41.59'. No currency code.")
    currency: str = Field(description="Required. The chosen row's currency column value alone, e.g. 'USD'.")


class Leg(BaseModel):
    """One one-way hop of the itinerary."""
    origin: str = Field(description="Origin as a 3-letter IATA code, e.g. LHR or LON. Never a city name.")
    destination: str = Field(description="Destination as a 3-letter IATA code, e.g. CDG or PAR. Never a city name.")
    departure_date: str | None = Field(None, description="Departure date (YYYY-MM-DD), if the user has settled on one.")
    offer: BestOffer | None = Field(None, description="The flight offer found for this leg. Leave unset; the flights search fills it in.")
