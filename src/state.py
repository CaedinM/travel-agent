from langchain.agents import AgentState
from models import BestOffer

class TripState(AgentState):
    origin: str
    destination: str
    season: str
    departure_date: str
    return_date: str
    trip_length: str
    budget: str
    flights_offer: BestOffer