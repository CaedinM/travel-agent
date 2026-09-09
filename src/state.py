from langchain.agents import AgentState
from models import Leg

class TripState(AgentState):
    origin: str
    legs: list[Leg]
    season: str
    departure_date: str
    return_date: str
    trip_length: str
    budget: str
    flight_preferences: str
