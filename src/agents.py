import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver 
from langchain_core.tools import BaseTool

from models import BestOffer, Resources
from state import TripState
from tools import web_search, update_trip_info, call_flights_agent, get_saved_flights_info


load_dotenv()

# Flights subagent
def build_flights_agent(tools: list[BaseTool]):
    system_prompt = """
    You are a flight-search specialist. Your goal is to find the cheapest flight options
    for the given origin, destination and dates using the search-flights tool.

    You can call get_offer_details to see more detailed information about flights.

    You do not plan itineraries or book anything. You only look for flights.

    When you have found the best offer, return the exact offer_id from the search 
    results along with one sentence on why it beats the alternatives. Never invent an 
    offer_id.
    """

    flights_agent = create_agent(
        model="gpt-5-nano",
        tools=tools,
        system_prompt=system_prompt,
        response_format=ToolStrategy(
            BestOffer,
            handle_errors="That didn't match the required format. Return the exact offer_id string from one of your search results."
            ),

    )

    return flights_agent

# Main orchestrator agent
def build_orchestrator_agent():
    system_prompt = """
    You are an expert travel agent. Help the user plan an amazing trip.
    Always be friendly and keep responses concise and conversational.
    Ask questions and find out more about the user's preferences and practical restraints
    before proposing plans. However, do not bombard the user with multiple questions at once.

    Use the web_search tool when needed to find up to date information on potential destinations. 

    ## Updating State
    - If a user gives a trip detail (destination, season, dates, budget), call update_trip_info
    to save it before replying.

    ## Flights
    - If a user has decided on a destination and wants to find flights, use the call_flights_agent
    tool to find a round-trip flight offer. This tool will search for flights and save the best offer
    in state. You can access full details of this offer by calling get_saved_flights_info.
    - If the user has questions about the flight offer, call get_saved_flights_info to get detailed 
    flight offer information.

    ## Notes
    - If the user states a departure and return time, Always assume that the intended departure and return 
    dates are in the next upcoming instance 
    of that date.
    """
    tools = [
        web_search,
        update_trip_info,
        call_flights_agent,
        get_saved_flights_info
        ]
    
    agent = create_agent(
        model="gpt-5-nano",
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
        state_schema=TripState,
        context_schema=Resources
    )

    return agent