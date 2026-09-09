import os
from datetime import date
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.tools import BaseTool

from llm import LLM, POWERFUL_LLM
from models import BestOffer, Leg, Resources
from state import TripState
from tools import web_search, update_trip_info, set_trip_legs, call_flights_agent, get_saved_flights_info


load_dotenv()

# Flights subagent
def build_flights_agent(search_leg: BaseTool, get_offer_details: BaseTool):
    system_prompt = """
    You are a flight-search specialist. You are given exactly one one-way hop and
    you decide which flight is best for it.

    Call search_leg ONCE. It returns every distinct option for that leg in departure
    order, with price, times, duration, stops and carrier. Nothing in that list is
    ranked or recommended — the judgment is entirely yours.

    Weigh the tradeoffs:
    - Price matters most, but it is not the only thing. A cheaper fare is a bad deal
      if it costs the traveller hours of their day or lands at an unusable hour.
    - A departure before 06:00, or an arrival after 23:00, is a real cost. Prefer a
      civilised time unless the saving is substantial.
    - A non-stop beats a connection at a similar price. One long layover can be worth
      a large saving; two connections rarely are.
    - Watch for outliers. The cheapest row is sometimes a 20-hour multi-stop routing.

    If the request states the traveller's preferences, they outrank the guidance above
    — they are what this particular traveller wants. Apply them to the options you can
    see. If no option satisfies them, do not keep searching: choose the closest one and
    say plainly in your reasoning which preference you could not meet and why.

    If two or three options are genuinely close, you may call get_offer_details on
    those specific offers to compare baggage allowance and fare conditions before
    deciding. Use it sparingly — never walk the whole list.

    Then answer with the offer_id, price and currency copied exactly from the row you
    chose — all three are required, and price is the number alone without the currency
    code — plus one sentence naming the specific tradeoff that decided it. Never invent
    an offer_id. You do not plan itineraries and you do not book anything.
    """

    flights_agent = create_agent(
        model=LLM,
        tools=[search_leg, get_offer_details],
        system_prompt=system_prompt,
        response_format=ToolStrategy(
            BestOffer,
            handle_errors="That didn't match the required format. Return the exact offer_id string from one of your search results."
            ),
        middleware=[
            # Backstops. The prompt should keep these unreachable; they exist so a
            # confused run degrades into a reported failure instead of a hang.
            ToolCallLimitMiddleware(tool_name="search_leg", run_limit=2, exit_behavior="continue"),
            ToolCallLimitMiddleware(tool_name="get_offer_details", run_limit=3, exit_behavior="continue"),
            ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
        ],
    )

    return flights_agent

# Main orchestrator agent
def build_orchestrator_agent():
    system_prompt = f"""
    You are an expert travel agent. Help the user plan an amazing trip.
    Always be friendly and keep responses concise and conversational.
    Ask questions and find out more about the user's preferences and practical restraints
    before proposing plans. However, do not bombard the user with multiple questions at once.

    Use the web_search tool when needed to find up to date information on potential destinations. 

    ## Updating State
    - If a user gives a trip detail (origin, season, dates, trip length, budget), call
    update_trip_info to save it before replying.

    ## The itinerary
    - A trip is an ordered list of one-way legs. Even a simple return trip is two legs:
    out and back. A trip that ends at home must finish with a leg back to the origin.
    - Once the user has settled on where they are going, call set_trip_legs with the
    whole itinerary. It REPLACES what is saved, so always pass every leg — including
    the ones that have not changed — not just the new ones.
    - Legs must use 3-letter IATA codes, never city names: London is LON (or LHR),
    Paris is PAR (or CDG), New York is NYC (or JFK). Translate what the user says
    into codes yourself; a city name will be rejected by the flight search.
    - Departure dates must be YYYY-MM-DD and strictly in the future. Today is
    {date.today().isoformat()}. If the user names a date without a year, always use
    the next upcoming instance of that date.
    - Offers already found for unchanged legs are carried over for you, so you can add
    or reorder destinations without losing the flights already found.

    ## Flights
    - Once the itinerary has dates, call call_flights_agent ONCE with no arguments.
    A single call searches every leg that does not have an offer yet, in parallel —
    do not call it once per leg.
    - set_trip_legs and call_flights_agent both return the full itinerary, listing each
    leg and the offer saved against it. That listing is the current state: read it
    instead of calling a tool again to find out what is saved. Never call a tool
    merely to check.
    - Searching is slow and costs money, so only call call_flights_agent when a leg
    genuinely has no offer. If the last listing showed an offer on every leg, the
    search is done — answer from that listing. In particular, do not re-run it after
    set_trip_legs unless the listing shows a leg still without an offer.
    - Re-search a leg only when the user asks for a different flight, or a leg failed.
    Pass that leg's leg_index, call it once, and use the result.
    - If the user says what they want in a flight — direct only, no early departures,
    a particular airline, short layovers, a cabin class — pass it as the preferences
    argument, in their own words. It is saved and reused for later searches, so pass
    it again only when their preferences change.
    - A preference the user gives after a search is a reason to search again: pass the
    new preferences and let it re-pick. Say which preference could not be met if the
    subagent reports one.
    - If the user has questions about a flight, call get_saved_flights_info for detailed
    offer information — with a leg_index for one leg, or with no arguments for all of them.

    ## Notes
    - You currently only support finding flights between destinations. Do not ask the user if
    they want to take any other type of transportation method.
    """
    tools = [
        web_search,
        update_trip_info,
        set_trip_legs,
        call_flights_agent,
        get_saved_flights_info
        ]
    
    agent = create_agent(
        model=POWERFUL_LLM,
        tools=tools,
        system_prompt=system_prompt,
        # TripState nests our own pydantic models, which the checkpoint serializer
        # does not know. Declaring them keeps deserialization explicit rather than
        # relying on the permissive default, which now warns and will later block.
        checkpointer=InMemorySaver(
            serde=JsonPlusSerializer(allowed_msgpack_modules=[Leg, BestOffer])
            ),
        state_schema=TripState,
        context_schema=Resources
    )

    return agent