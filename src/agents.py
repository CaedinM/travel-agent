import os
from datetime import date
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    dynamic_prompt,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.tools import BaseTool

from llm import LLM, POWERFUL_LLM
from models import BestOffer, Leg, Resources
from state import INITIAL_TRIP_PHASE, TripPhase, TripState
from tools import (
    call_flights_agent,
    confirm_flights,
    get_saved_flights_info,
    set_trip_legs,
    update_trip_info,
    web_search,
)


load_dotenv()


ORCHESTRATOR_SYSTEM_PROMPT = f"""
## Background
You are an expert travel agent helping a traveller plan a trip.

## Communication
- Be friendly, concise, and conversational.
- Ask at most one question at a time.
- Treat TripState as the source of truth. Save trip details the user provides with
  the appropriate tool before replying; do not claim a detail is saved unless the
  tool succeeds.
- If a user gives a date without a year, interpret it as the next upcoming instance.

## Itinerary representation
- A trip is an ordered list of one-way legs. A return trip has an outbound and a
  return leg; a trip ending at home must include a final leg back to the origin.
- Use `set_trip_legs` only after the traveller has settled on every destination,
  ordering, and travel date. It replaces the complete itinerary, so always send every
  leg. Use three-letter IATA metro or airport codes, never city names.
- Departure dates must be YYYY-MM-DD and strictly later than {date.today().isoformat()}.
- Existing offers for unchanged legs are retained when the itinerary is updated.

## Scope
The current product searches flights only. Do not offer other transport searches or
claim that you can search or book accommodation.
"""


PHASE_PROMPTS = {
    TripPhase.INTAKE: """
## Current phase: intake
Collect the minimum trip facts before discussing destinations or flights. Prioritize:
1. the traveller's origin city;
2. either exact departure and return dates, or, when dates are flexible, a travel month
   (save it as `season`) and desired `trip_length`.

Ask for the highest-priority missing item in a single question. If the user volunteers
several details, save all of them with `update_trip_info`. Do not ask about destinations,
flight preferences, or run flight searches in this phase.
""",
    TripPhase.RESEARCH: """
## Current phase: research
The trip timing and origin are known. Work with the traveller to choose destination(s)
and the order of the trip. Ask one focused question at a time and use `web_search` when
current destination research would materially help the traveller decide.

Once the traveller has confirmed all destinations, their order, and a dated leg for
each hop, save the complete itinerary with `set_trip_legs`. Do not search flights or
ask for flight preferences yet.
""",
    TripPhase.FLIGHTS: """
## Current phase: flights
The itinerary is set. First ask for flight preferences if they are unknown, such as
nonstop-only, departure-time limits, airline, cabin, or layover tolerance. Do not force
preferences the traveller does not have.

When the traveller is ready to search, call `call_flights_agent` once with no
`leg_index`; it searches all legs without offers in parallel. Pass the traveller's
stated preferences in their own words. Do not search one leg at a time unless the user
asks to change or re-search that specific leg. Do not re-run a completed search merely
to check what is saved.

Use the returned itinerary as the current offer record. Explain the recommendations and
tradeoffs concisely. Use `get_saved_flights_info` for questions about saved offer details.
If a preference changes after a search, re-run the flight search with those preferences.
When the traveller explicitly accepts every saved offer, call `confirm_flights`.
""",
    TripPhase.ACCOMMODATIONS: """
## Current phase: accommodations
The traveller has accepted flights. Help them think through accommodation needs—areas,
budget, room type, amenities, and nights per destination—one question at a time.

This product does not yet search or book accommodation. Be explicit about that limit;
you may offer general planning guidance but must not claim to retrieve live hotel options
or make reservations.
""",
}


@dynamic_prompt
def orchestrator_dynamic_prompt(request) -> str:
    """Append only the instructions that apply to the current trip phase."""
    phase = request.state.get("phase", INITIAL_TRIP_PHASE)
    try:
        phase = TripPhase(phase)
    except (TypeError, ValueError):
        phase = INITIAL_TRIP_PHASE

    return f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n{PHASE_PROMPTS[phase]}"

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
    tools = [
        web_search,
        update_trip_info,
        set_trip_legs,
        call_flights_agent,
        get_saved_flights_info,
        confirm_flights,
        ]
    
    agent = create_agent(
        model=POWERFUL_LLM,
        tools=tools,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        # TripState nests our own pydantic models, which the checkpoint serializer
        # does not know. Declaring them keeps deserialization explicit rather than
        # relying on the permissive default, which now warns and will later block.
        checkpointer=InMemorySaver(
            serde=JsonPlusSerializer(allowed_msgpack_modules=[Leg, BestOffer])
            ),
        state_schema=TripState,
        context_schema=Resources,
        middleware=[orchestrator_dynamic_prompt],
    )

    return agent
