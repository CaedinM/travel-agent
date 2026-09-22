"""Top-level trip-planning orchestrator."""

import os
from datetime import date

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from travel_agent.core.llm import POWERFUL_LLM
from travel_agent.core.models import (
    BestOffer,
    FlightPipelineMetric,
    HotelOption,
    Leg,
    Resources,
)
from travel_agent.core.state import INITIAL_TRIP_PHASE, TripPhase, TripState
from travel_agent.tools.trip import (
    call_legacy_flights_agent,
    confirm_flights,
    get_saved_flights_info,
    search_and_select_flights,
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
2. exact departure and return dates. If the user doesn't have them yet or is flexible,
ask for a  a travel month (save it as `season`) and desired `trip_length`.
3. the number of travellers.

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

When the traveller is ready to search, call `search_and_select_flights` once with no
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

    flight_tool = (
        "call_legacy_flights_agent"
        if os.environ.get("FLIGHT_PIPELINE", "jev").strip().lower() == "legacy"
        else "search_and_select_flights"
    )
    phase_prompt = PHASE_PROMPTS[phase].replace("search_and_select_flights", flight_tool)
    return f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n{phase_prompt}"


def build_orchestrator_agent():
    """Build the top-level agent that coordinates the trip workflow."""
    flight_search_tool = (
        call_legacy_flights_agent
        if os.environ.get("FLIGHT_PIPELINE", "jev").strip().lower() == "legacy"
        else search_and_select_flights
    )
    tools = [
        web_search,
        update_trip_info,
        set_trip_legs,
        flight_search_tool,
        get_saved_flights_info,
        confirm_flights,
    ]

    return create_agent(
        model=POWERFUL_LLM,
        tools=tools,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        # TripState nests our own pydantic models, which the checkpoint serializer
        # does not know. Declaring them keeps deserialization explicit rather than
        # relying on the permissive default, which now warns and will later block.
        checkpointer=InMemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[Leg, BestOffer, HotelOption, FlightPipelineMetric]
            )
        ),
        state_schema=TripState,
        context_schema=Resources,
        middleware=[orchestrator_dynamic_prompt],
    )
