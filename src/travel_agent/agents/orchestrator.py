"""Top-level trip-planning orchestrator."""

from datetime import date

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from travel_agent.core.llm import LLM
from travel_agent.core.models import (
    BestOffer,
    FlightPipelineMetric,
    FlightSearchPreferences,
    HotelOption,
    Leg,
    Resources,
)
from travel_agent.core.responses import DestinationSuggestion, PlannerResponse
from travel_agent.core.state import INITIAL_TRIP_PHASE, TripPhase, TripState
from travel_agent.tools.trip import (
    choose_prefetched_flights,
    confirm_flights,
    get_saved_flights_info,
    set_trip_legs,
    update_trip_info,
    web_search,
)

load_dotenv()


ORCHESTRATOR_SYSTEM_PROMPT = """
## Background
You are an expert travel agent helping a traveller plan a trip.

## Communication
- Be friendly, concise, and conversational.
- Ask at most one question at a time.
- Destination cards belong only in a reply presenting destination options and asking
  the traveller to choose or refine them. Keep `destination_suggestions` empty while
  asking for a departure airport, dates, traveller count, or other trip details.
  Never attach cards as a preview alongside an intake or clarification question.
- Treat TripState as the source of truth. Save trip details the user provides with
  the appropriate tool before replying; do not claim a detail is saved unless the
  tool succeeds.
- If a user gives a date without a year, interpret it as the next upcoming instance.

## Settle trip dates during planning
- Never suggest or save departure or return dates that are today or in the past.
  Both dates must be strictly after the current date provided for this turn, and
  the return date must be after the departure date. Check this before proposing dates.
- For a month or season without a year, use its next future occurrence. If this
  year's requested period has already passed, propose dates in the next year.
  If the traveller explicitly requests past dates, explain that they have passed
  and work with them to choose future dates rather than saving them.
- Ask for specific departure and return dates if either is missing. If the traveller
  has already supplied specific dates, save them without asking them to repeat or
  reconfirm those dates.
- If the traveller cannot provide specific dates, collect any missing general
  timeframe and desired trip length, one question at a time. Save flexible wording
  in `season` and the desired duration in `trip_length`.
- Use those preferences to suggest a concrete future departure and return date pair,
  state both dates clearly, and ask the traveller to confirm them. Do not save dates
  you suggested into departure_date or return_date until the traveller accepts them.
- If the traveller rejects or revises the proposal, work with their feedback and
  suggest revised dates. Continue until specific departure and return dates are
  agreed, then save them with `update_trip_info` in YYYY-MM-DD format.
- Resolve missing or unconfirmed dates before destination research or flight search.
  A season and trip length alone do not mean specific dates have been settled.

## Itinerary representation
- A trip is an ordered list of one-way legs. A return trip has an outbound and a
  return leg; a trip ending at home must include a final leg back to the origin.
- Use `set_trip_legs` only after the traveller has settled on every destination,
  ordering, and travel date. It replaces the complete itinerary, so always send every
  leg. Use three-letter IATA metro or airport codes, never city names.
- Departure dates must be YYYY-MM-DD and strictly later than today's date.
- Existing offers for unchanged legs are retained when the itinerary is updated.

## Scope
The current product searches flights only. Do not offer other transport searches or
claim that you can search or book accommodation.
"""


PHASE_PROMPTS = {
    TripPhase.INTAKE: """
## Current phase: intake
Converse with the user and collect the minimum trip facts
Collect the minimum trip facts before discussing destinations or flights. Prioritize:
1. the traveller's origin city;
2. exact departure and return dates. Date fields accept only future YYYY-MM-DD values,
never a month or season.
3. If the user cannot provide specific dates, follow the date proposal and confirmation
process above. Keep working with them until both dates are settled; do not save your
own suggested dates before confirmation.
4. the number of travellers.

Ask for the highest-priority missing item in a single question. If the user volunteers
several details, save all of them with `update_trip_info`. Do not ask about destinations,
flight preferences, or run flight searches in this phase.
""",
    TripPhase.RESEARCH: """
## Current phase: research
Check the saved trip timing first. If only a season or trip length is known, or either
specific date is missing, finish the date proposal and confirmation process above
before researching destinations. Once specific dates are settled and saved, work with
the traveller to choose destination(s) and the order of the trip. Ask one focused
question at a time and use `web_search` when current destination research would
materially help the traveller decide.

Before suggesting destinations, ensure the origin, exact departure and return dates,
and traveller count are saved. Ask for any missing detail first, without cards.
When presenting destinations for the traveller to choose or refine, populate the structured
`destination_suggestions` output with at most three well-matched options. Keep
the text response to a brief introduction and one focused question; do not repeat
the option details in prose or provide a long catalogue.

Once the traveller has confirmed all destinations, their order, and a dated leg for
each hop, save the complete itinerary with `set_trip_legs`. Do not search flights or
ask for flight preferences yet.
""",
    TripPhase.FLIGHTS: """
## Current phase: flights
The itinerary is set. First ask for flight preferences if they are unknown, such as
nonstop-only, departure/arrival-time limits, airline, or layover tolerance. Do not force
preferences the traveller does not have. Use judgment to normalize fuzzy preferences
without an unnecessary follow-up: for example, interpret "I don't want to fly too early
or land late" as departure_time_from="08:00" and arrival_time_to="20:00". State that
assumption in the preference summary and in your response after selection, so the traveller
can correct it. Ask a follow-up only when a reasonable default would materially change the trip.

Once the traveller's preferences are clear, call `choose_prefetched_flights` once to get
the best flight offer. Pass the `preferences` object with only confirmed facts. Use the
field descriptions exactly: cabin_class is one of economy, premium_economy, business, or
first; time fields are 24-hour HH:MM; and airline fields contain uppercase two-character
IATA codes. Use max_connections=0 for nonstop-only and leave unspecified preferences at
their defaults. Do not encode requirements only in the summary field.
Do not search one leg at a time unless the user asks to change or re-search that specific leg.

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

    return (
        f"Current date for this turn: {date.today().isoformat()}. "
        "All proposed and saved departure and return dates must be strictly after this date.\n\n"
        f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n{PHASE_PROMPTS[phase]}"
    )


def build_orchestrator_agent(checkpointer=None):
    """Build the top-level agent that coordinates the trip workflow."""
    tools = [
        web_search,
        update_trip_info,
        set_trip_legs,
        choose_prefetched_flights,
        get_saved_flights_info,
        confirm_flights,
    ]
    if checkpointer is None:
        # The command-line demo has no service lifecycle in which to own a
        # database-backed saver. Production always supplies AsyncPostgresSaver.
        checkpointer = InMemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[
                    Leg,
                    BestOffer,
                    FlightSearchPreferences,
                    HotelOption,
                    FlightPipelineMetric,
                    TripPhase,
                    DestinationSuggestion,
                    PlannerResponse,
                ]
            )
        )

    return create_agent(
        model=LLM,
        tools=tools,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        # TripState nests our own pydantic models, which the checkpoint serializer
        # does not know. Declaring them keeps deserialization explicit rather than
        # relying on the permissive default, which now warns and will later block.
        checkpointer=checkpointer,
        state_schema=TripState,
        context_schema=Resources,
        middleware=[orchestrator_dynamic_prompt],
        response_format=PlannerResponse,
    )
