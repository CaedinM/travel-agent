import asyncio
import json
from datetime import date
from functools import cache
from time import perf_counter

from dotenv import load_dotenv
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import Command
from tavily import TavilyClient

from travel_agent.core.models import BestOffer, FlightPipelineMetric, Leg
from travel_agent.core.workflow import apply_phase_transition
from travel_agent.flights.duffel import DuffelClient
from travel_agent.flights.flight_selector import (
    baggage_requested,
    enrich_baggage_details,
    parse_duffel_offers,
    prepare_candidates,
    select_best_offer,
)

load_dotenv()

@cache
def _tavily() -> TavilyClient:
    return TavilyClient()

@tool
def web_search(query: str) -> str:
    """Search the web for information"""
    return _tavily().search(query)


@tool
def update_trip_info(
    runtime: ToolRuntime,
    origin: str | None = None,
    travelers: int | None = None,
    season: str | None = None,
    departure_date: str | None = None,
    return_date: str | None = None,
    trip_length: str | None = None,
    budget: str | None = None
    ) -> Command:
    """Save any trip details the user has provided, including traveller count.
    Pass only the fields the user mentioned this turn and leave the rest as None.

    Destinations are not saved here — use set_trip_legs for the itinerary."""

    updates = {k: v for k, v in {
        "origin": origin,
        "travelers": travelers,
        "season": season,
        "departure_date": departure_date,
        "return_date": return_date,
        "trip_length": trip_length,
        "budget": budget,
        "messages": [ToolMessage("Success", tool_call_id=runtime.tool_call_id)]
    }.items() if v is not None}

    return Command(update=apply_phase_transition(runtime.state, updates))


def _leg_key(leg: Leg) -> tuple[str, str, str | None]:
    return (leg.origin.upper(), leg.destination.upper(), leg.departure_date)


def _describe_legs(legs: list[Leg]) -> str:
    lines = []
    for i, leg in enumerate(legs):
        when = leg.departure_date or "date not set"
        if leg.offer is None:
            status = "no offer yet"
        elif leg.offer.price:
            status = f"offer {leg.offer.offer_id} ({leg.offer.price} {leg.offer.currency or ''})".strip()
        else:
            status = f"offer {leg.offer.offer_id}"
        lines.append(f"  [{i}] {leg.origin} -> {leg.destination} on {when} — {status}")
    return "\n".join(lines)


@tool
def set_trip_legs(runtime: ToolRuntime, legs: list[Leg]) -> Command:
    """Set the full ordered itinerary as a list of one-way legs.

    This REPLACES the whole itinerary, so always pass every leg of the trip, not
    just the new ones. A simple return trip is two legs (out and back); a trip
    that ends at home must finish with a leg back to the origin.

    Leave `offer` unset — offers already found for legs that are unchanged
    (same origin, destination and departure date) are carried over automatically."""

    existing = {_leg_key(leg): leg.offer
                for leg in (runtime.state.get("legs") or [])
                if leg.offer is not None}

    merged = [
        leg if leg.offer is not None
        else leg.model_copy(update={"offer": existing.get(_leg_key(leg))})
        for leg in legs
    ]

    summary = f"Saved {len(merged)} leg(s):\n{_describe_legs(merged)}"

    updates = {
        "legs": merged,
        # Any itinerary replacement requires the traveller to reconfirm its offers.
        "flights_confirmed": False,
        "messages": [ToolMessage(summary, tool_call_id=runtime.tool_call_id)],
    }
    return Command(update=apply_phase_transition(runtime.state, updates))


def _mcp_json(raw) -> dict:
    """Unwrap an MCP tool result into the JSON object it carries.

    langchain-mcp-adapters returns a list of content blocks — typically
    [{"type": "text", "text": "<json>"}] — rather than the decoded payload."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        parts = []
        for block in raw:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if text:
                parts.append(text)
        raw = "".join(parts)
    return json.loads(raw)


def make_search_leg_tool(flights_tools_by_name: dict[str, BaseTool]) -> BaseTool:
    """Build the search tool the flights subagent gets."""
    from legacy.flight_options import dedupe, format_options, parse_offers

    search_flights = flights_tools_by_name["search_flights"]

    @tool
    async def search_leg(origin: str, destination: str, departure_date: str) -> str:
        """Search one-way flights for a single leg.

        Returns every distinct option in departure order with price, times,
        duration, stops and carrier. Call this once per leg."""
        raw = await search_flights.ainvoke({"params": {
            "type": "one_way",
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
        }})
        try:
            payload = _mcp_json(raw)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return f"Flight search returned an unreadable response: {str(raw)[:200]}"

        return format_options(dedupe(parse_offers(payload)))

    return search_leg


_MAX_CONCURRENT_SEARCHES = 3   # keeps clear of Duffel rate limits on long itineraries


async def _search_leg(
    agent, leg: Leg, semaphore: asyncio.Semaphore, preferences: str | None = None
    ) -> tuple[BestOffer | None, int, int | None, int | None]:
    started = perf_counter()
    message = (
        f"Find the best one-way flight from {leg.origin} to {leg.destination} "
        f"departing on {leg.departure_date}"
    )
    if preferences:
        message += (
            f"\n\nThe traveller's stated preferences for this trip: {preferences}\n"
            "Honour these when choosing. If nothing on this leg satisfies them, pick the "
            "closest option and say in your reasoning which preference you could not meet."
        )
    async with semaphore:
        result = await agent.ainvoke({"messages": [HumanMessage(content=message)]})
    input_tokens = 0
    output_tokens = 0
    found_usage = False
    for message in result.get("messages", []):
        usage = getattr(message, "usage_metadata", None) or {}
        if not isinstance(usage, dict):
            continue
        found_usage = True
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
    return (
        result.get("structured_response"),
        round((perf_counter() - started) * 1000),
        input_tokens if found_usage else None,
        output_tokens if found_usage else None,
    )


@tool
async def call_legacy_flights_agent(
    runtime: ToolRuntime,
    leg_index: int | None = None,
    preferences: str | None = None,
    ) -> Command | str:
    """Legacy benchmark: call the preserved MCP-backed flights subagent.

    Pass a leg_index to search (or re-search) just that leg. Omit it to fill in
    every leg that does not have an offer yet. Set the itinerary with
    set_trip_legs first.

    preferences: what the traveller wants in a flight, in plain English — for
    example "direct flights only", "nothing departing before 09:00", "prefer
    British Airways", "avoid long layovers". Pass this whenever the user has
    expressed a preference; it is saved and reused for later searches, so you
    only need to pass it again when their preferences change."""

    if runtime.context.pipeline != "legacy":
        return "Legacy flights subagent is disconnected. Set FLIGHT_PIPELINE=legacy to run it."

    legs = list(runtime.state.get("legs") or [])
    if not legs:
        return "Cannot search flights; no itinerary saved. Call set_trip_legs first."

    # A preference given this turn replaces the saved one; otherwise reuse what is
    # saved, so a follow-up re-search does not silently drop the user's criteria.
    saved_preferences = runtime.state.get("flight_preferences")
    preferences_changed = preferences is not None and preferences != saved_preferences
    preferences = preferences or saved_preferences

    if leg_index is None:
        if preferences_changed:
            # The criteria changed, so every leg needs re-picking — including legs
            # whose existing offer was chosen under the old criteria.
            targets = list(range(len(legs)))
        else:
            targets = [i for i, leg in enumerate(legs) if leg.offer is None]
            if not targets:
                return f"Every leg already has an offer:\n{_describe_legs(legs)}"
    else:
        if not 0 <= leg_index < len(legs):
            return f"No leg at index {leg_index}; the itinerary has {len(legs)} leg(s)."
        targets = [leg_index]

    # Check dates before spending an API call: Duffel rejects past dates with a bare
    # 422 whose explanation the MCP server discards, so catch them here instead.
    today = date.today()
    bad = []
    for i in targets:
        leg = legs[i]
        if not leg.departure_date:
            bad.append(f"[{i}] {leg.origin} -> {leg.destination}: no departure_date set.")
            continue
        try:
            when = date.fromisoformat(leg.departure_date)
        except ValueError:
            bad.append(f"[{i}] departure_date {leg.departure_date!r} is not a YYYY-MM-DD date.")
            continue
        if when <= today:
            bad.append(
                f"[{i}] departure_date {leg.departure_date} is not in the future "
                f"(today is {today.isoformat()})."
                )
    if bad:
        return (
            "Cannot search flights; fix these legs with set_trip_legs first:\n"
            + "\n".join(bad)
            )

    ## Search every target leg concurrently; one leg's failure must not sink the rest.
    # Created inside the call so it binds to this event loop.
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SEARCHES)
    results = await asyncio.gather(
        *(_search_leg(runtime.context.flights_agent, legs[i], semaphore, preferences)
          for i in targets),
        return_exceptions=True,
    )

    notes = []
    metrics = list(runtime.state.get("flight_pipeline_metrics") or [])
    for i, result in zip(targets, results):   # gather preserves input order
        leg = legs[i]
        if isinstance(result, Exception):
            notes.append(f"[{i}] {leg.origin} -> {leg.destination}: search failed ({result}).")
            metrics.append(FlightPipelineMetric(
                pipeline="legacy", leg_index=i, total_ms=0, error=str(result)
            ))
            continue
        pick, elapsed_ms, input_tokens, output_tokens = result
        metrics.append(FlightPipelineMetric(
            pipeline="legacy",
            leg_index=i,
            total_ms=elapsed_ms,
            selected_offer_id=pick.offer_id if pick else None,
            model="openai:gpt-5-mini",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=None if pick else "Subagent finished without selecting an offer.",
        ))
        if pick is None:
            notes.append(f"[{i}] {leg.origin} -> {leg.destination}: subagent finished without selecting an offer.")
        else:
            legs[i] = leg.model_copy(update={"offer": pick})

    summary = f"Itinerary:\n{_describe_legs(legs)}"
    if preferences:
        summary += f"\n\nSearched with preferences: {preferences}"
    if notes:
        summary += "\n\nProblems:\n" + "\n".join(notes)

    updates = {
        "legs": legs,
        "flight_pipeline_metrics": metrics[-100:],
        "messages": [ToolMessage(summary, tool_call_id=runtime.tool_call_id)],
        }
    if preferences:
        updates["flight_preferences"] = preferences

    return Command(update=apply_phase_transition(runtime.state, updates))


async def _search_and_select_direct_leg(
    *,
    client: DuffelClient,
    classifier,
    leg: Leg,
    leg_index: int,
    travelers: int,
    preferences: str | None,
    semaphore: asyncio.Semaphore,
) -> tuple[BestOffer | None, FlightPipelineMetric]:
    """Search one leg, then make a constrained Jev choice from its viable offers."""
    started = perf_counter()
    duffel_ms: int | None = None
    selection_ms: int | None = None
    candidates = 0
    try:
        duffel_started = perf_counter()
        async with semaphore:
            request = await client.search_one_way(
                origin=leg.origin,
                destination=leg.destination,
                departure_date=leg.departure_date or "",
                travelers=travelers,
            )
        duffel_ms = round((perf_counter() - duffel_started) * 1000)
        options = prepare_candidates(parse_duffel_offers(request))
        if baggage_requested(preferences):
            options = await enrich_baggage_details(client, options)
        candidates = len(options)
        if not options:
            raise ValueError("Duffel returned no viable flight offers.")

        selection_started = perf_counter()
        offer, confidence, input_tokens, output_tokens, model = await select_best_offer(
            classifier,
            origin=leg.origin,
            destination=leg.destination,
            departure_date=leg.departure_date or "",
            preferences=preferences,
            candidates=options,
        )
        selection_ms = round((perf_counter() - selection_started) * 1000)
        return offer, FlightPipelineMetric(
            pipeline="jev",
            leg_index=leg_index,
            duffel_ms=duffel_ms,
            selection_ms=selection_ms,
            total_ms=round((perf_counter() - started) * 1000),
            candidates=candidates,
            selected_offer_id=offer.offer_id,
            selection_confidence=confidence,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )
    except Exception as exc:  # Preserve partial success for the other legs.
        return None, FlightPipelineMetric(
            pipeline="jev",
            leg_index=leg_index,
            duffel_ms=duffel_ms,
            selection_ms=selection_ms,
            total_ms=round((perf_counter() - started) * 1000),
            candidates=candidates,
            error=str(exc),
        )


@tool
async def search_and_select_flights(
    runtime: ToolRuntime,
    leg_index: int | None = None,
    preferences: str | None = None,
) -> Command | str:
    """Search direct Duffel offers and select one per leg with Jev.

    Omit leg_index to fill every itinerary leg without an offer. Preferences are saved
    and reused; a new preference re-searches every leg because the old picks may no
    longer match the traveller's criteria.
    """
    if runtime.context.pipeline != "jev":
        return "Jev flight selection is disconnected. Set FLIGHT_PIPELINE=jev to run it."
    client = runtime.context.duffel_client
    classifier = runtime.context.classifier
    if not isinstance(client, DuffelClient) or classifier is None:
        return "Direct Duffel/Jev flight services are not configured."

    legs = list(runtime.state.get("legs") or [])
    if not legs:
        return "Cannot search flights; no itinerary saved. Call set_trip_legs first."
    travelers = runtime.state.get("travelers")
    if not isinstance(travelers, int) or travelers < 1:
        return "Cannot search flights until the traveller count is saved."

    saved_preferences = runtime.state.get("flight_preferences")
    preferences_changed = preferences is not None and preferences != saved_preferences
    preferences = preferences or saved_preferences
    if leg_index is None:
        targets = list(range(len(legs))) if preferences_changed else [
            index for index, leg in enumerate(legs) if leg.offer is None
        ]
        if not targets:
            return f"Every leg already has an offer:\n{_describe_legs(legs)}"
    elif 0 <= leg_index < len(legs):
        targets = [leg_index]
    else:
        return f"No leg at index {leg_index}; the itinerary has {len(legs)} leg(s)."

    today = date.today()
    invalid = []
    for index in targets:
        leg = legs[index]
        try:
            when = date.fromisoformat(leg.departure_date or "")
        except ValueError:
            invalid.append(f"[{index}] {leg.origin} -> {leg.destination}: departure date must be YYYY-MM-DD.")
            continue
        if when <= today:
            invalid.append(f"[{index}] departure date must be after {today.isoformat()}.")
    if invalid:
        return "Cannot search flights; fix these legs with set_trip_legs first:\n" + "\n".join(invalid)

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SEARCHES)
    results = await asyncio.gather(
        *(
            _search_and_select_direct_leg(
                client=client,
                classifier=classifier,
                leg=legs[index],
                leg_index=index,
                travelers=travelers,
                preferences=preferences,
                semaphore=semaphore,
            )
            for index in targets
        )
    )
    notes = []
    metrics = list(runtime.state.get("flight_pipeline_metrics") or [])
    for index, (offer, metric) in zip(targets, results):
        metrics.append(metric)
        if offer is None:
            notes.append(f"[{index}] {legs[index].origin} -> {legs[index].destination}: {metric.error}")
        else:
            legs[index] = legs[index].model_copy(update={"offer": offer})

    updates = {
        "legs": legs,
        # Keep the newest measurements while preventing unbounded checkpoint growth.
        "flight_pipeline_metrics": metrics[-100:],
        "messages": [ToolMessage(f"Itinerary:\n{_describe_legs(legs)}", tool_call_id=runtime.tool_call_id)],
    }
    if preferences:
        updates["flight_preferences"] = preferences
    if notes:
        updates["messages"] = [ToolMessage(
            f"Itinerary:\n{_describe_legs(legs)}\n\nProblems:\n" + "\n".join(notes),
            tool_call_id=runtime.tool_call_id,
        )]
    return Command(update=apply_phase_transition(runtime.state, updates))


@tool
def confirm_flights(runtime: ToolRuntime) -> Command | str:
    """Record that the traveller has accepted every saved flight offer.

    Call this only after the traveller explicitly confirms that the presented flights
    work for them. It does not accept a phase argument; the backend advances the
    workflow automatically when confirmation is valid.
    """
    legs = runtime.state.get("legs") or []
    if not legs or any(leg.offer is None for leg in legs):
        return (
            "Cannot confirm flights until every itinerary leg has a saved offer. "
            "Search any remaining legs first."
        )

    updates = {
        "flights_confirmed": True,
        "messages": [
            ToolMessage(
                "Flights confirmed. Moving on to accommodation planning.",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    }
    return Command(update=apply_phase_transition(runtime.state, updates))


@tool
async def get_saved_flights_info(runtime: ToolRuntime, leg_index: int | None = None) -> str:
    """Gets detailed information about the flight offers saved on the itinerary.

    Pass a leg_index for a single leg, or omit it for every leg that has an offer."""

    legs = runtime.state.get("legs") or []
    if not legs:
        return (
            "Can not get flight offer details; there is no itinerary saved. "
            "Call set_trip_legs, then call_flights_agent."
            )

    if leg_index is None:
        targets = [i for i, leg in enumerate(legs) if leg.offer is not None]
        if not targets:
            return (
                "Can not get flight offer details; no leg has an offer saved yet. "
                "Call the call_flights_agent tool first."
                )
    else:
        if not 0 <= leg_index < len(legs):
            return f"No leg at index {leg_index}; the itinerary has {len(legs)} leg(s)."
        if legs[leg_index].offer is None:
            return (
                f"Leg {leg_index} has no offer saved yet. "
                f"Call call_flights_agent with leg_index={leg_index} first."
                )
        targets = [leg_index]

    sections = []
    for i in targets:
        leg = legs[i]
        offer_id = leg.offer.offer_id
        header = f"[{i}] {leg.origin} -> {leg.destination} on {leg.departure_date}"

        try:
            if runtime.context.pipeline == "jev":
                client = runtime.context.duffel_client
                if not isinstance(client, DuffelClient):
                    raise RuntimeError("Direct Duffel client is not configured.")
                details = json.dumps(await client.get_offer(offer_id), indent=2)
            else:
                tools_by_name = runtime.context.flights_tools_by_name
                if tools_by_name is None:
                    raise RuntimeError("Legacy flights tools are not configured.")
                details = await tools_by_name["get_offer_details"].ainvoke(
                    {"params": {"offer_id": offer_id}}
                )
        except Exception as e:
            details = (
                f"Couldn't fetch details for offer {offer_id}: {e}."
                "The offer may have expired — consider re-running the flights search for this leg."
                )
        sections.append(f"{header}\n{details}")

    return "\n\n".join(sections)
