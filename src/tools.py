import os
import json
import asyncio
from dotenv import load_dotenv
from langchain.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from langchain.tools import tool, ToolRuntime
from langchain_core.tools import BaseTool
from langgraph.types import Command
from functools import cache
from datetime import date

from models import Leg, BestOffer
from flight_options import parse_offers, dedupe, format_options

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
    season: str | None = None,
    departure_date: str | None = None,
    return_date: str | None = None,
    trip_length: str | None = None,
    budget: str | None = None
    ) -> Command:
    """Save any trip details the user has provided. Pass only the fields
    the user mentioned this turn and leave the rest as None.

    Destinations are not saved here — use set_trip_legs for the itinerary."""

    updates = {k: v for k, v in {
        "origin": origin,
        "season": season,
        "departure_date": departure_date,
        "return_date": return_date,
        "trip_length": trip_length,
        "budget": budget,
        "messages": [ToolMessage("Success", tool_call_id=runtime.tool_call_id)]
    }.items() if v is not None}

    return Command(update=updates)


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

    return Command(update={
        "legs": merged,
        "messages": [ToolMessage(summary, tool_call_id=runtime.tool_call_id)],
        })


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
    ) -> BestOffer | None:
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
    return result.get("structured_response")


@tool
async def call_flights_agent(
    runtime: ToolRuntime,
    leg_index: int | None = None,
    preferences: str | None = None,
    ) -> Command | str:
    """Call the flights subagent to find a flight offer for one or more legs.

    Pass a leg_index to search (or re-search) just that leg. Omit it to fill in
    every leg that does not have an offer yet. Set the itinerary with
    set_trip_legs first.

    preferences: what the traveller wants in a flight, in plain English — for
    example "direct flights only", "nothing departing before 09:00", "prefer
    British Airways", "avoid long layovers". Pass this whenever the user has
    expressed a preference; it is saved and reused for later searches, so you
    only need to pass it again when their preferences change."""

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
    for i, pick in zip(targets, results):   # gather preserves input order
        leg = legs[i]
        if isinstance(pick, Exception):
            notes.append(f"[{i}] {leg.origin} -> {leg.destination}: search failed ({pick}).")
        elif pick is None:
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
        "messages": [ToolMessage(summary, tool_call_id=runtime.tool_call_id)],
        }
    if preferences:
        updates["flight_preferences"] = preferences

    return Command(update=updates)


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
            details = await runtime.context.flights_tools_by_name["get_offer_details"].ainvoke(
                {"params": {"offer_id": offer_id}}
                )
        except Exception as e:
            details = (
                f"Couldn't fetch details for offer {offer_id}: {e}."
                "The offer may have expired — consider re-running the flights search for this leg."
                )
        sections.append(f"{header}\n{details}")

    return "\n\n".join(sections)
