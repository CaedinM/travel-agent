import asyncio
from datetime import date
from functools import cache
from time import perf_counter
from typing import Annotated

from dotenv import load_dotenv
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import Field
from tavily import TavilyClient

from travel_agent.core.models import (
    BestOffer,
    FlightPipelineMetric,
    FlightSearchPreferences,
    Leg,
)
from travel_agent.core.state import validate_future_date, validate_trip_dates
from travel_agent.core.workflow import apply_phase_transition
from travel_agent.flights.duffel import DuffelClient
from travel_agent.flights.prefetch import FlightPrefetchCache
from travel_agent.flights.flight_selector import (
    baggage_requested,
    enrich_baggage_details,
    parse_duffel_offers,
    prepare_baggage_candidates,
    prepare_candidates,
    select_model_offer,
)

load_dotenv()

@cache
def _tavily() -> TavilyClient:
    return TavilyClient()

@tool
def web_search(query: str) -> str:
    """Search the web for information"""
    return _tavily().search(query)


def _date_validation_error(runtime: ToolRuntime, detail: str) -> Command:
    """Reject the whole update while giving the planner a corrective tool error."""
    return Command(update={"messages": [ToolMessage(
        content=(
            f"{detail} Provide a real calendar date formatted exactly as YYYY-MM-DD, "
            "for example 2027-05-10. Store flexible wording such as 'Spring' in "
            "`season`; retry once concrete dates are chosen."
        ),
        status="error",
        tool_call_id=runtime.tool_call_id,
    )]})


@tool
def update_trip_info(
    runtime: ToolRuntime,
    origin: str | None = None,
    travelers: int | None = None,
    season: str | None = None,
    departure_date: Annotated[str | None, Field(
        description="Outbound calendar date, exactly YYYY-MM-DD and strictly after today; today and past dates are rejected. Store months or seasons in season instead."
    )] = None,
    return_date: Annotated[str | None, Field(
        description="Return calendar date, exactly YYYY-MM-DD, strictly after today and after departure_date; today and past dates are rejected. Store months or seasons in season instead."
    )] = None,
    trip_length: str | None = None,
    budget: str | None = None
    ) -> Command:
    """Save any trip details the user has provided, including traveller count.
    Pass only the fields the user mentioned this turn and leave the rest as None.

    Destinations are not saved here — use set_trip_legs for the itinerary. Dates
    must be YYYY-MM-DD values strictly after today's date, with return_date after
    departure_date. Today and past dates cause a tool error without saving any
    fields. Store flexible wording such as a month or season in `season` instead."""

    proposed_departure = departure_date if departure_date is not None else runtime.state.get("departure_date")
    proposed_return = return_date if return_date is not None else runtime.state.get("return_date")
    try:
        validate_trip_dates(proposed_departure, proposed_return)
    except ValueError as exc:
        return _date_validation_error(runtime, f"Cannot save trip dates: {exc}")

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
    Every leg requires a future calendar departure_date formatted exactly as
    YYYY-MM-DD; store flexible months or seasons with update_trip_info instead.

    Leave `offer` unset — offers already found for legs that are unchanged
    (same origin, destination and departure date) are carried over automatically."""

    departure_dates: list[date] = []
    invalid_dates: list[str] = []
    for index, leg in enumerate(legs):
        try:
            departure_dates.append(validate_future_date(leg.departure_date, "departure_date"))
        except ValueError as exc:
            invalid_dates.append(
                f"[{index}] {leg.origin} -> {leg.destination}: {exc}"
            )
    if invalid_dates:
        return _date_validation_error(
            runtime,
            "Cannot save the itinerary until every leg has a future departure date:\n" + "\n".join(invalid_dates),
        )

    for index, (current_leg, next_leg) in enumerate(zip(legs, legs[1:])):
        current_date = departure_dates[index]
        next_date = departure_dates[index + 1]
        if next_date <= current_date:
            return _date_validation_error(
                runtime,
                "Cannot save the itinerary: each later leg must depart after the prior leg. "
                f"Leg [{index + 1}] {next_leg.origin} -> {next_leg.destination} on "
                f"{next_leg.departure_date} must be after leg [{index}] on "
                f"{current_leg.departure_date}."
            )

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


_MAX_CONCURRENT_SEARCHES = 3   # keeps clear of Duffel rate limits on long itineraries
async def _search_and_select_direct_leg(
    *,
    client: DuffelClient,
    selector_model,
    leg: Leg,
    leg_index: int,
    travelers: int,
    preferences: FlightSearchPreferences,
    semaphore: asyncio.Semaphore,
) -> tuple[BestOffer | None, FlightPipelineMetric]:
    """Search one leg, then make one bounded model choice from viable offers."""
    started = perf_counter()
    duffel_ms: int | None = None
    baggage_enrichment_ms: int | None = None
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
                preferences=preferences,
            )
        duffel_ms = round((perf_counter() - duffel_started) * 1000)
        options = prepare_candidates(parse_duffel_offers(request))
        if baggage_requested(preferences):
            baggage_started = perf_counter()
            options = await enrich_baggage_details(
                client, prepare_baggage_candidates(options)
            )
            baggage_enrichment_ms = round((perf_counter() - baggage_started) * 1000)
        candidates = len(options)
        if not options:
            raise ValueError("Duffel returned no viable flight offers.")

        selection_started = perf_counter()
        offer, input_tokens, output_tokens, model = await select_model_offer(
            selector_model=selector_model, preferences=preferences, candidates=options,
        )
        selection_ms = round((perf_counter() - selection_started) * 1000)
        return offer, FlightPipelineMetric(
            pipeline="model_selection",
            leg_index=leg_index,
            duffel_ms=duffel_ms,
            baggage_enrichment_ms=baggage_enrichment_ms,
            selection_ms=selection_ms,
            total_ms=round((perf_counter() - started) * 1000),
            candidates=candidates,
            selected_offer_id=offer.offer_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )
    except Exception as exc:  # Preserve partial success for the other legs.
        return None, FlightPipelineMetric(
            pipeline="model_selection",
            leg_index=leg_index,
            duffel_ms=duffel_ms,
            baggage_enrichment_ms=baggage_enrichment_ms,
            selection_ms=selection_ms,
            total_ms=round((perf_counter() - started) * 1000),
            candidates=candidates,
            error=str(exc),
        )


async def select_from_prefetched_requests(
    *, client: DuffelClient, selector_model, legs: list[Leg], travelers: int,
    preferences: FlightSearchPreferences,
    requests: dict[int, dict | Exception],
    include_selected: bool = False,
) -> tuple[list[Leg], list[FlightPipelineMetric], list[str]]:
    """Select from already-started Duffel requests without involving the chat agent."""
    async def select_one(index: int, leg: Leg):
        started = perf_counter()
        request = requests.get(index)
        if isinstance(request, Exception) or not isinstance(request, dict):
            return None, FlightPipelineMetric(
                pipeline="model_selection", leg_index=index, total_ms=round((perf_counter() - started) * 1000),
                error=str(request or "No prefetched offers available."),
            )
        baggage_ms: int | None = None
        try:
            options = prepare_candidates(parse_duffel_offers(request))
            if baggage_requested(preferences):
                baggage_started = perf_counter()
                options = await enrich_baggage_details(client, prepare_baggage_candidates(options))
                baggage_ms = round((perf_counter() - baggage_started) * 1000)
            if not options:
                raise ValueError("Duffel returned no viable flight offers.")
            selection_started = perf_counter()
            offer, input_tokens, output_tokens, model = await select_model_offer(
                selector_model=selector_model, preferences=preferences, candidates=options,
            )
            return offer, FlightPipelineMetric(
                pipeline="model_selection", leg_index=index, baggage_enrichment_ms=baggage_ms,
                selection_ms=round((perf_counter() - selection_started) * 1000),
                total_ms=round((perf_counter() - started) * 1000), candidates=len(options),
                selected_offer_id=offer.offer_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
            )
        except Exception as exc:
            return None, FlightPipelineMetric(
                pipeline="model_selection", leg_index=index, baggage_enrichment_ms=baggage_ms,
                total_ms=round((perf_counter() - started) * 1000), candidates=len(options) if 'options' in locals() else 0,
                error=str(exc),
            )

    targets = [
        (index, leg)
        for index, leg in enumerate(legs)
        if include_selected or leg.offer is None
    ]
    results = await asyncio.gather(*(select_one(index, leg) for index, leg in targets))
    updated = list(legs)
    metrics: list[FlightPipelineMetric] = []
    notes: list[str] = []
    for (index, leg), (offer, metric) in zip(targets, results):
        metrics.append(metric)
        if offer is None:
            notes.append(f"[{index}] {leg.origin} -> {leg.destination}: {metric.error}")
        else:
            updated[index] = leg.model_copy(update={"offer": offer})
    return updated, metrics, notes


@tool
async def choose_prefetched_flights(
    runtime: ToolRuntime, preferences: FlightSearchPreferences | None = None
) -> Command | str:
    """Choose flights with one bounded model decision from prefetched Duffel options.

    Call this only after the traveller's airline, cabin, connection, timing, and
    baggage preferences are sufficiently clear. Pass a concise, normalized summary
    of those preferences—not a bare conversational answer such as "yes".
    """
    if runtime.context.pipeline != "deterministic":
        return "Flight selection is unavailable."
    client = runtime.context.duffel_client
    selector_model = runtime.context.flight_selector_model
    cache = runtime.context.flight_prefetch
    legs = list(runtime.state.get("legs") or [])
    travelers = runtime.state.get("travelers")
    if not isinstance(client, DuffelClient) or selector_model is None or not isinstance(cache, FlightPrefetchCache):
        return "Direct Duffel flight services are not configured."
    if not legs or not isinstance(travelers, int) or travelers < 1:
        return "Cannot select flights until the itinerary and traveller count are saved."

    saved_preferences = runtime.state.get("flight_preferences")
    saved_preferences = (
        FlightSearchPreferences.model_validate(saved_preferences)
        if saved_preferences
        else None
    )
    normalized_preferences = preferences or saved_preferences or FlightSearchPreferences()
    preferences_changed = normalized_preferences != saved_preferences
    key = runtime.config.get("configurable", {}).get("thread_id")
    if not isinstance(key, str):
        return "Cannot identify this trip's prefetched flight options."
    requests = await cache.take(
        key, legs=legs, travelers=travelers, preferences=normalized_preferences
    )
    if requests is None:
        # A changed preference after a prior selection needs fresh offers for every
        # leg; the initial background prefetch only covers legs without offers.
        cache.start(
            key,
            client=client,
            legs=legs,
            travelers=travelers,
            preferences=normalized_preferences,
            include_selected=True,
        )
        requests = await cache.take(
            key, legs=legs, travelers=travelers, preferences=normalized_preferences
        )
    if requests is None:
        return "Could not start the flight search. Please try again."

    selected_legs, metrics, notes = await select_from_prefetched_requests(
        client=client, selector_model=selector_model, legs=legs, travelers=travelers,
        preferences=normalized_preferences,
        requests=requests,
        include_selected=preferences_changed,
    )
    updates = {
        "legs": selected_legs,
        "flight_preferences": normalized_preferences,
        "flights_confirmed": False,
        "flight_pipeline_metrics": [
            *(runtime.state.get("flight_pipeline_metrics") or []), *metrics
        ][-100:],
        "messages": [ToolMessage(
            f"Itinerary:\n{_describe_legs(selected_legs)}" + (
                "\n\nProblems:\n" + "\n".join(notes) if notes else ""
            ) + "\n\nApplied flight filters: " + normalized_preferences.applied_filter_summary() +
            "\nTell the traveller these filters, including any assumption, and offer to revise them.",
            tool_call_id=runtime.tool_call_id,
        )],
    }
    return Command(update=apply_phase_transition(runtime.state, updates))


@tool
async def search_and_select_flights(
    runtime: ToolRuntime,
    leg_index: int | None = None,
    preferences: FlightSearchPreferences | None = None,
) -> Command | str:
    """Search direct Duffel offers and select one per leg with a bounded model call.

    Omit leg_index to fill every itinerary leg without an offer. Preferences are saved
    and reused; a new preference re-searches every leg because the old picks may no
    longer match the traveller's criteria.
    """
    if runtime.context.pipeline != "deterministic":
        return "Flight selection is unavailable."
    client = runtime.context.duffel_client
    selector_model = runtime.context.flight_selector_model
    if not isinstance(client, DuffelClient) or selector_model is None:
        return "Direct Duffel flight services are not configured."

    legs = list(runtime.state.get("legs") or [])
    if not legs:
        return "Cannot search flights; no itinerary saved. Call set_trip_legs first."
    travelers = runtime.state.get("travelers")
    if not isinstance(travelers, int) or travelers < 1:
        return "Cannot search flights until the traveller count is saved."

    saved_preferences = runtime.state.get("flight_preferences")
    saved_preferences = (
        FlightSearchPreferences.model_validate(saved_preferences)
        if saved_preferences
        else None
    )
    preferences_changed = preferences is not None and preferences != saved_preferences
    preferences = preferences or saved_preferences or FlightSearchPreferences()
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
                selector_model=selector_model,
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
        "messages": [ToolMessage(
            f"Itinerary:\n{_describe_legs(legs)}\n\nApplied flight filters: "
            f"{preferences.applied_filter_summary()}\nTell the traveller these filters, "
            "including any assumption, and offer to revise them.",
            tool_call_id=runtime.tool_call_id,
        )],
    }
    if preferences:
        updates["flight_preferences"] = preferences
    if notes:
        updates["messages"] = [ToolMessage(
            f"Itinerary:\n{_describe_legs(legs)}\n\nProblems:\n" + "\n".join(notes) +
            f"\n\nApplied flight filters: {preferences.applied_filter_summary()}\n"
            "Tell the traveller these filters, including any assumption, and offer to revise them.",
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
            "Call set_trip_legs, then choose_prefetched_flights."
            )

    if leg_index is None:
        targets = [i for i, leg in enumerate(legs) if leg.offer is not None]
        if not targets:
            return (
                "Can not get flight offer details; no leg has an offer saved yet. "
                "Call the choose_prefetched_flights tool first."
                )
    else:
        if not 0 <= leg_index < len(legs):
            return f"No leg at index {leg_index}; the itinerary has {len(legs)} leg(s)."
        if legs[leg_index].offer is None:
            return (
                f"Leg {leg_index} has no offer saved yet. "
                f"Choose flights for that leg first."
                )
        targets = [leg_index]

    sections = []
    for i in targets:
        leg = legs[i]
        offer_id = leg.offer.offer_id
        header = f"[{i}] {leg.origin} -> {leg.destination} on {leg.departure_date}"

        try:
            client = runtime.context.duffel_client
            if not isinstance(client, DuffelClient):
                raise RuntimeError("Direct Duffel client is not configured.")
            details = json.dumps(await client.get_offer(offer_id), indent=2)
        except Exception as e:
            details = (
                f"Couldn't fetch details for offer {offer_id}: {e}."
                "The offer may have expired — consider re-running the flights search for this leg."
                )
        sections.append(f"{header}\n{details}")

    return "\n\n".join(sections)
