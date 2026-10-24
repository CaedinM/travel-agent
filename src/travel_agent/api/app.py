"""HTTP interface for the travel-planning agent."""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager

import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel, Field

from travel_agent.agents.orchestrator import build_orchestrator_agent
from travel_agent.api.auth import current_user_id
from travel_agent.api.security import content_security_policy
from travel_agent.api.webhooks import handle_clerk_webhook
from travel_agent.core.models import BestOffer, FlightPipelineMetric, FlightSearchPreferences
from travel_agent.core.models import HotelOption, Leg
from travel_agent.core.responses import DestinationSuggestion, PlannerResponse
from travel_agent.core.state import INITIAL_TRIP_PHASE, TripPhase, TripSnapshot
from travel_agent.core.token_limits import (
    CHAT_TOKEN_RESERVATION,
    FLIGHT_SELECTION_TOKENS_PER_LEG,
    FREE_MONTHLY_TOKEN_LIMIT,
    PAID_MONTHLY_TOKEN_LIMIT,
)
from travel_agent.flights.duffel import DuffelClient
from travel_agent.infrastructure.database import (
    create_resumed_thread,
    create_trip_and_thread,
    initialize_database,
    list_trips,
    load_trip_state,
    save_trip_state,
    set_trip_status,
    reserve_monthly_tokens,
    settle_monthly_tokens,
    trip_for_thread,
)
from travel_agent.infrastructure.resources import open_resources, validate_chat_message_length
from travel_agent.infrastructure.phoenix import configure_runtime_tracing, runtime_chat_turn
from travel_agent.tools.trip import select_from_prefetched_requests

load_dotenv()
redis_url = os.environ["REDIS_URL"]
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    """A message sent by a frontend user to the travel agent."""

    message: str = Field(min_length=1, description="The user's message to the agent.")
    thread_id: str | None = Field(
        default=None,
        description="An ID returned by a previous request to continue that conversation.",
    )
    trip_id: str | None = Field(
        default=None,
        description="An existing trip to resume in a new chat thread.",
    )


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    trip_id: str
    destination_suggestions: list[DestinationSuggestion] = Field(default_factory=list)


class TripSummaryResponse(BaseModel):
    trip_id: str
    title: str
    status: str
    state_version: int
    created_at: str
    updated_at: str


class ResumeTripResponse(BaseModel):
    trip_id: str
    thread_id: str


class FlightSelectionRequest(BaseModel):
    """Preferences submitted after the flight-search prefetch has begun."""

    preferences: FlightSearchPreferences = Field(
        default_factory=FlightSearchPreferences,
        description="Structured preferences using Duffel's filter vocabulary.",
    )


class ClerkConfigResponse(BaseModel):
    """The safe-to-expose Clerk configuration needed by the browser SDK."""

    publishable_key: str


class TripProgressResponse(BaseModel):
    """The small, client-safe subset of a trip used by the planning sidebar."""

    thread_id: str
    phase: TripPhase
    origin: str | None = None
    departure_date: str | None = None
    return_date: str | None = None
    season: str | None = None
    destinations: list[str] = Field(default_factory=list)
    travelers: int | None = None
    flights_ready: bool = False
    accommodations_ready: bool = False
    direct_flight_search: bool = False
    flight_offers: list["FlightOfferCardResponse"] = Field(default_factory=list)


class FlightOfferCardResponse(BaseModel):
    """The selected offer facts needed by the chat's visual flight cards."""

    leg_index: int
    origin: str
    destination: str
    departure_date: str | None = None
    offer_id: str
    price: str
    currency: str
    departure: str | None = None
    arrival: str | None = None
    duration_minutes: int | None = None
    stops: int | None = None
    carriers: list[str] = Field(default_factory=list)
    included_checked_baggage: str | None = None
    additional_checked_baggage: list[str] = Field(default_factory=list)


class FlightMetricsResponse(BaseModel):
    """Persisted per-leg measurements for the bounded flight-selection workflow."""

    thread_id: str
    metrics: list[FlightPipelineMetric] = Field(default_factory=list)


def trip_progress(thread_id: str, state: dict) -> TripProgressResponse:
    """Project persisted state into the sidebar contract without exposing messages/offers."""
    origin = state.get("origin")
    legs = state.get("legs") or []

    def field(leg: object, name: str):
        return leg.get(name) if isinstance(leg, dict) else getattr(leg, name, None)

    destinations = [
        destination
        for leg in legs
        if (destination := field(leg, "destination")) and destination != origin
    ]
    flights_ready = bool(legs) and all(field(leg, "offer") is not None for leg in legs)
    offers = []
    for index, leg in enumerate(legs):
        raw_offer = field(leg, "offer")
        if raw_offer is None:
            continue
        offer = BestOffer.model_validate(raw_offer)
        offers.append(FlightOfferCardResponse(
            leg_index=index,
            origin=field(leg, "origin") or "",
            destination=field(leg, "destination") or "",
            departure_date=field(leg, "departure_date"),
            offer_id=offer.offer_id,
            price=offer.price,
            currency=offer.currency,
            departure=offer.departure,
            arrival=offer.arrival,
            duration_minutes=offer.duration_minutes,
            stops=offer.stops,
            carriers=offer.carriers,
            included_checked_baggage=offer.included_checked_baggage,
            additional_checked_baggage=offer.additional_checked_baggage,
        ))
    try:
        phase = TripPhase(state.get("phase", INITIAL_TRIP_PHASE))
    except (TypeError, ValueError):
        phase = INITIAL_TRIP_PHASE

    return TripProgressResponse(
        thread_id=thread_id,
        phase=phase,
        origin=origin,
        departure_date=state.get("departure_date"),
        return_date=state.get("return_date"),
        season=state.get("season"),
        destinations=destinations,
        travelers=state.get("travelers"),
        flights_ready=flights_ready,
        direct_flight_search=True,
        flight_offers=offers,
    )


def persisted_trip_state(state: dict) -> dict:
    """Validate and strip message/transient fields before saving a trip."""
    return TripSnapshot.model_validate(state).model_dump(mode="json")


def trip_title(state: dict) -> str:
    """Give the trip list a useful, non-LLM-generated label."""
    legs = state.get("legs") or []
    origin = state.get("origin")
    first = legs[0] if legs else None
    destination = first.get("destination") if isinstance(first, dict) else getattr(first, "destination", None)
    if origin and destination:
        return f"{origin} to {destination}"
    if destination:
        return f"Trip to {destination}"
    return "Untitled trip"


async def save_current_trip_state(request: Request, clerk_user_id: str, trip_id: str, state: dict) -> None:
    """Save the durable planning snapshot after a successful state change."""
    snapshot = persisted_trip_state(state)
    saved_version = await asyncio.to_thread(
        save_trip_state, clerk_user_id, trip_id, snapshot, trip_title(snapshot)
    )
    if saved_version is None:
        raise HTTPException(status_code=404, detail="Trip not found")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open shared service connections once and close them at shutdown."""
    app.state.phoenix_tracing_enabled = configure_runtime_tracing()
    await asyncio.to_thread(initialize_database)
    redis_client = redis.from_url(redis_url)
    app.state.redis = redis_client
    try:
        serde = JsonPlusSerializer(
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
        async with AsyncPostgresSaver.from_conn_string(
            os.environ["DATABASE_URL"], serde=serde
        ) as checkpointer:
            await checkpointer.setup()
            async with open_resources() as resources:
                app.state.resources = resources
                app.state.agent = build_orchestrator_agent(checkpointer)
                yield
    finally:
        await redis_client.aclose()


app = FastAPI(title="Travel Agent API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")


@app.middleware("http")
async def apply_security_headers(request: Request, call_next):
    """Limit browser capabilities and untrusted resource origins for every response."""
    response = await call_next(request)
    response.headers.setdefault(
        "Content-Security-Policy",
        content_security_policy(os.environ.get("CLERK_PUBLISHABLE_KEY")),
    )
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Permissions-Policy", "camera=(), geolocation=(), microphone=()")
    return response

LIMIT = 10
WINDOW = 60

_INCREMENT_WITH_EXPIRY = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return {count, redis.call('TTL', KEYS[1])}
"""


async def rate_limit(request: Request) -> None:
    """Enforce the per-client fixed-window limit for costly API requests."""
    client_id = request.client.host if request.client else "unknown"
    key = f"rate_limit:{client_id}"

    count, ttl = await request.app.state.redis.eval(
        _INCREMENT_WITH_EXPIRY, 1, key, WINDOW
    )
    if count > LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {ttl}s.",
            headers={"Retry-After": str(max(ttl, 0))},
        )


def _reported_token_count(value: object) -> int:
    """Read total provider-reported tokens from a LangChain message or metric."""
    usage = value if isinstance(value, dict) else getattr(value, "usage_metadata", None)
    if not isinstance(usage, dict):
        return 0
    total_tokens = usage.get("total_tokens")
    if isinstance(total_tokens, int) and total_tokens >= 0:
        return total_tokens
    return sum(
        token_count
        for token_count in (usage.get("input_tokens"), usage.get("output_tokens"))
        if isinstance(token_count, int) and token_count >= 0
    )


def _token_usage_from_result(
    result: dict, previous_message_count: int, previous_metric_count: int
) -> int:
    """Count only model calls made by the current request, not prior chat history."""
    messages = list(result.get("messages") or [])
    metrics = list(result.get("flight_pipeline_metrics") or [])
    message_tokens = sum(
        _reported_token_count(message) for message in messages[previous_message_count:]
    )
    metric_tokens = sum(
        _reported_token_count(
            {
                "input_tokens": getattr(metric, "input_tokens", None)
                if not isinstance(metric, dict)
                else metric.get("input_tokens"),
                "output_tokens": getattr(metric, "output_tokens", None)
                if not isinstance(metric, dict)
                else metric.get("output_tokens"),
            }
        )
        for metric in metrics[previous_metric_count:]
    )
    return message_tokens + metric_tokens


async def reserve_token_budget(clerk_user_id: str, reservation_tokens: int):
    """Hold monthly token budget before work that can invoke an OpenAI model."""
    reservation = await asyncio.to_thread(
        reserve_monthly_tokens,
        clerk_user_id,
        reservation_tokens,
        FREE_MONTHLY_TOKEN_LIMIT,
        PAID_MONTHLY_TOKEN_LIMIT,
    )
    if reservation is None:
        raise HTTPException(
            status_code=402,
            detail="Your monthly AI allowance has been used. Upgrade to continue planning.",
        )
    return reservation


async def settle_token_budget(clerk_user_id: str, reservation, actual_tokens: int) -> None:
    """Record actual reported usage and release unused reserved token budget."""
    try:
        await asyncio.to_thread(
            settle_monthly_tokens, clerk_user_id, reservation, actual_tokens
        )
    except Exception:
        logger.exception("database.token_usage_settlement_failed")


@app.get("/", include_in_schema=False)
async def chat_interface() -> FileResponse:
    """Serve the small browser client alongside the API."""
    return FileResponse("frontend/dist/index.html")


@app.get("/auth/config", response_model=ClerkConfigResponse)
async def clerk_config() -> ClerkConfigResponse:
    """Provide Clerk's publishable key without placing configuration in source."""
    publishable_key = os.environ.get("CLERK_PUBLISHABLE_KEY")
    if not publishable_key:
        raise HTTPException(status_code=503, detail="Clerk authentication is not configured.")
    return ClerkConfigResponse(publishable_key=publishable_key)


@app.post("/webhooks/clerk", status_code=200)
async def clerk_webhook(request: Request) -> dict[str, str]:
    """Accept Clerk's signed lifecycle events without session authentication."""
    return await handle_clerk_webhook(request)


@app.get("/trips", response_model=list[TripSummaryResponse])
async def get_trips(user_id: str = Depends(current_user_id)) -> list[TripSummaryResponse]:
    """List the signed-in user's saved trips for a resume picker."""
    rows = await asyncio.to_thread(list_trips, user_id)
    return [
        TripSummaryResponse(
            trip_id=str(row["trip_id"]), title=row["title"], status=row["status"],
            state_version=row["state_version"], created_at=row["created_at"].isoformat(),
            updated_at=row["updated_at"].isoformat(),
        )
        for row in rows
    ]


@app.post("/trips/{trip_id}/resume", response_model=ResumeTripResponse)
async def resume_trip(
    trip_id: str, request: Request, user_id: str = Depends(current_user_id)
) -> ResumeTripResponse:
    """Start a fresh chat thread linked to an owned saved trip."""
    thread_id = str(uuid.uuid4())
    created = await asyncio.to_thread(create_resumed_thread, user_id, trip_id, thread_id)
    if not created:
        raise HTTPException(status_code=404, detail="Trip not found")
    state = await asyncio.to_thread(load_trip_state, user_id, trip_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    await request.app.state.agent.aupdate_state(
        {"configurable": {"thread_id": f"{user_id}:{thread_id}"}},
        persisted_trip_state(state),
    )
    return ResumeTripResponse(trip_id=trip_id, thread_id=thread_id)


@app.post("/trips/{trip_id}/archive", status_code=204)
async def archive_trip(trip_id: str, user_id: str = Depends(current_user_id)) -> None:
    """Hide an owned trip from future resume actions without deleting its data."""
    archived = await asyncio.to_thread(set_trip_status, user_id, trip_id, "archived")
    if not archived:
        raise HTTPException(status_code=404, detail="Trip not found")


@app.get("/trips/{thread_id}/progress", response_model=TripProgressResponse)
async def get_trip_progress(
    thread_id: str, request: Request, user_id: str = Depends(current_user_id)
) -> TripProgressResponse:
    """Return the persisted planning fields for one chat thread."""
    trip_id = await asyncio.to_thread(trip_for_thread, user_id, thread_id)
    if trip_id is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
    snapshot = await request.app.state.agent.aget_state(config)
    state = dict(snapshot.values or {})
    if not state:
        raise HTTPException(status_code=404, detail="Trip not found")
    return trip_progress(thread_id, state)


@app.get("/trips/{thread_id}/flight-metrics", response_model=FlightMetricsResponse)
async def get_flight_metrics(
    thread_id: str, request: Request, user_id: str = Depends(current_user_id)
) -> FlightMetricsResponse:
    """Return per-leg timing and selection measurements for this trip."""
    trip_id = await asyncio.to_thread(trip_for_thread, user_id, thread_id)
    if trip_id is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
    snapshot = await request.app.state.agent.aget_state(config)
    state = dict(snapshot.values or {})
    if not state:
        raise HTTPException(status_code=404, detail="Trip not found")
    return FlightMetricsResponse(
        thread_id=thread_id,
        metrics=list(state.get("flight_pipeline_metrics") or []),
    )


@app.post(
    "/trips/{thread_id}/flights/select",
    response_model=ChatResponse,
    dependencies=[Depends(rate_limit)],
)
async def select_flights_from_preferences(
    thread_id: str,
    payload: FlightSelectionRequest,
    request: Request,
    user_id: str = Depends(current_user_id),
) -> ChatResponse:
    """Finish a prefetched flight search without a preliminary chat-model turn."""
    trip_id = await asyncio.to_thread(trip_for_thread, user_id, thread_id)
    if trip_id is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
    snapshot = await request.app.state.agent.aget_state(config)
    state = dict(snapshot.values or {})
    if not state:
        raise HTTPException(status_code=404, detail="Trip not found")
    if state.get("phase") != TripPhase.FLIGHTS:
        raise HTTPException(status_code=409, detail="Flight preferences can only be submitted in the flights phase.")
    if request.app.state.resources.pipeline != "deterministic":
        raise HTTPException(status_code=503, detail="Flight selection is unavailable.")
    legs = list(state.get("legs") or [])
    travelers = state.get("travelers")
    client = request.app.state.resources.duffel_client
    selector_model = request.app.state.resources.flight_selector_model
    if not legs or not isinstance(travelers, int) or travelers < 1:
        raise HTTPException(status_code=409, detail="The itinerary and traveler count must be saved before searching flights.")
    if not isinstance(client, DuffelClient) or selector_model is None:
        raise HTTPException(status_code=503, detail="Flight-search services are unavailable.")

    cache = request.app.state.resources.flight_prefetch
    preferences = payload.preferences
    cached = await cache.take(
        config["configurable"]["thread_id"],
        legs=legs,
        travelers=travelers,
        preferences=preferences,
    )
    if cached is None:
        cache.start(
            config["configurable"]["thread_id"],
            client=client,
            legs=legs,
            travelers=travelers,
            preferences=preferences,
        )
        cached = await cache.take(
            config["configurable"]["thread_id"],
            legs=legs,
            travelers=travelers,
            preferences=preferences,
        )
    if cached is None:  # Defensive: start always creates an entry for valid uncovered legs.
        raise HTTPException(status_code=500, detail="Could not start the flight search.")

    token_reservation = await reserve_token_budget(
        user_id, len(legs) * FLIGHT_SELECTION_TOKENS_PER_LEG
    )
    metrics: list[FlightPipelineMetric] = []
    try:
        selected_legs, metrics, notes = await select_from_prefetched_requests(
            client=client, selector_model=selector_model, legs=legs, travelers=travelers,
            preferences=preferences, requests=cached, include_selected=True,
        )
    finally:
        actual_tokens = sum(
            _reported_token_count(
                {
                    "input_tokens": metric.input_tokens,
                    "output_tokens": metric.output_tokens,
                }
            )
            for metric in metrics
        )
        await settle_token_budget(user_id, token_reservation, actual_tokens)
    await request.app.state.agent.aupdate_state(
        config,
        {
            "legs": selected_legs,
            "flight_preferences": preferences,
            "flights_confirmed": False,
            "flight_pipeline_metrics": [
                *(state.get("flight_pipeline_metrics") or []), *metrics
            ][-100:],
        },
    )
    updated_snapshot = await request.app.state.agent.aget_state(config)
    await save_current_trip_state(
        request, user_id, str(trip_id), dict(updated_snapshot.values or {})
    )
    selected_lines = []
    for index, leg in enumerate(selected_legs):
        if leg.offer is not None:
            selected_lines.append(
                f"[{index}] {leg.origin} → {leg.destination}: "
                f"{leg.offer.price} {leg.offer.currency} ({leg.offer.offer_id})"
            )
    summary = "Selected flights:\n" + "\n".join(selected_lines)
    if notes:
        summary += " Some legs could not be selected: " + "; ".join(notes)
    else:
        summary += " Review the recommendations and tell me if you want changes or would like to confirm them."
    summary += "\n\nApplied flight filters: " + preferences.applied_filter_summary()
    summary += "\nYou can revise any of these filters if they do not match your intent."
    return ChatResponse(response=summary, thread_id=thread_id, trip_id=str(trip_id))


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
async def chat(
    payload: ChatRequest,
    request: Request,
    user_id: str = Depends(current_user_id),
) -> ChatResponse:
    """Send a message to the agent and return its final reply."""
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be blank")
    try:
        validate_chat_message_length(message)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    if payload.thread_id and payload.trip_id:
        raise HTTPException(status_code=422, detail="Send either thread_id or trip_id, not both")

    is_new_thread = payload.thread_id is None
    thread_id = payload.thread_id or str(uuid.uuid4())
    restored_state: dict | None = None
    if payload.thread_id:
        trip_uuid = await asyncio.to_thread(trip_for_thread, user_id, thread_id)
        if trip_uuid is None:
            raise HTTPException(status_code=404, detail="Trip not found")
        trip_id = str(trip_uuid)
    elif payload.trip_id:
        trip_id = payload.trip_id
        restored_state = await asyncio.to_thread(load_trip_state, user_id, trip_id)
        if restored_state is None:
            raise HTTPException(status_code=404, detail="Trip not found")

    token_reservation = await reserve_token_budget(user_id, CHAT_TOKEN_RESERVATION)
    if payload.trip_id:
        created = await asyncio.to_thread(create_resumed_thread, user_id, payload.trip_id, thread_id)
        if not created:
            await settle_token_budget(user_id, token_reservation, 0)
            raise HTTPException(status_code=404, detail="Trip not found")
    elif payload.thread_id is None:
        trip_id = str(await asyncio.to_thread(create_trip_and_thread, user_id, thread_id))

    # The browser only sees the opaque UUID. Prefixing the persisted key with the
    # Clerk user ID prevents a guessed UUID from reading another user's trip.
    config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
    previous_snapshot = await request.app.state.agent.aget_state(config)
    previous_state = dict(previous_snapshot.values or {})
    previous_message_count = len(previous_state.get("messages") or [])
    previous_metric_count = len(previous_state.get("flight_pipeline_metrics") or [])
    agent_input = {"messages": [HumanMessage(content=message)]}
    if restored_state is not None:
        # A new chat receives planning facts but never historical chat messages.
        agent_input.update(persisted_trip_state(restored_state))
    elif is_new_thread:
        agent_input["phase"] = INITIAL_TRIP_PHASE

    result: dict = {}
    with runtime_chat_turn(thread_id, message, turn_index=previous_message_count // 2) as turn_span:
        try:
            result = await request.app.state.agent.ainvoke(
                agent_input,
                config=config,
                context=request.app.state.resources,
            )
        finally:
            await settle_token_budget(
                user_id,
                token_reservation,
                _token_usage_from_result(
                    result, previous_message_count, previous_metric_count
                ),
            )
        structured_response = result.get("structured_response")
        if isinstance(structured_response, PlannerResponse):
            structured_response = structured_response.for_trip_state(result)
            response = structured_response.response
            destination_suggestions = structured_response.destination_suggestions
        else:
            content = result["messages"][-1].content
            response = content if isinstance(content, str) else str(content)
            destination_suggestions = []
        if turn_span is not None:
            turn_span.set_attribute("output.value", response)
    await save_current_trip_state(request, user_id, trip_id, result)

    # As soon as the itinerary reaches flights, begin Duffel work while the user is
    # reading the prompt for airline/cabin/connection preferences.
    if result.get("phase") == TripPhase.FLIGHTS:
        legs = list(result.get("legs") or [])
        travelers = result.get("travelers")
        client = request.app.state.resources.duffel_client
        if isinstance(client, DuffelClient) and isinstance(travelers, int) and travelers > 0:
            prefetch = request.app.state.resources.flight_prefetch
            if prefetch is None:
                return ChatResponse(response=response, thread_id=thread_id, trip_id=trip_id,
                                    destination_suggestions=destination_suggestions)
            prefetch.start(
                config["configurable"]["thread_id"], client=client, legs=legs, travelers=travelers
            )

    return ChatResponse(response=response, thread_id=thread_id, trip_id=trip_id,
                        destination_suggestions=destination_suggestions)
