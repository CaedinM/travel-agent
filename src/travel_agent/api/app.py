"""HTTP interface for the travel-planning agent."""

import os
import uuid
from contextlib import asynccontextmanager

import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field

from travel_agent.agents.orchestrator import build_orchestrator_agent
from travel_agent.api.auth import current_user_id
from travel_agent.core.models import FlightPipelineMetric
from travel_agent.core.state import INITIAL_TRIP_PHASE, TripPhase
from travel_agent.infrastructure.resources import open_resources

load_dotenv()
redis_url = os.environ["REDIS_URL"]

class ChatRequest(BaseModel):
    """A message sent by a frontend user to the travel agent."""

    message: str = Field(min_length=1, description="The user's message to the agent.")
    thread_id: str | None = Field(
        default=None,
        description="An ID returned by a previous request to continue that conversation.",
    )


class ChatResponse(BaseModel):
    response: str
    thread_id: str


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


class FlightMetricsResponse(BaseModel):
    """Persisted pipeline measurements for comparing legacy and Jev runs."""

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
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open shared service connections once and close them at shutdown."""
    redis_client = redis.from_url(redis_url)
    app.state.redis = redis_client
    try:
        async with open_resources() as resources:
            app.state.resources = resources
            app.state.agent = build_orchestrator_agent()
            yield
    finally:
        await redis_client.aclose()


app = FastAPI(title="Travel Agent API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

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
    """Enforce the per-client fixed-window limit for chat requests."""
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


@app.get("/", include_in_schema=False)
async def chat_interface() -> FileResponse:
    """Serve the small browser client alongside the API."""
    return FileResponse("frontend/index.html")


@app.get("/auth/config", response_model=ClerkConfigResponse)
async def clerk_config() -> ClerkConfigResponse:
    """Provide Clerk's publishable key without placing configuration in source."""
    publishable_key = os.environ.get("CLERK_PUBLISHABLE_KEY")
    if not publishable_key:
        raise HTTPException(status_code=503, detail="Clerk authentication is not configured.")
    return ClerkConfigResponse(publishable_key=publishable_key)


@app.get("/trips/{thread_id}/progress", response_model=TripProgressResponse)
async def get_trip_progress(
    thread_id: str, request: Request, user_id: str = Depends(current_user_id)
) -> TripProgressResponse:
    """Return the persisted planning fields for one chat thread."""
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
    """Return per-leg timing and token measurements for flight pipeline benchmarks."""
    config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
    snapshot = await request.app.state.agent.aget_state(config)
    state = dict(snapshot.values or {})
    if not state:
        raise HTTPException(status_code=404, detail="Trip not found")
    return FlightMetricsResponse(
        thread_id=thread_id,
        metrics=list(state.get("flight_pipeline_metrics") or []),
    )


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

    is_new_trip = payload.thread_id is None
    thread_id = payload.thread_id or str(uuid.uuid4())
    # The browser only sees the opaque UUID. Prefixing the persisted key with the
    # Clerk user ID prevents a guessed UUID from reading another user's trip.
    config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
    agent_input = {"messages": [HumanMessage(content=message)]}
    if is_new_trip:
        agent_input["phase"] = INITIAL_TRIP_PHASE

    result = await request.app.state.agent.ainvoke(
        agent_input,
        config=config,
        context=request.app.state.resources,
    )
    content = result["messages"][-1].content
    response = content if isinstance(content, str) else str(content)

    return ChatResponse(response=response, thread_id=thread_id)
