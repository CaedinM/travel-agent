"""HTTP interface for the travel-planning agent."""

from contextlib import asynccontextmanager
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain.messages import HumanMessage

from agents import build_orchestrator_agent
from resources import open_resources
from state import INITIAL_TRIP_PHASE, TripPhase


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
    """Open the MCP connection once, and close it when the server stops."""
    async with open_resources() as resources:
        app.state.resources = resources
        app.state.agent = build_orchestrator_agent()
        yield


app = FastAPI(title="Travel Agent API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/", include_in_schema=False)
async def chat_interface() -> FileResponse:
    """Serve the small browser client alongside the API."""
    return FileResponse("frontend/index.html")


@app.get("/trips/{thread_id}/progress", response_model=TripProgressResponse)
async def get_trip_progress(thread_id: str, request: Request) -> TripProgressResponse:
    """Return the persisted planning fields for one chat thread."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await request.app.state.agent.aget_state(config)
    state = dict(snapshot.values or {})
    if not state:
        raise HTTPException(status_code=404, detail="Trip not found")
    return trip_progress(thread_id, state)


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Send a message to the agent and return its final reply."""
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be blank")

    is_new_trip = payload.thread_id is None
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
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
