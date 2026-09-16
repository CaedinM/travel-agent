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
from state import INITIAL_TRIP_PHASE


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
