import os
from dotenv import load_dotenv
from langchain.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from functools import cache

tavily_client = TavilyClient()

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
    destination: str | None = None,
    season: str | None = None,
    departure_date: str | None = None,
    return_date: str | None = None,
    trip_length: str | None = None,
    budget: str | None = None
    ) -> Command:
    """Save any trip details the user has provided. Pass only the fields
    the user mentioned this turn and leave the rest as None."""

    updates = {k: v for k, v in {
        "origin": origin,
        "destination": destination,
        "season": season,
        "departure_date": departure_date,
        "return_date": return_date,
        "trip_length": trip_length,
        "budget": budget,
        "messages": [ToolMessage("Success", tool_call_id=runtime.tool_call_id)]
    }.items() if v is not None}
    
    return Command(update=updates)


@tool
async def call_flights_agent(runtime: ToolRuntime) -> Command | str:
    """Call the flights subagent to find a flight offer."""

    missing = [k for k in ("origin", "destination", "departure_date", "return_date")
        if not runtime.state.get(k)]
    if missing:
        return f"Cannot search flights; missing: {', '.join(missing)}"
    
    origin = runtime.state["origin"]
    destination = runtime.state["destination"]
    departure_date = runtime.state["departure_date"]
    return_date = runtime.state["return_date"]

    message = f"""
    Find round-trip flights from {origin} to {destination} departing
    on {departure_date} and returning on {return_date}
    """

    result = await runtime.context.flights_agent.ainvoke({"messages": [HumanMessage(content=message)]})

    ## Process the json response from the subagent and update the TripState to contain the Offer.
    pick = result.get("structured_response")

    if pick is None:
        return "The flights subagent finished without selecting an offer. Try the search again."

    return Command(update={"flights_offer": pick, "messages": [ToolMessage("Success", tool_call_id=runtime.tool_call_id)]})
    

@tool
async def get_saved_flights_info(runtime: ToolRuntime) -> str:
    """Gets detailed information about the flight offer currently saved in state."""
    
    offer = runtime.state.get("flights_offer")
    
    if offer is None:
        return (
            "Can not get flight offer details; There is no flight offer saved in state. "
            "If you want to get a flight offer call the call_flights_agent tool first."
            )

    offer_id = runtime.state.get("flights_offer").offer_id

    try:
        return await runtime.context.flights_tools_by_name["get_offer_details"].ainvoke({"params": {"offer_id": offer_id}})
    except Exception as e:
        return (
            f"Couldn't fetch details for offer {offer_id}: {e}."
            "The offer may have expired — consider re-running the flights search."
            )