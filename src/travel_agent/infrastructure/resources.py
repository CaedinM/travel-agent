import os
from contextlib import asynccontextmanager

from langchain_typesafe import TypeSafeClassifier

from travel_agent.core.models import Resources
from travel_agent.flights.duffel import DuffelClient
from travel_agent.flights.prefetch import FlightPrefetchCache


@asynccontextmanager
async def open_resources(*, legacy_model=None):
    """Open only the services needed by the selected benchmark pipeline.

    Jev/direct-Duffel is the default. Set FLIGHT_PIPELINE=legacy to intentionally
    reconnect the preserved MCP/subagent implementation for an apples-to-apples run.
    """
    pipeline = os.environ.get("FLIGHT_PIPELINE", "jev").strip().lower()
    if pipeline == "legacy":
        from langchain_mcp_adapters.tools import load_mcp_tools
        from legacy.flights import build_flights_agent
        from legacy.mcp_client import build_mcp_client
        from travel_agent.tools.trip import make_search_leg_tool

        client = build_mcp_client()
        async with client.session("flights") as session:
            tools = await load_mcp_tools(session)
            tools_by_name = {tool.name: tool for tool in tools}
            yield Resources(
                pipeline="legacy",
                flights_agent=build_flights_agent(
                    make_search_leg_tool(tools_by_name),
                    tools_by_name["get_offer_details"],
                    model=legacy_model,
                ),
                flights_tools_by_name=tools_by_name,
            )
        return
    if pipeline != "jev":
        raise ValueError("FLIGHT_PIPELINE must be either 'jev' or 'legacy'.")

    duffel_client = DuffelClient.from_environment()
    classifier = TypeSafeClassifier(model="jev-1.13.0")
    flight_prefetch = FlightPrefetchCache()
    try:
        yield Resources(
            pipeline="jev", duffel_client=duffel_client, classifier=classifier,
            flight_prefetch=flight_prefetch,
        )
    finally:
        flight_prefetch.close()
        await duffel_client.aclose()
        if classifier.async_client is not None:
            await classifier.async_client.aclose()
        if classifier.client is not None:
            classifier.client.close()
