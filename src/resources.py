from contextlib import asynccontextmanager

from langchain_mcp_adapters.tools import load_mcp_tools

from mcp_client import build_mcp_client
from agents import build_flights_agent
from tools import make_search_leg_tool

from models import Resources

@asynccontextmanager
async def open_resources():
    client = build_mcp_client()
    async with client.session("flights") as session:
        tools = await load_mcp_tools(session)
        tools_by_name = {t.name: t for t in tools}
        yield Resources(
            flights_agent=build_flights_agent(
                make_search_leg_tool(tools_by_name),
                tools_by_name["get_offer_details"],
                ),
            flights_tools_by_name=tools_by_name,
            )
