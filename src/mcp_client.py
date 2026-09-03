from langchain_mcp_adapters.client import MultiServerMCPClient
import shutil
import asyncio

import os
from dotenv import load_dotenv

load_dotenv()

def build_mcp_client():
    is_live = os.environ.get("LIVE_MODE", "false").strip().lower() == "true"
    if is_live:
        duffel_key = os.environ["DUFFEL_LIVE_API_KEY"]
    else:
        duffel_key = os.environ["DUFFEL_TEST_API_KEY"]

    client = MultiServerMCPClient(
        {
            "flights": {
                "transport": "stdio",
                "command": shutil.which("flights-mcp") or "flights-mcp",
                "args": [],
                "env": {"DUFFEL_API_KEY_LIVE": duffel_key},
            }
        }
    )
    return client

async def get_mcp_tools(client: MultiServerMCPClient):
    tools = await client.get_tools()
    return tools