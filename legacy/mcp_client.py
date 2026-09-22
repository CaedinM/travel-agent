import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

def build_mcp_client():
    is_live = os.environ.get("LIVE_MODE", "false").strip().lower() == "true"
    if is_live:
        duffel_key = os.environ["DUFFEL_LIVE_API_KEY"]
    else:
        duffel_key = os.environ["DUFFEL_TEST_API_KEY"]

    venv_command = Path(sys.executable).with_name("flights-mcp")
    command = (
        str(venv_command)
        if venv_command.is_file()
        else shutil.which("flights-mcp") or "flights-mcp"
    )
    client = MultiServerMCPClient(
        {
            "flights": {
                "transport": "stdio",
                "command": command,
                "args": [],
                "env": {"DUFFEL_API_KEY_LIVE": duffel_key},
            }
        }
    )
    return client

async def get_mcp_tools(client: MultiServerMCPClient):
    tools = await client.get_tools()
    return tools
