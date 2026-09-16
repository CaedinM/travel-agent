# Travel Agentic

Plan and book your next trip by simply chatting with an agent.

## Run the web app

Requires Python 3.12.

From the project root, create an environment and install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template and provide the required keys:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required for the planning agents. |
| `TAVILY_API_KEY` | Required for destination research. |
| `DUFFEL_TEST_API_KEY` | Required for flight search using Duffel test data. |
| `DUFFEL_LIVE_API_KEY` | Required only when `LIVE_MODE=true`. |
| `LANGSMITH_*` | Optional tracing configuration. Set `LANGSMITH_TRACING=false` to disable tracing. |

Start the app:

```bash
uvicorn api:app --app-dir src --reload
```


## Stack

- Frontend: React 18.3.1 and React DOM, loaded in-browser
- Backend: Python 3.12, FastAPI, and Uvicorn
- Agents: LangChain, LangGraph, and OpenAI (`gpt-5` / `gpt-5-mini`)
- Flight search: Duffel via the `flights-mcp` MCP server
- Destination research: Tavily
- State: In-memory LangGraph checkpointer
- Observability: LangSmith (optional)

## Stack

Frontend: React 18.3.1
Backend: FastAPI, LangChain/LangGraph

## Project layout

| Path | Purpose |
|---|---|
| `src/api.py` | FastAPI server and chat/progress endpoints |
| `src/agents.py` | Planning and flight-search agents |
| `src/tools.py` | Agent tools and flight-search integration |
| `src/models.py`, `src/state.py` | Itinerary, offers, and conversation state |
| `src/resources.py`, `src/mcp_client.py` | MCP session and agent resources |
| `frontend/` | Browser chat interface and styles |
