# Travel Agentic

Plan and book your next trip by simply chatting with an agent.

## Stack

- Frontend: React 18.3.1
- Backend: FastAPI, LangChain/LangGraph, Uvicorn
- Models: OpenAI (`gpt-5` / `gpt-5-mini`)
- Flight Search: Duffel via the `flights-mcp` MCP server
- Web Search: Tavily
- Caching: Redis
- Auth: Clerk
- Observability: LangSmith (optional)

## Project layout

| Path | Purpose |
|---|---|
| `src/api.py` | FastAPI server and chat/progress endpoints |
| `src/agents/` | Planning orchestrator and specialized travel agents |
| `src/tools.py` | Agent tools and flight-search integration |
| `src/models.py`, `src/state.py` | Itinerary, offers, and conversation state |
| `src/resources.py`, `src/mcp_client.py` | MCP session and agent resources |
| `frontend/` | Browser chat interface and styles |


## Setup

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
| `CLERK_PUBLISHABLE_KEY` | Required by the browser to display Clerk sign-in and account controls. |
| `CLERK_SECRET_KEY` | Required by the API to verify signed-in users. Keep it server-side. |
| `CLERK_JWT_KEY` | Optional Clerk PEM public key for networkless session-token verification. |
| `CLERK_AUTHORIZED_PARTIES` | Comma-separated allowed app origins, e.g. `http://localhost:8000`. |
| `TAVILY_API_KEY` | Required for destination research. |
| `DUFFEL_TEST_API_KEY` | Required for flight search using Duffel test data. |
| `DUFFEL_LIVE_API_KEY` | Required only when `LIVE_MODE=true`. |
| `LANGSMITH_*` | Optional tracing configuration. Set `LANGSMITH_TRACING=false` to disable tracing. |
| `REDIS_URL` | Required for chat rate limiting. Use `redis://localhost:6379/0` with the local Redis setup below. |


### Local Development
The API connects to Redis for chat rate limiting, but does not start Redis itself.
Run Redis separately before starting the app. With Docker Desktop installed and
running, create the local Redis container once:

```bash
docker run -d --name travel-agent-redis -p 127.0.0.1:6379:6379 redis:7-alpine
```

Set the matching value in `.env`:

```env
REDIS_URL=redis://localhost:6379/0
```

On later development sessions, start the existing container instead of creating
a new one:

```bash
docker start travel-agent-redis
```

If you use a different host port, update `REDIS_URL` to match it.

Start the app:

```bash
uvicorn api:app --app-dir src --reload
```