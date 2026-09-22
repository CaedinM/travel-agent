# Travel Agentic
Chat with an AI agent to plan and book your next trip.

## Features:
- Converse with the agent to plan a trip.
- 
- Chat interface also displays trip details as they are mentioned.

## The Agent:
- Uses dynamic prompts to correctly guide users through trip planning.
- Avoids mistakes and hallucinations by saving important details in state rather than relying on message history for context.
- Integrates with duffel API to find flights for the user based on their preferences (specific airline, time of day, cabin, etc.)
- Utilizes Jev to optimize latency and token usage for decision making such as choosing a flight offer.

## Stack

- Frontend: React 18.3.1
- Backend: FastAPI, LangChain/LangGraph, Uvicorn
- Models: OpenAI (`gpt-5` / `gpt-5-mini`), TypeSafe (`jev-1.13.0`)
- Flights: Duffel.com API
- Web Search: Tavily
- Auth: Clerk
- Rate Limiting: Redis
- Observability: LangSmith (optional)

## Project layout

| Path | Purpose |
|---|---|
| `src/travel_agent/api/` | FastAPI server, authentication, and HTTP contracts |
| `src/travel_agent/agents/` | Planning orchestrator and specialized travel agents |
| `src/travel_agent/tools/` | Agent tools and flight-search workflows |
| `src/travel_agent/core/` | Itinerary models, conversation state, workflow, and LLM configuration |
| `src/travel_agent/flights/` | Duffel client plus flight parsing and selection |
| `src/travel_agent/infrastructure/` | MCP session and runtime resource wiring |
| `frontend/` | Browser chat interface and styles |


## Developer Setup

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
| `TYPESAFE_API_KEY` | Required for Jev flight selection (the default flight pipeline). |
| `FLIGHT_PIPELINE` | Optional: `jev` (default) or `legacy` to benchmark the preserved MCP/subagent path. |
| `LANGSMITH_*` | Optional tracing configuration. Set `LANGSMITH_TRACING=false` to disable tracing. |
| `REDIS_URL` | Required for chat rate limiting. Use `redis://localhost:6379/0` with the local Redis setup below. |

## Running Locally
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
uvicorn travel_agent.api.app:app --app-dir src --reload
```

## Jev 
### Flight pipeline benchmarking
Travel Agentic uses the Jev Classifier model for selecting flights based on
ambiguous user preferences. The depreciated system used a subagent and an MCP tool
to find flights. The new system uses a deterministic call to the Duffel.com API,
drops disqualified candidates and then serves the rest to Jev along with the user's
preferences to choose the "best" option.

To observe the latency and token cost gains, I ran both systems against a test set
of 10 mock TripStates. Experiment can be found in `experiments/flights_evaluation/`
The results:

| System | Average end-to-end time per request | Total estimated token cost |
|---|---|---|
| Deterministic Duffel search + Jev Classifier | ~2.33s | $0.00358 |
| GPT-5-mini + flights-mcp | ~14.91s | $0.02943 |

**Jev is roughly 6.4× faster and 8.2× cheaper**
