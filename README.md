# Travel Agentic
Chat with an AI agent to plan your next trip and find flight recommendations.
The application does not book flights or accommodation.

## Features:
- Converse with the agent to plan a trip.
- Chat interface displays trip details as they are mentioned.

## The Agent:
- Uses dynamic prompts to correctly guide users through trip planning.
- Avoids mistakes and hallucinations by saving important details in state rather than relying on message history for context.
- Integrates with duffel API to find flights for the user based on their preferences (specific airline, time of day, cabin, etc.)

## Stack

- Frontend: React 18.3.1
- Backend: FastAPI, LangChain/LangGraph, Uvicorn
- Models: OpenAI (`gpt-5` / `gpt-5-mini`)
- Flights: Duffel.com API
- Web Search: Tavily
- Auth: Clerk
- Rate Limiting: Redis
- Observability: LangSmith (optional)
- Deployment: Cloudflare Workers + Containers

## Project layout

| Path | Purpose |
|---|---|
| `src/travel_agent/api/` | FastAPI server, authentication, and HTTP contracts |
| `src/travel_agent/agents/` | Planning orchestrator and specialized travel agents |
| `src/travel_agent/tools/` | Agent tools and flight-search workflows |
| `src/travel_agent/core/` | Itinerary models, conversation state, workflow, and LLM configuration |
| `src/travel_agent/flights/` | Duffel client plus flight parsing and selection |
| `src/travel_agent/infrastructure/` | Database ownership and shared runtime resource wiring |
| `frontend/` | Browser chat interface and styles |


## Local development

Docker Compose is the recommended local development environment. It builds the
Python 3.12 API image and starts the API, PostgreSQL, and Redis together, so a
contributor does not need to install Python, create a virtual environment, or
create database services manually.

### Prerequisites

- Docker Desktop, running locally.
- Credentials for the external services used by the app.

### First-time setup

Clone the repository, then create the local environment file:

```bash
cp .env.example .env
```

Add the external-service credentials to `.env`. Docker Compose configures the
local database and cache automatically; leave `DATABASE_URL` and `REDIS_URL`
out of this file.

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
| `LANGSMITH_*` | Optional tracing configuration. Set `LANGSMITH_TRACING=false` to disable tracing. |
| `CLERK_WEBHOOK_SIGNING_SECRET` | Signing secret for the Clerk `user.created` webhook. |

Start the complete stack:

```bash
docker compose up --build
```

The API is available at `http://127.0.0.1:8000`. On subsequent starts, the
image is already built, so use:

```bash
docker compose up
```

Python source files are mounted into the API container and Uvicorn reloads
after code changes. The browser client is a production bundle; rebuild the API
image with `docker compose build api` after changing files under `frontend/`.

### Running the agent: CLI vs. frontend

Use the CLI when you want to exercise the planning agent directly. It creates a
conversation thread in the agent runtime and does not go through the browser,
FastAPI authentication, trip persistence, or Clerk. It is the fastest way to
try a conversation while developing agent prompts, workflow, research, or
flight-selection behavior:

```bash
docker compose exec api python -m travel_agent.cli
```

Use the frontend at `http://127.0.0.1:8000` when you want to test the complete
product flow: Clerk sign-in, authenticated API requests, user-scoped saved
trips, PostgreSQL persistence, Redis rate limiting, and the browser UI. This
path requires valid `CLERK_PUBLISHABLE_KEY` and `CLERK_SECRET_KEY` values in
`.env`, plus a Clerk user who can sign in to that development instance.

Run automated tests inside the same environment:

```bash
docker compose exec api pytest
```

Stop the local stack with:

```bash
docker compose down
```

The database and Redis volumes are retained between runs. Use the following
only when you intentionally want to delete all local database and cache data:

```bash
docker compose down -v
```

## Cloudflare deployment

Production uses Cloudflare Workers + Containers to run the Python API and its
bundled frontend. In Workers Builds, set **Build command** to
`npm ci && npm run check:cloudflare && npm run build` and **Deploy command** to
`npm run deploy`.

See [the Cloudflare deployment guide](docs/cloudflare.md) for required runtime
secrets, external Postgres/Redis configuration, Clerk settings, and Render data
migration. This requires a Workers Paid plan and incurs container usage charges.

## Agent evaluation

The reproducible evaluation dataset and scoring code live in `evals/`. It has
30 synthetic cases across standard, incomplete, correction, provider-failure,
and boundary conversations. The deterministic checks are ordinary pytest tests
and should run on every change:

```bash
pytest tests/test_evals.py
```

Inspect the current dataset with:

```bash
python -m evals.run --list
```

The `evals.harness` module accepts the real orchestrator through an injected
adapter. `run_decision` restores a labelled state for a single turn;
`run_conversation` runs a fixture-backed traveller for up to the scenario's
turn limit. Both return an evidence-bearing report with independent workflow,
completion, and efficiency results.

For manual quality comparisons, configure `PHOENIX_COLLECTOR_ENDPOINT` and
`PHOENIX_API_KEY`, then call `evals.phoenix.configure()` before a run and
`evals.phoenix.record_report()` afterward with an opaque session identifier.
The recorder emits one trace per turn and nested tool spans. LLM judges in
`evals.judges` are advisory until a 100-example human-labelled calibration set
achieves at least 80% failure recall and specificity; deterministic invariants
remain the CI gate.

Run the advisory conversation-quality judges only after a manual or synthetic
conversation has completed:

```bash
python -m evals.conversation_quality --limit 100
```

The runner reads only traces tagged `evaluation` or runtime traces tagged
`manual`; it excludes `production` traces. It writes pass/fail annotations for
grounded communication, context continuity, progress and recovery, naturalness,
decision support when applicable, and session repetition. Reviewers add matching
human annotations in Phoenix, then compare agreement with:

```bash
python -m evals.conversation_quality --calibrate --limit 500
```

## Flight selection

Flight search uses Duffel. The application filters ineligible offers and sends the
remaining candidates, together with the traveller's preferences, to Jev to select
the best option.
