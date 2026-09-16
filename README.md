# Travel Agent

A conversational travel planning agent. You describe a trip in plain English and it
builds an itinerary, then finds a flight for every leg.

Trips are modelled as an ordered list of one-way legs, so a simple return trip and a
multi-city trip are the same thing — London → Paris → Rome → London is three legs. Each
leg carries its own flight offer.

Two agents:

- **Orchestrator** — talks to the user, keeps the itinerary in state, decides when to search.
- **Flights subagent** — given one leg, searches it and picks the best flight. All the
  choosing happens here; it weighs price against departure time, duration and stops, and
  honours preferences like "direct only" or "nothing before 09:00".

Legs are searched in parallel, so a three-leg trip takes about as long as one leg.

## Stack

| | |
|---|---|
| Orchestration | LangChain / LangGraph agents, OpenAI models |
| Flight data | [Duffel](https://duffel.com) via the `flights-mcp` MCP server (stdio) |
| Web search | Tavily |
| State | `TripState`, persisted per-thread by an in-memory checkpointer |
| Tracing | LangSmith (optional) |

Source layout, all under `src/`:

| file | |
|---|---|
| `main.py` | REPL entry point |
| `agents.py` | both agents and their prompts |
| `llm.py` | which model each agent runs on — `LLM` and `POWERFUL_LLM` |
| `tools.py` | orchestrator tools + the subagent's search tool |
| `flight_options.py` | trims raw search results to a compact option table |
| `models.py`, `state.py` | `Leg`, `BestOffer`, `TripState` |
| `mcp_client.py`, `resources.py` | MCP session and agent wiring |

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs `flights-mcp` from git as a console script into `.venv/bin`.

Then copy the env template and fill it in:

```bash
cp .env.example .env
```

| variable | |
|---|---|
| `OPENAI_API_KEY` | required |
| `TAVILY_API_KEY` | required, for web search |
| `DUFFEL_TEST_API_KEY` | required — test data, no real bookings |
| `DUFFEL_LIVE_API_KEY` | only if `LIVE_MODE=true` |
| `LANGSMITH_*` | optional; set `LANGSMITH_TRACING=false` to skip |


## Run

```bash
source .venv/bin/activate
python src/main.py
```

Type `exit` or `quit` to leave. Quitting this way also flushes LangSmith traces —
killing the process mid-search leaves a trace stuck as pending.

```
> I want to fly from London to Paris on 2026-11-06, then Rome on 2026-11-10, then home on 2026-11-15
> find me flights
> only direct flights, nothing before 9am
> how much is the Rome leg?
```

Dates must be in the future; Duffel rejects anything else.

**Activate the venv first.** The agent launches `flights-mcp` via `PATH`, and if another
Python's copy is found first the MCP server dies immediately with `McpError: Connection
closed`.

## Frontend

The React chat interface is served by the FastAPI application—there is no separate
Node install or frontend development server to run.

Start the app from the project root:

```bash
source .venv/bin/activate
uvicorn api:app --app-dir src --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The frontend sends
messages to the local `/chat` endpoint and automatically retains the conversation ID
for follow-up messages. Stop the server with `Ctrl+C`.

## Web API

Start the HTTP server after installing the updated dependencies:

```bash
source .venv/bin/activate
uvicorn api:app --app-dir src --reload
```

Send a message with `POST /chat`:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"I want to fly from London to Paris next month"}'
```

The response includes the agent's reply and a `thread_id`. Send that same `thread_id`
with later messages to preserve the itinerary and conversation state:

```json
{"message":"Find me flights", "thread_id":"<thread-id>"}
```

The current planning checklist for a conversation is available at:

```text
GET /trips/<thread-id>/progress
```

It returns the collected origin, dates, itinerary destination codes, traveller count,
and workflow status without returning the chat history or flight-offer details.
