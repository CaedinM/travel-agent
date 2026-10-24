# Railway deployment

The production application runs as three services in one Railway project:
`travel-agent` (FastAPI plus the bundled browser UI), `Postgres`, and `Redis`.
Use the same production environment and US West region for all three services.
The databases use persistent Railway volumes and private networking.

## Build and start

Railway builds the repository-root Dockerfile. `railway.json` supplies the
production start command, listens on Railway's `PORT`, and checks `/health` with
a 300-second startup timeout. No separate frontend build or deploy command is
needed: the image builds the frontend before packaging the Python application.

Railway currently accepts this configuration but warns that JSON/TOML config is
deprecated and will stop working on December 1, 2026. Migrate it to Railway's
Infrastructure as Code format before that deadline; the CLI provides
`railway config migrate` for that transition.

Keep one application instance while flight prefetches live in process memory.
Conversation checkpoints and saved trips live in Postgres and survive restarts.
The Docker Compose development target remains separate from the production image.

## Runtime configuration

Create official Postgres and Redis services, leaving public database access
disabled. In the application service, configure Railway variable references:

```text
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}
```

These references assume the service names shown above. Do not copy public
database URLs into the application when all three services run in this project.

Supply application secrets through Railway variables, never Docker build
arguments or committed files:

- `OPENAI_API_KEY`, `TAVILY_API_KEY`, `TYPESAFE_API_KEY`.
- `DUFFEL_TEST_API_KEY` for the initial test-mode deployment.
- `CLERK_PUBLISHABLE_KEY` and `CLERK_SECRET_KEY` from the existing development
  Clerk instance; `CLERK_JWT_KEY` is optional.
- `CLERK_WEBHOOK_SIGNING_SECRET`, obtained from the webhook endpoint configured
  for this deployment.
- `CLERK_AUTHORIZED_PARTIES`, set to the exact generated Railway HTTPS origin.

Set these deployment defaults explicitly:

```text
LIVE_MODE=false
LANGSMITH_TRACING=false
PHOENIX_TRACING_ENABLED=false
PHOENIX_TRACE_POPULATION=production
PHOENIX_CAPTURE_CONTENT=false
```

The token allowance settings in `.env.example` remain optional overrides.
Provider credentials stay server-side; only Clerk's publishable key is returned
to the browser through `/auth/config`.

## Domain and Clerk

Generate an HTTPS domain for the application service and set
`CLERK_AUTHORIZED_PARTIES` to that origin. Continue serving both `/` and the API
on this domain. Configure Clerk's `user.created` webhook to:

```text
https://YOUR_RAILWAY_DOMAIN/webhooks/clerk
```

Store that endpoint's signing secret in the application variables. The webhook
rejects invalid signatures; do not disable signature verification to test it.
Do not remove Cloudflare challenge domains from the browser security policy:
they are used by Clerk independently of the application hosting platform.

The application reads Railway's edge-set `X-Real-IP` for per-client rate limits
only when Railway identifies the environment. It validates that value and falls
back to the socket address. Uvicorn does not accept arbitrary proxy chains.

## Deployment and validation

Sign in with `railway login`, link the dedicated project, and deploy the current
checkout with `railway up --service travel-agent`. Inspect deployment status and
logs until `/health` returns `200` with `{"status":"ok"}`. Health probes test
Postgres and Redis without calling model, research, or flight providers.

Verify the homepage and static assets, Clerk sign-in, chat, trip listing/resume,
and a Duffel test-mode flight search. Confirm a second user cannot access the
first user's trip. Restart the application and verify saved trips and
conversation checkpoints remain available.

This workspace is on Hobby. Railway requires Pro for native manual and scheduled
volume backups, so native backups are intentionally unconfigured for this initial
test deployment. Before storing important trips, upgrade and enable daily
Postgres backups with a restore check, or implement an external backup process.
The initial migration starts with empty databases and does not delete or import
any Render database.

After the migration changes are committed and pushed, connect the application
service to `CaedinM/travel-agent`, branch `main`, with repository root `/`.
Keep runtime settings in Railway and build/deployment settings in `railway.json`.

## Local checks

```bash
pytest
npm run test:frontend
docker build --platform linux/amd64 --target production -t travel-agent:railway .
docker compose up --build
```

The regular local app runs at `http://localhost:8000`. Its database/cache URLs
are provided by Compose, while `.env` supplies provider credentials. Railway
private URLs work inside Railway and must not replace the Compose URLs.
