# Deploy to Cloudflare Workers + Containers

The Worker forwards requests to the existing Python 3.12 FastAPI container.
This preserves native Python dependencies, direct Postgres/Redis connections,
Clerk authentication, and background Duffel prefetches. It is a Containers
deployment behind a Worker, rather than a native Python Worker, so `pywrangler`
and a Python Workers `pyproject.toml` are not used.

## Requirements

- A Cloudflare account on a Workers Paid plan with Containers enabled. Container
  usage is billed separately; see [Containers pricing](https://developers.cloudflare.com/containers/pricing/).
- Node.js 24 and Docker Desktop running for local builds/deployment.
- Externally reachable Postgres and Redis services. Cloudflare does not provision
  either service from `compose.yaml`.
- The existing application provider credentials and a production Clerk instance.

## Build and deploy commands

In Cloudflare **Workers & Pages → your Worker → Settings → Builds**, use the
repository root and set:

- **Build command:** `npm ci && npm run check:cloudflare && npm run build`
- **Deploy command:** `npm run deploy`

Workers Builds can build the Dockerfile during deployment. The deploy command
runs `wrangler deploy`, which builds and pushes the container image as well as
publishing the Worker. The Dockerfile also builds the frontend, so local deploys
do not depend on a pre-existing `frontend/dist` directory. The separate dashboard
build validates the frontend before publishing.

For a local deployment:

```bash
npm ci
npx wrangler login
npm run check:cloudflare
npm run test:cloudflare
npm run build:container
npm run deploy
```

`build:container` validates a Linux AMD64 image locally; Wrangler manages its own
image build and upload. Node/TypeScript never replaces the Python backend.
For details see [Cloudflare's deployment guide](https://developers.cloudflare.com/containers/guides/deploy/).

Disable automatic non-production branch deploys initially. The default
`wrangler versions upload` does not update container images or provide a
full-app preview URL. A staging deployment needs its own Worker configuration
and secrets and must use a full `wrangler deploy`.

## Application secrets and settings

Set these under the Worker's **Settings → Variables and Secrets**, or enter each
one at the interactive prompt from `npx wrangler secret put NAME`:

- `DATABASE_URL`: external Postgres connection URL, including provider-required
  TLS settings (normally `sslmode=require`). Use a direct or session-pooled URL;
  the LangGraph checkpointer uses prepared statements and needs a compatible
  connection. A Render internal database hostname is not reachable here.
- `REDIS_URL`: externally reachable Redis URL, normally `rediss://` for TLS.
- `OPENAI_API_KEY`, `TAVILY_API_KEY`, `TYPESAFE_API_KEY`.
- `DUFFEL_TEST_API_KEY` (default test mode).
- `CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`,
  `CLERK_WEBHOOK_SIGNING_SECRET`.
- `CLERK_AUTHORIZED_PARTIES`: comma-separated exact origins, such as
  `https://travel.example.com`. Include the workers.dev origin if it is used.

For example:

```bash
npx wrangler secret put DATABASE_URL
npx wrangler secret put REDIS_URL
```

Repeat for the other required settings. Secrets entered at these prompts are
runtime bindings, not Docker build arguments. Do not put credentials in
`wrangler.jsonc`, build logs, or the frontend. Workers Builds build-time variables
alone do not configure the running application.

Optional settings from `.env.example` can also be supplied as Worker bindings.
For real flight offers, supply `DUFFEL_LIVE_API_KEY` and deliberately change
`LIVE_MODE` to `"true"` in `wrangler.jsonc`. The committed configuration disables
LangSmith and Phoenix tracing. To enable them, update their flags and supply the
provider settings; Phoenix runtime traces are always tagged `production` by the
Worker. Never set the trace population to `manual` for deployed traffic.

The Worker explicitly allowlists settings forwarded into the container in
`cloudflare/runtime.ts`. Add any new application setting there when adding it to
the Python application. Non-secret variables managed in `wrangler.jsonc` should
be edited there, since a deploy can overwrite dashboard values.

After changing runtime secrets, stop the existing container instance from the
Cloudflare dashboard so its next start receives the new environment. Deploying
unchanged container configuration may not restart an already-running process.

## Storage, routing, and cutover from Render

The deployment uses one `basic` instance with a stable ID, sleeping after ten
minutes without activity. This keeps the existing in-process flight-prefetch
cache useful across requests. A restart can lose a prefetch; the flight-selection
endpoint starts a fresh search on a cache miss. Conversations use the existing
Postgres checkpointer and trip snapshots; the container disk is not durable.
Do not raise `max_instances` or introduce random routing without first addressing
the shared prefetch cache and concurrent chat updates.

The Worker forwards the Cloudflare client IP to the private Uvicorn listener,
replacing incoming forwarding headers so Redis limits apply per client. The
frontend remains served by FastAPI at `/` and `/static`, preserving same-origin
authentication and security headers. No separate Pages deployment is needed.

Before switching traffic:

1. Back up the Render database. If moving Postgres, migrate the entire database,
   including application tables and LangGraph checkpoint tables. Pause writes
   during the final copy to avoid losing trips or token usage. If retaining the
   database temporarily, use its external TLS URL and keep that service running.
2. Provision/migrate Redis and configure its external TLS endpoint. Network
   access rules must permit connections from the container.
3. Configure the Worker secrets, deploy, and allow several minutes for first-time
   container provisioning. Attach the production custom domain in Cloudflare.
4. Add the new origin/domain to Clerk's production application settings and
   `CLERK_AUTHORIZED_PARTIES`. Point Clerk's signed webhook to
   `https://YOUR_DOMAIN/webhooks/clerk` and use that endpoint's signing secret.
5. Verify the homepage, sign-in, saved trip list/resume, a new chat, and a Duffel
   test-mode search. Verify that another signed-in user cannot access the trip.
6. Stop the old Render API only after those checks pass. Retain the database
   backup and any still-used Postgres/Redis services.

No account resources are created by adding these files. Credentials, service
provisioning, data migration, domain setup, and a live deployment are separate
steps; local tests do not verify those external integrations.

## Local Worker testing

The normal `docker compose up --build` workflow is unchanged. It uses the
Dockerfile's `development` target, which includes pytest and the test fixtures;
Cloudflare uses the final `production` target.

To exercise the Worker proxy locally, copy `.dev.vars.example` to `.dev.vars`,
fill in credentials and reachable database/cache URLs, then run:

```bash
npm run dev:cloudflare
```

Open `http://localhost:8787`. The regular Compose Postgres host port is 5433;
inside the local Worker container use `host.docker.internal` instead of
`localhost`. Compose Redis is private to its network, so use a reachable Redis
endpoint or publish it explicitly for this workflow. Both `.dev.vars` and local
Wrangler state are ignored by Git and excluded from Docker build context.

Run the deterministic proxy checks with:

```bash
npm run check:cloudflare
npm run test:cloudflare
```
