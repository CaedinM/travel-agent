/** Only application settings may cross from Worker bindings into the container. */
const APPLICATION_SETTINGS = [
  "DATABASE_URL", "REDIS_URL", "OPENAI_API_KEY", "TAVILY_API_KEY",
  "TYPESAFE_API_KEY", "DUFFEL_TEST_API_KEY", "DUFFEL_LIVE_API_KEY", "LIVE_MODE",
  "CLERK_PUBLISHABLE_KEY", "CLERK_SECRET_KEY", "CLERK_JWT_KEY",
  "CLERK_AUTHORIZED_PARTIES", "CLERK_WEBHOOK_SIGNING_SECRET",
  "FREE_MONTHLY_TOKEN_LIMIT", "PAID_MONTHLY_TOKEN_LIMIT", "CHAT_TOKEN_RESERVATION",
  "FLIGHT_SELECTION_TOKENS_PER_LEG", "LANGSMITH_TRACING", "LANGSMITH_ENDPOINT",
  "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "PHOENIX_COLLECTOR_ENDPOINT",
  "PHOENIX_API_KEY", "PHOENIX_TRACING_ENABLED", "PHOENIX_RUNTIME_PROJECT",
  "PHOENIX_CAPTURE_CONTENT", "PHOENIX_AGENT_MODEL_VERSION", "PHOENIX_AGENT_PROMPT_VERSION",
] as const;

export function containerEnvironment(bindings: object): Record<string, string> {
  const values = bindings as Record<string, unknown>;
  const environment: Record<string, string> = {
    PHOENIX_TRACE_POPULATION: "production",
  };
  for (const name of APPLICATION_SETTINGS) {
    const value = values[name];
    if (typeof value === "string" && value !== "") environment[name] = value;
  }
  return environment;
}

/** Uvicorn trusts this private proxy; discard client-supplied forwarding chains. */
export function containerRequest(request: Request): Request {
  const forwarded = new Request(request);
  forwarded.headers.delete("Forwarded");
  forwarded.headers.delete("X-Forwarded-Host");
  forwarded.headers.delete("X-Real-IP");
  forwarded.headers.delete("X-Forwarded-For");
  const clientIp = request.headers.get("CF-Connecting-IP");
  if (clientIp) forwarded.headers.set("X-Forwarded-For", clientIp);
  forwarded.headers.set("X-Forwarded-Proto", new URL(request.url).protocol.slice(0, -1));
  return forwarded;
}
