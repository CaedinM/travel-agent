import { Container, getContainer } from "@cloudflare/containers";
import { containerEnvironment, containerRequest } from "./runtime";

interface Env {
  TRAVEL_AGENT: DurableObjectNamespace<TravelAgentContainer>;
}

export class TravelAgentContainer extends Container<Env> {
  defaultPort = 8000;
  sleepAfter = "10m";
  envVars = containerEnvironment(this.env);
  // The image has no public listener; only this Worker can forward to Uvicorn.
  entrypoint = [
    "uvicorn", "travel_agent.api.app:app", "--app-dir", "src",
    "--host", "0.0.0.0", "--port", "8000",
    "--proxy-headers", "--forwarded-allow-ips", "*",
  ];
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // A stable instance keeps background prefetch work available to the next
    // request. Durable trip state and checkpoints remain in external Postgres.
    const container = getContainer(env.TRAVEL_AGENT, "travel-agent");
    return container.fetch(containerRequest(request));
  },
} satisfies ExportedHandler<Env>;
