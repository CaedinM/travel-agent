import assert from "node:assert/strict";
import test from "node:test";
import { containerEnvironment, containerRequest } from "./runtime.ts";

test("only allowlisted, nonempty app settings reach the container", () => {
  assert.deepEqual(containerEnvironment({
    DATABASE_URL: "postgresql://database", OPENAI_API_KEY: "test-key",
    CLERK_JWT_KEY: "line one\nline two", REDIS_URL: "", LIVE_MODE: false,
    CLOUDFLARE_API_TOKEN: "deployment-only", TRAVEL_AGENT: { binding: true },
    PHOENIX_TRACE_POPULATION: "manual",
  }), {
    PHOENIX_TRACE_POPULATION: "production", DATABASE_URL: "postgresql://database",
    OPENAI_API_KEY: "test-key", CLERK_JWT_KEY: "line one\nline two",
  });
});

test("the trusted Cloudflare IP replaces spoofed forwarding headers", async () => {
  const original = new Request("https://travel.example/chat", {
    method: "POST", body: '{"message":"hello"}', headers: {
      "CF-Connecting-IP": "203.0.113.7", "X-Forwarded-For": "spoofed",
      "X-Forwarded-Proto": "http", "X-Forwarded-Host": "evil.example",
      "Forwarded": "for=spoofed", "X-Real-IP": "spoofed",
      "Authorization": "Bearer session", "Content-Type": "application/json",
    },
  });
  const forwarded = containerRequest(original);
  assert.equal(forwarded.headers.get("X-Forwarded-For"), "203.0.113.7");
  assert.equal(forwarded.headers.get("X-Forwarded-Proto"), "https");
  for (const name of ["Forwarded", "X-Forwarded-Host", "X-Real-IP"]) {
    assert.equal(forwarded.headers.get(name), null);
  }
  assert.equal(forwarded.headers.get("Authorization"), "Bearer session");
  assert.equal(forwarded.url, original.url);
  assert.equal(await forwarded.text(), '{"message":"hello"}');
  assert.equal(original.headers.get("X-Forwarded-For"), "spoofed");
});

test("missing Cloudflare IP does not allow a client-supplied forwarding chain", () => {
  const forwarded = containerRequest(new Request("http://localhost/chat", {
    headers: { "X-Forwarded-For": "spoofed" },
  }));
  assert.equal(forwarded.headers.get("X-Forwarded-For"), null);
  assert.equal(forwarded.headers.get("X-Forwarded-Proto"), "http");
});
