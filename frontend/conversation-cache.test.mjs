import assert from "node:assert/strict";
import test from "node:test";
import {
  CONVERSATION_CACHE_TTL_MS,
  cacheKeyForUser,
  clearConversationCache,
  readConversationCache,
  writeConversationCache,
} from "./conversation-cache.mjs";

function storage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
}

const conversation = (lastMessageAt = 1_000) => ({
  version: 1,
  lastMessageAt,
  messages: [{ id: "message-1", role: "assistant", content: "Welcome", destinationSuggestions: [] }],
  threadId: "thread-1",
  tripId: "trip-1",
  progress: { phase: "intake" },
});

test("restores the complete cached conversation only for its signed-in user", () => {
  const browserStorage = storage();
  const cached = conversation();
  assert.equal(writeConversationCache({ storage: browserStorage, userId: "user-a", conversation: cached }), true);
  assert.deepEqual(readConversationCache({ storage: browserStorage, userId: "user-a", now: () => 2_000 }), cached);
  assert.equal(readConversationCache({ storage: browserStorage, userId: "user-b", now: () => 2_000 }), null);
});

test("clears expired, malformed, and explicitly discarded conversations", () => {
  const browserStorage = storage();
  writeConversationCache({ storage: browserStorage, userId: "user-a", conversation: conversation() });
  assert.equal(readConversationCache({ storage: browserStorage, userId: "user-a", now: () => 1_000 + CONVERSATION_CACHE_TTL_MS }), null);

  browserStorage.setItem(cacheKeyForUser("user-a"), "not json");
  assert.equal(readConversationCache({ storage: browserStorage, userId: "user-a", now: () => 2_000 }), null);

  writeConversationCache({ storage: browserStorage, userId: "user-a", conversation: conversation() });
  clearConversationCache({ storage: browserStorage, userId: "user-a" });
  assert.equal(readConversationCache({ storage: browserStorage, userId: "user-a", now: () => 2_000 }), null);
});
