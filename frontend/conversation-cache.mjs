export const CONVERSATION_CACHE_VERSION = 1;
export const CONVERSATION_CACHE_TTL_MS = 14 * 24 * 60 * 60 * 1000;

const CACHE_PREFIX = "travel-agentic:conversation:";

export const cacheKeyForUser = (userId) => `${CACHE_PREFIX}${encodeURIComponent(userId)}`;

const isMessage = (message) => (
  message
  && typeof message.id === "string"
  && (message.role === "user" || message.role === "assistant")
  && typeof message.content === "string"
  && (message.destinationSuggestions === undefined || Array.isArray(message.destinationSuggestions))
);

const isCacheEntry = (entry) => (
  entry
  && entry.version === CONVERSATION_CACHE_VERSION
  && typeof entry.lastMessageAt === "number"
  && Array.isArray(entry.messages)
  && entry.messages.every(isMessage)
  && (entry.threadId === null || typeof entry.threadId === "string")
  && (entry.tripId === null || typeof entry.tripId === "string")
  && (entry.progress === null || typeof entry.progress === "object")
);

export function readConversationCache({ storage, userId, now = Date.now }) {
  if (!storage || !userId) return null;
  const key = cacheKeyForUser(userId);
  try {
    const entry = JSON.parse(storage.getItem(key) ?? "null");
    if (!isCacheEntry(entry) || now() - entry.lastMessageAt >= CONVERSATION_CACHE_TTL_MS) {
      storage.removeItem(key);
      return null;
    }
    return entry;
  } catch {
    return null;
  }
}

export function writeConversationCache({ storage, userId, conversation }) {
  if (!storage || !userId || !isCacheEntry(conversation)) return false;
  try {
    storage.setItem(cacheKeyForUser(userId), JSON.stringify(conversation));
    return true;
  } catch {
    return false;
  }
}

export function clearConversationCache({ storage, userId }) {
  if (!storage || !userId) return;
  try {
    storage.removeItem(cacheKeyForUser(userId));
  } catch {
    // Browser privacy settings can block local storage. The live conversation still works.
  }
}
