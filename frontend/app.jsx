import React, { Component, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Clerk } from "@clerk/clerk-js";
import { ui } from "@clerk/ui";
import "./styles.css";
import {
  CONVERSATION_CACHE_TTL_MS,
  clearConversationCache,
  readConversationCache,
  writeConversationCache,
} from "./conversation-cache.mjs";

const starters = ["Plan a week in Japan this spring", "Find me a long weekend in Lisbon", "Build a food-first trip through Italy"];
const MAX_CHAT_MESSAGE_LENGTH = 400;
const asText = (content) => (typeof content === "string" ? content : String(content ?? ""));
const messageId = () => globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;

function ArrowUp() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 17V7M7.5 11.5 12 7l4.5 4.5" /></svg>; }
function Sparkle() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 2 .85 5.15L18 8l-5.15.85L12 14l-.85-5.15L6 8l5.15-.85L12 2ZM19 15l.43 2.57L22 18l-2.57.43L19 21l-.43-2.57L16 18l2.57-.43L19 15Z" /></svg>; }
function Check() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12.5 4.2 4.2L19 7" /></svg>; }

const phaseNames = { intake: "Intake", research: "Research", flights: "Flights", accommodations: "Accommodations" };
const formatDate = (date) => {
  const match = typeof date === "string" && date.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  return match ? `${match[2]}/${match[3]}/${match[1].slice(-2)}` : "";
};
const formatTime = (timestamp) => {
  const date = timestamp ? new Date(timestamp) : null;
  return date && !Number.isNaN(date.valueOf())
    ? new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(date)
    : "Time unavailable";
};
const formatDuration = (minutes) => {
  if (!Number.isFinite(minutes)) return "Duration unavailable";
  const hours = Math.floor(minutes / 60);
  const remainder = minutes % 60;
  return `${hours}h${remainder ? ` ${remainder}m` : ""}`;
};

function ProgressItem({ label, complete, children }) {
  return <li className={`progress-item ${complete ? "complete" : ""}`}>
    <span className="progress-check">{complete && <Check />}</span>
    <div><p>{label}</p>{children && <span className="progress-value">{children}</span>}</div>
  </li>;
}

function TripProgress({ progress }) {
  const hasExactDates = Boolean(progress?.departure_date && progress?.return_date);
  const dates = hasExactDates
    ? `${formatDate(progress.departure_date)} → ${formatDate(progress.return_date)}`
    : "";
  const destinations = progress?.destinations?.join(" · ") ?? "";
  return <aside className="trip-progress" aria-label="Trip planning progress">
    <div className="progress-heading"><p className="eyebrow">TRIP LEDGER</p><h2>Planning<br /><em>progress</em></h2><span className="phase-pill">{progress ? phaseNames[progress.phase] ?? "Planning" : "Not started"}</span></div>
    <ol className="progress-list">
      <ProgressItem label="Origin" complete={Boolean(progress?.origin)}>{progress?.origin}</ProgressItem>
      <ProgressItem label="Dates" complete={Boolean(dates)}>{dates}</ProgressItem>
      <ProgressItem label="Destinations" complete={Boolean(destinations)}>{destinations}</ProgressItem>
      <ProgressItem label="Travelers" complete={Number.isInteger(progress?.travelers)}>{progress?.travelers}</ProgressItem>
      <ProgressItem label="Flights" complete={Boolean(progress?.flights_ready)} />
      <ProgressItem label="Accommodations" complete={Boolean(progress?.accommodations_ready)} />
    </ol>
  </aside>;
}

function Welcome({ onStarter }) {
  return <div className="welcome"><div className="mark"><Sparkle /></div><p className="eyebrow">YOUR NEXT STORY STARTS HERE</p><h1>Where would you<br /><em>like to go?</em></h1><p className="intro">Tell me the feeling you’re chasing. I’ll turn it into a considered trip, down to the flights.</p><div className="starters">{starters.map((starter) => <button key={starter} onClick={() => onStarter(starter)}>{starter}<span>↗</span></button>)}</div></div>;
}

function Message({ message }) {
  return <article className={`message ${message.role}`}>
    {message.role === "assistant" && <div className="avatar"><Sparkle /></div>}
    <div className="message-body">{asText(message.content)}{message.destinationSuggestions?.length ? <DestinationSuggestions suggestions={message.destinationSuggestions} /> : null}</div>
  </article>;
}

function DestinationSuggestions({ suggestions }) {
  return <section className="destination-suggestions" aria-label="Destination suggestions">
    {suggestions.map((suggestion) => <article key={`${suggestion.country}-${suggestion.city}`} className="destination-card">
      <p>{suggestion.country}</p><h3>{suggestion.city}</h3><span>{suggestion.description}</span>
      {suggestion.reasoning ? <small>{suggestion.reasoning}</small> : null}
    </article>)}
  </section>;
}

function FlightOfferCard({ offer }) {
  const carriers = offer.carriers?.length ? offer.carriers.join(" · ") : "Airline unavailable";
  const stops = offer.stops === 0 ? "Nonstop" : offer.stops === 1 ? "1 stop" : Number.isInteger(offer.stops) ? `${offer.stops} stops` : "Stops unavailable";
  const bag = offer.included_checked_baggage && !offer.included_checked_baggage.startsWith("Unknown")
    ? offer.included_checked_baggage
    : offer.additional_checked_baggage?.[0] ? `Extra bag: ${offer.additional_checked_baggage[0]}` : "Baggage details unavailable";
  return <article className="flight-card" aria-label={`Selected flight from ${offer.origin} to ${offer.destination}`}>
    <div className="flight-card-topline"><span>SELECTED FLIGHT · {String(offer.leg_index + 1).padStart(2, "0")}</span><strong>{offer.price} <small>{offer.currency}</small></strong></div>
    <div className="flight-route"><div><b>{offer.origin}</b><time>{formatTime(offer.departure)}</time></div><div className="flight-line"><i /><span>{formatDuration(offer.duration_minutes)}</span><i /></div><div><b>{offer.destination}</b><time>{formatTime(offer.arrival)}</time></div></div>
    <div className="flight-meta"><span>{carriers}</span><span>{stops}</span></div>
    <div className="flight-bag"><span aria-hidden="true">◒</span>{bag}</div>
  </article>;
}

function FlightOfferCards({ offers }) {
  if (!offers?.length) return null;
  return <section className="flight-offers" aria-label="Selected flight offers">
    <div className="flight-offers-heading"><p className="eyebrow">YOUR ITINERARY</p><span>{offers.length} {offers.length === 1 ? "leg" : "legs"} selected</span></div>
    <div className="flight-offers-row">{offers.map((offer) => <FlightOfferCard key={`${offer.leg_index}-${offer.offer_id}`} offer={offer} />)}</div>
  </section>;
}

function MessageStream({ messages, isSending, error, onStarter, offers }) {
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages.length, isSending, error]);

  return <section className={`conversation ${messages.length ? "has-messages" : ""}`} aria-live="polite" aria-busy={isSending}>
    {!messages.length ? <Welcome onStarter={onStarter} /> : <div className="message-list">
      {messages.map((message) => <Message key={message.id} message={message} />)}
      {!isSending && <FlightOfferCards offers={offers} />}
      {isSending && <article className="message assistant"><div className="avatar"><Sparkle /></div><div className="typing" aria-label="Agent is thinking"><i /><i /><i /></div></article>}
      {error && <p className="stream-error" role="alert">{error}</p>}
    </div>}
    <div ref={endRef} />
  </section>;
}

function ChatComposer({ disabled, tokenLimitReached, onSend }) {
  const [draft, setDraft] = useState("");
  const textareaRef = useRef(null);
  useEffect(() => textareaRef.current?.focus(), []);
  const resize = () => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
  };
  const submit = (event) => {
    event?.preventDefault();
    const message = draft.trim();
    if (!message || disabled) return;
    setDraft("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    onSend(message);
  };
  return <footer className="composer-zone"><form className="composer" onSubmit={submit}>
    <textarea ref={textareaRef} value={draft} disabled={tokenLimitReached} maxLength={MAX_CHAT_MESSAGE_LENGTH} onChange={(event) => { setDraft(event.target.value); resize(); }} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) submit(event); }} placeholder={tokenLimitReached ? "Upgrade to continue planning" : "Ask anything about your next trip…"} rows="1" aria-label="Message travel concierge" />
    <button className="send" type="submit" disabled={!draft.trim() || disabled} aria-label="Send message"><ArrowUp /></button>
  </form><div className="composer-meta"><span className="composer-counter" aria-live="polite">{draft.length}/{MAX_CHAT_MESSAGE_LENGTH}</span><p className="composer-note">{tokenLimitReached ? "Your monthly AI allowance is used. Paid access is coming soon." : "Travel Agentic AI can make mistakes. Check important details."}</p></div></footer>;
}

class ChatErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { failed: false }; }
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? <main className="app-shell fallback"><p>Something interrupted the chat interface.</p><button onClick={() => window.location.reload()}>Reload chat</button></main> : this.props.children; }
}

function UserMenu({ clerk }) {
  const userButtonRef = useRef(null);
  useEffect(() => {
    const container = userButtonRef.current;
    if (!container) return undefined;
    clerk.mountUserButton(container);
    return () => clerk.unmountUserButton(container);
  }, [clerk]);
  return <div className="user-menu" ref={userButtonRef} aria-label="Account menu" />;
}

function App({ clerk }) {
  const [messages, setMessages] = useState([]);
  const [threadId, setThreadId] = useState(null);
  const [tripId, setTripId] = useState(null);
  const [progress, setProgress] = useState(null);
  const [savedTrips, setSavedTrips] = useState([]);
  const [showSavedTrips, setShowSavedTrips] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [tokenLimitReached, setTokenLimitReached] = useState(false);
  const [error, setError] = useState("");
  const [cacheNotice, setCacheNotice] = useState("");
  const progressRequest = useRef(0);
  const conversationVersion = useRef(0);
  const cacheLoaded = useRef(false);
  const userId = clerk.user?.id ?? null;
  const browserStorage = (() => { try { return window.localStorage; } catch { return null; } })();
  const persistConversation = (nextMessages, nextThreadId, nextTripId, nextProgress, messageAdded = false) => {
    if (!cacheLoaded.current || !userId) return;
    const previous = readConversationCache({ storage: browserStorage, userId });
    if (!messageAdded && !previous) return;
    const lastMessageAt = messageAdded ? Date.now() : previous?.lastMessageAt ?? Date.now();
    const saved = writeConversationCache({ storage: browserStorage, userId, conversation: {
      version: 1, lastMessageAt, messages: nextMessages, threadId: nextThreadId, tripId: nextTripId, progress: nextProgress,
    } });
    if (!saved) setCacheNotice("This browser could not save your conversation for later.");
  };
  useEffect(() => {
    if (!userId || cacheLoaded.current) return;
    cacheLoaded.current = true;
    const cached = readConversationCache({ storage: browserStorage, userId });
    if (!cached) return;
    setMessages(cached.messages);
    setThreadId(cached.threadId);
    setTripId(cached.tripId);
    setProgress(cached.progress);
    if (cached.threadId) refreshProgress(cached.threadId, cached);
  }, [userId]);
  useEffect(() => {
    if (!userId || !browserStorage) return undefined;
    const clearIfExpired = () => {
      const cached = readConversationCache({ storage: browserStorage, userId });
      if (cached) return;
      conversationVersion.current += 1;
      setMessages([]); setThreadId(null); setTripId(null); setProgress(null);
    };
    const interval = window.setInterval(clearIfExpired, CONVERSATION_CACHE_TTL_MS);
    window.addEventListener("focus", clearIfExpired);
    return () => { window.clearInterval(interval); window.removeEventListener("focus", clearIfExpired); };
  }, [userId]);
  const refreshProgress = async (nextThreadId, conversationForCache = null) => {
    const requestId = ++progressRequest.current;
    try {
      const token = await clerk.session?.getToken();
      const response = await fetch(`/trips/${encodeURIComponent(nextThreadId)}/progress`, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
      const data = await response.json().catch(() => null);
      if (!response.ok || !data) throw new Error("Trip progress is unavailable.");
      if (requestId === progressRequest.current) {
        setProgress(data);
        if (conversationForCache) {
          persistConversation(
            conversationForCache.messages,
            conversationForCache.threadId,
            conversationForCache.tripId,
            data,
          );
        }
        return data;
      }
    } catch {
      if (requestId === progressRequest.current) setProgress(null);
    }
    return null;
  };
  const loadSavedTrips = async () => {
    try {
      const token = await clerk.session?.getToken();
      const response = await fetch("/trips", { headers: token ? { Authorization: `Bearer ${token}` } : {} });
      const data = await response.json().catch(() => []);
      if (response.ok && Array.isArray(data)) setSavedTrips(data.filter((trip) => trip.status === "active"));
    } catch { setSavedTrips([]); }
  };
  useEffect(() => { loadSavedTrips(); }, [clerk]);
  const resumeSavedTrip = async (savedTripId) => {
    const version = ++conversationVersion.current;
    try {
      const token = await clerk.session?.getToken();
      const response = await fetch(`/trips/${encodeURIComponent(savedTripId)}/resume`, { method: "POST", headers: token ? { Authorization: `Bearer ${token}` } : {} });
      const data = await response.json().catch(() => null);
      if (!response.ok || !data?.thread_id) throw new Error(data?.detail || "That trip could not be resumed.");
      progressRequest.current += 1;
      if (version !== conversationVersion.current) return;
      setThreadId(data.thread_id);
      setTripId(data.trip_id);
      const restoredMessages = [{ id: messageId(), role: "assistant", content: "I’ve loaded this trip plan. What would you like to work on next?" }];
      setMessages(restoredMessages);
      setError("");
      setShowSavedTrips(false);
      const restoredProgress = await refreshProgress(data.thread_id, {
        messages: restoredMessages,
        threadId: data.thread_id,
        tripId: data.trip_id,
      });
      if (version !== conversationVersion.current) return;
      persistConversation(restoredMessages, data.thread_id, data.trip_id, restoredProgress, true);
    } catch (resumeError) {
      if (version === conversationVersion.current) setError(resumeError instanceof Error ? resumeError.message : "That trip could not be resumed.");
    }
  };
  const sendMessage = async (content) => {
    const message = content.trim();
    if (!message || isSending || tokenLimitReached) return;
    const version = conversationVersion.current;
    const userMessage = { id: messageId(), role: "user", content: message };
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    persistConversation(nextMessages, threadId, tripId, progress, true);
    setError("");
    setIsSending(true);
    try {
      const token = await clerk.session?.getToken();
      const response = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) }, body: JSON.stringify({ message, thread_id: threadId }) });
      const data = await response.json().catch(() => null);
      if (version !== conversationVersion.current) return;
      if (response.status === 402) setTokenLimitReached(true);
      if (!response.ok || !data) throw new Error(data?.detail || "The agent couldn’t respond just now.");
      const nextThreadId = data.thread_id ?? threadId;
      setThreadId(nextThreadId);
      setTripId(data.trip_id ?? tripId);
      const refreshedProgress = nextThreadId ? await refreshProgress(nextThreadId, {
        messages: nextMessages,
        threadId: nextThreadId,
        tripId: data.trip_id ?? tripId,
      }) : progress;
      if (version !== conversationVersion.current) return;
      loadSavedTrips();
      const assistantMessage = { id: messageId(), role: "assistant", content: asText(data.response), destinationSuggestions: data.destination_suggestions ?? [] };
      const completedMessages = [...nextMessages, assistantMessage];
      setMessages(completedMessages);
      persistConversation(completedMessages, nextThreadId, data.trip_id ?? tripId, refreshedProgress, true);
    } catch (requestError) {
      if (version === conversationVersion.current) setError(requestError instanceof Error ? requestError.message : "Something went wrong. Please try again.");
    } finally { if (version === conversationVersion.current) setIsSending(false); }
  };
  const resetConversation = () => { conversationVersion.current += 1; progressRequest.current += 1; clearConversationCache({ storage: browserStorage, userId }); setMessages([]); setThreadId(null); setTripId(null); setProgress(null); setError(""); setCacheNotice(""); };
  return <main className="app-shell"><header className="topbar"><a className="wordmark" href="/" aria-label="Travel Agentic home">Travel Agentic<span>.</span></a><div className="saved-trips"><button className="saved-trips-toggle" onClick={() => { setShowSavedTrips((visible) => !visible); if (!showSavedTrips) loadSavedTrips(); }}>Saved trips</button>{showSavedTrips && <div className="saved-trips-menu">{savedTrips.length ? savedTrips.map((trip) => <button key={trip.trip_id} onClick={() => resumeSavedTrip(trip.trip_id)}>{trip.title}</button>) : <p>No saved trips yet.</p>}</div>}</div><button className="new-chat" onClick={resetConversation}>New conversation</button><UserMenu clerk={clerk} /></header><div className="workspace"><div className="chat-column">{cacheNotice && <p className="stream-error" role="status">{cacheNotice}</p>}<MessageStream messages={messages} isSending={isSending} error={error} onStarter={sendMessage} offers={progress?.flights_ready ? progress.flight_offers : []} /><ChatComposer disabled={isSending || tokenLimitReached} tokenLimitReached={tokenLimitReached} onSend={sendMessage} /></div><TripProgress progress={progress} /></div></main>;
}

function SignIn({ clerk }) {
  const signInRef = useRef(null);
  useEffect(() => {
    const container = signInRef.current;
    if (!container) return undefined;
    clerk.mountSignIn(container, {
      forceRedirectUrl: window.location.href,
      appearance: {
        variables: { colorPrimary: "#1f5c47", fontFamily: "Manrope, sans-serif", borderRadius: "12px" },
        elements: {
          rootBox: "clerk-root",
          card: "clerk-card",
          headerTitle: "clerk-heading",
          headerSubtitle: "clerk-subtitle",
          socialButtonsBlockButton: "clerk-social-button",
          formButtonPrimary: "clerk-submit",
        },
      },
    });
    return () => clerk.unmountSignIn(container);
  }, [clerk]);
  return <main className="auth-shell"><section className="auth-content" aria-label="Sign in"><a className="wordmark" href="/">Travel Agentic<span>.</span></a><p className="auth-intro">Sign in to save private trip plans and conversations.</p><div ref={signInRef} /></section></main>;
}

function BootError({ message }) { return <main className="auth-shell"><section className="auth-card"><a className="wordmark" href="/">Travel Agentic<span>.</span></a><p className="error">{message}</p></section></main>; }

async function bootstrap() {
  try {
    const configResponse = await fetch("/auth/config");
    const config = await configResponse.json().catch(() => null);
    if (!configResponse.ok || !config?.publishable_key) throw new Error("Clerk is not configured for this app.");
    const clerk = new Clerk(config.publishable_key);
    await clerk.load({ ui });
    const screen = clerk.isSignedIn ? <App clerk={clerk} /> : <SignIn clerk={clerk} />;
    createRoot(document.getElementById("root")).render(<ChatErrorBoundary>{screen}</ChatErrorBoundary>);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to load sign-in.";
    createRoot(document.getElementById("root")).render(<BootError message={message} />);
  }
}

bootstrap();
