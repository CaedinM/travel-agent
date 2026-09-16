import React, { Component, useEffect, useRef, useState } from "https://esm.sh/react@18.3.1";
import { createRoot } from "https://esm.sh/react-dom@18.3.1/client";

const starters = ["Plan a week in Japan this spring", "Find me a long weekend in Lisbon", "Build a food-first trip through Italy"];
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
    : progress?.season ?? "";
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
    <div className="message-body">{asText(message.content)}</div>
  </article>;
}

function MessageStream({ messages, isSending, error, onStarter }) {
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages.length, isSending, error]);

  return <section className={`conversation ${messages.length ? "has-messages" : ""}`} aria-live="polite" aria-busy={isSending}>
    {!messages.length ? <Welcome onStarter={onStarter} /> : <div className="message-list">
      {messages.map((message) => <Message key={message.id} message={message} />)}
      {isSending && <article className="message assistant"><div className="avatar"><Sparkle /></div><div className="typing" aria-label="Agent is thinking"><i /><i /><i /></div></article>}
      {error && <p className="stream-error" role="alert">{error}</p>}
    </div>}
    <div ref={endRef} />
  </section>;
}

function ChatComposer({ disabled, onSend }) {
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
    <textarea ref={textareaRef} value={draft} onChange={(event) => { setDraft(event.target.value); resize(); }} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) submit(event); }} placeholder="Ask anything about your next trip…" rows="1" aria-label="Message travel concierge" />
    <button className="send" type="submit" disabled={!draft.trim() || disabled} aria-label="Send message"><ArrowUp /></button>
  </form><p className="composer-note">Travel Agentic AI can make mistakes. Check important details.</p></footer>;
}

class ChatErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { failed: false }; }
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? <main className="app-shell fallback"><p>Something interrupted the chat interface.</p><button onClick={() => window.location.reload()}>Reload chat</button></main> : this.props.children; }
}

function App() {
  const [messages, setMessages] = useState([]);
  const [threadId, setThreadId] = useState(null);
  const [progress, setProgress] = useState(null);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");
  const progressRequest = useRef(0);
  const refreshProgress = async (nextThreadId) => {
    const requestId = ++progressRequest.current;
    try {
      const response = await fetch(`/trips/${encodeURIComponent(nextThreadId)}/progress`);
      const data = await response.json().catch(() => null);
      if (!response.ok || !data) throw new Error("Trip progress is unavailable.");
      if (requestId === progressRequest.current) setProgress(data);
    } catch {
      if (requestId === progressRequest.current) setProgress(null);
    }
  };
  const sendMessage = async (content) => {
    const message = content.trim();
    if (!message || isSending) return;
    setMessages((current) => [...current, { id: messageId(), role: "user", content: message }]);
    setError("");
    setIsSending(true);
    try {
      const response = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, thread_id: threadId }) });
      const data = await response.json().catch(() => null);
      if (!response.ok || !data) throw new Error(data?.detail || "The agent couldn’t respond just now.");
      const nextThreadId = data.thread_id ?? threadId;
      setThreadId(nextThreadId);
      if (nextThreadId) await refreshProgress(nextThreadId);
      setMessages((current) => [...current, { id: messageId(), role: "assistant", content: asText(data.response) }]);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Something went wrong. Please try again.");
    } finally { setIsSending(false); }
  };
  const resetConversation = () => { progressRequest.current += 1; setMessages([]); setThreadId(null); setProgress(null); setError(""); };
  return <main className="app-shell"><header className="topbar"><a className="wordmark" href="/" aria-label="Travel Agentic home">Travel Agentic<span>.</span></a><button className="new-chat" onClick={resetConversation}>New conversation</button></header><div className="workspace"><div className="chat-column"><MessageStream messages={messages} isSending={isSending} error={error} onStarter={sendMessage} /><ChatComposer disabled={isSending} onSend={sendMessage} /></div><TripProgress progress={progress} /></div></main>;
}

createRoot(document.getElementById("root")).render(<ChatErrorBoundary><App /></ChatErrorBoundary>);
