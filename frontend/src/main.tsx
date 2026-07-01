import React from "react";
import ReactDOM from "react-dom/client";
import { AlertCircle, Bot, CheckCircle2, Send, UserRound } from "lucide-react";
import "./styles.css";

type Role = "user" | "assistant";

type Message = {
  role: Role;
  content: string;
};

type Health = {
  api: string;
  ollama: string;
  model: string;
  model_available: boolean;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const initialMessages: Message[] = [
  {
    role: "assistant",
    content:
      "Bonjour, je suis l'assistant financier TechCorp. Donnez-moi un cas business, un document a synthetiser ou une question d'analyse.",
  },
];

function App() {
  const [messages, setMessages] = React.useState<Message[]>(initialMessages);
  const [input, setInput] = React.useState("");
  const [health, setHealth] = React.useState<Health | null>(null);
  const [isSending, setIsSending] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const bottomRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    fetch(`${API_BASE_URL}/api/health`)
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage(event: React.FormEvent) {
    event.preventDefault();
    const content = input.trim();
    if (!content || isSending) return;

    const nextMessages: Message[] = [...messages, { role: "user", content }];
    setMessages([...nextMessages, { role: "assistant", content: "" }]);
    setInput("");
    setIsSending(true);
    setError(null);

    try {
      const payloadMessages = nextMessages
        .filter((message) => message.content.trim() !== "" || message.role === "user")
        .map((message) => ({
          role: message.role,
          content: message.content,
        }));

      const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: payloadMessages,
          temperature: 0.3,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error("Le backend n'a pas pu joindre le modele.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const rawEvent of events) {
          const dataLine = rawEvent.split("\n").find((line) => line.startsWith("data: "));
          if (!dataLine) continue;
          const payload = JSON.parse(dataLine.slice(6));
          if (payload.token) {
            setMessages((current) => {
              const copy = [...current];
              const last = copy[copy.length - 1];
              copy[copy.length - 1] = { ...last, content: last.content + payload.token };
              return copy;
            });
          }
        }
      }
    } catch (err) {
      const detail = err instanceof Error ? err.message : "Erreur inconnue.";
      setError(detail);
      setMessages((current) => {
        const copy = [...current];
        copy[copy.length - 1] = {
          role: "assistant",
          content: "Je n'arrive pas a contacter le modele. Verifiez Ollama et le modele configure.",
        };
        return copy;
      });
    } finally {
      setIsSending(false);
    }
  }

  const ready = health?.ollama === "ok" && health.model_available;

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">TechCorp Industries</p>
            <h1>Console IA Finance</h1>
          </div>
          <div className={`status ${ready ? "ready" : "warning"}`}>
            {ready ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
            <span>{health ? `${health.model} ${ready ? "pret" : "a verifier"}` : "API indisponible"}</span>
          </div>
        </header>

        <div className="chat-panel">
          <div className="messages" aria-live="polite">
            {messages.map((message, index) => (
              <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
                <div className="avatar">{message.role === "assistant" ? <Bot size={18} /> : <UserRound size={18} />}</div>
                <p>{message.content || "..."}</p>
              </article>
            ))}
            <div ref={bottomRef} />
          </div>

          {error && <p className="error">{error}</p>}

          <form className="composer" onSubmit={sendMessage}>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Posez une question finance, risque, marche ou strategie..."
              rows={3}
            />
            <button type="submit" disabled={isSending || !input.trim()} aria-label="Envoyer">
              <Send size={20} />
            </button>
          </form>
        </div>
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
