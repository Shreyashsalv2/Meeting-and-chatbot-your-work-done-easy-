"use client";

import { Clock, MessageSquareText, Send, Sparkles, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { Spinner } from "@/components/ui/feedback";
import { api } from "@/lib/api";
import { formatDuration } from "@/lib/format";
import type { ChatCitation } from "@/lib/types";
import { cn } from "@/lib/utils";

type Msg = { role: "user" | "assistant"; content: string; citations?: ChatCitation[] };

// Light guard against abuse of the shared Groq key on the public site.
const MAX_USER_MESSAGES = 20;

export function MeetingChat({
  meetingId,
  onSeek,
}: {
  meetingId: number;
  onSeek?: (startTime: number) => void;
}) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  const userCount = messages.filter((m) => m.role === "user").length;
  const capped = userCount >= MAX_USER_MESSAGES;

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading, open]);

  async function send() {
    const q = input.trim();
    if (!q || loading || capped) return;
    const history = messages.slice(-8);
    setMessages((m) => [...m, { role: "user", content: q }]);
    setInput("");
    setLoading(true);
    try {
      const { answer, citations } = await api.chatWithMeeting(meetingId, q, history);
      setMessages((m) => [...m, { role: "assistant", content: answer, citations }]);
    } catch {
      toast.error("Couldn't get an answer — try again.");
      setMessages((m) => [
        ...m,
        { role: "assistant", content: "Sorry, I couldn't answer that just now." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      {/* Floating toggle */}
      <button
        onClick={() => setOpen((o) => !o)}
        aria-label={open ? "Close chat" : "Ask about this meeting"}
        className="fixed bottom-5 right-5 z-40 flex h-12 items-center gap-2 rounded-full bg-brand px-4 text-brand-ink shadow-lg transition hover:bg-brand-strong"
      >
        {open ? <X size={18} /> : <MessageSquareText size={18} />}
        {!open && <span className="text-sm font-medium">Ask this meeting</span>}
      </button>

      {/* Chat panel */}
      {open && (
        <div className="fixed bottom-20 right-5 z-40 flex h-[28rem] w-[22rem] max-w-[calc(100vw-2.5rem)] flex-col overflow-hidden rounded-2xl border border-line bg-card shadow-2xl">
          <div className="flex items-center gap-2 border-b border-line px-4 py-3">
            <Sparkles size={16} className="text-brand" />
            <span className="text-sm font-semibold text-ink">Ask about this meeting</span>
          </div>

          <div ref={listRef} className="scroll-slim flex-1 space-y-3 overflow-y-auto p-3">
            {messages.length === 0 && !loading && (
              <p className="px-1 pt-1 text-sm text-muted">
                Ask anything about this meeting — I answer from the transcript, and I&apos;ll tell
                you if something isn&apos;t covered.
              </p>
            )}
            {messages.map((m, i) => (
              <div
                key={i}
                className={cn(
                  "flex flex-col gap-1.5",
                  m.role === "user" ? "items-end" : "items-start",
                )}
              >
                <div
                  className={cn(
                    "max-w-[85%] whitespace-pre-wrap rounded-2xl px-3 py-2 text-sm",
                    m.role === "user" ? "bg-brand text-brand-ink" : "bg-panel text-ink",
                  )}
                >
                  {m.content}
                </div>
                {m.role === "assistant" && m.citations && m.citations.length > 0 && (
                  <div className="flex max-w-[85%] flex-wrap gap-1">
                    {m.citations.map((c, j) => (
                      <button
                        key={j}
                        onClick={() => c.start_time != null && onSeek?.(c.start_time)}
                        title={c.snippet}
                        className="inline-flex items-center gap-1 rounded-full border border-line bg-canvas px-2 py-0.5 text-xs text-muted transition hover:border-brand hover:text-brand"
                      >
                        <Clock size={11} />
                        {c.speaker ? `${c.speaker} · ` : ""}
                        {formatDuration(c.start_time ?? 0)}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="flex items-center gap-2 rounded-2xl bg-panel px-3 py-2 text-sm text-muted">
                  <Spinner /> thinking…
                </div>
              </div>
            )}
          </div>

          <div className="border-t border-line p-2">
            {capped ? (
              <p className="px-2 py-1.5 text-center text-xs text-muted">
                Message limit reached for this session.
              </p>
            ) : (
              <div className="flex items-center gap-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && send()}
                  placeholder="Ask a question…"
                  className="flex-1 rounded-lg border border-line bg-canvas px-3 py-2 text-sm text-ink outline-none transition placeholder:text-muted focus:border-brand"
                />
                <button
                  onClick={send}
                  disabled={loading || !input.trim()}
                  aria-label="Send"
                  className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-brand text-brand-ink transition hover:bg-brand-strong disabled:opacity-50"
                >
                  <Send size={16} />
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
