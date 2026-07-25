"use client";

import { Bot, Clock, Download, Send, Sparkles, Wrench } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { Spinner } from "@/components/ui/feedback";
import { api } from "@/lib/api";
import { formatDuration } from "@/lib/format";
import type {
  AssistantArtifact,
  AssistantOffer,
  AssistantResponse,
  AssistantStep,
  ChatCitation,
} from "@/lib/types";
import { cn } from "@/lib/utils";

type Msg = {
  role: "user" | "assistant";
  content: string;
  route?: string;
  citations?: ChatCitation[];
  steps?: AssistantStep[];
  artifact?: AssistantArtifact | null;
  offers?: AssistantOffer[];
};

// Guard the shared Groq key on the public site.
const MAX_USER_MESSAGES = 20;

const ROUTE_META: Record<string, { label: string; cls: string }> = {
  no_retrieval: { label: "Direct answer", cls: "bg-panel text-muted" },
  single_meeting: { label: "One meeting", cls: "bg-brand-soft text-brand" },
  semantic_all: { label: "All meetings", cls: "bg-brand-soft text-brand" },
  agentic: { label: "Agent + tools", cls: "bg-amber-100 text-amber-700 dark:bg-amber-500/15 dark:text-amber-400" },
};

const EXAMPLES = [
  "What did we decide about onboarding across all meetings?",
  "In the Weekly Engineering Sync, what caused the database outage?",
  "Help me schedule a follow-up for the database outage action items",
];

function downloadArtifact(a: AssistantArtifact) {
  const blob = new Blob([a.content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = a.filename || "assistant-output.txt";
  // The link must be in the DOM, and the blob URL must NOT be revoked
  // synchronously — revoking too early aborts the download before it starts.
  document.body.appendChild(link);
  link.click();
  setTimeout(() => {
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, 200);
}

export default function AssistantChat() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  const capped = messages.filter((m) => m.role === "user").length >= MAX_USER_MESSAGES;

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  // Patch the last message (the streaming assistant bubble) in place.
  function patchLast(patch: Partial<Msg> | ((prev: Msg) => Partial<Msg>)) {
    setMessages((m) => {
      const copy = [...m];
      const i = copy.length - 1;
      if (i < 0) return copy;
      const p = typeof patch === "function" ? patch(copy[i]) : patch;
      copy[i] = { ...copy[i], ...p };
      return copy;
    });
  }

  async function send(text?: string) {
    const q = (text ?? input).trim();
    if (!q || loading || capped) return;
    const history = messages.slice(-8).map((m) => ({ role: m.role, content: m.content }));
    // Add the user turn AND an empty assistant bubble that streams in.
    setMessages((m) => [
      ...m,
      { role: "user", content: q },
      { role: "assistant", content: "" },
    ]);
    setInput("");
    setLoading(true);
    try {
      await api.askAssistantStream(q, history, {
        onToken: (t) => patchLast((prev) => ({ content: prev.content + t })),
        onMeta: (meta) =>
          patchLast({
            route: meta.route,
            citations: meta.citations,
            steps: meta.steps,
            artifact: meta.artifact ?? null,
            offers: meta.offers,
          }),
      });
      // If the stream produced nothing, show a fallback.
      patchLast((prev) => ({
        content: prev.content || "Sorry, I couldn't answer that just now.",
      }));
    } catch {
      toast.error("Couldn't get an answer — try again.");
      patchLast((prev) => ({
        content: prev.content || "Sorry, I couldn't answer that just now.",
      }));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col px-4 py-6 md:px-8">
      <div className="mb-4 shrink-0">
        <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight text-ink">
          <Sparkles size={22} className="text-brand" /> Assistant
        </h1>
        <p className="mt-1 text-sm text-muted">
          Ask across all your meetings and get work done from here — I answer, research, and draft
          deliverables. For actions I can&apos;t automate yet, I&apos;ll guide you through them.
        </p>
      </div>

      <div ref={listRef} className="scroll-slim flex-1 space-y-4 overflow-y-auto pb-4">
        {messages.length === 0 && !loading && (
          <div className="rounded-2xl border border-line bg-card p-5">
            <p className="text-sm text-muted">Try asking:</p>
            <div className="mt-3 flex flex-col gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => send(ex)}
                  className="rounded-lg border border-line bg-canvas px-3 py-2 text-left text-sm text-ink transition hover:border-brand hover:text-brand"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={cn("flex flex-col gap-2", m.role === "user" ? "items-end" : "items-start")}
          >
            {m.role === "assistant" && m.route && (
              <span
                className={cn(
                  "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide",
                  ROUTE_META[m.route]?.cls ?? "bg-panel text-muted",
                )}
              >
                {m.route === "agentic" ? <Wrench size={11} /> : <Bot size={11} />}
                {ROUTE_META[m.route]?.label ?? m.route}
              </span>
            )}

            <div
              className={cn(
                "max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm",
                m.role === "user" ? "bg-brand text-brand-ink" : "bg-panel text-ink",
              )}
            >
              {m.content ? (
                m.content
              ) : m.role === "assistant" ? (
                <span className="inline-flex items-center gap-2 text-muted">
                  <Spinner /> thinking…
                </span>
              ) : null}
            </div>

            {/* Tool-call trace (agentic branch — populated in Phase E) */}
            {m.steps && m.steps.length > 0 && (
              <details className="max-w-[85%] rounded-lg border border-line bg-card px-3 py-2 text-xs text-muted">
                <summary className="cursor-pointer font-medium text-ink">
                  {m.steps.length} tool {m.steps.length === 1 ? "step" : "steps"}
                </summary>
                <ol className="mt-2 space-y-2">
                  {m.steps.map((s, j) => (
                    <li key={j}>
                      <span className="font-mono font-semibold text-brand">{s.tool}</span>
                      <span className="text-muted"> ({s.input})</span>
                      <p className="mt-0.5 line-clamp-3 whitespace-pre-wrap">{s.output}</p>
                    </li>
                  ))}
                </ol>
              </details>
            )}

            {/* Cross-meeting citations */}
            {m.citations && m.citations.length > 0 && (
              <div className="flex max-w-[85%] flex-wrap gap-1.5">
                {m.citations.map((c, j) => (
                  <Link
                    key={j}
                    href={
                      c.start_time != null
                        ? `/meetings/${c.meeting_id}?t=${Math.floor(c.start_time)}`
                        : `/meetings/${c.meeting_id}`
                    }
                    title={c.snippet}
                    className="inline-flex items-center gap-1 rounded-full border border-line bg-canvas px-2 py-0.5 text-xs text-muted transition hover:border-brand hover:text-brand"
                  >
                    <Clock size={11} />
                    {c.meeting_title ? `${c.meeting_title} · ` : ""}
                    {formatDuration(c.start_time ?? 0)}
                  </Link>
                ))}
              </div>
            )}

            {/* Downloadable artifact (export-to-text — Phase E) */}
            {m.artifact && (
              <button
                onClick={() => downloadArtifact(m.artifact!)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-brand bg-brand-soft px-3 py-1.5 text-xs font-medium text-brand transition hover:bg-brand hover:text-brand-ink"
              >
                <Download size={13} /> {m.artifact.filename}
              </button>
            )}

            {/* Proactive download suggestions (only when nothing was produced yet) */}
            {m.role === "assistant" && !m.artifact && m.offers && m.offers.length > 0 && (
              <div className="flex max-w-[85%] flex-wrap gap-1.5">
                {m.offers.map((o, j) => (
                  <button
                    key={j}
                    onClick={() => send(o.prompt)}
                    disabled={loading || capped}
                    className="inline-flex items-center gap-1 rounded-full border border-brand/40 bg-brand-soft px-3 py-1 text-xs font-medium text-brand transition hover:bg-brand hover:text-brand-ink disabled:opacity-50"
                  >
                    {o.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="shrink-0 border-t border-line pt-3">
        {capped ? (
          <p className="py-2 text-center text-xs text-muted">
            Message limit reached for this session.
          </p>
        ) : (
          <div className="flex items-center gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()}
              placeholder="Ask about your meetings…"
              className="h-11 flex-1 rounded-xl border border-line bg-card px-4 text-sm text-ink outline-none transition placeholder:text-muted focus:border-brand focus:ring-2 focus:ring-brand/20"
            />
            <button
              onClick={() => send()}
              disabled={loading || !input.trim()}
              aria-label="Send"
              className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-brand text-brand-ink transition hover:bg-brand-strong disabled:opacity-50"
            >
              <Send size={17} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
