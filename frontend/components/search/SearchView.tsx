"use client";

import { FileText, MessageSquareText, Search, Sparkles, Tag, User } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

import { CenteredSpinner, EmptyState } from "@/components/ui/feedback";
import { useSearch, type SearchMode } from "@/lib/hooks";
import type { SearchMatch } from "@/lib/types";
import { cn } from "@/lib/utils";

const FIELD_META: Record<
  SearchMatch["field"],
  { icon: React.ComponentType<{ size?: number }>; label: string }
> = {
  title: { icon: FileText, label: "Title" },
  participant: { icon: User, label: "Participant" },
  summary: { icon: MessageSquareText, label: "Summary" },
  transcript: { icon: Tag, label: "Transcript" },
};

export default function SearchView() {
  const router = useRouter();
  const params = useSearchParams();
  const initial = params.get("q") ?? "";

  const [value, setValue] = useState(initial);
  const [query, setQuery] = useState(initial);
  const [mode, setMode] = useState<SearchMode>("keyword");

  useEffect(() => {
    setValue(initial);
    setQuery(initial);
  }, [initial]);

  const { data, isFetching } = useSearch(query, mode);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const q = value.trim();
    setQuery(q);
    router.replace(q ? `/search?q=${encodeURIComponent(q)}` : "/search");
  }

  function hrefFor(m: SearchMatch): string {
    return m.field === "transcript" && m.start_time != null
      ? `/meetings/${m.meeting_id}?t=${Math.floor(m.start_time)}`
      : `/meetings/${m.meeting_id}`;
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 md:px-8">
      <h1 className="text-2xl font-semibold tracking-tight text-ink">Search</h1>
      <p className="mb-4 mt-1 text-sm text-muted">
        {mode === "semantic"
          ? "Semantic search — finds transcript moments by meaning, even without exact words (RAG-Fusion)."
          : "Search across every meeting title, participant, summary, and transcript."}
      </p>

      {/* Keyword | Semantic mode toggle */}
      <div className="mb-4 inline-flex rounded-lg border border-line bg-card p-0.5">
        {(["keyword", "semantic"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition",
              mode === m ? "bg-brand text-brand-ink" : "text-muted hover:text-ink",
            )}
          >
            {m === "semantic" ? <Sparkles size={14} /> : <Search size={14} />}
            {m === "semantic" ? "Semantic" : "Keyword"}
          </button>
        ))}
      </div>

      <form onSubmit={submit} className="relative mb-6">
        <Search
          size={18}
          className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-muted"
        />
        <input
          autoFocus
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Search everything…"
          className="h-12 w-full rounded-xl border border-line bg-card pl-11 pr-4 text-[15px] text-ink outline-none transition placeholder:text-muted focus:border-brand focus:ring-2 focus:ring-brand/20"
        />
      </form>

      {!query ? (
        <p className="text-sm text-muted">Type a term above to search.</p>
      ) : isFetching && !data ? (
        <CenteredSpinner />
      ) : data && data.count > 0 ? (
        <>
          <p className="mb-3 text-sm text-muted">
            {data.count} {data.count === 1 ? "result" : "results"} for “{query}”
          </p>
          <div className="space-y-2">
            {data.results.map((m, i) => {
              const meta = FIELD_META[m.field];
              return (
                <Link
                  key={`${m.meeting_id}-${m.field}-${i}`}
                  href={hrefFor(m)}
                  className="flex items-start gap-3 rounded-xl border border-line bg-card p-4 transition hover:border-brand/40 hover:shadow-sm"
                >
                  <span className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-brand-soft text-brand">
                    <meta.icon size={16} />
                  </span>
                  <span className="min-w-0">
                    <span className="flex items-center gap-2">
                      <span className="truncate font-medium text-ink">
                        {m.meeting_title}
                      </span>
                      <span className="shrink-0 text-[11px] font-semibold uppercase tracking-wide text-muted">
                        {meta.label}
                      </span>
                    </span>
                    <span className="mt-0.5 line-clamp-2 text-sm text-muted">
                      {m.snippet}
                    </span>
                  </span>
                </Link>
              );
            })}
          </div>
        </>
      ) : (
        <EmptyState
          icon={<Search size={30} />}
          title="No results"
          description={`Nothing matched “${query}”. Try another term.`}
        />
      )}
    </div>
  );
}
