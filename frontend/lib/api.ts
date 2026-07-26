// Typed client for the FastAPI backend.
import type {
  ActionItem,
  ActionItemInput,
  AssistantResponse,
  AuthMe,
  ChatResponse,
  MeetingCreateInput,
  MeetingDetail,
  MeetingListItem,
  MeetingUpdateInput,
  SearchResults,
  SortOption,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    credentials: "include", // send the session cookie
    ...init,
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export interface MeetingQuery {
  search?: string;
  participant?: string;
  meeting_date?: string;
  sort?: SortOption;
}

function queryString(params: MeetingQuery): string {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
  });
  const s = q.toString();
  return s ? `?${s}` : "";
}

export const api = {
  listMeetings: (params: MeetingQuery = {}) =>
    request<MeetingListItem[]>(`/meetings${queryString(params)}`),

  getMeeting: (id: number) => request<MeetingDetail>(`/meetings/${id}`),

  createMeeting: (body: MeetingCreateInput) =>
    request<MeetingDetail>(`/meetings`, { method: "POST", body: JSON.stringify(body) }),

  updateMeeting: (id: number, body: MeetingUpdateInput) =>
    request<MeetingDetail>(`/meetings/${id}`, { method: "PATCH", body: JSON.stringify(body) }),

  deleteMeeting: (id: number) =>
    request<void>(`/meetings/${id}`, { method: "DELETE" }),

  regenerateSummary: (id: number) =>
    request<MeetingDetail>(`/meetings/${id}/generate-summary`, { method: "POST" }),

  uploadMeeting: async (form: FormData): Promise<MeetingDetail> => {
    // Let the browser set the multipart boundary — do NOT set Content-Type.
    const res = await fetch(`${API_BASE}/meetings/upload`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`Upload failed (${res.status})`);
    return res.json();
  },

  addActionItem: (meetingId: number, body: ActionItemInput) =>
    request<ActionItem>(`/meetings/${meetingId}/action-items`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  updateActionItem: (id: number, body: Partial<ActionItemInput>) =>
    request<ActionItem>(`/action-items/${id}`, { method: "PATCH", body: JSON.stringify(body) }),

  deleteActionItem: (id: number) =>
    request<void>(`/action-items/${id}`, { method: "DELETE" }),

  search: (q: string) => request<SearchResults>(`/search?q=${encodeURIComponent(q)}`),

  // RAG-Fusion semantic search (multi-query + reciprocal-rank fusion) across transcripts.
  semanticSearch: (q: string) =>
    request<SearchResults>(`/search/semantic?q=${encodeURIComponent(q)}`),

  chatWithMeeting: (
    id: number,
    question: string,
    history: { role: string; content: string }[] = [],
  ) =>
    request<ChatResponse>(`/meetings/${id}/chat`, {
      method: "POST",
      body: JSON.stringify({ question, history }),
    }),

  exportUrl: (id: number, format: "md" | "txt") =>
    `${API_BASE}/meetings/${id}/export?format=${format}`,

  // Unified assistant — Adaptive RAG router (+ Agentic RAG tool subgraph).
  askAssistant: (
    question: string,
    history: { role: string; content: string }[] = [],
  ) =>
    request<AssistantResponse>(`/assistant`, {
      method: "POST",
      body: JSON.stringify({ question, history }),
    }),

  // Streaming variant (SSE): onToken fires per token, onMeta once at the end.
  askAssistantStream: async (
    question: string,
    history: { role: string; content: string }[],
    handlers: {
      onToken: (t: string) => void;
      onMeta: (meta: Omit<AssistantResponse, "answer">) => void;
    },
  ): Promise<void> => {
    const res = await fetch(`${API_BASE}/assistant/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ question, history }),
    });
    if (!res.ok || !res.body) throw new Error(`Stream failed (${res.status})`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        const line = part.replace(/^data: /, "").trim();
        if (!line || line === "[DONE]") continue;
        try {
          const obj = JSON.parse(line);
          if (typeof obj.token === "string") handlers.onToken(obj.token);
          else if (obj.meta) handlers.onMeta(obj.meta);
        } catch {
          /* ignore partial frame */
        }
      }
    }
  },

  // --- Auth ---
  authMe: () => request<AuthMe>(`/auth/me`),
  logout: () => request<{ ok: boolean }>(`/auth/logout`, { method: "POST" }),
  googleLoginUrl: () => `${API_BASE}/auth/google/login`,
};
