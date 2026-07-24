// TypeScript mirror of the backend Pydantic schemas (the API contract).

export type GeneratedBy = "seeded" | "groq" | "mock";

export interface Participant {
  id: number;
  name: string;
  email?: string | null;
  role?: string | null;
}

export interface Segment {
  id: number;
  speaker: string;
  text: string;
  start_time: number;
  end_time: number;
  order_index: number;
}

export interface Summary {
  id: number;
  overview: string;
  generated_by: GeneratedBy;
  created_at: string;
  updated_at: string;
}

export interface ActionItem {
  id: number;
  text: string;
  assignee?: string | null;
  completed: boolean;
  due_date?: string | null;
  order_index: number;
}

export interface Topic {
  id: number;
  title: string;
  start_time?: number | null;
  order_index: number;
}

export interface Tag {
  id: number;
  name: string;
}

export interface MeetingListItem {
  id: number;
  title: string;
  description?: string | null;
  meeting_date: string;
  duration_seconds: number;
  created_at: string;
  participants: Participant[];
  tags: Tag[];
}

export interface MeetingDetail extends MeetingListItem {
  media_url?: string | null;
  updated_at: string;
  segments: Segment[];
  summary?: Summary | null;
  action_items: ActionItem[];
  topics: Topic[];
}

export interface SearchMatch {
  meeting_id: number;
  meeting_title: string;
  field: "title" | "participant" | "summary" | "transcript";
  snippet: string;
  segment_id?: number | null;
  start_time?: number | null;
}

export interface SearchResults {
  query: string;
  count: number;
  results: SearchMatch[];
}

// A transcript moment a chat/assistant answer was grounded in (RAG citations).
export interface ChatCitation {
  meeting_id?: number | null;
  meeting_title?: string | null;
  speaker?: string | null;
  start_time?: number | null;
  snippet: string;
}

export interface ChatResponse {
  answer: string;
  citations: ChatCitation[];
}

// --- Request payloads ---
export interface ParticipantInput {
  name: string;
  email?: string | null;
  role?: string | null;
}

export interface MeetingCreateInput {
  title: string;
  description?: string | null;
  meeting_date?: string | null;
  participants?: ParticipantInput[];
  transcript_text?: string | null;
  tags?: string[];
  generate_summary?: boolean;
}

export interface MeetingUpdateInput {
  title?: string;
  description?: string | null;
  participants?: ParticipantInput[];
  tags?: string[];
}

export interface ActionItemInput {
  text: string;
  assignee?: string | null;
  completed?: boolean;
}

export type SortOption = "recent" | "oldest" | "title";
