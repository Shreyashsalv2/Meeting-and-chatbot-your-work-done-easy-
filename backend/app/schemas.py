"""Pydantic request/response schemas (the API contract).

Kept separate from the ORM models so the wire format is explicit and stable.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from .models import GeneratedBy


class ORMModel(BaseModel):
    """Base for read models that are populated from ORM objects."""

    model_config = ConfigDict(from_attributes=True)


# --- Participants -----------------------------------------------------------
class ParticipantCreate(BaseModel):
    name: str
    email: Optional[str] = None
    role: Optional[str] = None


class ParticipantRead(ORMModel):
    id: int
    name: str
    email: Optional[str] = None
    role: Optional[str] = None


# --- Transcript segments ----------------------------------------------------
class SegmentCreate(BaseModel):
    speaker: str
    text: str
    start_time: float = 0.0
    end_time: float = 0.0


class SegmentRead(ORMModel):
    id: int
    speaker: str
    text: str
    start_time: float
    end_time: float
    order_index: int


# --- Summary ----------------------------------------------------------------
class SummaryRead(ORMModel):
    id: int
    overview: str
    generated_by: GeneratedBy
    created_at: datetime
    updated_at: datetime


# --- Action items -----------------------------------------------------------
class ActionItemCreate(BaseModel):
    text: str
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    completed: bool = False


class ActionItemUpdate(BaseModel):
    text: Optional[str] = None
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    completed: Optional[bool] = None


class ActionItemRead(ORMModel):
    id: int
    text: str
    assignee: Optional[str] = None
    completed: bool
    due_date: Optional[datetime] = None
    order_index: int


# --- Topics -----------------------------------------------------------------
class TopicRead(ORMModel):
    id: int
    title: str
    start_time: Optional[float] = None
    order_index: int


# --- Tags -------------------------------------------------------------------
class TagRead(ORMModel):
    id: int
    name: str


# --- Meetings ---------------------------------------------------------------
class MeetingCreate(BaseModel):
    """Create a meeting from a pasted transcript, structured segments, or a form.

    Provide ``transcript_text`` (raw paste/upload) OR ``segments`` (structured).
    """

    title: str
    description: Optional[str] = None
    meeting_date: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    media_url: Optional[str] = None
    participants: Optional[list[ParticipantCreate]] = None
    transcript_text: Optional[str] = None
    segments: Optional[list[SegmentCreate]] = None
    tags: Optional[list[str]] = None
    generate_summary: bool = True


class MeetingUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    meeting_date: Optional[datetime] = None
    media_url: Optional[str] = None
    participants: Optional[list[ParticipantCreate]] = None
    tags: Optional[list[str]] = None


class MeetingListItem(ORMModel):
    """Lightweight shape for the library/dashboard cards."""

    id: int
    title: str
    description: Optional[str] = None
    meeting_date: datetime
    duration_seconds: int
    created_at: datetime
    participants: list[ParticipantRead] = []
    tags: list[TagRead] = []


class MeetingDetail(ORMModel):
    """Full meeting payload for the detail view."""

    id: int
    title: str
    description: Optional[str] = None
    meeting_date: datetime
    duration_seconds: int
    media_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    participants: list[ParticipantRead] = []
    segments: list[SegmentRead] = []
    summary: Optional[SummaryRead] = None
    action_items: list[ActionItemRead] = []
    topics: list[TopicRead] = []
    tags: list[TagRead] = []


# --- Search -----------------------------------------------------------------
class SearchMatch(BaseModel):
    meeting_id: int
    meeting_title: str
    field: str  # "title" | "participant" | "summary" | "transcript"
    snippet: str
    segment_id: Optional[int] = None
    start_time: Optional[float] = None


class SearchResults(BaseModel):
    query: str
    count: int
    results: list[SearchMatch]


# --- Chat -------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[list[ChatMessage]] = None


class ChatCitation(BaseModel):
    """A transcript moment an answer was grounded in (for deep-link chips)."""

    meeting_id: Optional[int] = None
    meeting_title: Optional[str] = None
    speaker: Optional[str] = None
    start_time: Optional[float] = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[ChatCitation] = []


# --- Assistant (unified Adaptive + Agentic RAG) -----------------------------
class AssistantStep(BaseModel):
    """One tool call made by the agentic branch (for the visible trace)."""

    tool: str
    input: str
    output: str


class AssistantArtifact(BaseModel):
    """A downloadable text deliverable produced by a tool (e.g. export-to-text)."""

    filename: str
    content: str


class AssistantOffer(BaseModel):
    """A proactive suggestion chip. Clicking it sends ``prompt`` back as a message."""

    label: str
    prompt: str


class AssistantRequest(BaseModel):
    question: str
    history: Optional[list[ChatMessage]] = None


class AssistantResponse(BaseModel):
    answer: str
    route: str  # which branch fired: no_retrieval | single_meeting | semantic_all | agentic
    task_kind: Optional[str] = None  # factual | actionable | creative
    citations: list[ChatCitation] = []
    steps: list[AssistantStep] = []
    artifact: Optional[AssistantArtifact] = None
    offers: list[AssistantOffer] = []
