"""SQLModel ORM models — the database schema.

Relationships (all children cascade-delete with their meeting):

    Meeting 1---* Participant
    Meeting 1---* TranscriptSegment
    Meeting 1---1 Summary
    Meeting 1---* ActionItem
    Meeting 1---* Topic
    Meeting *---* Tag   (via MeetingTagLink)
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field, Relationship, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GeneratedBy(str, Enum):
    """Provenance of a summary so the UI can label AI vs seeded content."""

    seeded = "seeded"
    groq = "groq"
    mock = "mock"


# --- Many-to-many link table for tags ---------------------------------------
class MeetingTagLink(SQLModel, table=True):
    __tablename__ = "meeting_tags"

    meeting_id: Optional[int] = Field(
        default=None, foreign_key="meetings.id", primary_key=True
    )
    tag_id: Optional[int] = Field(
        default=None, foreign_key="tags.id", primary_key=True
    )


class Meeting(SQLModel, table=True):
    __tablename__ = "meetings"

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    description: Optional[str] = None
    meeting_date: datetime = Field(default_factory=utcnow, index=True)
    duration_seconds: int = 0
    media_url: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    participants: list["Participant"] = Relationship(
        back_populates="meeting",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    segments: list["TranscriptSegment"] = Relationship(
        back_populates="meeting",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "order_by": "TranscriptSegment.order_index",
        },
    )
    summary: Optional["Summary"] = Relationship(
        back_populates="meeting",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "uselist": False},
    )
    action_items: list["ActionItem"] = Relationship(
        back_populates="meeting",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "order_by": "ActionItem.order_index",
        },
    )
    topics: list["Topic"] = Relationship(
        back_populates="meeting",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "order_by": "Topic.order_index",
        },
    )
    tags: list["Tag"] = Relationship(
        back_populates="meetings", link_model=MeetingTagLink
    )


class Participant(SQLModel, table=True):
    __tablename__ = "participants"

    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: int = Field(foreign_key="meetings.id", index=True)
    name: str
    email: Optional[str] = None
    role: Optional[str] = None

    meeting: Optional[Meeting] = Relationship(back_populates="participants")


class TranscriptSegment(SQLModel, table=True):
    __tablename__ = "transcript_segments"

    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: int = Field(foreign_key="meetings.id", index=True)
    speaker: str
    start_time: float = 0.0  # seconds from start of recording
    end_time: float = 0.0
    text: str
    order_index: int = 0

    meeting: Optional[Meeting] = Relationship(back_populates="segments")


class Summary(SQLModel, table=True):
    __tablename__ = "summaries"

    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: int = Field(foreign_key="meetings.id", index=True, unique=True)
    overview: str = ""
    generated_by: GeneratedBy = Field(default=GeneratedBy.mock)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    meeting: Optional[Meeting] = Relationship(back_populates="summary")


class ActionItem(SQLModel, table=True):
    __tablename__ = "action_items"

    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: int = Field(foreign_key="meetings.id", index=True)
    text: str
    assignee: Optional[str] = None
    completed: bool = False
    due_date: Optional[datetime] = None
    order_index: int = 0
    created_at: datetime = Field(default_factory=utcnow)

    meeting: Optional[Meeting] = Relationship(back_populates="action_items")


class Topic(SQLModel, table=True):
    """A key topic / chapter. ``start_time`` (when present) makes it clickable."""

    __tablename__ = "topics"

    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: int = Field(foreign_key="meetings.id", index=True)
    title: str
    start_time: Optional[float] = None
    order_index: int = 0

    meeting: Optional[Meeting] = Relationship(back_populates="topics")


class User(SQLModel, table=True):
    """A Google-authenticated user. Holds identity + the Google OAuth tokens used by
    the Calendar tool. (Meetings are shared across users for now — no per-user FK yet.)"""

    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    google_sub: str = Field(unique=True, index=True)
    email: str = Field(index=True)
    name: Optional[str] = None
    picture: Optional[str] = None
    google_access_token: Optional[str] = None
    google_refresh_token: Optional[str] = None
    google_token_expiry: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Tag(SQLModel, table=True):
    __tablename__ = "tags"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)

    meetings: list[Meeting] = Relationship(
        back_populates="tags", link_model=MeetingTagLink
    )
