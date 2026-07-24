"""Reusable persistence logic shared across routers.

Keeps routers thin and centralizes the non-trivial parts: turning a create
payload into a meeting graph, and (re)applying AI-generated insights.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlmodel import Session, select

from . import models, schemas, transcript_parser
from .models import GeneratedBy
from .services import groq_service

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def index_meeting_safe(meeting: "models.Meeting") -> None:
    """Best-effort RAG (re)index of a meeting. Must never break CRUD."""
    try:
        from .services.rag import vector_store

        vector_store.index_meeting(meeting)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG index failed for meeting %s: %s", getattr(meeting, "id", "?"), exc)


def deindex_meeting_safe(meeting_id: int) -> None:
    """Best-effort removal of a meeting's chunks from the vector store."""
    try:
        from .services.rag import vector_store

        vector_store.remove_meeting(meeting_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG de-index failed for meeting %s: %s", meeting_id, exc)


def _estimate_end(start: float, text: str) -> float:
    words = len(text.split())
    return round(start + max(2.0, words / 2.5), 2)


def get_or_create_tags(session: Session, names: list[str] | None) -> list[models.Tag]:
    tags: list[models.Tag] = []
    for raw in names or []:
        name = raw.strip()
        if not name:
            continue
        tag = session.exec(select(models.Tag).where(models.Tag.name == name)).first()
        if tag is None:
            tag = models.Tag(name=name)
            session.add(tag)
            session.flush()
        tags.append(tag)
    return tags


def build_segments(payload: schemas.MeetingCreate) -> list[dict]:
    """Normalize the create payload into segment dicts."""
    if payload.segments:
        segments: list[dict] = []
        prev_end = 0.0
        for i, s in enumerate(payload.segments):
            start = s.start_time if s.start_time else prev_end
            end = s.end_time if s.end_time and s.end_time > start else _estimate_end(start, s.text)
            prev_end = end
            segments.append(
                {
                    "speaker": s.speaker,
                    "text": s.text,
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "order_index": i,
                }
            )
        return segments
    if payload.transcript_text:
        return transcript_parser.parse_transcript(payload.transcript_text)
    return []


def create_meeting(session: Session, payload: schemas.MeetingCreate) -> models.Meeting:
    segments = build_segments(payload)

    if payload.participants:
        participants = [
            models.Participant(name=p.name, email=p.email, role=p.role)
            for p in payload.participants
        ]
    else:
        participants = [
            models.Participant(name=name)
            for name in transcript_parser.participants_from_segments(segments)
        ]

    duration = payload.duration_seconds
    if not duration and segments:
        duration = int(max(s["end_time"] for s in segments))

    meeting = models.Meeting(
        title=payload.title,
        description=payload.description,
        meeting_date=payload.meeting_date or _utcnow(),
        duration_seconds=duration or 0,
        media_url=payload.media_url,
        participants=participants,
        segments=[models.TranscriptSegment(**seg) for seg in segments],
        tags=get_or_create_tags(session, payload.tags),
    )
    session.add(meeting)
    session.flush()  # assign PKs

    if payload.generate_summary and segments:
        transcript_text = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)
        insights = groq_service.generate_meeting_insights(payload.title, transcript_text)
        apply_insights(session, meeting, insights)

    session.commit()
    session.refresh(meeting)
    index_meeting_safe(meeting)  # add this meeting's transcript to the RAG index
    return meeting


def apply_insights(session: Session, meeting: models.Meeting, insights: dict) -> None:
    """Replace a meeting's summary, action items, and topics with fresh ones."""
    if meeting.summary is not None:
        session.delete(meeting.summary)
        session.flush()
        meeting.summary = None

    meeting.summary = models.Summary(
        overview=insights.get("overview", ""),
        generated_by=insights.get("generated_by", GeneratedBy.mock),
    )
    # Reassigning these lists lets delete-orphan clean up the previous rows.
    meeting.action_items = [
        models.ActionItem(text=a["text"], assignee=a.get("assignee"), order_index=i)
        for i, a in enumerate(insights.get("action_items", []))
    ]
    meeting.topics = [
        models.Topic(title=t["title"], start_time=t.get("start_time"), order_index=i)
        for i, t in enumerate(insights.get("topics", []))
    ]
    meeting.updated_at = _utcnow()
    session.add(meeting)


def transcript_as_text(meeting: models.Meeting) -> str:
    return "\n".join(f"{s.speaker}: {s.text}" for s in meeting.segments)
