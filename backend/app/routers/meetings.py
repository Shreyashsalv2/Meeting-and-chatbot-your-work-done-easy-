"""Meeting CRUD + creation-from-transcript endpoints."""
from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from sqlmodel import Session, select

from .. import crud, models, schemas
from ..database import get_session
from ..services import groq_service

router = APIRouter(prefix="/api/meetings", tags=["meetings"])


def _get_or_404(session: Session, meeting_id: int) -> models.Meeting:
    meeting = session.get(models.Meeting, meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@router.get("", response_model=list[schemas.MeetingListItem])
def list_meetings(
    session: Session = Depends(get_session),
    search: Optional[str] = Query(None, description="Match title, description, or participant"),
    participant: Optional[str] = Query(None),
    meeting_date: Optional[date] = Query(None, description="Filter to a single day (YYYY-MM-DD)"),
    sort: str = Query("recent", pattern="^(recent|oldest|title)$"),
):
    meetings = list(session.exec(select(models.Meeting)).all())

    if search:
        s = search.lower()
        meetings = [
            m for m in meetings
            if s in m.title.lower()
            or (m.description and s in m.description.lower())
            or any(s in p.name.lower() for p in m.participants)
        ]
    if participant:
        p = participant.lower()
        meetings = [m for m in meetings if any(p in part.name.lower() for part in m.participants)]
    if meeting_date:
        meetings = [m for m in meetings if m.meeting_date.date() == meeting_date]

    if sort == "oldest":
        meetings.sort(key=lambda m: m.meeting_date)
    elif sort == "title":
        meetings.sort(key=lambda m: m.title.lower())
    else:  # recent
        meetings.sort(key=lambda m: m.meeting_date, reverse=True)
    return meetings


@router.post("", response_model=schemas.MeetingDetail, status_code=201)
def create_meeting(payload: schemas.MeetingCreate, session: Session = Depends(get_session)):
    if not payload.transcript_text and not payload.segments:
        raise HTTPException(status_code=422, detail="Provide either transcript_text or segments")
    return crud.create_meeting(session, payload)


@router.post("/upload", response_model=schemas.MeetingDetail, status_code=201)
async def upload_meeting(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    participants: Optional[str] = Form(None, description="Comma-separated names"),
    generate_summary: bool = Form(True),
    session: Session = Depends(get_session),
):
    raw = (await file.read()).decode("utf-8", errors="ignore")
    meeting_title = title or (
        file.filename.rsplit(".", 1)[0] if file.filename else "Untitled Meeting"
    )
    part_list = None
    if participants:
        part_list = [
            schemas.ParticipantCreate(name=n.strip())
            for n in participants.split(",")
            if n.strip()
        ]
    payload = schemas.MeetingCreate(
        title=meeting_title,
        transcript_text=raw,
        participants=part_list,
        generate_summary=generate_summary,
    )
    return crud.create_meeting(session, payload)


@router.get("/{meeting_id}", response_model=schemas.MeetingDetail)
def get_meeting(meeting_id: int, session: Session = Depends(get_session)):
    return _get_or_404(session, meeting_id)


@router.patch("/{meeting_id}", response_model=schemas.MeetingDetail)
def update_meeting(
    meeting_id: int,
    payload: schemas.MeetingUpdate,
    session: Session = Depends(get_session),
):
    meeting = _get_or_404(session, meeting_id)
    data = payload.model_dump(exclude_unset=True)

    for field in ("title", "description", "meeting_date", "media_url"):
        if field in data:
            setattr(meeting, field, data[field])

    if data.get("participants") is not None:
        for existing in list(meeting.participants):
            session.delete(existing)
        session.flush()
        meeting.participants = [
            models.Participant(name=p["name"], email=p.get("email"), role=p.get("role"))
            for p in data["participants"]
        ]
    if data.get("tags") is not None:
        meeting.tags = crud.get_or_create_tags(session, data["tags"])

    meeting.updated_at = crud._utcnow()
    session.add(meeting)
    session.commit()
    session.refresh(meeting)
    crud.index_meeting_safe(meeting)  # refresh RAG metadata (e.g. title)
    return meeting


@router.delete("/{meeting_id}", status_code=204)
def delete_meeting(meeting_id: int, session: Session = Depends(get_session)):
    meeting = _get_or_404(session, meeting_id)
    session.delete(meeting)
    session.commit()
    crud.deindex_meeting_safe(meeting_id)  # drop its chunks from the RAG index
    return Response(status_code=204)


@router.post("/{meeting_id}/generate-summary", response_model=schemas.MeetingDetail)
def regenerate_summary(meeting_id: int, session: Session = Depends(get_session)):
    meeting = _get_or_404(session, meeting_id)
    if not meeting.segments:
        raise HTTPException(status_code=400, detail="Meeting has no transcript to summarize")
    insights = groq_service.generate_meeting_insights(
        meeting.title, crud.transcript_as_text(meeting)
    )
    crud.apply_insights(session, meeting, insights)
    session.commit()
    session.refresh(meeting)
    return meeting


@router.post("/{meeting_id}/chat", response_model=schemas.ChatResponse)
def chat_about_meeting(
    meeting_id: int,
    payload: schemas.ChatRequest,
    session: Session = Depends(get_session),
):
    meeting = _get_or_404(session, meeting_id)
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")
    history = [{"role": m.role, "content": m.content} for m in (payload.history or [])]
    answer = groq_service.chat_with_meeting(
        crud.transcript_as_text(meeting), payload.question, history
    )
    return schemas.ChatResponse(answer=answer)
