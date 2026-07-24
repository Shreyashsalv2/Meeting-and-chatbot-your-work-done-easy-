"""Unified assistant endpoint — Adaptive RAG router (+ Agentic RAG subgraph).

One conversation, one endpoint. The Adaptive router decides per message whether to
answer directly, retrieve within one meeting, retrieve across all, or hand off to the
tool-using agent — and the single response shape carries everything the UI needs
(answer, which route fired, citations, tool steps, and any downloadable artifact).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from .. import models, schemas
from ..database import get_session

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


@router.post("", response_model=schemas.AssistantResponse)
def assistant(payload: schemas.AssistantRequest, session: Session = Depends(get_session)):
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    history = [{"role": m.role, "content": m.content} for m in (payload.history or [])]
    meetings_index = [
        {"id": m.id, "title": m.title}
        for m in session.exec(select(models.Meeting)).all()
    ]

    from ..services.rag import adaptive_rag

    result = adaptive_rag.answer(payload.question, history, meetings_index)
    return schemas.AssistantResponse(**result)
