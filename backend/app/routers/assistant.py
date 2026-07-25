"""Unified assistant endpoint — Adaptive RAG router (+ Agentic RAG subgraph).

One conversation, one endpoint. The Adaptive router decides per message whether to
answer directly, retrieve within one meeting, retrieve across all, or hand off to the
tool-using agent — and the single response shape carries everything the UI needs
(answer, which route fired, citations, tool steps, and any downloadable artifact).
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from .. import models, schemas
from ..database import get_session

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


def _history_and_index(payload: schemas.AssistantRequest, session: Session):
    history = [{"role": m.role, "content": m.content} for m in (payload.history or [])]
    meetings_index = [
        {"id": m.id, "title": m.title}
        for m in session.exec(select(models.Meeting)).all()
    ]
    return history, meetings_index


@router.post("", response_model=schemas.AssistantResponse)
def assistant(payload: schemas.AssistantRequest, session: Session = Depends(get_session)):
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    history, meetings_index = _history_and_index(payload, session)
    from ..services.rag import adaptive_rag

    result = adaptive_rag.answer(payload.question, history, meetings_index)
    return schemas.AssistantResponse(**result)


@router.post("/stream")
def assistant_stream(payload: schemas.AssistantRequest, session: Session = Depends(get_session)):
    """Streaming variant (SSE): emits `data: {"token": …}` events, then a final
    `data: {"meta": {route, citations, steps, artifact, offers}}` and `data: [DONE]`."""
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    history, meetings_index = _history_and_index(payload, session)
    from ..services.rag import adaptive_rag

    def sse():
        try:
            for kind, data in adaptive_rag.answer_stream(payload.question, history, meetings_index):
                key = "token" if kind == "token" else "meta"
                yield f"data: {json.dumps({key: data})}\n\n"
        except Exception:  # noqa: BLE001 - never break the stream
            yield f"data: {json.dumps({'token': 'Sorry, something went wrong.'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
