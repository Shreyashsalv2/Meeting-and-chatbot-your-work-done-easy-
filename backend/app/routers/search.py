"""Global search across all meetings (titles, participants, summaries, transcript)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from .. import models, schemas
from ..database import get_session

router = APIRouter(prefix="/api", tags=["search"])

MAX_TRANSCRIPT_HITS_PER_MEETING = 3


def _snippet(text: str, needle: str, width: int = 60) -> str:
    low = text.lower()
    idx = low.find(needle.lower())
    if idx == -1:
        return text[: width * 2].strip()
    start = max(0, idx - width)
    end = min(len(text), idx + len(needle) + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


@router.get("/search", response_model=schemas.SearchResults)
def global_search(
    q: str = Query(..., min_length=1),
    session: Session = Depends(get_session),
):
    needle = q.lower().strip()
    results: list[schemas.SearchMatch] = []
    meetings = session.exec(select(models.Meeting)).all()

    for m in meetings:
        if needle in m.title.lower():
            results.append(
                schemas.SearchMatch(
                    meeting_id=m.id, meeting_title=m.title, field="title", snippet=m.title
                )
            )
        for p in m.participants:
            if needle in p.name.lower():
                results.append(
                    schemas.SearchMatch(
                        meeting_id=m.id,
                        meeting_title=m.title,
                        field="participant",
                        snippet=p.name,
                    )
                )
                break
        if m.summary and needle in m.summary.overview.lower():
            results.append(
                schemas.SearchMatch(
                    meeting_id=m.id,
                    meeting_title=m.title,
                    field="summary",
                    snippet=_snippet(m.summary.overview, needle),
                )
            )
        hits = 0
        for seg in m.segments:
            if needle in seg.text.lower():
                results.append(
                    schemas.SearchMatch(
                        meeting_id=m.id,
                        meeting_title=m.title,
                        field="transcript",
                        snippet=_snippet(seg.text, needle),
                        segment_id=seg.id,
                        start_time=seg.start_time,
                    )
                )
                hits += 1
                if hits >= MAX_TRANSCRIPT_HITS_PER_MEETING:
                    break

    return schemas.SearchResults(query=q, count=len(results), results=results)


@router.get("/search/semantic", response_model=schemas.SearchResults)
def semantic_search(q: str = Query(..., min_length=1)):
    """RAG-Fusion semantic search: multi-query + reciprocal-rank fusion over all
    transcripts. Returns the same SearchMatch shape as keyword search (so the UI and
    deep-linking are unchanged), just ranked by meaning instead of substring match."""
    from ..services.rag import fusion_rag

    matches = fusion_rag.search(q, top_n=8)
    results = [schemas.SearchMatch(**m) for m in matches]
    return schemas.SearchResults(query=q, count=len(results), results=results)
