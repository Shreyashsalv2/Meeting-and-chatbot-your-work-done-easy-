"""Shared RAG foundation: embeddings, vector store, indexing, retrieval.

This module is the single dependency-inversion seam for all four RAG techniques.
The graphs import ``get_llm`` / ``retriever`` / ``similarity_search`` from here and
never touch Chroma / fastembed / Groq directly, so moving to a hosted embeddings
API or a different vector store is a one-function change.

Design notes (kept in sync with docs/RAG_LEARNING_NOTES.md):
- **Embeddings**: fastembed (ONNX runtime, no torch) → local, no API key, small.
- **Chunking**: we group consecutive speaker turns up to ``rag_chunk_size`` rather
  than blindly character-splitting, so every chunk keeps the *timestamp of its
  first utterance*. That timestamp is what powers click-to-seek citations.
- **One collection** for all meetings; ``meeting_id`` metadata drives per-meeting
  filtering (Self-RAG, single-meeting routing) — this is why Chroma over FAISS.
- All indexing is **best-effort** and callers wrap it so it can never break CRUD.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from ...config import settings

if TYPE_CHECKING:  # avoid importing heavy libs at module load
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_COLLECTION = "meeting_segments"


# --- Seams (Dependency Inversion): swap these to change providers -------------
@lru_cache(maxsize=1)
def get_embeddings():
    """Local, no-key embeddings (fastembed / ONNX — no torch).

    THE single swap point for a hosted embeddings API later.
    """
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=settings.embeddings_model)


def llm_available() -> bool:
    """Whether the Groq-backed LLM can be constructed (key present)."""
    return bool(settings.groq_api_key)


def get_llm(temperature: float = 0.2, **kwargs):
    """The main (strong) ChatGroq. Raises if no key — callers guard with ``llm_available``."""
    return _chat_groq(settings.groq_model, temperature, **kwargs)


def get_fast_llm(temperature: float = 0.0, **kwargs):
    """The cheap/fast ChatGroq (separate daily budget). For routing/grading + fallback."""
    return _chat_groq(settings.groq_fast_model, temperature, **kwargs)


def _chat_groq(model: str, temperature: float, **kwargs):
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured")
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=temperature,
        **kwargs,
    )


# --- Rate-limit resilience ---------------------------------------------------
class RateLimited(Exception):
    """Raised when both the main and fast models are rate-limited. Carries a hint."""

    def __init__(self, retry_hint: Optional[str] = None):
        self.retry_hint = retry_hint
        super().__init__("Groq rate limit reached on all models")


def is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "rate_limit" in s or "rate limit" in s


def rate_limit_retry_hint(exc: Exception) -> Optional[str]:
    """Extract Groq's 'try again in X' hint, if present."""
    import re

    m = re.search(r"try again in ([0-9hms\.]+)", str(exc))
    return m.group(1) if m else None


def rate_limit_message(hint: Optional[str] = None) -> str:
    """User-facing message when the AI is rate-limited (shared across the RAG modules)."""
    when = f" Please try again in ~{hint}." if hint else " Please try again in a few minutes."
    return (
        "⏳ The shared AI key has hit today's usage limit, so I can't answer right now."
        f"{when} (This is a free-tier quota, not a bug — it resets on a daily schedule.)"
    )


def resilient_invoke(messages, temperature: float = 0.2):
    """Invoke the strong model; on a rate limit, transparently retry on the fast model.

    Returns the response's text content. Raises ``RateLimited`` only if BOTH models are
    rate-limited, so the app degrades (strong → fast) before it ever fails."""
    try:
        return (get_llm(temperature).invoke(messages).content or "").strip()
    except Exception as exc:  # noqa: BLE001
        if not is_rate_limit(exc):
            raise
        hint = rate_limit_retry_hint(exc)
        try:
            return (get_fast_llm(temperature).invoke(messages).content or "").strip()
        except Exception as exc2:  # noqa: BLE001
            if is_rate_limit(exc2):
                raise RateLimited(rate_limit_retry_hint(exc2) or hint) from exc2
            raise


@lru_cache(maxsize=1)
def get_store():
    """The persistent Chroma vector store (telemetry disabled)."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from langchain_chroma import Chroma

    client = chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    return Chroma(
        client=client,
        collection_name=_COLLECTION,
        embedding_function=get_embeddings(),
    )


# --- Chunking ----------------------------------------------------------------
def _chunk_meeting(meeting) -> tuple[list["Document"], list[str]]:
    """Group consecutive speaker turns into ~``rag_chunk_size`` chunks.

    Each chunk carries the metadata of its *first* segment (notably ``start_time``),
    so a retrieved chunk can deep-link to that moment. Returns (documents, ids)
    with stable ids ``m{meeting_id}-c{n}`` so re-indexing upserts cleanly.
    """
    from langchain_core.documents import Document

    segments = list(meeting.segments)
    docs: list[Document] = []
    ids: list[str] = []

    buf: list[str] = []
    buf_len = 0
    anchor = None  # first segment in the current buffer

    def flush():
        nonlocal buf, buf_len, anchor
        if not buf or anchor is None:
            return
        idx = len(docs)
        docs.append(
            Document(
                page_content="\n".join(buf),
                metadata={
                    "meeting_id": int(meeting.id),
                    "meeting_title": meeting.title,
                    "speaker": anchor.speaker,
                    "start_time": float(anchor.start_time),
                    "order_index": int(anchor.order_index),
                },
            )
        )
        ids.append(f"m{meeting.id}-c{idx}")
        buf, buf_len, anchor = [], 0, None

    for seg in segments:
        line = f"{seg.speaker}: {seg.text}"
        if anchor is None:
            anchor = seg
        buf.append(line)
        buf_len += len(line)
        if buf_len >= settings.rag_chunk_size:
            flush()
    flush()
    return docs, ids


def _meta_docs(meeting) -> tuple[list["Document"], list[str]]:
    """Index the meeting's SUMMARY, ACTION ITEMS, and TOPICS as their own retrievable
    documents (so the assistant can answer from them, per the info-priority spec).
    They carry a ``kind`` and no meaningful timestamp (start_time=0.0)."""
    from langchain_core.documents import Document

    mid, title = int(meeting.id), meeting.title
    base = {"meeting_id": mid, "meeting_title": title, "start_time": 0.0}
    docs: list["Document"] = []
    ids: list[str] = []

    if meeting.summary and meeting.summary.overview:
        docs.append(
            Document(
                page_content=f"Summary of '{title}': {meeting.summary.overview}",
                metadata={**base, "kind": "summary", "speaker": "Summary"},
            )
        )
        ids.append(f"m{mid}-summary")

    for i, a in enumerate(meeting.action_items):
        who = f" (assignee: {a.assignee})" if a.assignee else ""
        status = "done" if a.completed else "open"
        docs.append(
            Document(
                page_content=f"Action item [{status}] from '{title}': {a.text}{who}",
                metadata={**base, "kind": "action_item", "speaker": a.assignee or "Action item"},
            )
        )
        ids.append(f"m{mid}-action-{i}")

    for i, t in enumerate(meeting.topics):
        docs.append(
            Document(
                page_content=f"Topic in '{title}': {t.title}",
                metadata={**base, "kind": "topic", "speaker": "Topic",
                          "start_time": float(t.start_time or 0.0)},
            )
        )
        ids.append(f"m{mid}-topic-{i}")

    return docs, ids


# --- Indexing (best-effort; callers must guard) ------------------------------
def remove_meeting(meeting_id: int) -> None:
    store = get_store()
    existing = store.get(where={"meeting_id": int(meeting_id)})
    stale = existing.get("ids") or []
    if stale:
        store.delete(ids=stale)


def index_meeting(meeting) -> int:
    """(Re)index a meeting: transcript chunks PLUS its summary, action items, and topics.
    Idempotent upsert (remove_meeting clears all of the meeting's docs first)."""
    docs, ids = _chunk_meeting(meeting)
    meta_docs, meta_ids = _meta_docs(meeting)
    docs += meta_docs
    ids += meta_ids
    remove_meeting(meeting.id)  # clear old docs (all kinds) first
    if docs:
        get_store().add_documents(docs, ids=ids)
    return len(docs)


def reindex_all(session) -> int:
    """Rebuild the whole index from the DB. Returns total meetings indexed."""
    from sqlmodel import select

    from ...models import Meeting

    meetings = session.exec(select(Meeting)).all()
    for m in meetings:
        index_meeting(m)
    return len(meetings)


def count() -> int:
    """Number of chunks currently in the store (for verification/health)."""
    try:
        return get_store()._collection.count()
    except Exception:  # noqa: BLE001
        return len(get_store().get().get("ids") or [])


def ensure_indexed(session) -> None:
    """Index everything if the store is empty (safe to call on every startup)."""
    try:
        if count() == 0:
            n = reindex_all(session)
            logger.info("RAG: indexed %d meetings into Chroma on startup.", n)
    except Exception as exc:  # noqa: BLE001 - never block startup
        logger.warning("RAG startup indexing skipped: %s", exc)


# --- Retrieval ---------------------------------------------------------------
def similarity_search(
    query: str, meeting_id: Optional[int] = None, k: Optional[int] = None
) -> list["Document"]:
    """Top-k chunks for ``query``, optionally filtered to one meeting."""
    flt = {"meeting_id": int(meeting_id)} if meeting_id is not None else None
    return get_store().similarity_search(query, k=k or settings.rag_top_k, filter=flt)


def retriever(meeting_id: Optional[int] = None, k: Optional[int] = None):
    """A LangChain retriever, optionally scoped to one meeting via metadata filter."""
    search_kwargs: dict = {"k": k or settings.rag_top_k}
    if meeting_id is not None:
        search_kwargs["filter"] = {"meeting_id": int(meeting_id)}
    return get_store().as_retriever(search_kwargs=search_kwargs)
