"""RAG #3 — Cross-session memory: the assistant remembers past conversations.

Each assistant turn is persisted to the durable ``chat_turns`` table (Postgres) AND
indexed into a per-user Chroma collection (``vector_store``). On a later message —
even in a brand-new session — the router recalls the most relevant past turns and
folds them into the prompt, so "what did we decide about X last week?" works.

Two seams keep this decoupled from the RAG graphs:
- ``save_turn``  — persist + index one user→assistant exchange (best-effort).
- ``recall``     — fetch a formatted memory block for a question (or "" if none).

Everything is per ``user_id`` and best-effort: memory can never break an answer.
"""
from __future__ import annotations

import logging

from sqlmodel import Session

from ...database import engine
from ...models import ChatTurn
from . import vector_store as vs

logger = logging.getLogger(__name__)

_MEMORY_HEADER = (
    "Relevant notes from your earlier conversations with this user (use them for "
    "continuity and to recall past decisions/preferences; don't repeat them verbatim "
    "unless asked, and never treat them as meeting FACTS):"
)


def save_turn(user_id: int, question: str, answer: str) -> None:
    """Persist a user question + assistant answer and index both for recall.

    Best-effort: any failure is logged and swallowed so it never affects the response.
    """
    if not user_id:
        return
    try:
        with Session(engine) as session:
            user_turn = ChatTurn(user_id=user_id, role="user", content=question or "")
            asst_turn = ChatTurn(user_id=user_id, role="assistant", content=answer or "")
            session.add(user_turn)
            session.add(asst_turn)
            session.commit()
            session.refresh(user_turn)
            session.refresh(asst_turn)
        for t in (user_turn, asst_turn):
            try:
                vs.add_memory(user_id, t.id, t.role, t.content)
            except Exception:  # noqa: BLE001
                continue
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory.save_turn failed: %s", exc)


def recall(user_id: int, question: str, k: int = 4) -> str:
    """Return a formatted memory block of this user's relevant past turns, or ""."""
    if not user_id or not (question or "").strip():
        return ""
    try:
        docs = vs.search_memory(user_id, question, k=k)
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory.recall failed: %s", exc)
        return ""
    lines = [d.page_content.strip() for d in docs if d.page_content.strip()]
    if not lines:
        return ""
    body = "\n".join(f"- {ln}" for ln in lines)
    return f"{_MEMORY_HEADER}\n{body}"
