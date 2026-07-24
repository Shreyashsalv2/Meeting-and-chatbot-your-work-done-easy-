"""RAG #1 — Self-RAG for the per-meeting chat.

"Self-RAG" = the model reflects on its own pipeline instead of blindly answering:
it **grades** whether retrieved chunks are relevant, **rewrites** the question and
retries when they aren't, and **grades its own answer** (grounded? does it actually
answer?) before returning. That self-correction is why this is a LangGraph *graph*
and not a straight chain.

Graph shape::

    START → retrieve → grade_documents → (has relevant docs?)
                                          ├─ yes ─────────────► generate → grade_generation
                                          ├─ no & retries left ► transform_query → retrieve
                                          └─ no & out of retries ► not_covered → END
    grade_generation → (useful?)
                        ├─ yes ─────────────────────► END
                        └─ no & retries left ► transform_query → retrieve

All grading uses tiny yes/no LLM prompts (Groq-friendly) and everything is bounded by
``settings.self_rag_max_retries`` so it can never loop forever. Returns the answer plus
**citations** (speaker + timestamp) drawn only from the chunks it actually used.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ...config import settings
from . import vector_store as vs

logger = logging.getLogger(__name__)


# --- Graph state -------------------------------------------------------------
class SelfRAGState(TypedDict, total=False):
    question: str          # current (possibly rewritten) retrieval query
    original: str          # the user's original question (used for answering)
    meeting_id: int
    history: list[dict]
    documents: list[Document]
    generation: str
    citations: list[dict]
    retries: int


# --- Prompts -----------------------------------------------------------------
_GEN_SYSTEM = (
    "You are answering a question about ONE meeting, using ONLY the transcript excerpts "
    "provided as Context. Answer strictly from the Context; do not use outside knowledge. "
    "If the Context does not contain the answer, reply exactly: "
    "\"I don't know based on this meeting.\" Be concise and reference speakers when helpful."
)


def _llm(temperature: float = 0.0):
    return vs.get_llm(temperature=temperature)


# --- Nodes -------------------------------------------------------------------
def _retrieve(state: SelfRAGState) -> dict:
    docs = vs.similarity_search(state["question"], meeting_id=state["meeting_id"])
    return {"documents": docs}


def _grade_documents(state: SelfRAGState) -> dict:
    """Keep only the chunks relevant to the question (one LLM call for all)."""
    docs = state.get("documents") or []
    if not docs:
        return {"documents": []}
    numbered = "\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
    prompt = (
        f"Question: {state['original']}\n\n"
        f"Transcript excerpts:\n{numbered}\n\n"
        "Which excerpts are relevant to answering the question? "
        "Reply with the numbers only, comma-separated (e.g. '1,3'). "
        "If none are relevant, reply 'none'."
    )
    try:
        raw = _llm(0.0).invoke([HumanMessage(content=prompt)]).content or ""
    except Exception as exc:  # noqa: BLE001 - fail open (keep all) on grader error
        logger.warning("grade_documents failed: %s", exc)
        return {"documents": docs}
    if "none" in raw.lower():
        return {"documents": []}
    idxs = {int(n) for n in re.findall(r"\d+", raw)}
    kept = [d for i, d in enumerate(docs) if (i + 1) in idxs]
    return {"documents": kept or docs}  # fail open if parsing found nothing


def _transform_query(state: SelfRAGState) -> dict:
    """Rewrite the query for better retrieval; count it against the retry budget."""
    prompt = (
        "Rewrite the following question into a concise, standalone search query that "
        "would retrieve relevant meeting-transcript passages. Return only the rewritten query.\n\n"
        f"Question: {state['original']}"
    )
    try:
        better = (_llm(0.0).invoke([HumanMessage(content=prompt)]).content or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("transform_query failed: %s", exc)
        better = state["question"]
    return {"question": better or state["question"], "retries": state.get("retries", 0) + 1}


def _generate(state: SelfRAGState) -> dict:
    docs = state.get("documents") or []
    context = "\n\n".join(
        f"({d.metadata.get('speaker','?')} @ {int(d.metadata.get('start_time',0))}s) {d.page_content}"
        for d in docs
    )
    messages: list = [SystemMessage(content=f"{_GEN_SYSTEM}\n\nContext:\n{context}")]
    for turn in (state.get("history") or [])[-8:]:
        role, content = turn.get("role"), str(turn.get("content", "")).strip()
        if not content:
            continue
        messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    messages.append(HumanMessage(content=state["original"]))

    answer = (_llm(0.2).invoke(messages).content or "").strip()
    citations = _build_citations(docs, answer)
    return {"generation": answer or "I don't know based on this meeting.", "citations": citations}


def _not_covered(state: SelfRAGState) -> dict:
    return {"generation": "I don't know based on this meeting.", "citations": []}


def _build_citations(docs: list[Document], answer: str) -> list[dict]:
    if _is_unknown(answer):
        return []
    seen: set = set()
    cites: list[dict] = []
    for d in docs:
        key = (d.metadata.get("meeting_id"), d.metadata.get("start_time"))
        if key in seen:
            continue
        seen.add(key)
        cites.append(
            {
                "meeting_id": d.metadata.get("meeting_id"),
                "meeting_title": d.metadata.get("meeting_title"),
                "speaker": d.metadata.get("speaker"),
                "start_time": d.metadata.get("start_time"),
                "snippet": d.page_content[:160],
            }
        )
    return cites


def _is_unknown(answer: str) -> bool:
    a = answer.lower()
    return "don't know based on this meeting" in a or "dont know based on this meeting" in a


# --- Edges (decisions) -------------------------------------------------------
def _decide_after_docs(state: SelfRAGState) -> str:
    if state.get("documents"):
        return "generate"
    if state.get("retries", 0) < settings.self_rag_max_retries:
        return "transform_query"
    return "not_covered"


def _grade_generation(state: SelfRAGState) -> str:
    """Is the answer grounded in the context AND does it answer the question?"""
    answer = state.get("generation", "")
    if _is_unknown(answer):
        return "end"  # a truthful "I don't know" is an acceptable terminal state
    if state.get("retries", 0) >= settings.self_rag_max_retries:
        return "end"  # out of budget — accept what we have
    docs = state.get("documents") or []
    context = "\n".join(d.page_content for d in docs)
    prompt = (
        "You are checking an answer. Reply with two tokens only, like 'yes yes'.\n"
        "First token: is the ANSWER supported by the CONTEXT (not invented)? yes/no.\n"
        "Second token: does the ANSWER actually address the QUESTION? yes/no.\n\n"
        f"QUESTION: {state['original']}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
    )
    try:
        raw = (_llm(0.0).invoke([HumanMessage(content=prompt)]).content or "").lower()
    except Exception as exc:  # noqa: BLE001 - fail open (accept) on grader error
        logger.warning("grade_generation failed: %s", exc)
        return "end"
    tokens = re.findall(r"yes|no", raw)
    grounded = tokens[0] == "yes" if len(tokens) >= 1 else True
    answers = tokens[1] == "yes" if len(tokens) >= 2 else True
    if grounded and answers:
        return "end"
    return "transform_query"  # bad answer → rewrite & retrieve again (bounded)


# --- Graph -------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_graph():
    g = StateGraph(SelfRAGState)
    g.add_node("retrieve", _retrieve)
    g.add_node("grade_documents", _grade_documents)
    g.add_node("transform_query", _transform_query)
    g.add_node("generate", _generate)
    g.add_node("not_covered", _not_covered)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        _decide_after_docs,
        {"generate": "generate", "transform_query": "transform_query", "not_covered": "not_covered"},
    )
    g.add_edge("transform_query", "retrieve")
    g.add_conditional_edges(
        "generate", _grade_generation, {"end": END, "transform_query": "transform_query"}
    )
    g.add_edge("not_covered", END)
    return g.compile()


# --- Public entry ------------------------------------------------------------
def answer(
    meeting_id: int,
    question: str,
    history: Optional[list[dict]] = None,
    fallback_transcript: Optional[str] = None,
) -> dict:
    """Answer a question about one meeting via the Self-RAG graph.

    Returns ``{"answer": str, "citations": list}``. Never raises: on any failure it
    degrades to the naive full-transcript chat (if a transcript is supplied) or a
    friendly message.
    """
    if not vs.llm_available():
        return {
            "answer": "The chat assistant isn't available right now (no AI key is configured).",
            "citations": [],
        }
    try:
        final = _get_graph().invoke(
            {
                "question": question,
                "original": question,
                "meeting_id": meeting_id,
                "history": history or [],
                "retries": 0,
            }
        )
        return {"answer": final.get("generation", ""), "citations": final.get("citations", [])}
    except Exception as exc:  # noqa: BLE001 - never crash the endpoint
        logger.warning("Self-RAG failed (%s); falling back to naive chat.", exc)
        if fallback_transcript is not None:
            from .. import groq_service

            return {
                "answer": groq_service.chat_with_meeting(fallback_transcript, question, history),
                "citations": [],
            }
        return {"answer": "Sorry, I couldn't answer that just now.", "citations": []}
