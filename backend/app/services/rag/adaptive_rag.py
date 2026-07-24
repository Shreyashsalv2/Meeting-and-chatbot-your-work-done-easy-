"""RAG #2 — Adaptive RAG: the unified assistant's top-level brain.

Naive RAG retrieves for *every* question — wasteful for "hi", wrong for "email a
summary". **Adaptive RAG routes first**: a classifier looks at the question (and the
conversation so far) and picks the cheapest strategy that can answer it:

- ``no_retrieval``   — greeting / small talk / not about meeting content → answer directly.
- ``single_meeting`` — about one specific meeting → retrieve *within that meeting*.
- ``semantic_all``   — spans or doesn't name a meeting → retrieve across *all* meetings.
- ``agentic``        — needs an action/tool (look something up, produce a document) →
                       hand off to the Agentic RAG subgraph (wired in Phase E).

Graph shape::

    START → route_query ──┬─ no_retrieval ─► answer_direct ─────────► END
                          ├─ single_meeting ► retrieve(one) ► generate ► END
                          ├─ semantic_all ──► retrieve(all) ► generate ► END
                          └─ agentic ───────► run_agent ──────────────► END   (Phase E)

This module owns only the routing + retrieval branches; the agentic branch is a thin
seam (``_run_agent``) so Phase E adds a capability without editing the router (OCP).
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from . import vector_store as vs

logger = logging.getLogger(__name__)

VALID_ROUTES = {"no_retrieval", "single_meeting", "semantic_all", "agentic"}


class AssistantState(TypedDict, total=False):
    question: str
    history: list[dict]
    meetings_index: list[dict]   # [{"id": int, "title": str}, ...]
    route: str
    meeting_id: Optional[int]
    documents: list[Document]
    generation: str
    citations: list[dict]
    steps: list[dict]
    artifact: Optional[dict]


_ANSWER_SYSTEM = (
    "You are the assistant for a meeting-notes app. Answer the user using ONLY the "
    "Context excerpts from their meetings. If the Context is empty or lacks the answer, "
    "say so briefly. Be concise; when useful, mention which meeting and who said it."
)
_DIRECT_SYSTEM = (
    "You are the assistant for a meeting-notes app. The user's message doesn't require "
    "looking up meeting content. Reply briefly and helpfully. Do not invent meeting details."
)


# --- Routing -----------------------------------------------------------------
def _route_query(state: AssistantState) -> dict:
    index = state.get("meetings_index") or []
    listing = "\n".join(f"{m['id']}: {m['title']}" for m in index) or "(none)"
    prompt = (
        "Classify how to answer the user's message. Options:\n"
        '- "no_retrieval": greeting/small talk, or not about meeting content.\n'
        '- "single_meeting": about ONE specific meeting from the list (give its id).\n'
        '- "semantic_all": about meeting content generally, or spanning meetings.\n'
        '- "agentic": the user wants an ACTION — look something up externally, and/or '
        "produce/draft/export a document.\n\n"
        f"Meetings:\n{listing}\n\n"
        f"Message: {state['question']}\n\n"
        'Respond with JSON only: {"route": "...", "meeting_id": <id or null>}'
    )
    route, meeting_id = "semantic_all", None
    try:
        raw = vs.get_llm(0.0).invoke([HumanMessage(content=prompt)]).content or ""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        cand = str(data.get("route", "")).strip()
        if cand in VALID_ROUTES:
            route = cand
        mid = data.get("meeting_id")
        meeting_id = int(mid) if isinstance(mid, (int, str)) and str(mid).isdigit() else None
    except Exception as exc:  # noqa: BLE001 - default to semantic_all on any error
        logger.warning("route_query failed (%s); defaulting to semantic_all.", exc)
    if route == "single_meeting" and meeting_id is None:
        route = "semantic_all"  # can't focus without a target
    return {"route": route, "meeting_id": meeting_id}


def _route_selector(state: AssistantState) -> str:
    return state.get("route", "semantic_all")


# --- Retrieval + generation branches -----------------------------------------
def _retrieve_single(state: AssistantState) -> dict:
    docs = vs.similarity_search(state["question"], meeting_id=state.get("meeting_id"))
    return {"documents": docs}


def _retrieve_all(state: AssistantState) -> dict:
    return {"documents": vs.similarity_search(state["question"])}


def _generate(state: AssistantState) -> dict:
    docs = state.get("documents") or []
    context = "\n\n".join(
        f"[{d.metadata.get('meeting_title','?')} | {d.metadata.get('speaker','?')} "
        f"@ {int(d.metadata.get('start_time',0))}s] {d.page_content}"
        for d in docs
    )
    messages = _with_history(f"{_ANSWER_SYSTEM}\n\nContext:\n{context or '(none)'}", state)
    answer = (vs.get_llm(0.2).invoke(messages).content or "").strip()
    return {"generation": answer, "citations": _citations(docs)}


def _answer_direct(state: AssistantState) -> dict:
    messages = _with_history(_DIRECT_SYSTEM, state)
    answer = (vs.get_llm(0.3).invoke(messages).content or "").strip()
    return {"generation": answer, "citations": []}


def _run_agent(state: AssistantState) -> dict:
    """Agentic branch. Real tool-agent is wired in Phase E; stub answers meanwhile."""
    try:
        from . import agentic_rag  # noqa: F401  (present from Phase E)

        return agentic_rag.run(state["question"], state.get("history"))
    except Exception:  # noqa: BLE001 - Phase D: no agent yet → fall back to semantic answer
        docs = vs.similarity_search(state["question"])
        context = "\n\n".join(d.page_content for d in docs)
        messages = _with_history(f"{_ANSWER_SYSTEM}\n\nContext:\n{context or '(none)'}", state)
        answer = (vs.get_llm(0.2).invoke(messages).content or "").strip()
        return {"generation": answer, "citations": _citations(docs)}


# --- Helpers -----------------------------------------------------------------
def _with_history(system: str, state: AssistantState) -> list:
    messages: list = [SystemMessage(content=system)]
    for turn in (state.get("history") or [])[-8:]:
        role, content = turn.get("role"), str(turn.get("content", "")).strip()
        if not content:
            continue
        messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    messages.append(HumanMessage(content=state["question"]))
    return messages


def _citations(docs: list[Document]) -> list[dict]:
    seen, cites = set(), []
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


# --- Graph -------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_graph():
    g = StateGraph(AssistantState)
    g.add_node("route_query", _route_query)
    g.add_node("answer_direct", _answer_direct)
    g.add_node("retrieve_single", _retrieve_single)
    g.add_node("retrieve_all", _retrieve_all)
    g.add_node("generate", _generate)
    g.add_node("run_agent", _run_agent)

    g.add_edge(START, "route_query")
    g.add_conditional_edges(
        "route_query",
        _route_selector,
        {
            "no_retrieval": "answer_direct",
            "single_meeting": "retrieve_single",
            "semantic_all": "retrieve_all",
            "agentic": "run_agent",
        },
    )
    g.add_edge("retrieve_single", "generate")
    g.add_edge("retrieve_all", "generate")
    g.add_edge("answer_direct", END)
    g.add_edge("generate", END)
    g.add_edge("run_agent", END)
    return g.compile()


# --- Public entry ------------------------------------------------------------
def answer(
    question: str,
    history: Optional[list[dict]] = None,
    meetings_index: Optional[list[dict]] = None,
) -> dict:
    """Run the unified assistant. Returns {answer, route, citations, steps, artifact}."""
    if not vs.llm_available():
        return {
            "answer": "The assistant isn't available right now (no AI key is configured).",
            "route": "no_retrieval",
            "citations": [],
            "steps": [],
            "artifact": None,
        }
    try:
        final = _get_graph().invoke(
            {
                "question": question,
                "history": history or [],
                "meetings_index": meetings_index or [],
            }
        )
        return {
            "answer": final.get("generation", ""),
            "route": final.get("route", "semantic_all"),
            "citations": final.get("citations", []),
            "steps": final.get("steps", []),
            "artifact": final.get("artifact"),
        }
    except Exception as exc:  # noqa: BLE001 - never crash the endpoint
        logger.warning("Adaptive RAG failed: %s", exc)
        return {
            "answer": "Sorry, I couldn't answer that just now.",
            "route": "no_retrieval",
            "citations": [],
            "steps": [],
            "artifact": None,
        }
