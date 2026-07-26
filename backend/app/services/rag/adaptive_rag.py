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

from ...config import settings
from . import vector_store as vs

logger = logging.getLogger(__name__)

VALID_ROUTES = {"no_retrieval", "single_meeting", "semantic_all", "agentic"}
VALID_TASK_KINDS = {"factual", "actionable", "creative"}


class AssistantState(TypedDict, total=False):
    question: str
    history: list[dict]
    meetings_index: list[dict]   # [{"id": int, "title": str}, ...]
    route: str
    meeting_id: Optional[int]
    task_kind: str               # factual | actionable | creative → temperature
    offer_download: bool         # is the answer worth saving as a document?
    documents: list[Document]
    generation: str
    citations: list[dict]
    steps: list[dict]
    artifact: Optional[dict]
    offers: list[dict]
    google_creds: object  # google.oauth2 Credentials for the Calendar tool (or None)
    user_id: Optional[int]  # who's asking → cross-session memory recall (RAG #3)


def temp_for(task_kind: Optional[str]) -> float:
    """Map a task kind to a generation temperature (precise → warmer)."""
    return {
        "factual": settings.temp_factual,
        "actionable": settings.temp_actionable,
        "creative": settings.temp_creative,
    }.get(task_kind or "factual", settings.temp_actionable)


def _build_offers(state: AssistantState) -> list[dict]:
    """Proactive download suggestions when the answer is deliverable-worthy."""
    if not state.get("offer_download"):
        return []
    return [
        {
            "label": "📄 Save a research brief",
            "prompt": "Research this and save a downloadable prep brief based on our conversation.",
        },
        {
            "label": "📝 Save a chat summary",
            "prompt": "Save a summary of this conversation as a downloadable document.",
        },
    ]


_ANSWER_SYSTEM = (
    "You are an AI Meeting Intelligence Assistant — a warm, sharp teammate with recall of the "
    "user's meetings. Answer using the Context excerpts from their meetings and the conversation "
    "so far; reference meetings by title and who said what. If the Context lacks the answer, say "
    "so plainly — never invent decisions, dates, participants, or action items (grounding applies "
    "to meeting FACTS). Be conversational and interactive (like ChatGPT/Claude), not curt; keep it "
    "concise, and proactively offer a helpful next step, a related meeting, or unfinished work when "
    "it genuinely helps."
)
_DIRECT_SYSTEM = (
    "You are an AI Meeting Intelligence Assistant for the user's meetings — warm, friendly, and "
    "interactive like ChatGPT/Claude. This message is small talk or general, so just engage "
    "naturally and helpfully. Feel free to converse and suggest, but don't invent meeting details; "
    "when useful, proactively offer what you can do (e.g. summarize a meeting, pull up action "
    "items, research a topic, or draft something)."
)


# Obvious "do an action" phrasing → route straight to the agent (skips the classifier LLM).
_ACTION_RE = re.compile(
    r"\b(schedule|re-?schedule|book|set up a|remind|reminder|e-?mail|send|invite|"
    r"add (?:it|this|that|them)?\s*to (?:my )?(?:calendar|tasks?|to-?do)|create (?:a )?task|"
    r"draft (?:an? )?(?:email|message)|put (?:it|this) on (?:my )?calendar)\b",
    re.IGNORECASE,
)


def _memory_block(state: AssistantState) -> str:
    """Recall this user's relevant past-conversation turns (RAG #3), or "" if none."""
    uid = state.get("user_id")
    if not uid:
        return ""
    try:
        from . import memory

        return memory.recall(int(uid), state["question"])
    except Exception:  # noqa: BLE001 - memory must never break an answer
        return ""


def _augment(system: str, extra: str) -> str:
    """Append a non-empty memory/context block to a system prompt."""
    return f"{system}\n\n{extra}" if extra else system


# --- Routing -----------------------------------------------------------------
def _route_query(state: AssistantState) -> dict:
    q = state["question"]
    # Free pre-check: obvious action requests go straight to the agent — no LLM call.
    if _ACTION_RE.search(q):
        return {"route": "agentic", "meeting_id": None,
                "task_kind": "actionable", "offer_download": False}
    index = state.get("meetings_index") or []
    listing = "\n".join(f"{m['id']}: {m['title']}" for m in index) or "(none)"
    hist = state.get("history") or []
    convo = "\n".join(
        f"{t.get('role')}: {str(t.get('content', ''))[:180]}" for t in hist[-2:]
    ) or "(none)"
    prompt = (
        "Classify how to answer the message; use the recent conversation to resolve references "
        "like 'this meeting' / 'that task'.\n"
        "route:\n"
        '- "no_retrieval": greeting/small talk, or not about meeting content.\n'
        '- "single_meeting": about ONE specific meeting from the list (give its id); resolve '
        '"this/that meeting" from the conversation.\n'
        '- "semantic_all": about meeting content generally, or spanning meetings.\n'
        '- "agentic": wants an action/document produced, OR an EXTERNAL lookup / explanation of a '
        'general concept or technology (e.g. "explain OAuth 2.0", "what is a statement timeout") — '
        "even if it's tied to a meeting task; the meetings don't explain general concepts.\n"
        "task_kind: factual | actionable | creative.  offer_download: true if worth saving as a "
        "document, else false.\n\n"
        f"Recent conversation:\n{convo}\n\n"
        f"Meetings:\n{listing}\n\n"
        f"Message: {q}\n\n"
        'Respond with JSON only: {"route": "...", "meeting_id": <id or null>, '
        '"task_kind": "...", "offer_download": true|false}'
    )
    route, meeting_id = "semantic_all", None
    task_kind, offer_download = "factual", False
    try:
        # Cheap 8B classifier (the deterministic pre-check already handles actions).
        raw = vs.get_fast_llm(0.0).invoke([HumanMessage(content=prompt)]).content or ""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        cand = str(data.get("route", "")).strip()
        if cand in VALID_ROUTES:
            route = cand
        mid = data.get("meeting_id")
        meeting_id = int(mid) if isinstance(mid, (int, str)) and str(mid).isdigit() else None
        tk = str(data.get("task_kind", "")).strip()
        if tk in VALID_TASK_KINDS:
            task_kind = tk
        offer_download = bool(data.get("offer_download", False))
    except Exception as exc:  # noqa: BLE001 - default to semantic_all on any error
        logger.warning("route_query failed (%s); defaulting to semantic_all.", exc)
    if route == "single_meeting" and meeting_id is None:
        route = "semantic_all"  # can't focus without a target
    return {
        "route": route,
        "meeting_id": meeting_id,
        "task_kind": task_kind,
        "offer_download": offer_download,
    }


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
    system = _augment(_ANSWER_SYSTEM, _memory_block(state))
    messages = _with_history(f"{system}\n\nContext:\n{context or '(none)'}", state)
    answer = vs.resilient_invoke(messages, temp_for(state.get("task_kind")))
    return {"generation": answer, "citations": _citations(docs), "offers": _build_offers(state)}


def _answer_direct(state: AssistantState) -> dict:
    messages = _with_history(_augment(_DIRECT_SYSTEM, _memory_block(state)), state)
    answer = vs.resilient_invoke(messages, temp_for(state.get("task_kind")))
    return {"generation": answer, "citations": [], "offers": _build_offers(state)}


def _run_agent(state: AssistantState) -> dict:
    """Agentic branch — delegate to the tool-agent subgraph (RAG #4)."""
    try:
        from . import agentic_rag

        return agentic_rag.run(
            state["question"],
            state.get("history"),
            state.get("meetings_index"),
            temperature=temp_for(state.get("task_kind")),
            google_creds=state.get("google_creds"),
            memory=_memory_block(state),
        )
    except vs.RateLimited:
        raise
    except Exception:  # noqa: BLE001 - fall back to a semantic answer
        docs = vs.similarity_search(state["question"])
        context = "\n\n".join(d.page_content for d in docs)
        messages = _with_history(f"{_ANSWER_SYSTEM}\n\nContext:\n{context or '(none)'}", state)
        answer = vs.resilient_invoke(messages, 0.2)
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
    google_creds=None,
    user_id: Optional[int] = None,
) -> dict:
    """Run the unified assistant. Returns {answer, route, citations, steps, artifact}."""
    if not vs.llm_available():
        return {
            "answer": "The assistant isn't available right now (no AI key is configured).",
            "route": "no_retrieval",
            "task_kind": None,
            "citations": [],
            "steps": [],
            "artifact": None,
            "offers": [],
        }
    try:
        final = _get_graph().invoke(
            {
                "question": question,
                "history": history or [],
                "meetings_index": meetings_index or [],
                "google_creds": google_creds,
                "user_id": user_id,
            }
        )
        return {
            "answer": final.get("generation", ""),
            "route": final.get("route", "semantic_all"),
            "task_kind": final.get("task_kind"),
            "citations": final.get("citations", []),
            "steps": final.get("steps", []),
            "artifact": final.get("artifact"),
            "offers": final.get("offers", []),
        }
    except vs.RateLimited as rl:
        return {
            "answer": vs.rate_limit_message(rl.retry_hint),
            "route": "no_retrieval",
            "task_kind": None,
            "citations": [],
            "steps": [],
            "artifact": None,
            "offers": [],
        }
    except Exception as exc:  # noqa: BLE001 - never crash the endpoint
        logger.warning("Adaptive RAG failed: %s", exc)
        return {
            "answer": "Sorry, I couldn't answer that just now.",
            "route": "no_retrieval",
            "task_kind": None,
            "citations": [],
            "steps": [],
            "artifact": None,
            "offers": [],
        }


# --- Streaming (SSE) — token-neutral (streams the existing generation call) --
def _context_from(docs) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('meeting_title','?')} | {d.metadata.get('speaker','?')} "
        f"@ {int(d.metadata.get('start_time',0))}s] {d.page_content}"
        for d in docs
    )


def _stream_generation(messages, temperature: float):
    """Yield answer tokens from the strong model, falling back to 8B on a pre-stream
    rate limit; yields the honest rate-limit message if both are exhausted."""
    yielded = False
    try:
        for chunk in vs.get_llm(temperature).stream(messages):
            if chunk.content:
                yielded = True
                yield chunk.content
        return
    except Exception as exc:  # noqa: BLE001
        if yielded or not vs.is_rate_limit(exc):
            return
    try:
        for chunk in vs.get_fast_llm(temperature).stream(messages):
            if chunk.content:
                yielded = True
                yield chunk.content
    except Exception as exc:  # noqa: BLE001
        if not yielded and vs.is_rate_limit(exc):
            yield vs.rate_limit_message(vs.rate_limit_retry_hint(exc))


def answer_stream(question, history=None, meetings_index=None, google_creds=None, user_id=None):
    """Generator yielding ('token', str) events then a final ('meta', dict). Never raises."""
    base_meta = {"route": "no_retrieval", "task_kind": None, "citations": [],
                 "steps": [], "artifact": None, "offers": []}
    if not vs.llm_available():
        yield ("token", "The assistant isn't available right now (no AI key is configured).")
        yield ("meta", base_meta)
        return
    state = {"question": question, "history": history or [], "meetings_index": meetings_index or [],
             "google_creds": google_creds, "user_id": user_id}
    try:
        state.update(_route_query(state))
        route = state.get("route", "semantic_all")

        if route == "agentic":
            # True agentic streaming (Phase 4): tools run inside the graph, then the
            # agent's final answer streams token-by-token (LangGraph stream_mode="messages").
            from . import agentic_rag

            for kind, data in agentic_rag.run_stream(
                question,
                state.get("history"),
                state.get("meetings_index"),
                temperature=temp_for(state.get("task_kind")),
                google_creds=state.get("google_creds"),
                memory=_memory_block(state),
            ):
                if kind == "meta":
                    data = {"route": "agentic", "task_kind": state.get("task_kind"), **data}
                yield (kind, data)
            return

        mem = _memory_block(state)
        if route == "no_retrieval":
            docs = []
            messages = _with_history(_augment(_DIRECT_SYSTEM, mem), state)
        else:
            docs = (vs.similarity_search(question, meeting_id=state.get("meeting_id"))
                    if route == "single_meeting" else vs.similarity_search(question))
            messages = _with_history(
                f"{_augment(_ANSWER_SYSTEM, mem)}\n\nContext:\n{_context_from(docs) or '(none)'}",
                state,
            )

        for tok in _stream_generation(messages, temp_for(state.get("task_kind"))):
            yield ("token", tok)
        yield ("meta", {
            "route": route, "task_kind": state.get("task_kind"),
            "citations": _citations(docs), "steps": [], "artifact": None,
            "offers": _build_offers(state),
        })
    except vs.RateLimited as rl:
        yield ("token", vs.rate_limit_message(rl.retry_hint))
        yield ("meta", base_meta)
    except Exception as exc:  # noqa: BLE001
        logger.warning("answer_stream failed: %s", exc)
        yield ("token", "Sorry, I couldn't answer that just now.")
        yield ("meta", base_meta)
