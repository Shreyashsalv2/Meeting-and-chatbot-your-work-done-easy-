"""RAG #4 — Agentic RAG: a tool-calling agent (the Adaptive router's ``agentic`` branch).

The RAGs so far follow a *fixed* pipeline we designed. An **agent** is different: we hand
the LLM a set of tools and let *it* decide which to call, in what order, and when to stop —
looping until it has what it needs. That's the LangGraph ReAct loop:

    START → agent → (did it ask for a tool?) ──yes──► tools → agent → …  (repeat)
                                              └──no──► END

Each pass, the agent sees the *results of the previous tool call* and picks the next move,
so "search the meetings, then look the unfamiliar term up on Wikipedia, then export a doc"
falls out naturally — no branch is hard-coded. Three tools, all no-OAuth / no-credentials:

1. ``search_meetings``    — CUSTOM: retrieves from the user's transcripts (our vector store).
2. ``wikipedia``          — PRE-MADE: LangChain's WikipediaQueryRun, for external background.
3. ``export_meeting_text``— CUSTOM: reuses the app's existing export builder to produce a
   downloadable text document (returned to the UI as an ``artifact``).

Bounded by ``settings.agent_max_steps`` so the loop can't run away. Returns the same
``{generation, citations, steps, artifact}`` shape the assistant response already carries,
so the UI needs no changes — the step trace + download button light up automatically.
"""
from __future__ import annotations

import json
import logging
from typing import Annotated, Optional, TypedDict

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ...config import settings
from . import vector_store as vs

logger = logging.getLogger(__name__)

_wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=800)
)

# Action categories we do NOT have integrations for yet (pending OAuth). Shrink this
# list as real tools are added so the agent's messaging stays accurate.
_UNSUPPORTED_ACTIONS = (
    "scheduling or changing calendar events, sending email or chat messages, creating tasks "
    "in a task tracker, or taking any action inside another app/service"
)

_AGENT_SYSTEM = (
    "You are the assistant for a meeting-notes app — the user's single command center. Act as an "
    "agent with tools.\n"
    "WHAT YOU CAN DO:\n"
    "- `search_meetings`: ground answers in the user's own meetings. Prefer this for anything "
    "about their meetings.\n"
    "- `wikipedia`: research external knowledge, concepts, and HOW-TO / best practices.\n"
    "- `create_document`: save a COMPOSED deliverable you write yourself (brief, summary, draft, "
    "checklist, how-to guide) as a downloadable document (NOT a raw transcript).\n"
    "- `export_meeting_text`: the raw full-meeting export (transcript + summary + action items + "
    "topics) for a meeting id — use ONLY when the user explicitly wants the whole meeting.\n\n"
    f"WHAT YOU CANNOT DO YET: you have NO external-action integrations — you cannot {_UNSUPPORTED_ACTIONS}. "
    "Those need integrations that aren't set up yet (pending OAuth).\n\n"
    "IF THE USER ASKS FOR AN ACTION YOU CAN'T DO YET: (a) briefly and honestly say you don't have "
    "that integration yet — NEVER pretend you performed it; (b) call `wikipedia` to research how "
    "the task is normally done / best practices; (c) give concise step-by-step guidance the user "
    "can follow themselves; (d) produce what you CAN — e.g. draft the email/message text to copy, "
    "or a checklist/how-to — and offer to save it via `create_document`.\n\n"
    "DELIVERABLES you CAN produce and how:\n"
    "1. RESEARCH / PREP BRIEF (default for actionable work) → `search_meetings` for the relevant "
    "action items/context, then `wikipedia` to research the key topic/concept, then "
    "`create_document` whose content is a synthesized brief = the meeting-grounded action items "
    "PLUS the Wikipedia findings, written as prose. Do NOT paste the transcript.\n"
    "2. CHAT SUMMARY → summarize the conversation so far, then `create_document` with that recap "
    "(no external research needed unless asked).\n"
    "3. HOW-TO GUIDE / DRAFT (for unsupported actions) → `wikipedia` for the how-to, then "
    "`create_document` with step-by-step guidance or the draft text.\n"
    "4. FULL MEETING EXPORT → `export_meeting_text(meeting_id)` only when explicitly asked for the "
    "whole meeting.\n\n"
    "Call tools as needed, then give a concise, clean final answer referencing meetings by title. "
    "Do NOT narrate your plan or write tool names/backticks (e.g. `search_meetings`, "
    "`create_document`) in your answer — just use the tools silently and present the result. Do "
    "not invent meeting content."
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def run(
    question: str,
    history: Optional[list[dict]] = None,
    meetings_index: Optional[list[dict]] = None,
    temperature: Optional[float] = None,
) -> dict:
    """Run the tool-agent. Returns {generation, citations, steps, artifact}. Never raises."""
    if not vs.llm_available():
        return {
            "generation": "The assistant isn't available right now (no AI key is configured).",
            "citations": [],
            "steps": [],
            "artifact": None,
        }

    # Per-run collectors (closures capture these so tools can report structured output).
    citations: list[dict] = []
    steps: list[dict] = []
    artifact: dict = {}

    @tool
    def search_meetings(query: str) -> str:
        """Search the user's meeting transcripts for passages relevant to a query.
        Returns matching moments with their meeting title, speaker, and timestamp."""
        docs = vs.similarity_search(query, k=5)
        if not docs:
            return "No relevant passages found in the meetings."
        lines = []
        for d in docs:
            md = d.metadata
            citations.append(
                {
                    "meeting_id": md.get("meeting_id"),
                    "meeting_title": md.get("meeting_title"),
                    "speaker": md.get("speaker"),
                    "start_time": md.get("start_time"),
                    "snippet": d.page_content[:160],
                }
            )
            lines.append(
                f"(meeting {md.get('meeting_id')} '{md.get('meeting_title')}' "
                f"@ {int(md.get('start_time', 0))}s) {d.page_content}"
            )
        return "\n".join(lines)

    @tool
    def wikipedia(query: str) -> str:
        """Look up concise external background about a topic or term on Wikipedia."""
        try:
            return _wiki.run(query)[:900] or "No Wikipedia result."
        except Exception as exc:  # noqa: BLE001
            return f"Wikipedia lookup failed: {exc}"

    @tool
    def create_document(title: str, content: str) -> str:
        """Save a COMPOSED deliverable you wrote (e.g. a prep/research brief = action items plus
        external background, or a chat summary) as a downloadable text document. This is NOT a
        meeting export — do not paste the raw transcript; write the synthesized document body."""
        from ...routers.export import _slug

        artifact["filename"] = f"{_slug(title)}.txt"
        artifact["content"] = content
        return f"Saved '{title}' as {artifact['filename']} (offered to the user as a download)."

    @tool
    def export_meeting_text(meeting_id: int) -> str:
        """Raw full-meeting export (summary + action items + full transcript) for a meeting id.
        Use ONLY when the user explicitly wants the whole meeting — for composed briefs use
        create_document instead."""
        from sqlmodel import Session

        from ... import models
        from ...database import engine
        from ...routers.export import _render_text, _slug

        try:
            with Session(engine) as session:
                meeting = session.get(models.Meeting, int(meeting_id))
                if meeting is None:
                    return f"No meeting with id {meeting_id}."
                artifact["filename"] = f"{_slug(meeting.title)}.txt"
                artifact["content"] = _render_text(meeting)
                title = meeting.title
            return f"Exported '{title}' as {artifact['filename']} (offered to the user as a download)."
        except Exception as exc:  # noqa: BLE001
            return f"Export failed: {exc}"

    tools = [search_meetings, wikipedia, create_document, export_meeting_text]
    tools_by_name = {t.name: t for t in tools}
    # Bound temperature: honor the task's warmth but keep tool-calling reliable.
    agent_temp = min(temperature if temperature is not None else 0.1, 0.4)
    llm = vs.get_llm(agent_temp).bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """Invoke the tool-bound LLM, resilient to Groq's occasional malformed tool call.

        llama-3.3-70b sometimes emits a bad tool-call format → Groq 400 'tool_use_failed'.
        We retry once, then fall back to a no-tools answer so the agent never dead-ends
        (it can still synthesize from any tool results already gathered)."""
        msgs = state["messages"]
        try:
            return {"messages": [llm.invoke(msgs)]}
        except Exception as exc:  # noqa: BLE001
            if "tool_use_failed" in str(exc):
                try:
                    return {"messages": [llm.invoke(msgs)]}  # one retry (stochastic)
                except Exception:  # noqa: BLE001
                    pass
            try:
                return {"messages": [vs.get_llm(0.2).invoke(msgs)]}  # answer without tools
            except Exception:  # noqa: BLE001
                return {"messages": [AIMessage(content="I couldn't complete that tool step.")]}

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        out = []
        for tc in getattr(last, "tool_calls", []) or []:
            name, args = tc["name"], tc.get("args", {})
            try:
                result = str(tools_by_name[name].invoke(args))
            except Exception as exc:  # noqa: BLE001
                result = f"tool error: {exc}"
            steps.append({"tool": name, "input": json.dumps(args)[:200], "output": result[:600]})
            out.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        return {"messages": out}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and len(steps) < settings.agent_max_steps:
            return "tools"
        return "end"

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    g.add_edge("tools", "agent")
    graph = g.compile()

    listing = "\n".join(f"{m['id']}: {m['title']}" for m in (meetings_index or [])) or "(none)"
    system = f"{_AGENT_SYSTEM}\n\nAvailable meetings (id: title):\n{listing}"
    messages: list = [SystemMessage(content=system)]
    for turn in (history or [])[-6:]:
        role, content = turn.get("role"), str(turn.get("content", "")).strip()
        if content:
            messages.append(
                HumanMessage(content=content) if role == "user" else AIMessage(content=content)
            )
    messages.append(HumanMessage(content=question))

    try:
        final = graph.invoke({"messages": messages})
        answer = (final["messages"][-1].content or "").strip()
    except Exception as exc:  # noqa: BLE001 - never crash the assistant
        logger.warning("Agentic RAG failed: %s", exc)
        return {
            "generation": "Sorry, I couldn't complete that request just now.",
            "citations": citations[:5],
            "steps": steps,
            "artifact": artifact or None,
        }

    # Dedupe citations by (meeting, timestamp).
    seen, deduped = set(), []
    for c in citations:
        key = (c.get("meeting_id"), c.get("start_time"))
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    return {
        "generation": answer or "Done.",
        "citations": deduped,
        "steps": steps,
        "artifact": artifact or None,
    }
