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
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
)

# Tools that need sign-in/OAuth and aren't wired yet. Shrink this as real integrations
# land so the agent's messaging stays accurate.
_FUTURE_TOOLS = "Google Calendar (events), Google Tasks, Gmail (email), Google Drive, and Google Docs"

_AGENT_SYSTEM = (
    "You are an AI Meeting Intelligence Assistant — an intelligent teammate with recall of the "
    "user's meetings, not a generic chatbot. Turn meeting information into actionable project "
    "intelligence: remember decisions, track action items and unfinished work, connect related "
    "meetings, and help the user move work forward.\n\n"

    "INFORMATION PRIORITY — answer from the earliest source that has the answer:\n"
    "1) the user's meetings (transcripts, summaries, action items) via `search_meetings` → "
    "2) the conversation so far → 3) earlier tool results → 4) external knowledge via `wikipedia` "
    "→ 5) general knowledge. NEVER search externally if the answer is already in the meetings or "
    "this conversation.\n\n"

    "TOOLS:\n"
    "- `search_meetings`: retrieve from the user's own meetings (who said what, with timestamps). "
    "Prefer it for anything about their meetings.\n"
    "- `wikipedia`: external background / how-to / best practices. Use ONLY when the meetings are "
    "insufficient or the user wants general knowledge — and briefly say WHY (e.g. 'this isn't "
    "covered in your meetings, so I'll look up OAuth 2.0 for reliable background').\n"
    "- `create_document`: save a composed deliverable you write (summary, action-item list, key "
    "notes, prep/research brief, draft, checklist, how-to, or Wikipedia findings) as a downloadable "
    "text file — NOT a raw transcript.\n"
    "- `export_meeting_text(meeting_id)`: the raw full-meeting export — only when the user "
    "explicitly wants the whole meeting.\n\n"

    "BEHAVIOR:\n"
    "- Continuity: the conversation is continuous. Resolve references like 'that task', 'the API', "
    "'the previous decision' from context; don't ask for clarification when context makes it clear.\n"
    "- Action items: when the user refers to a task, tie it to the matching action item — explain "
    "what it is, why it exists (cite the meeting), and suggest concrete next steps + useful resources.\n"
    "- Proactivity (only when it genuinely helps, not every turn): note a related meeting ('also "
    "discussed in …'), unfinished/pending work, downstream effects, or a sensible next action.\n"
    "- Downloads (when relevant, not every reply): remind the user they can download the result "
    "(summary, action items, key notes, chat history, or Wikipedia findings) as a text file.\n\n"

    f"TOOLS YOU DON'T HAVE YET (pending sign-in/OAuth): {_FUTURE_TOOLS}. You cannot schedule events, "
    "send email, or create tasks yet. If the user asks for one: do NOT say a flat 'I can't' and do "
    "NOT pretend you did it. Instead — (a) say you'll do it directly once that integration is "
    "enabled, (b) offer the closest alternative now (prepare the event/email/task details, or a "
    "checklist), and (c) if useful, look up how it's done via `wikipedia` and offer to save it. "
    "Example: 'I can't create the calendar event yet — once Google Calendar is connected I'll add "
    "it automatically. For now, here are the details you can paste in…'\n\n"

    "ACCURACY: never invent decisions, meetings, participants, dates, action items, deadlines, tool "
    "outputs, or documents. If the meetings don't contain something, say so plainly (e.g. 'the "
    "meeting history doesn't record a confirmed approval for that'). Accuracy over completeness.\n\n"

    "STYLE & TONE: be warm, natural, and genuinely interactive — like a sharp, friendly teammate "
    "(think how ChatGPT/Claude converse), NOT a rigid lookup tool. Greet and engage, keep it "
    "concise and action-oriented, use bullets when they help, and proactively offer a helpful "
    "suggestion or next step. Grounding applies to meeting FACTS only — never invent those — but "
    "you may still converse, reason, and suggest freely. Reference meetings by title. Do NOT "
    "narrate your plan or write tool names/backticks in the answer — use tools silently and "
    "present the result."
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
            if vs.is_rate_limit(exc):
                # Main model out of daily budget → keep tool-calling on the fast model.
                try:
                    fast = vs.get_fast_llm(agent_temp).bind_tools(tools)
                    return {"messages": [fast.invoke(msgs)]}
                except Exception as exc2:  # noqa: BLE001
                    if vs.is_rate_limit(exc2):
                        raise vs.RateLimited(
                            vs.rate_limit_retry_hint(exc2) or vs.rate_limit_retry_hint(exc)
                        ) from exc2
            if "tool_use_failed" in str(exc):
                try:
                    return {"messages": [llm.invoke(msgs)]}  # one retry (stochastic)
                except Exception:  # noqa: BLE001
                    pass
            try:
                # Answer without tools, strong→fast fallback (may raise RateLimited).
                return {"messages": [AIMessage(content=vs.resilient_invoke(msgs, 0.2))]}
            except vs.RateLimited:
                raise
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
    except vs.RateLimited as rl:
        return {
            "generation": vs.rate_limit_message(rl.retry_hint),
            "citations": citations[:5],
            "steps": steps,
            "artifact": artifact or None,
        }
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
