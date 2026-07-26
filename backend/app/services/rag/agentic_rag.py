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

# Tools that need sign-in/OAuth and aren't wired yet. Shrink this as real integrations land.
_FUTURE_TOOLS = "Google Calendar, Google Tasks, Gmail, Google Drive, and Google Docs"

_AGENT_SYSTEM = (
    "You are an AI Meeting Intelligence Assistant — a warm, sharp teammate with recall of the "
    "user's meetings (not a generic chatbot). Turn meeting info into action: remember decisions, "
    "track action items, connect related meetings, move work forward.\n"
    "PRIORITY: answer from the user's meetings first (search_meetings → transcripts, summaries, "
    "action items), then the conversation. Use wikipedia ONLY when the meetings can't answer — and "
    "briefly say why.\n"
    "TOOLS: search_meetings (their meetings); wikipedia (external background/how-to); "
    "create_document (SAVE a composed deliverable — summary, action items, notes, brief, draft, "
    "how-to, or wiki findings — as a downloadable file); export_meeting_text(id) (raw full meeting, "
    "only if explicitly asked).\n"
    "DOWNLOADS: you CAN save downloadable files with create_document. NEVER say you 'can't save "
    "files', never tell the user to copy-paste, and never dump a long document inline — call "
    "create_document instead. Whenever the user wants to keep/save/download something, or after a "
    "wikipedia lookup, use create_document and note it's downloadable.\n"
    f"FUTURE TOOLS (not connected yet, pending sign-in): {_FUTURE_TOOLS}. You can't schedule "
    "events, send email, or create tasks yet. If asked: don't fake it and don't just say 'I "
    "can't' — say you'll do it directly once that's connected, and give the closest alternative "
    "now (prepare the details / a draft / a checklist, and offer to save it). E.g. 'I can't create "
    "the calendar event yet — once Google Calendar is connected I'll add it automatically; for now "
    "here are the details you can paste in…'\n"
    "GROUNDING + TONE: never invent meeting FACTS (decisions, dates, people, action items, "
    "deadlines) — if the meetings don't cover it, say so, then still help. Otherwise be warm, "
    "conversational, and proactive (offer a next step or related meeting). Be concise; bullets when "
    "useful; reference meetings by title. Don't narrate your tool plan or write tool "
    "names/backticks — use tools silently and present the result."
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def run(
    question: str,
    history: Optional[list[dict]] = None,
    meetings_index: Optional[list[dict]] = None,
    temperature: Optional[float] = None,
    google_creds=None,
) -> dict:
    """Run the tool-agent. Returns {generation, citations, steps, artifact}. Never raises.
    ``google_creds`` (when present) enables the live Google Calendar tools."""
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

    # --- Google Calendar tools (only when the user has connected Google) ---
    calendar_tools = []
    if google_creds is not None:
        from googleapiclient.discovery import build as _gbuild

        def _svc():
            return _gbuild("calendar", "v3", credentials=google_creds, cache_discovery=False)

        _tzc: dict = {}

        def _caltz() -> str:
            """The user's primary-calendar timezone (Google requires it on dateTimes)."""
            if "v" not in _tzc:
                try:
                    _tzc["v"] = _svc().calendars().get(calendarId="primary").execute().get(
                        "timeZone", "UTC"
                    )
                except Exception:  # noqa: BLE001
                    _tzc["v"] = "UTC"
            return _tzc["v"]

        @tool
        def list_calendar_events(days_ahead: int = 7) -> str:
            """List the user's upcoming Google Calendar events for the next N days
            (returns each event's id, title, and start time — use the id to amend/delete)."""
            import datetime as _dt

            now = _dt.datetime.now(_dt.timezone.utc)
            tmax = now + _dt.timedelta(days=max(1, days_ahead))
            try:
                res = (
                    _svc().events().list(
                        calendarId="primary", timeMin=now.isoformat(), timeMax=tmax.isoformat(),
                        singleEvents=True, orderBy="startTime", maxResults=10,
                    ).execute()
                )
            except Exception as exc:  # noqa: BLE001
                return f"Couldn't read the calendar: {exc}"
            items = res.get("items", [])
            if not items:
                return "No upcoming events in that window."
            return "\n".join(
                f"- id={e['id']} | {e.get('summary','(no title)')} | "
                f"{e.get('start',{}).get('dateTime', e.get('start',{}).get('date',''))}"
                for e in items
            )

        @tool
        def create_calendar_event(title: str, start: str, end: str, description: str = "") -> str:
            """Create a Google Calendar event. start/end are ISO datetimes
            (e.g. 2026-07-31T15:00:00). Returns the event id + link."""
            tz = _caltz()
            body = {"summary": title, "description": description,
                    "start": {"dateTime": start, "timeZone": tz},
                    "end": {"dateTime": end, "timeZone": tz}}
            try:
                ev = _svc().events().insert(calendarId="primary", body=body).execute()
            except Exception as exc:  # noqa: BLE001
                return f"Couldn't create the event: {exc}"
            return f"Created '{title}' (id={ev['id']}). Link: {ev.get('htmlLink','')}"

        @tool
        def update_calendar_event(event_id: str, title: str = "", start: str = "",
                                  end: str = "", description: str = "") -> str:
            """Amend/reschedule an existing event by id (pass only the fields to change)."""
            patch = {}
            if title:
                patch["summary"] = title
            if description:
                patch["description"] = description
            if start:
                patch["start"] = {"dateTime": start, "timeZone": _caltz()}
            if end:
                patch["end"] = {"dateTime": end, "timeZone": _caltz()}
            if not patch:
                return "Nothing to update — provide at least one field to change."
            try:
                ev = _svc().events().patch(calendarId="primary", eventId=event_id, body=patch).execute()
            except Exception as exc:  # noqa: BLE001
                return f"Couldn't update the event: {exc}"
            return f"Updated event {event_id}. Link: {ev.get('htmlLink','')}"

        @tool
        def delete_calendar_event(event_id: str) -> str:
            """Cancel/delete a calendar event by id."""
            try:
                _svc().events().delete(calendarId="primary", eventId=event_id).execute()
            except Exception as exc:  # noqa: BLE001
                return f"Couldn't delete the event: {exc}"
            return f"Deleted event {event_id}."

        calendar_tools = [
            list_calendar_events, create_calendar_event,
            update_calendar_event, delete_calendar_event,
        ]

    tools = [search_meetings, wikipedia, create_document, export_meeting_text] + calendar_tools
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
    import datetime as _date

    _today = _date.date.today().isoformat()
    cal_note = (
        f"\n\nGoogle Calendar IS connected — today is {_today}. You CAN list, create, "
        "amend/reschedule, and delete the user's real events with the calendar tools. Do it, then "
        "confirm (include the event link). Resolve relative dates to concrete ones from today "
        "(e.g. 'Friday' → the upcoming Friday's date, THIS year) and pass start/end as "
        "'YYYY-MM-DDTHH:MM:SS'. List first to get an event id before amending/deleting."
        if google_creds is not None
        else "\n\nGoogle Calendar is NOT connected for this user, so you cannot create real "
        "events yet — prepare the details and suggest they connect Google."
    )
    system = f"{_AGENT_SYSTEM}\n\nAvailable meetings (id: title):\n{listing}{cal_note}"
    messages: list = [SystemMessage(content=system)]
    for turn in (history or [])[-4:]:  # trim history to save tokens
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

    # Safety net: always offer a download after a Wikipedia lookup that produced no file,
    # so the user always has a download choice (even if the model forgot create_document).
    used_wiki = any(s.get("tool") == "wikipedia" for s in steps)
    offers = (
        [{"label": "📄 Download the findings",
          "prompt": "Save the information above as a downloadable text file."}]
        if used_wiki and not artifact
        else []
    )
    return {
        "generation": answer or "Done.",
        "citations": deduped,
        "steps": steps,
        "artifact": artifact or None,
        "offers": offers,
    }
