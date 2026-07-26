# RAG Learning Notes

A build-along guide to the four RAG techniques added to this project. Written as we
build so you can read it end-to-end afterwards and understand *why*, not just *what*.

> **Mental model to hold onto:**
> **LangChain = the plumbing** (chunking, embeddings, retrieval, talking to the LLM, tools).
> **LangGraph = the control flow** (a *graph* of steps with branches, loops, and retries).
> A plain "retrieve then answer" is a straight line. The moment you want to *grade* what you
> retrieved, *route* between strategies, *retry* on a bad answer, or let an agent *loop over
> tools*, you want a graph — that's LangGraph.

---

## 0. What is RAG, in one paragraph

An LLM only knows what's in its training data + what you put in the prompt. **RAG
(Retrieval-Augmented Generation)** means: before answering, *retrieve* the most relevant
pieces of your own data and put them in the prompt, so the model answers from *your* facts
instead of guessing. The hard parts are (a) turning documents into searchable pieces
(*chunking + embeddings + a vector store*) and (b) deciding *how* to retrieve and *whether to
trust* what you got (*that's where the named techniques differ*).

---

## 1. The shared foundation (Phase A) ✅

Everything the four RAGs need lives in one module:
[`backend/app/services/rag/vector_store.py`](../backend/app/services/rag/vector_store.py).

### The pieces (LangChain plumbing)

| Piece | What it does | In our code |
|---|---|---|
| **Embeddings** | Turn text into a vector (list of numbers) so "similar meaning" ≈ "close vectors". | `get_embeddings()` → `FastEmbedEmbeddings` (fastembed, local ONNX model `bge-small-en-v1.5`, **no API key, no torch**) |
| **Chunking** | Split a transcript into retrievable pieces. | `_chunk_meeting()` — groups consecutive speaker turns up to `rag_chunk_size` chars |
| **Vector store** | Store chunks + vectors; find nearest ones to a query. | `get_store()` → **Chroma**, persisted to `backend/chroma/` |
| **Retriever** | The query interface over the store (optionally filtered). | `retriever()` / `similarity_search()` |
| **LLM** | The model that writes the final answer. | `get_llm()` → `ChatGroq` (reuses the existing Groq key) |

### Two design decisions worth understanding

1. **Why one Chroma collection with a `meeting_id` on every chunk (not one store per
   meeting)?** Because two of our RAGs need *both* "search inside one meeting" (Self-RAG) and
   "search across all meetings" (Adaptive/Fusion/Agent). With metadata on each chunk we get
   both from one store: pass `filter={"meeting_id": id}` for the first, omit it for the second.
   Chroma supports this metadata filtering natively — that's why we picked it over FAISS.

2. **Why group speaker turns instead of using LangChain's `RecursiveCharacterTextSplitter`?**
   The textbook splitter cuts on character count and would happily slice mid-sentence, losing
   which *speaker* said it and *when*. We keep each chunk anchored to the **timestamp of its
   first utterance** (`start_time` in the chunk's metadata). That timestamp is exactly what
   lets a citation deep-link into the transcript (`?t=56`). Precise citations > textbook
   chunking, for this app. (Trade-off: chunk "overlap" doesn't apply in turn-grouping mode.)

### How it stays wired to the app (best-effort, never breaks CRUD)

- On **create** → `crud.index_meeting_safe()` adds the new meeting's chunks.
- On **update** → re-index (keeps the `meeting_title` metadata fresh).
- On **delete** → `crud.deindex_meeting_safe()` removes its chunks.
- On **startup** → `vector_store.ensure_indexed()` rebuilds the index if empty (the Chroma dir
  is ephemeral on Render, just like the SQLite file — so we rebuild from the DB on boot).

Every one of those is wrapped in `try/except`: if the RAG layer ever fails, meeting CRUD keeps
working. This mirrors the project's existing rule that *AI must never break meeting creation*.

### Design principles baked in (so later changes are cheap)

- **Dependency Inversion** — graphs depend on `get_llm` / `get_embeddings` / `retriever`, never
  on Chroma/fastembed/Groq classes. Moving to a hosted embeddings API = edit `get_embeddings()`
  only.
- **Single Responsibility** — one module per RAG; routers stay thin; schemas are the contract.
- **Open/Closed** — adding a RAG = new module + a new route value; adding an agent tool = append
  to a list. Existing graphs don't change.

### Verify it yourself
```bash
cd backend && venv/bin/python - <<'PY'
from sqlmodel import Session
from app.database import engine
from app.services.rag import vector_store as vs
with Session(engine) as s: vs.reindex_all(s)
print("chunks:", vs.count())
for d in vs.similarity_search("onboarding activation drop", k=3):
    print(d.metadata["meeting_id"], d.metadata["start_time"], d.page_content[:60])
PY
```

---

## 2. RAG #1 — Self-RAG (per-meeting chat)  ✅ Phase B

**File:** [`backend/app/services/rag/self_rag.py`](../backend/app/services/rag/self_rag.py) ·
**Wired at:** `POST /api/meetings/{id}/chat` (`routers/meetings.py`) · **UI:** `MeetingChat.tsx`

### The idea
Plain ("naive") RAG does: retrieve → stuff into prompt → answer, and *hopes* the chunks
were relevant and the answer is faithful. **Self-RAG adds reflection**: the model grades its
own retrieval and its own answer, and *corrects* itself. Concretely we added three checks the
naive version doesn't have:
1. **Grade documents** — are the retrieved chunks actually relevant? Drop the ones that aren't.
2. **Rewrite & retry** — if nothing relevant came back, rewrite the question and retrieve again.
3. **Grade the answer** — is it *grounded* in the context (not invented) and does it *answer*
   the question? If not, retry (bounded).

This is why it's a **graph**: naive RAG is a straight line; Self-RAG has branches and a loop.

### The graph
```
START → retrieve → grade_documents ──has relevant?──┬─ yes ─────────► generate → grade_generation
                                                    ├─ no, retry left ► transform_query → retrieve
                                                    └─ no, no retries ► not_covered → END
grade_generation ──useful?──┬─ yes / out of budget ─► END
                            └─ no ► transform_query → retrieve
```
- **`retrieve`** → `vector_store.similarity_search(q, meeting_id=...)` — note the `meeting_id`
  filter: this chat only sees *its* meeting's chunks.
- **`grade_documents`** → one LLM call: "which of these numbered excerpts are relevant?" We keep
  those. (One call for all chunks, not one per chunk — cheaper.)
- **`transform_query`** → one LLM call rewriting the question into a better search query. This is
  the only place `retries` is incremented, so the loop is guaranteed to end after
  `self_rag_max_retries`.
- **`generate`** → the answer, from kept chunks + last-8 chat history, with a hard rule to say
  *"I don't know based on this meeting"* when the context lacks the answer.
- **`grade_generation`** → one LLM call returning two yes/no tokens (grounded? answers?). If bad
  and we still have budget, loop back through `transform_query`.

### Citations
The chunks that survive grading carry `speaker` + `start_time` metadata, so the answer ships
with **citations** the UI renders as chips. Clicking a chip calls `onSeek(start_time)` →
`seekTo()` in `MeetingDetailView`, which moves the (simulated) player to that moment. Grounding
you can *click*.

### Key implementation choices (and why)
- **String grading, not function-calling.** Graders ask for `yes/no` (or "1,3") text and parse
  it. Simpler and more robust across models than structured/tool output, and every grader
  **fails open** (on error it assumes "keep"/"accept") so a flaky grader never blocks the user.
- **One shared `retries` counter, incremented only in `transform_query`.** LangGraph conditional
  edges *decide* but can't *mutate* state, so the increment lives in a node. One increment site =
  easy to reason about the bound.
- **Never raises.** `answer()` catches everything and falls back to the old naive
  full-transcript chat (`groq_service.chat_with_meeting`), then to a friendly message.

### Verify
Ask on `/meetings/1`: *"Who owns the search rebuild spec and by when?"* → grounded answer +
a `Marcus Rodriguez · 1:54` chip that seeks the player. Ask *"What is the capital of France?"*
→ "I don't know based on this meeting." with no chips.

## 3. RAG #3 — RAG-Fusion / Multi-Query (semantic search)  ✅ Phase C

**File:** [`backend/app/services/rag/fusion_rag.py`](../backend/app/services/rag/fusion_rag.py) ·
**Wired at:** `GET /api/search/semantic` (`routers/search.py`) · **UI:** `SearchView.tsx` (Keyword | Semantic toggle)

### The idea
Two independent upgrades over the app's old substring search:
1. **Semantic** — search by *meaning* (embeddings), so "reduce churn / keep users engaged" finds
   the onboarding & activation discussions even though those exact words never appear.
2. **Multi-query fusion (RAG-Fusion)** — one phrasing is a narrow lens. The LLM rewrites the
   query into several phrasings, we retrieve for *each*, then merge the ranked lists. Passages
   that rank well across *many* phrasings win — more robust than any single query.

### The graph
```
START → generate_queries → retrieve (one ranked list per sub-query) → fuse (RRF) → END
```
- **`generate_queries`** → LLM produces 3 alternative phrasings; we search the original **+** all 3.
- **`retrieve`** → `similarity_search` for each sub-query (no `meeting_id` filter → across *all*
  meetings). Each returns a ranked list.
- **`fuse`** → **Reciprocal Rank Fusion**: a passage's score = Σ `1 / (K + rank)` over the lists
  it appears in (`K = 60`). We sort by that. RRF uses only *ranks*, never raw similarity scores,
  so there's nothing to calibrate across queries — that's why it's the standard fusion method.

### Why it reuses the existing UI
`fusion_rag.search()` returns dicts shaped exactly like `SearchMatch` (`field="transcript"` +
`start_time`), so the `/search` page renders them with the same result cards and the same
click-to-timestamp deep-links — the only new UI is a **Keyword | Semantic** toggle. Adding a
whole new capability with almost no new UI is the payoff of keeping a stable response contract.

### Degradation (never breaks)
No Groq key → skip multi-query, fall back to a single semantic query. Any error → single query,
then empty results. Search never throws.

### Verify
On `/search`, switch to **Semantic** and search *"reduce customer churn and keep users
engaged"*. Keyword mode → **0 results** (no literal match). Semantic mode → the onboarding,
activation, and marketing moments, each deep-linking into the transcript.

## 4. RAG #2 — Adaptive RAG (the unified assistant's router)  ✅ Phase D

**File:** [`backend/app/services/rag/adaptive_rag.py`](../backend/app/services/rag/adaptive_rag.py) ·
**Wired at:** `POST /api/assistant` (`routers/assistant.py`) · **UI:** new `/assistant` page (`AssistantChat.tsx`)

### The idea
Not every question deserves a retrieval. "Hi" needs none; "what caused the outage in the
Weekly Sync?" needs *one* meeting; "action items about onboarding across meetings?" needs *all*
of them; "draft me a doc" needs *tools*. **Adaptive RAG routes first**, then does the minimum
work that answers the question. This is also the app's single "assistant" surface — the router
*is* the brain, and the agent (RAG #4) is just one of its branches.

### The graph
```
START → route_query ──┬─ no_retrieval ─► answer_direct ─────────► END
                      ├─ single_meeting ► retrieve_single ► generate ► END
                      ├─ semantic_all ──► retrieve_all ► generate ► END
                      └─ agentic ───────► run_agent ──────────────► END
```
- **`route_query`** → the classifier. We feed it the question, recent history, **and the list of
  meetings (id + title)**. It returns JSON `{"route": ..., "meeting_id": ...}`. Giving it the
  meeting list is what lets `single_meeting` know *which* meeting to focus (it returns the id).
  If it picks `single_meeting` but can't name one, we downgrade to `semantic_all`.
- **`answer_direct`** → replies without any retrieval (greetings/meta).
- **`retrieve_single` / `retrieve_all`** → the same `similarity_search`, the only difference is
  the `meeting_id` filter. Both flow into one `generate` node.
- **`generate`** → answers from context with cross-meeting citations + a `route` badge.
- **`run_agent`** → the seam to RAG #4 (Phase E). In Phase D it's a stub that falls back to a
  semantic answer.

### One conversation, not two modes
The endpoint returns **one stable shape**: `{answer, route, citations, steps, artifact}`. The UI
shows a **route badge** (so you can *see* the router's decision), cross-meeting citation chips
that deep-link into transcripts, and — already wired but empty until Phase E — a tool-step trace
and a download button. Because the shape is fixed, Phase E adds the agent with **zero** frontend
changes.

### Why this is good design (OCP in action)
`run_agent` tries to `import agentic_rag` and call it; if that module isn't there yet, it falls
back. So Phase E adds a whole new capability by *adding a file*, not editing the router. Same
story for routes: they're keys in a dict, not hard-coded branches.

### Verify
On `/assistant`: "hi" → **Direct answer** badge, no citations. "In the Weekly Engineering Sync,
what caused the outage?" → **One meeting** badge, citations only from that meeting. "What did we
decide about onboarding across all meetings?" → **All meetings** badge, citations spanning several.

## 5. RAG #4 — Agentic RAG (tool-calling agent subgraph)  ✅ Phase E

**File:** [`backend/app/services/rag/agentic_rag.py`](../backend/app/services/rag/agentic_rag.py) ·
**Reached via:** the Adaptive router's `agentic` branch (`POST /api/assistant`) · **UI:** the same `/assistant` chat (step trace + download)

### The idea
Every RAG before this ran a pipeline *we* designed. An **agent** flips it: we give the LLM
tools and let *it* plan. It looks at the question, calls a tool, sees the result, decides the
next tool, and loops until it can answer. The control flow is emergent, not hard-coded — that's
what "agentic" means.

### The graph (the ReAct loop)
```
START → agent → (asked for a tool?) ──yes──► tools → agent → …  (repeat, bounded)
                                    └──no──► END
```
- **`agent`** → `ChatGroq.bind_tools([...])`. The model replies either with a final answer *or*
  with `tool_calls`.
- **`tools`** → we execute each requested tool and feed the results back as `ToolMessage`s.
- **`should_continue`** → loops back to `agent` while there are tool calls *and* we're under
  `agent_max_steps` (the runaway-loop guard).

### The three tools (1 custom retriever + 1 pre-made + 1 custom action)
| Tool | Kind | What it does |
|---|---|---|
| `search_meetings` | **custom** | Retrieves from the user's transcripts (our vector store). This is the "RAG" inside the agent, and it's what fills the **citations**. |
| `wikipedia` | **pre-made** | LangChain's `WikipediaQueryRun` — external background, no key/OAuth. |
| `export_meeting_text` | **custom** | Reuses the app's existing export builder (`routers/export._render_text`) to produce a **downloadable text doc**, returned as the `artifact`. |

"Next tool depends on the last result" is real here: *"search my meetings for the outage, then
explain the term via Wikipedia"* → the agent runs `search_meetings`, reads what it found, then
calls `wikipedia` on the term — we didn't wire that order, it decided.

### How steps / citations / artifact get out of the agent
Tools return *strings* to the LLM, but the UI wants structure. So the tools are **closures**
defined inside `run()` that also append to local `citations` / `artifact` lists, and the `tools`
node records each call into `steps`. All three come back in the response, and the assistant UI
(built in Phase D) already renders them: a **route badge**, an expandable **tool-step trace**,
and a **Download** button for the artifact — zero new frontend code.

### Two real bugs we hit (see gotchas)
1. The agent guessed **wrong meeting ids** — fixed by handing it the `id: title` list in its
   prompt (a router already had it).
2. Groq's llama-3.3-70b sometimes emits a **malformed tool call** → 400 `tool_use_failed` —
   fixed with a retry-then-answer-without-tools fallback so the agent never dead-ends.

### Verify
On `/assistant`: *"Draft and export a document for the Weekly Engineering Sync meeting"* → an
**Agent + tools** badge, a `export_meeting_text` step, and a **Download** button for
`weekly-engineering-sync.txt`. *"Search my meetings for the outage, then explain a statement
timeout via Wikipedia"* → a two-step trace (`search_meetings` → `wikipedia`) + citations.

---

## Recap: the four techniques side by side

| RAG | Core move | LangGraph shape | Where |
|---|---|---|---|
| **Self-RAG** | reflect: grade docs & grade own answer, retry | loop with self-critique | per-meeting chat |
| **RAG-Fusion** | multi-query + reciprocal rank fusion | fan-out → fuse | `/search` semantic |
| **Adaptive RAG** | route to the cheapest strategy | router with branches | `/assistant` brain |
| **Agentic RAG** | let the LLM pick tools in a loop | ReAct agent ↔ tools | `/assistant` agent branch |

Same foundation under all four (§1); each adds one idea. That's the whole point — RAG isn't one
thing, it's a family of control-flow patterns over retrieval, and LangGraph is how you express them.

---

## Verification (Phase F)

All four were driven in a real browser end-to-end (plus `next build` and the backend `/docs`):
- **Self-RAG** — asked "who owns the search rebuild spec and by when?" on `/meetings/1` → grounded
  answer + a `Marcus Rodriguez · 1:54` citation chip; clicking it seeked the player to 1:55.
- **RAG-Fusion** — `/search` in Semantic mode for "reduce customer churn and keep users engaged"
  (no literal match) → **8** relevant transcript results.
- **Adaptive RAG** — `/assistant`: "onboarding across all meetings" → **All meetings** badge + chips
  spanning several meetings; a greeting → direct answer.
- **Agentic RAG** — `/assistant`: "draft and export a document for the Weekly Engineering Sync" →
  **Agent + tools** badge, an `export_meeting_text({meeting_id: 2})` step, and a downloadable
  `weekly-engineering-sync.txt`; "search then explain via Wikipedia" → `search_meetings → wikipedia`.

Build/API gates: `next build` compiles all routes; `/api/assistant` and `/api/search/semantic` are
in the OpenAPI schema; existing CRUD/chat contracts unchanged.

---

## Follow-up: context-aware download offers + task-based temperature

**Files:** `adaptive_rag.py` (classify + offers + `temp_for`), `agentic_rag.py` (`create_document`
tool + task temperature), `AssistantChat.tsx` (chips), `config.py` (temperature tiers).

### The idea
Two upgrades to the assistant:
1. **Don't make the user command a download.** When the router senses the answer is *deliverable-worthy*
   (actionable/creative work), the assistant proactively shows **suggestion chips** — 📄 *Save a research
   brief* and 📝 *Save a chat summary* — so the user just clicks instead of typing an export command.
2. **Match the LLM's temperature to the task.** Factual lookups → `0.15` (precise), actionable drafting →
   `0.3`, creative/ideation → `0.6`.

### How it works (reusing what already exists)
- The **same** router LLM call now also returns `task_kind` and `offer_download` — no extra latency. `task_kind`
  → `temp_for()` → the temperature used by `generate` / `answer_direct` / the agent.
- When `offer_download` is true on a **non-agentic** answer, the response carries `offers` (a list). The UI
  renders them as chips; clicking one calls the existing `send(offer.prompt)` — so the chip is just a canned
  message that flows through the normal pipeline. Zero new endpoints or API methods.
- The chip's prompt is an **action**, so the router sends it to the `agentic` branch, where a **new
  `create_document(title, content)` tool** saves a *composed* deliverable. The agent writes the body itself:
  for a research brief it runs `search_meetings → wikipedia → create_document` (so Wikipedia is now part of the
  actionable pipeline); for a chat summary it summarizes the conversation. This is deliberately **distinct from
  `export_meeting_text`**, which still dumps the whole meeting transcript — three separate downloads, not one.

### Gotchas
- **Actionable phrasing can skip the offer.** "Prepare a plan" is classified as an *action* → routes straight to
  `agentic` (no chip). The chip is for actionable *Q&A* ("what are the action items?"). Both are valid; the chip
  just covers the case where the user didn't explicitly ask for a file.
- **llama sometimes answers without calling tools** even when it should draft/save. The chip's prompt is very
  explicit ("Research this and save a downloadable prep brief"), which makes tool use reliable; the bare
  "prepare a plan" phrasing is where it occasionally no-ops.
- **Agent temperature is clamped to ≤ 0.4** even for creative tasks — higher temperatures make Groq's
  tool-calling emit malformed calls more often (`tool_use_failed`). Precise generation gets the full range; the
  tool-using agent stays conservative.
- **The `agent_max_steps` cap bounds cycles, not raw steps** — one agent turn can request several tools at once,
  so you may see more than `agent_max_steps` entries in the trace. Still bounded, just not 1:1.

---

## Follow-up: capability-aware assistant (honest limits + Wikipedia how-to)

**Files:** `agentic_rag.py` (`_UNSUPPORTED_ACTIONS` + `_AGENT_SYSTEM`), `adaptive_rag.py` (route), `AssistantChat.tsx` (copy).

### The idea
The `/assistant` agent is the **command center** — the user should do everything from here. But some actions
need integrations we don't have yet (Calendar, email, task trackers — pending OAuth). Instead of failing or
pretending, the agent is now **capability-aware**: when asked for something it can't do, it (1) honestly says it
lacks that integration, (2) uses **Wikipedia as a how-to helper** to research how it's done, (3) gives
step-by-step guidance, and (4) produces what it *can* — e.g. drafts the email text and saves it via
`create_document`. So the user stays in one place even for not-yet-automated actions. This is the bridge to the
future OAuth phase: when real action tools land, they slot into the same agent and shrink `_UNSUPPORTED_ACTIONS`.

### How it works
- `_AGENT_SYSTEM` now spells out **what it CAN do** and **what it CANNOT do yet** (`_UNSUPPORTED_ACTIONS`, a
  single editable constant), plus the fallback rule (decline honestly → Wikipedia how-to → guidance → draft).
- The router's `agentic` branch was broadened so "do X / schedule / email / remind" requests reach the agent
  (that's where the capability-aware handling lives).
- Verified: "email the action items" → honest decline + `wikipedia`/`create_document` → downloadable draft;
  "schedule a meeting" → honest decline + clean step-by-step guidance; **never falsely claims to have acted**.

### Gotchas
- **The model narrates its tool plan.** llama would write "First I'll `search_meetings`, then `wikipedia`…"
  straight into the answer (ugly, leaks tool names). Fix: an explicit "do NOT narrate your plan or write tool
  names/backticks — use tools silently and present the result" line in the prompt. Cleared the leakage.
- **Simple how-tos skip Wikipedia.** For common-knowledge actions (scheduling a meeting) the model answers from
  its own knowledge (steps=[]) rather than calling Wikipedia. That's fine — the honest decline + clean guidance
  is what matters; Wikipedia fires when the how-to genuinely needs external detail (e.g. email best practices).
- **Honesty is prompt-enforced, not guaranteed.** We assert in tests that the answer never says "I've
  scheduled/sent…". Worth a periodic check if the prompt changes.

---

## Follow-up: surviving Groq rate limits (per-model budgets, 8B fallback, honest errors)

**Files:** `config.py` (`groq_fast_model`, `agent_max_steps` 6→4), `vector_store.py` (`get_fast_llm`,
`resilient_invoke`, `RateLimited`, `rate_limit_*`), plus `adaptive_rag.py` / `self_rag.py` / `fusion_rag.py` /
`agentic_rag.py` (route cheap calls to 8B; honest messages).

### What broke, and why
The assistant went fully non-responsive — "Sorry, I couldn't answer that just now" / "I couldn't complete that
tool step" on *everything*. The log showed the real cause: **Groq's free-tier daily token cap** for
`llama-3.3-70b-versatile` was exhausted (`429 … tokens per day (TPD): Limit 100000, Used ~99737`). Our RAG stack
fires *many* 70B calls per turn (router + graders + multi-query + the agent loop), so heavy use drained the day's
budget; after that every call 429'd and fell through to the generic error strings. **Not a code bug — a quota.**

### The fix (two ideas)
1. **Rate limits are per-model.** `llama-3.1-8b-instant` has a *separate, larger* daily budget. So the cheap
   "housekeeping" calls (routing, Self-RAG graders, RAG-Fusion multi-query) now run on **8B** via `get_fast_llm`,
   and only user-facing final generation + the agent's tool-calling use 70B. This alone makes the 70B budget last
   far longer.
2. **Degrade, don't die.** `resilient_invoke()` tries 70B and, on a 429, transparently retries the *same*
   messages on 8B — so when 70B is capped the assistant keeps answering (lower quality) instead of erroring. The
   agent does the same (rebinds tools to 8B). Only when **both** models are rate-limited do we raise a typed
   `RateLimited`, which the entry points turn into an **honest** message: "⏳ hit today's usage limit, try again
   in ~X min (free-tier quota, not a bug)" — parsed from Groq's own retry hint.

Also trimmed tokens: `agent_max_steps` 6→4, wiki payload 800→500 chars.

### Verified
Live (with 70B exhausted): the assistant **responds** on the 8B fallback instead of erroring. Simulated
both-models-limited: returns the honest "try again in ~5m0s" message. Retrieval unaffected (embeddings don't use
the LLM).

### Gotchas / lessons
- **Token-hungry graphs meet free tiers.** Multi-call RAG (grade + route + fuse + agent loops) burns quota fast.
  Push every non-user-facing call to the cheapest model that can do it.
- **Per-model budgets are a feature.** Splitting work across `70b` and `8b-instant` doubles effective headroom
  and gives a natural fallback tier.
- **Fail honestly.** An opaque "Sorry" made a *quota* problem look like a broken app. Parsing the provider's
  retry hint and saying "it's a daily limit, back in X min" is the difference between "broken" and "busy".
- **Degraded ≠ dead.** Falling back to a weaker model keeps the product usable during exhaustion; note the
  quality dip is expected.

---

## Follow-up: the Meeting-Intelligence persona (grounded + interactive)

**Files:** `agentic_rag._AGENT_SYSTEM`, `adaptive_rag` (`_route_query` + `_ANSWER_SYSTEM`/`_DIRECT_SYSTEM`),
`vector_store.index_meeting`/`_meta_docs`, `config.py` (temps).

### The idea
The assistant should feel like an **intelligent teammate with perfect recall** of the meetings: answer from
meeting knowledge first (priority: transcripts → summaries → action items → conversation → Wikipedia →
general), be **action-item aware**, **proactive** with suggestions, use Wikipedia only when meetings fall short
(and say *why* + offer the result as a download), and frame unavailable actions as *"once <Google tool> is
connected I'll do X; for now here's the closest alternative"* — never faking. Crucially, it's **grounded but
interactive**: never invent meeting *facts*, but converse warmly and suggest freely (not a rigid lookup tool).

### The two fixes that made it real (beyond prompt wording)
1. **Routing** — action requests must reach the `agentic` branch (that's where the future-tools handling lives).
   The router was on the cheap 8B model and misrouted "schedule a follow-up Friday about the outage" to
   `semantic_all` (the topic words pulled it there), so it gave a flat "not in the meetings" instead of the
   future-tools framing. Fix: route on the **strong model with 8B fallback** (`resilient_invoke`) + explicit
   action examples ("the action verb wins over the topic"). Now it routes `agentic` and frames correctly.
2. **Knowledge sources** — retrieval indexed only transcript turns, so "answer from summaries / action items"
   had no data. `_meta_docs` now also indexes each meeting's **summary**, **action items** (with assignee +
   open/done), and **topics** as their own `kind`-tagged documents (15 → 65 docs on the seed set). So
   action-item questions answer from actual action-item records, not just transcript guesses.

### Balance: grounded ≠ robotic
The "never hallucinate" rule is scoped to **meeting facts** (decisions, dates, participants, action items). The
prompt explicitly tells the model to otherwise be warm, conversational, and proactive — so a greeting gets a
friendly, helpful reply, not "that's not in the meetings." Temperature stays 0.1–0.3 for consistency; the human
tone comes from the persona, not from cranking temperature.

### Gotchas
- **Cheap models misroute.** Moving routing to 8B saved little (routing is one small call) but hurt accuracy on
  action-vs-topic classification. Lesson: spend the strong model on the *decision* that changes behavior; save
  8B for grading/rephrasing where accuracy matters less.
- **Meta docs have no timestamp.** Summary/action-item docs use `start_time=0.0` (Chroma metadata can't be
  None), so their citation chips show the meeting title without a deep-link — expected.

---

## Follow-up: streaming, reliable downloads, and cutting Groq tokens

**Files:** `assistant.py` (`/stream`), `adaptive_rag.py` (`answer_stream`, action pre-check, 8B router),
`agentic_rag.py` (condensed prompt, download safety-net), `AssistantChat.tsx` + `api.ts` (SSE client).

### Streaming (SSE), token-neutral
`POST /api/assistant/stream` returns `text/event-stream`; `answer_stream` yields `data: {"token": …}` events then
a final `data: {"meta": …}` and `data: [DONE]`. It streams the **existing** generation call — no extra LLM call:
retrieval/direct branches stream `ChatGroq.stream()`; the agent runs tools first then its answer streams (chunked,
since tools finish before the final text). Frontend consumes the stream with a `fetch` + `ReadableStream` reader
and appends tokens to the live bubble; metadata (route/citations/steps/artifact/offers) applies at the end.

### Reliable downloads (the transcript bug)
The model used to refuse to save ("I'm a text AI, I can't save files") and dump walls of text. Fixed two ways:
(1) a blunt prompt rule — *you CAN save via create_document; never say you can't, never tell the user to
copy-paste*; (2) a deterministic safety-net in `agentic_rag.run` — if `wikipedia` ran but no file was produced,
auto-append a "📄 Download the findings" offer. So a download choice always appears after a lookup.

### Where the Groq tokens went — and how we cut them
**What consumed them (per assistant turn):**
- The **system prompt is paid on *every* LLM call** — and the agent loop makes up to `agent_max_steps` calls,
  each re-sending the (long) prompt + accumulating message history + tool outputs. This is the dominant cost.
- The **router** was a call on the **70B** model every turn.
- Self-RAG's per-meeting chat runs retrieve→grade→generate→grade (~3–4 calls).

**How we cut it:**
| Lever | Before | After |
|---|---|---|
| Agent system prompt | ~2.9k chars (~720 tok) ×≤4 calls | condensed ~2.0k chars (~510 tok) ×≤4 calls |
| Routing | 70B call every turn | **regex pre-check → 0 calls for actions**; else cheap **8B** |
| Agent tool cycles | `agent_max_steps` 6 | **4** |
| History in prompts | last 6 turns (agent) | **last 4** (agent), last 2 (router) |
| Wiki payload | 800 chars | 500 |
| Streaming | — | reuses the existing call (**no extra**) |

Net per turn: a greeting/meeting-Q keeps 2 calls but the router is now 8B not 70B; an **action turn drops the
whole 70B routing call** (regex) and runs a shorter agent prompt over fewer, shorter steps. Total tokens go
**down**, which is what kept the daily-cap pain from returning.

### Gotcha: stale HMR during multi-file edits
Streaming appeared "stuck on thinking…" in the browser while the raw SSE stream (tested via `fetch` in the
console) worked perfectly — the Next.js dev server had a **stale bundle** across the several files I changed at
once. `rm -rf .next` + restarting `next dev` fixed it. Lesson: when a feature spans api + component + types and
"the code is obviously right," suspect the dev bundle before the code.

---

## Follow-up: cross-session memory (the assistant remembers past conversations) ✅ Phase 3

**Files:** `models.py` (`ChatTurn`), `vector_store.py` (`user_memory` collection + `add_memory`/`search_memory`/
`reindex_memory`), NEW `services/rag/memory.py` (`save_turn`/`recall`), `adaptive_rag.py` (thread `user_id`,
`_memory_block`, fold into every branch), `agentic_rag.py` (`memory=` kwarg), `assistant.py` (pass `user_id`,
persist each turn).

### The idea
History in the request only survives one browser session. **Cross-session memory** lets the assistant recall
things from *previous* sessions ("what did we decide about the launch last week?", "my cat's name is Mochi").
It's retrieval, same as the meetings — just over a different corpus (the user's own past turns), scoped per user.

### How it works
1. **Persist** — after every assistant turn, `memory.save_turn(user_id, question, answer)` writes two `chat_turns`
   rows (user + assistant) to **Postgres** (durable) and indexes each into a Chroma collection.
2. **Index** — a **separate** `user_memory` collection (not `meeting_segments`), every doc tagged `user_id`.
   Separation matters: memory must never leak into meeting retrieval, and meeting chunks must never pollute
   recall. Doc id `mem-{turn_id}` → idempotent upsert.
3. **Recall** — on each new message, `_memory_block(state)` runs `search_memory(user_id, question, k=4)`
   (filtered by `user_id`) and, if there are hits, prepends a labelled block to the system prompt of *whatever*
   branch fired (direct, retrieval, or agent). Local embeddings → **recall costs zero Groq tokens**.
4. **Durability** — Chroma is ephemeral on Render, so on startup `ensure_indexed` calls `reindex_memory` to
   rebuild the memory index from `chat_turns` (Postgres is the source of truth), exactly like meetings.

### Design principles
- **Same seam as meetings** (DIP): the graphs call `memory.recall`/`vs.search_memory`, never Chroma directly.
- **Best-effort everywhere**: save and recall are wrapped so memory can never break an answer (or CRUD).
- **Per-user isolation by metadata filter** — the one thing that must not regress; every read passes `user_id`.
- **Grounding still holds**: the memory block is explicitly labelled "not meeting FACTS" so the model uses it
  for continuity/preferences, not as a source of invented meeting decisions.

### Verify
Seed a fact in one "session" (`save_turn`), then in a fresh call with **no history** ask about it — the answer
recalls it. Tested: "my cat is Mochi" → later "what is my cat called?" → *"Your cat's name is Mochi."* (route
`no_retrieval`, zero history, memory-only).

### Gotchas
- Recall runs **before** the current turn is persisted, so the model never "recalls" the question it's answering.
- Only fires when a `user_id` exists (i.e. logged in). Un-authed / auth-disabled → memory is silently skipped.
- Keep memory in its own collection — reusing `meeting_segments` would need `kind` filtering on every meeting
  query and risk cross-contamination; a second collection is simpler and safer.

---

## Problems & Gotchas log

Real issues hit while building, why they happened, and how we resolved them — kept for the
teaching pass.

### Already hit
- **Gmail plain password won't work as a tool.** Google disabled password/"less secure app"
  login in 2022. SMTP needs a 16-char **App Password** (requires 2FA); **Google Calendar /
  Drive / Gmail-API are OAuth-only** with *no* password path at all. → We chose OAuth-free
  tools for the agent this pass (custom retriever + Wikipedia + export-to-text). Calendar/email
  are documented as a future extension.
- **`sentence-transformers` drags in torch (~hundreds of MB).** Heavy install, slow cold
  starts, too big for Render's 512 MB free tier. → Switched to **fastembed** (ONNX runtime, no
  torch) via `FastEmbedEmbeddings`. Local, no key, small. `get_embeddings()` is the single swap
  point if we later want a hosted API.
- **`groq` got downgraded 1.5.0 → 0.37.1** when `langchain-groq` was installed (its pin caps
  groq lower). The existing `groq_service.py` still works (the `client.chat.completions.create`
  API is unchanged), verified by booting the app. Flagged so we re-test the live Groq path in
  Phase B.
- **Chunks were too coarse** — at `chunk_size=900` we got only 2 chunks/meeting, so citations
  pointed at the chunk's start (e.g. `@0s`) even when the answer was at `@56s`. → Lowered to
  `500` (≈3 chunks/meeting, 15 total) for precise citations. Also removed a dead
  `rag_chunk_overlap` config knob (unused by turn-grouping) to avoid misleading config.
- **`langchain-community` prints a deprecation warning** (it's being sunset; `FastEmbedEmbeddings`
  lives there). Harmless for now; if it's ever removed we move to a standalone fastembed
  integration package — again only `get_embeddings()` changes.
- **Chroma telemetry noise** — disabled via `ChromaSettings(anonymized_telemetry=False)` in
  `get_store()`.
- **LangGraph conditional edges can't mutate state** (Phase B). A conditional edge is a pure
  function `state → next_node_name`; it can't bump a counter. So the retry counter is incremented
  inside the `transform_query` *node*, and the edge only reads it. Keeping a single increment site
  makes the loop bound obvious.
- **Graders must fail open** (Phase B). If a yes/no grader errors or returns garbage, defaulting to
  "no" could loop forever or wrongly refuse. We default to "keep the docs / accept the answer" so a
  flaky grader degrades to naive RAG instead of breaking the chat.
- **`groq` 0.37.1 live path re-verified** (Phase B): Self-RAG calls Groq via `ChatGroq` and the
  existing `groq_service` still works — the earlier version downgrade caused no issues.
- **Routing needs to know *which* meeting** (Phase D). `single_meeting` is useless if the router
  can't identify the target. Fix: pass the meeting list (id + title) into the classifier prompt and
  have it return the `meeting_id`; if it can't, downgrade to `semantic_all`. Lesson: a router is
  only as good as the context you give it to decide with.
- **Making the agent branch additive** (Phase D). The router's `run_agent` node imports the
  (not-yet-existing) `agentic_rag` module inside a `try/except` and falls back. So Phase E adds the
  agent by creating one file — no edits to the router. This is the Open/Closed principle paying off.
- **Next.js 16 build check** (Phase D). `tsc` type-checks but doesn't validate App-Router page
  wiring; `next build` does. Ran it to confirm the new `/assistant` route compiles (it did) — worth
  doing whenever you add a route in this Next 16 project.
- **Groq `tool_use_failed` 400** (Phase E). llama-3.3-70b occasionally emits a malformed tool call
  (`<function=name{...}>` instead of JSON) and Groq rejects the whole request with a 400. Fix: catch
  it in the agent node, retry once (sampling may fix it), then fall back to a **no-tools** answer so
  the agent still responds from whatever it already gathered. Lesson: open-weights tool-calling is
  not 100% reliable — the agent loop must tolerate a bad tool turn.
- **The agent guessed wrong meeting ids** (Phase E). Asked to export "the Weekly Engineering Sync",
  it *searched* for that title and picked ids 5 then 23 (wrong / nonexistent) because a title isn't
  in the transcript text, so semantic search doesn't reliably map title→id. Fix: put the actual
  `id: title` list in the agent's system prompt. Lesson: don't make an agent *derive* facts you
  already have — give them to it.
- **Getting structured data out of string-returning tools** (Phase E). Tools return strings to the
  LLM, but the UI needs citations/artifact objects. Solution: define the tools as **closures inside
  `run()`** that append to local lists, so each call records structured output as a side effect
  while still returning a string to the model.

### Anticipated (watch for these in later phases)
- Extra LLM calls (grading, routing, multi-query, agent steps) eat Groq tokens/latency → keep
  retrieved context to `top_k`, cap Self-RAG retries and agent steps via config.
- Self-RAG could loop on empty retrieval → `self_rag_max_retries` cap + a graceful "not covered".
- The agent could loop calling tools → `agent_max_steps` cap.
- Hallucinated citations → the answer must only cite chunks actually retrieved.
- Chroma dir is ephemeral on redeploy → startup `ensure_indexed()` rebuilds it.
