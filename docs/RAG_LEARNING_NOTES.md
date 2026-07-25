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
