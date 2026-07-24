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

## 2. RAG #1 — Self-RAG (per-meeting chat)  ⏳ Phase B
_To be written when built._

## 3. RAG #3 — RAG-Fusion / Multi-Query (semantic search)  ⏳ Phase C
_To be written when built._

## 4. RAG #2 — Adaptive RAG (the unified assistant's router)  ⏳ Phase D
_To be written when built._

## 5. RAG #4 — Agentic RAG (tool-calling agent subgraph)  ⏳ Phase E
_To be written when built._

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

### Anticipated (watch for these in later phases)
- Extra LLM calls (grading, routing, multi-query, agent steps) eat Groq tokens/latency → keep
  retrieved context to `top_k`, cap Self-RAG retries and agent steps via config.
- Self-RAG could loop on empty retrieval → `self_rag_max_retries` cap + a graceful "not covered".
- The agent could loop calling tools → `agent_max_steps` cap.
- Hallucinated citations → the answer must only cite chunks actually retrieved.
- Chroma dir is ephemeral on redeploy → startup `ensure_indexed()` rebuilds it.
