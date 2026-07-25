"""RAG #3 — RAG-Fusion (Multi-Query) for semantic search.

A single search query is a narrow lens: it only matches passages phrased like the
query. **RAG-Fusion** widens it: the LLM rewrites the query into several alternative
phrasings, we retrieve for *each*, then merge the ranked lists with **Reciprocal Rank
Fusion (RRF)** — passages that rank highly across *many* phrasings float to the top.
The result is semantic search that's robust to how the user happened to word things,
and it beats the app's old substring search (which misses synonyms entirely).

Graph shape::

    START → generate_queries → retrieve (one ranked list per sub-query) → fuse (RRF) → END

RRF score for a passage = Σ over sub-queries of 1 / (K + rank_in_that_list). No model
scores to calibrate, just ranks — simple and strong. Returns `SearchMatch`-shaped dicts
(field="transcript" + start_time) so the existing search UI and deep-linking just work.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from . import vector_store as vs

logger = logging.getLogger(__name__)

_RRF_K = 60          # standard RRF constant (dampens the weight of top ranks)
_N_ALT_QUERIES = 3   # alternative phrasings to generate (plus the original)


class FusionState(TypedDict, total=False):
    query: str
    top_n: int
    subqueries: list[str]
    ranked_lists: list[list[Document]]
    fused: list[Document]


def _generate_queries(state: FusionState) -> dict:
    q = state["query"]
    prompt = (
        f"Generate {_N_ALT_QUERIES} alternative search queries that rephrase the question "
        "below to retrieve relevant meeting-transcript passages (use synonyms and related "
        "terms). Return each query on its own line, with no numbering or extra text.\n\n"
        f"Question: {q}"
    )
    try:
        raw = vs.get_fast_llm(0.3).invoke([HumanMessage(content=prompt)]).content or ""
        alts = [ln.strip("-• ").strip() for ln in raw.splitlines() if ln.strip()]
    except Exception as exc:  # noqa: BLE001 - degrade to single-query semantic search
        logger.warning("multi-query generation failed: %s", exc)
        alts = []
    seen, subqueries = set(), []
    for cand in [q, *alts]:
        low = cand.lower()
        if cand and low not in seen:
            seen.add(low)
            subqueries.append(cand)
    return {"subqueries": subqueries[: _N_ALT_QUERIES + 1]}


def _retrieve(state: FusionState) -> dict:
    top_k = state.get("top_n", 8)
    ranked_lists = [vs.similarity_search(sq, k=top_k) for sq in state["subqueries"]]
    return {"ranked_lists": ranked_lists}


def _fuse(state: FusionState) -> dict:
    """Reciprocal Rank Fusion across the per-query ranked lists."""
    scores: dict = {}
    doc_by_key: dict = {}
    for ranked in state.get("ranked_lists", []):
        for rank, doc in enumerate(ranked):
            key = (doc.metadata.get("meeting_id"), doc.metadata.get("start_time"))
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            doc_by_key[key] = doc
    ordered = sorted(scores, key=lambda k: scores[k], reverse=True)
    fused = [doc_by_key[k] for k in ordered][: state.get("top_n", 8)]
    return {"fused": fused}


@lru_cache(maxsize=1)
def _get_graph():
    g = StateGraph(FusionState)
    g.add_node("generate_queries", _generate_queries)
    g.add_node("retrieve", _retrieve)
    g.add_node("fuse", _fuse)
    g.add_edge(START, "generate_queries")
    g.add_edge("generate_queries", "retrieve")
    g.add_edge("retrieve", "fuse")
    g.add_edge("fuse", END)
    return g.compile()


def _to_match(d: Document) -> dict:
    return {
        "meeting_id": d.metadata.get("meeting_id"),
        "meeting_title": d.metadata.get("meeting_title"),
        "field": "transcript",
        "snippet": d.page_content[:200],
        "start_time": d.metadata.get("start_time"),
    }


def search(query: str, top_n: int = 8) -> list[dict]:
    """Semantic search across all meetings. Returns SearchMatch-shaped dicts. Never raises."""
    q = (query or "").strip()
    if not q:
        return []
    fused: list[Document] = []
    try:
        if vs.llm_available():
            fused = _get_graph().invoke({"query": q, "top_n": top_n}).get("fused", [])
        else:
            fused = vs.similarity_search(q, k=top_n)  # no key → single-query semantic
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG-Fusion failed (%s); falling back to single-query.", exc)
        try:
            fused = vs.similarity_search(q, k=top_n)
        except Exception:  # noqa: BLE001
            fused = []
    return [_to_match(d) for d in fused]
