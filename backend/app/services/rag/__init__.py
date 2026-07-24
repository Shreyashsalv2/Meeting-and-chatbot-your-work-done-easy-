"""RAG package: shared foundation + the four named RAG techniques.

Modules:
- ``vector_store``  — embeddings, Chroma store, indexing, retrieval (the shared seam)
- ``self_rag``      — RAG #1: Self-RAG on the per-meeting chat
- ``adaptive_rag``  — RAG #2: Adaptive router (the unified assistant's brain)
- ``fusion_rag``    — RAG #3: RAG-Fusion / multi-query semantic search
- ``agentic_rag``   — RAG #4: tool-calling agent (subgraph of the router)

Everything depends ONLY on the seams exposed by ``vector_store`` (get_embeddings /
get_llm / get_store / retriever / ...), never on concrete Chroma/fastembed/Groq
classes — so swapping a provider is a one-function change (Dependency Inversion).
"""
