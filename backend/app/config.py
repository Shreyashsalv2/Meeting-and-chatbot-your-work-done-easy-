"""Application configuration.

Settings are read from environment variables (and a local ``.env`` file when
present). Secrets such as ``GROQ_API_KEY`` live ONLY here / in the environment –
never hard-coded and never committed.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    database_url: str = "sqlite:///./fireflies.db"

    # --- Groq (LLM) ---
    # Empty by default so the app runs fully offline with a mock fallback.
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    # Cheap/fast model with a SEPARATE (larger) daily token budget. Used for the many
    # auxiliary calls (routing, grading, multi-query) and as the fallback when the main
    # model is rate-limited, so the assistant keeps working instead of going dark.
    groq_fast_model: str = "llama-3.1-8b-instant"

    # --- RAG (LangChain + LangGraph) ---
    # Local, no-key embeddings via fastembed (ONNX — no torch). This is the single
    # swap point: change ``embeddings_model`` / ``get_embeddings()`` to move to a
    # hosted embeddings API later without touching any RAG graph.
    embeddings_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: str = "./chroma"          # persistent vector store location
    # We chunk by grouping consecutive speaker turns up to this many chars (not a
    # blind character split) so each chunk keeps the timestamp of its first turn —
    # that's what powers click-to-seek citations. Overlap is therefore N/A.
    rag_chunk_size: int = 500             # chars per transcript chunk
    rag_top_k: int = 5                    # retrieved chunks per query
    self_rag_max_retries: int = 1         # query-rewrite retries in Self-RAG
    agent_max_steps: int = 4              # tool-call cap in Agentic RAG (lower = fewer tokens)

    # Task-based generation temperature. The assistant prioritizes consistency over
    # creativity (meeting intelligence must be deterministic + low-hallucination), so
    # all tiers stay in the 0.1–0.3 range.
    temp_factual: float = 0.15
    temp_actionable: float = 0.25
    temp_creative: float = 0.3

    # --- CORS / frontend ---
    # Production frontend origin (the Vercel URL). Localhost is always allowed.
    frontend_url: str = "http://localhost:3000"

    @property
    def cors_origins(self) -> list[str]:
        """Explicit allowlist: local dev origins plus the configured frontend."""
        origins = {
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        }
        if self.frontend_url:
            origins.add(self.frontend_url.rstrip("/"))
        return sorted(origins)

    # Allow any Vercel preview/production deployment without re-configuring.
    cors_origin_regex: str = r"https://.*\.vercel\.app"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
