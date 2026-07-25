"""FastAPI application entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db
from .routers import action_items, assistant, auth, export, meetings, search
from .seed import seed_if_empty


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_if_empty()
    _ensure_rag_indexed()
    yield


def _ensure_rag_indexed() -> None:
    """Build the vector index on startup if empty (best-effort — never blocks boot)."""
    try:
        from sqlmodel import Session

        from .database import engine
        from .services.rag import vector_store

        with Session(engine) as session:
            vector_store.ensure_indexed(session)
    except Exception:  # noqa: BLE001
        pass


app = FastAPI(
    title="Fireflies Clone API",
    description="Meeting notes & transcription platform — meetings, transcripts, AI summaries.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meetings.router)
app.include_router(action_items.router)
app.include_router(search.router)
app.include_router(export.router)
app.include_router(assistant.router)
app.include_router(auth.router)


@app.get("/api/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/", tags=["meta"])
def root():
    return {"name": "Fireflies Clone API", "docs": "/docs", "health": "/api/health"}
