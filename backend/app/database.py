"""Database engine, session management, and initialization."""
from collections.abc import Generator

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from .config import settings


def _normalized_db_url(url: str) -> str:
    """Managed Postgres providers (Neon, Supabase, Render…) hand out ``postgres://`` or
    ``postgresql://`` URLs, but SQLAlchemy needs an explicit driver — pin psycopg (v3)."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


DATABASE_URL = _normalized_db_url(settings.database_url)
_IS_SQLITE = DATABASE_URL.startswith("sqlite")

# ``check_same_thread`` is required for SQLite when used across FastAPI's
# threadpool. It is ignored by other databases.
connect_args = {"check_same_thread": False} if _IS_SQLITE else {}

# ``pool_pre_ping`` keeps pooled Postgres connections healthy (managed DBs drop idle
# connections); harmless for SQLite.
engine = create_engine(
    DATABASE_URL, echo=False, connect_args=connect_args, pool_pre_ping=not _IS_SQLITE
)


@event.listens_for(Engine, "connect")
def _enable_sqlite_fk(dbapi_connection, _connection_record):
    """Enforce foreign-key constraints on SQLite (off by default)."""
    # Only relevant for SQLite; guard so other drivers are untouched.
    if _IS_SQLITE:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def init_db() -> None:
    """Create all tables. Import models first so they register with metadata."""
    from . import models  # noqa: F401  (ensures models are registered)

    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    with Session(engine) as session:
        yield session
