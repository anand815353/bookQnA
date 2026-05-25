# app/db.py
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from app.models import Base
from app.services.lexical_index import ensure_lexical_fts_schema

DATABASE_URL = "sqlite:///./data/app.db"

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

BOOKS_COLUMN_MIGRATIONS = [
    ("total_sections", "INTEGER"),
    ("total_chunks", "INTEGER NOT NULL DEFAULT 0"),
    ("processed_chunks", "INTEGER NOT NULL DEFAULT 0"),
    ("failed_chunks", "INTEGER NOT NULL DEFAULT 0"),
    ("progress_percent", "INTEGER NOT NULL DEFAULT 0"),
    ("current_step", "VARCHAR"),
    ("last_processed_chunk_index", "INTEGER"),
    ("started_at", "DATETIME"),
    ("finished_at", "DATETIME"),
]

CHAT_SESSIONS_COLUMN_MIGRATIONS = [
    ("summary_text", "TEXT"),
]


def _ensure_books_columns() -> None:
    with engine.begin() as conn:
        table_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='books'")
        ).fetchone()
        if not table_exists:
            return

        existing_columns = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(books)")).fetchall()
        }

        for column_name, column_sql in BOOKS_COLUMN_MIGRATIONS:
            if column_name in existing_columns:
                continue
            conn.execute(text(f"ALTER TABLE books ADD COLUMN {column_name} {column_sql}"))


def _ensure_chat_sessions_columns() -> None:
    with engine.begin() as conn:
        table_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_sessions'")
        ).fetchone()
        if not table_exists:
            return

        existing_columns = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(chat_sessions)")).fetchall()
        }

        for column_name, column_sql in CHAT_SESSIONS_COLUMN_MIGRATIONS:
            if column_name in existing_columns:
                continue
            conn.execute(text(f"ALTER TABLE chat_sessions ADD COLUMN {column_name} {column_sql}"))


def init_db():
    Base.metadata.create_all(bind=engine)
    _ensure_books_columns()
    _ensure_chat_sessions_columns()
    ensure_lexical_fts_schema(engine)
