# app/services/lexical_index.py
"""SQLite FTS5 lexical index for chunk bodies; Chroma remains source of truth for vectors/metadata."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Extended schema: UNINDEXED metadata for traceability; body is the only indexed column.
_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS lexical_chunks_fts USING fts5(
    chunk_id UNINDEXED,
    book_id UNINDEXED,
    page_start UNINDEXED,
    page_end UNINDEXED,
    section_id UNINDEXED,
    chapter_title UNINDEXED,
    body,
    tokenize = 'porter unicode61'
);
"""

_INSERT_SQL = text(
    """
    INSERT INTO lexical_chunks_fts(
        chunk_id, book_id, page_start, page_end, section_id, chapter_title, body
    ) VALUES (
        :chunk_id, :book_id, :page_start, :page_end, :section_id, :chapter_title, :body
    )
    """
)


def _fts_has_expected_columns(conn: Any) -> bool:
    try:
        conn.execute(
            text(
                "SELECT page_start, page_end, section_id, chapter_title, body "
                "FROM lexical_chunks_fts LIMIT 0"
            )
        )
        return True
    except Exception:
        return False


def ensure_lexical_fts_schema(engine: Engine) -> None:
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='lexical_chunks_fts'"
                )
            ).fetchone()
            if row and not _fts_has_expected_columns(conn):
                logger.warning(
                    "lexical_fts_schema_migrating dropping legacy lexical_chunks_fts "
                    "(re-run: python -m app.services.lexical_index rebuild)"
                )
                conn.execute(text("DROP TABLE IF EXISTS lexical_chunks_fts"))
                row = None
            if not row:
                conn.execute(text(_FTS_DDL))
    except Exception:
        logger.exception("lexical_fts_schema_failed")


def delete_book(engine: Engine, book_id: str) -> None:
    """Remove all FTS rows for a book (alias: clear_book_index)."""
    clear_book_index(engine, book_id)


def clear_book_index(engine: Engine, book_id: str) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM lexical_chunks_fts WHERE book_id = :bid"),
                {"bid": book_id},
            )
    except Exception:
        logger.exception("lexical_fts_delete_book_failed book_id=%s", book_id)


def _normalize_body(body: str) -> str:
    return (body or "").replace("\x00", " ")


def index_chunk_rows(engine: Engine, book_id: str, rows: list[dict[str, Any]]) -> int:
    """Replace FTS rows for one book. Each row dict: chunk_id, book_id, page_start, page_end, section_id, chapter_title, body."""
    if not rows:
        logger.info("lexical_fts_index_skipped_empty book_id=%s", book_id)
        return 0
    try:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM lexical_chunks_fts WHERE book_id = :bid"),
                {"bid": book_id},
            )
            payload = []
            for r in rows:
                bid = str(r.get("book_id", book_id))
                page_start = int(r.get("page_start", 0) or 0)
                raw_pe = r.get("page_end")
                if raw_pe is None or str(raw_pe).strip() == "":
                    page_end = page_start
                else:
                    page_end = int(raw_pe)
                if page_end <= 0:
                    page_end = page_start
                payload.append(
                    {
                        "chunk_id": str(r["chunk_id"]),
                        "book_id": bid,
                        "page_start": page_start,
                        "page_end": page_end,
                        "section_id": str(r.get("section_id", "") or ""),
                        "chapter_title": str(r.get("chapter_title", "") or ""),
                        "body": _normalize_body(str(r.get("body", "") or "")),
                    }
                )
            conn.execute(_INSERT_SQL, payload)
        logger.info(
            "lexical_fts_index_completed book_id=%s chunks_indexed=%s",
            book_id,
            len(rows),
        )
        return len(rows)
    except Exception:
        logger.exception("lexical_fts_index_chunks_failed book_id=%s", book_id)
        return 0


def index_chunks(engine: Engine, rows: list[tuple[str, str, str]]) -> None:
    """Bulk insert/replace: legacy (chunk_id, book_id, body) tuples; metadata columns default empty/zero."""
    if not rows:
        return
    book_id = rows[0][1]
    dict_rows = [
        {
            "chunk_id": r[0],
            "book_id": r[1],
            "page_start": 0,
            "page_end": 0,
            "section_id": "",
            "chapter_title": "",
            "body": r[2],
        }
        for r in rows
    ]
    index_chunk_rows(engine, book_id, dict_rows)


def _iter_chroma_chunks_for_book(collection: Any, book_id: str, *, batch_size: int = 256):
    offset = 0
    while True:
        batch = collection.get(
            where={"book_id": book_id},
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        yield ids, batch.get("documents") or [], batch.get("metadatas") or []
        offset += len(ids)


def index_chunks_for_book(engine: Engine, book_id: str, vectorstore: Any) -> int:
    """Rebuild SQLite FTS for one book from Chroma chunk rows."""
    collection = vectorstore._collection
    total = 0
    all_rows: list[dict[str, Any]] = []
    for raw_ids, documents, metadatas in _iter_chroma_chunks_for_book(collection, book_id):
        for i, cid in enumerate(raw_ids):
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            text_body = documents[i] if i < len(documents) else ""
            contextual = str(meta.get("contextual_text", "") or "").strip()
            body = f"{contextual}\n{text_body}" if contextual else (text_body or "")
            page_start = int(meta.get("page_start", 0) or 0)
            page_end = int(meta.get("page_end", page_start) or page_start)
            all_rows.append(
                {
                    "chunk_id": str(cid),
                    "book_id": book_id,
                    "page_start": page_start,
                    "page_end": page_end,
                    "section_id": str(meta.get("section_id", "") or ""),
                    "chapter_title": str(meta.get("chapter_title", "") or ""),
                    "body": body,
                }
            )
            total += 1
    indexed = index_chunk_rows(engine, book_id, all_rows)
    logger.info(
        "lexical_fts_reindex_from_chroma book_id=%s chroma_chunks_seen=%s fts_rows_written=%s",
        book_id,
        total,
        indexed,
    )
    return indexed


def rebuild_all_books_from_chroma(engine: Engine, vectorstore: Any) -> dict[str, int]:
    """Rebuild FTS for all books marked indexed in SQLite."""
    out: dict[str, int] = {}
    try:
        with engine.connect() as conn:
            book_ids = [
                str(r[0])
                for r in conn.execute(
                    text("SELECT id FROM books WHERE status = 'indexed'")
                ).fetchall()
            ]
    except Exception:
        logger.exception("lexical_fts_rebuild_list_books_failed")
        return out
    if not book_ids:
        logger.warning("lexical_fts_rebuild_no_indexed_books_in_sql")
        return out
    logger.info("lexical_fts_rebuild_all_started book_count=%s", len(book_ids))
    for bid in book_ids:
        out[bid] = index_chunks_for_book(engine, bid, vectorstore)
    logger.info(
        "lexical_fts_rebuild_all_completed books=%s total_chunks_indexed=%s",
        len(out),
        sum(out.values()),
    )
    return out


def count_chunks(engine: Engine, book_id: str | None = None) -> int:
    try:
        with engine.connect() as conn:
            if book_id:
                n = conn.execute(
                    text("SELECT COUNT(*) FROM lexical_chunks_fts WHERE book_id = :bid"),
                    {"bid": book_id},
                ).scalar()
            else:
                n = conn.execute(text("SELECT COUNT(*) FROM lexical_chunks_fts")).scalar()
            return int(n or 0)
    except Exception:
        logger.exception("lexical_fts_count_failed book_id=%s", book_id)
        return 0


def _fts_match_expression(query: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", (query or "").lower())
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return " OR ".join(tokens)


def search_chunk_ids(
    engine: Engine,
    query: str,
    book_ids: list[str] | None,
    *,
    limit: int,
) -> list[str]:
    match_expr = _fts_match_expression(query)
    if not match_expr:
        return []
    try:
        with engine.connect() as conn:
            if book_ids:
                placeholders = ", ".join(f":b{i}" for i in range(len(book_ids)))
                params: dict[str, object] = {f"b{i}": bid for i, bid in enumerate(book_ids)}
                params["lim"] = max(1, int(limit))
                params["match"] = match_expr
                sql = f"""
                    SELECT chunk_id
                    FROM lexical_chunks_fts
                    WHERE book_id IN ({placeholders})
                      AND lexical_chunks_fts MATCH :match
                    ORDER BY bm25(lexical_chunks_fts)
                    LIMIT :lim
                """
                result = conn.execute(text(sql), params)
            else:
                result = conn.execute(
                    text(
                        """
                        SELECT chunk_id
                        FROM lexical_chunks_fts
                        WHERE lexical_chunks_fts MATCH :match
                        ORDER BY bm25(lexical_chunks_fts)
                        LIMIT :lim
                        """
                    ),
                    {"match": match_expr, "lim": max(1, int(limit))},
                )
            hit_ids = [str(row[0]) for row in result.fetchall() if row[0]]
            logger.debug(
                "lexical_fts_search_done match_expr=%s book_filter=%s hit_count=%s limit=%s",
                match_expr,
                bool(book_ids),
                len(hit_ids),
                limit,
            )
            return hit_ids
    except Exception:
        logger.exception("lexical_fts_search_failed")
        return []


def _cli_rebuild(book_id: str | None) -> int:
    from app.db import engine as app_engine, init_db
    from app.services.retrieval import get_vectorstore

    init_db()
    vs = get_vectorstore()
    if book_id:
        n = index_chunks_for_book(app_engine, book_id, vs)
        print(f"Indexed {n} chunks for book_id={book_id!r}")
        return n
    stats = rebuild_all_books_from_chroma(app_engine, vs)
    total = sum(stats.values())
    print(f"Rebuilt FTS for {len(stats)} book(s), {total} chunk rows.")
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SQLite FTS lexical index utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    p_rebuild = sub.add_parser("rebuild", help="Rebuild FTS rows from Chroma chunk store")
    p_rebuild.add_argument("--book-id", default=None, help="Limit rebuild to one book id")
    args = parser.parse_args(argv)
    if args.command == "rebuild":
        _cli_rebuild(args.book_id)
    else:
        parser.error("unknown command")


if __name__ == "__main__":
    main(sys.argv[1:])
