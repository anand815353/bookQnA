# app/services/book_metadata.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from app.models import Book, BookSection

logger = logging.getLogger(__name__)

MAX_HEADINGS = 24


def load_planner_book_hints(db: "Session", book_ids: list[str] | None) -> dict[str, object]:
    """
    Returns kwargs for plan_retrieval_query: book_title, toc_headings, chapter_titles.
    Empty dict if nothing to add.
    """
    if not book_ids:
        return {}
    try:
        books = db.query(Book).filter(Book.id.in_(book_ids)).all()
        if not books:
            return {}
        titles = [((b.title or "").strip()) for b in books if (b.title or "").strip()]
        book_title = "; ".join(titles)[:400] if titles else None

        sections = (
            db.query(BookSection)
            .filter(BookSection.book_id.in_(book_ids))
            .order_by(BookSection.page_start)
            .limit(400)
            .all()
        )
        toc_headings: list[str] = []
        chapter_titles: list[str] = []
        seen_toc: set[str] = set()
        seen_ch: set[str] = set()

        for s in sections:
            src = (s.source_of_structure or "").strip().lower()
            t = (s.title or "").strip()
            if t and src == "toc" and t not in seen_toc:
                toc_headings.append(t)
                seen_toc.add(t)
            ch = (s.chapter_title or "").strip()
            if ch and ch not in seen_ch:
                chapter_titles.append(ch)
                seen_ch.add(ch)
            if len(toc_headings) >= MAX_HEADINGS and len(chapter_titles) >= MAX_HEADINGS:
                break

        out: dict[str, object] = {}
        if book_title:
            out["book_title"] = book_title
        if toc_headings:
            out["toc_headings"] = toc_headings[:MAX_HEADINGS]
        if chapter_titles:
            out["chapter_titles"] = chapter_titles[:MAX_HEADINGS]
        return out
    except Exception:
        logger.exception("book_metadata_load_failed")
        return {}
