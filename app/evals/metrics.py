import re
from typing import Any

from app.evals.models import EvalCase

ABSTENTION_MARKERS = (
    "i do not know",
    "i don't know",
    "not enough context",
    "not in the context",
    "cannot answer",
    "insufficient information",
)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def book_matches(expected_book: str | None, evidence: dict[str, Any]) -> bool:
    if not expected_book:
        return False
    expected = _normalize_text(expected_book)
    if not expected:
        return False

    book_id = _normalize_text(evidence.get("book_id"))
    book_title = _normalize_text(evidence.get("book_title"))

    if expected == book_id or expected == book_title:
        return True
    if book_title and expected in book_title:
        return True
    return False


def chapter_matches(expected_chapter: str | None, evidence: dict[str, Any]) -> bool:
    if not expected_chapter:
        return False
    expected = _normalize_text(expected_chapter)
    if not expected:
        return False

    for field in ("chapter_title", "subchapter_title", "section_title"):
        value = _normalize_text(evidence.get(field))
        if value and (expected == value or expected in value):
            return True
    return False


def page_matches(expected_pages: list[int], evidence: dict[str, Any]) -> bool:
    if not expected_pages:
        return False

    try:
        start = int(evidence.get("page_start") or 0)
        end = int(evidence.get("page_end") or start)
    except (TypeError, ValueError):
        return False

    lo = min(start, end)
    hi = max(start, end)
    return any(lo <= page <= hi for page in expected_pages)


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float | None:
    if not expected_keywords:
        return None
    answer_text = _normalize_text(answer)
    if not answer_text:
        return 0.0

    keywords = [kw for kw in (_normalize_text(value) for value in expected_keywords) if kw]
    if not keywords:
        return None
    hits = sum(1 for kw in keywords if kw in answer_text)
    return hits / len(keywords)


def is_abstention_answer(answer: str) -> bool:
    normalized = _normalize_text(answer)
    if not normalized:
        return False
    return any(marker in normalized for marker in ABSTENTION_MARKERS)


def abstention_correctness(answer: str, *, answerable: bool) -> int:
    abstained = is_abstention_answer(answer)
    return int(abstained == (not answerable))


def retrieval_metrics_for_case(case: EvalCase, evidence_items: list[dict[str, Any]]) -> dict[str, float | int | None]:
    book_hit = None
    if case.expected_book:
        book_hit = int(any(book_matches(case.expected_book, item) for item in evidence_items))

    page_hit = None
    if case.expected_pages:
        page_hit = int(any(page_matches(case.expected_pages, item) for item in evidence_items))

    chapter_hit = None
    if case.expected_chapter:
        chapter_hit = int(any(chapter_matches(case.expected_chapter, item) for item in evidence_items))

    top_k_hit = None
    if case.expected_book or case.expected_pages or case.expected_chapter:
        filtered = evidence_items
        if case.expected_book:
            filtered = [item for item in filtered if book_matches(case.expected_book, item)]
        if case.expected_chapter:
            filtered = [item for item in filtered if chapter_matches(case.expected_chapter, item)]
        if case.expected_pages:
            filtered = [item for item in filtered if page_matches(case.expected_pages, item)]
        top_k_hit = int(bool(filtered))

    return {
        "correct_book_hit": book_hit,
        "correct_page_hit": page_hit,
        "chapter_hit": chapter_hit,
        "top_k_evidence_hit": top_k_hit,
    }
