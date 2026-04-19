from dataclasses import dataclass, field
from typing import Any

DEFAULT_TEST_TYPE = "concept_lookup"
KNOWN_TEST_TYPES = {
    "follow_up",
    "ambiguous_short",
    "concept_lookup",
    "chapter_summary",
    "locate_section_page",
    "compare_concepts",
    "exact_phrase_lookup",
    "abstention",
}


def normalize_test_type(value: str | None) -> str:
    if not value:
        return DEFAULT_TEST_TYPE
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or DEFAULT_TEST_TYPE


@dataclass(slots=True)
class EvalCase:
    case_id: str
    question: str
    expected_book: str | None = None
    expected_pages: list[int] = field(default_factory=list)
    expected_chapter: str | None = None
    expected_answer_keywords: list[str] = field(default_factory=list)
    answerable: bool = True
    test_type: str = DEFAULT_TEST_TYPE
    query_book_ids: list[str] | None = None
    recent_history: list[dict[str, str]] | None = None
    citation_usefulness_score_manual: float | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
