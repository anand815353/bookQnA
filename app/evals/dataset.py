import json
from pathlib import Path
from typing import Any

from app.evals.models import EvalCase, normalize_test_type


def _normalize_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def _normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, list):
        items = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def _normalize_expected_pages(value: Any) -> list[int]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    pages: set[int] = set()

    for item in raw_items:
        if item is None:
            continue

        if isinstance(item, int):
            pages.add(item)
            continue

        if isinstance(item, str) and item.strip().isdigit():
            pages.add(int(item.strip()))
            continue

        if isinstance(item, dict):
            start = item.get("start") or item.get("page_start")
            end = item.get("end") or item.get("page_end") or start
            if start is None:
                continue
            try:
                start_i = int(start)
                end_i = int(end)
            except (TypeError, ValueError):
                continue
            for page in range(min(start_i, end_i), max(start_i, end_i) + 1):
                pages.add(page)
            continue

        if isinstance(item, (tuple, list)) and len(item) == 2:
            try:
                start_i = int(item[0])
                end_i = int(item[1])
            except (TypeError, ValueError):
                continue
            for page in range(min(start_i, end_i), max(start_i, end_i) + 1):
                pages.add(page)

    return sorted(pages)


def _load_rows(dataset_path: Path) -> list[dict[str, Any]]:
    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(json.loads(stripped))
        return rows

    if suffix == ".json":
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            cases = payload.get("cases")
            if isinstance(cases, list):
                return cases
        raise ValueError("JSON dataset must be a list or an object with a 'cases' list.")

    raise ValueError(
        f"Unsupported dataset format '{dataset_path.suffix}'. Use .json or .jsonl."
    )


def load_eval_cases(dataset_path: str | Path) -> list[EvalCase]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")

    rows = _load_rows(path)
    cases: list[EvalCase] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Eval row #{idx} is not an object: {row!r}")

        question = str(row.get("question", "")).strip()
        if not question:
            raise ValueError(f"Eval row #{idx} is missing a non-empty 'question'.")

        case_id = str(
            row.get("case_id")
            or row.get("id")
            or f"case_{idx:03d}"
        ).strip()
        expected_book = row.get("expected_book")
        if isinstance(expected_book, list):
            expected_book = str(expected_book[0]).strip() if expected_book else None
        elif expected_book is not None:
            expected_book = str(expected_book).strip() or None

        expected_chapter = row.get("expected_chapter")
        if expected_chapter is not None:
            expected_chapter = str(expected_chapter).strip() or None

        manual_score = row.get("citation_usefulness_score_manual")
        if manual_score is None:
            manual_score = row.get("citation_usefulness_score")
        if manual_score is not None:
            try:
                manual_score = float(manual_score)
            except (TypeError, ValueError):
                manual_score = None

        case = EvalCase(
            case_id=case_id,
            question=question,
            expected_book=expected_book,
            expected_pages=_normalize_expected_pages(row.get("expected_pages")),
            expected_chapter=expected_chapter,
            expected_answer_keywords=_normalize_str_list(row.get("expected_answer_keywords")),
            answerable=_normalize_bool(row.get("answerable"), default=True),
            test_type=normalize_test_type(row.get("test_type")),
            query_book_ids=_normalize_str_list(row.get("query_book_ids")) or None,
            citation_usefulness_score_manual=manual_score,
            notes=str(row.get("notes", "")).strip() or None,
            metadata={
                key: value
                for key, value in row.items()
                if key
                not in {
                    "case_id",
                    "id",
                    "question",
                    "expected_book",
                    "expected_pages",
                    "expected_chapter",
                    "expected_answer_keywords",
                    "answerable",
                    "test_type",
                    "query_book_ids",
                    "citation_usefulness_score_manual",
                    "citation_usefulness_score",
                    "notes",
                }
            },
        )
        cases.append(case)

    return cases
