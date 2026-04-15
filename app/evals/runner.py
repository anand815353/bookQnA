from collections import defaultdict
from typing import Any, Callable

from app.evals.metrics import (
    abstention_correctness,
    keyword_coverage,
    retrieval_metrics_for_case,
)
from app.evals.models import EvalCase

RetrievalFn = Callable[[str, int, list[str] | None], list[dict[str, Any]]]
AnswerFn = Callable[[str, int, list[str] | None], dict[str, Any]]

RETRIEVAL_METRIC_KEYS = (
    "correct_book_hit",
    "correct_page_hit",
    "chapter_hit",
    "top_k_evidence_hit",
)
ANSWER_METRIC_KEYS = (
    "correct_book_hit",
    "correct_page_hit",
    "chapter_hit",
    "top_k_evidence_hit",
    "answer_keyword_coverage",
    "abstention_correctness",
    "citation_usefulness_score_manual",
)


def default_retrieval_adapter(question: str, top_k: int, book_ids: list[str] | None) -> list[dict[str, Any]]:
    from app.services.retrieval import search_parent_evidence

    parent_evidence = search_parent_evidence(
        question=question,
        top_k=top_k,
        book_ids=book_ids,
    )
    rows: list[dict[str, Any]] = []
    for item in parent_evidence:
        rows.append(
            {
                "book_id": item.get("book_id"),
                "book_title": item.get("book_title"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "chapter_title": item.get("chapter_title"),
                "subchapter_title": item.get("subchapter_title"),
                "section_title": item.get("section_title"),
                "confidence": item.get("confidence"),
            }
        )
    return rows


def default_answer_adapter(question: str, top_k: int, book_ids: list[str] | None) -> dict[str, Any]:
    from app.services.qa import answer_question

    return answer_question(question=question, top_k=top_k, book_ids=book_ids)


def _summarize_metrics(
    results: list[dict[str, Any]],
    *,
    metric_keys: tuple[str, ...],
) -> dict[str, Any]:
    summary = {
        "cases_total": len(results),
        "metrics": {},
        "by_test_type": {},
    }

    for metric in metric_keys:
        values = [
            row["metrics"].get(metric)
            for row in results
            if row["metrics"].get(metric) is not None
        ]
        summary["metrics"][metric] = {
            "value": round(sum(values) / len(values), 4) if values else None,
            "count": len(values),
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["test_type"]].append(row)

    for test_type, rows in grouped.items():
        by_type = {"cases_total": len(rows), "metrics": {}}
        for metric in metric_keys:
            values = [
                row["metrics"].get(metric)
                for row in rows
                if row["metrics"].get(metric) is not None
            ]
            by_type["metrics"][metric] = {
                "value": round(sum(values) / len(values), 4) if values else None,
                "count": len(values),
            }
        summary["by_test_type"][test_type] = by_type

    return summary


def evaluate_retrieval(
    cases: list[EvalCase],
    *,
    top_k: int = 8,
    retrieval_fn: RetrievalFn | None = None,
) -> dict[str, Any]:
    retrieval = retrieval_fn or default_retrieval_adapter
    results: list[dict[str, Any]] = []

    for case in cases:
        evidence = retrieval(case.question, top_k, case.query_book_ids)
        metrics = retrieval_metrics_for_case(case, evidence)
        results.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "test_type": case.test_type,
                "answerable": case.answerable,
                "expected_book": case.expected_book,
                "expected_pages": case.expected_pages,
                "expected_chapter": case.expected_chapter,
                "metrics": metrics,
                "evidence_count": len(evidence),
                "evidence": evidence[:top_k],
            }
        )

    return {
        "summary": _summarize_metrics(results, metric_keys=RETRIEVAL_METRIC_KEYS),
        "results": results,
    }


def evaluate_answers(
    cases: list[EvalCase],
    *,
    top_k: int = 8,
    answer_fn: AnswerFn | None = None,
) -> dict[str, Any]:
    answer = answer_fn or default_answer_adapter
    results: list[dict[str, Any]] = []

    for case in cases:
        payload = answer(case.question, top_k, case.query_book_ids) or {}
        answer_text = str(payload.get("answer", "") or "")
        citations = payload.get("citations", []) or []
        citation_rows = [dict(item) for item in citations if isinstance(item, dict)]

        metrics = retrieval_metrics_for_case(case, citation_rows)
        metrics["answer_keyword_coverage"] = keyword_coverage(
            answer_text, case.expected_answer_keywords
        )
        metrics["abstention_correctness"] = abstention_correctness(
            answer_text,
            answerable=case.answerable,
        )
        metrics["citation_usefulness_score_manual"] = case.citation_usefulness_score_manual

        results.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "test_type": case.test_type,
                "answerable": case.answerable,
                "expected_book": case.expected_book,
                "expected_pages": case.expected_pages,
                "expected_chapter": case.expected_chapter,
                "expected_answer_keywords": case.expected_answer_keywords,
                "metrics": metrics,
                "grounded": bool(payload.get("grounded")),
                "answer": answer_text,
                "citation_count": len(citation_rows),
                "citations": citation_rows[:top_k],
            }
        )

    return {
        "summary": _summarize_metrics(results, metric_keys=ANSWER_METRIC_KEYS),
        "results": results,
    }
