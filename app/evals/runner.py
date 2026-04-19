import inspect
from collections import defaultdict
from typing import Any, Callable

from app.evals.metrics import (
    abstention_correctness,
    keyword_coverage,
    retrieval_metrics_for_case,
)
from app.evals.models import EvalCase, KNOWN_TEST_TYPES

RetrievalFn = Callable[..., list[dict[str, Any]]]
AnswerFn = Callable[..., dict[str, Any]]

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
PLANNER_FOCUS_TEST_TYPES = (
    "follow_up",
    "ambiguous_short",
    "exact_phrase_lookup",
    "chapter_summary",
    "concept_lookup",
    "abstention",
)
CRITICAL_REGRESSION_BUCKETS = {
    "exact_phrase_lookup",
    "abstention",
}


def _supports_parameter(func: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters


def _planner_kwargs_from_case(case: EvalCase | None) -> dict[str, Any]:
    if case is None:
        return {}

    metadata = case.metadata or {}
    return {
        "recent_history": case.recent_history,
        "book_title": metadata.get("planner_book_title"),
        "toc_headings": metadata.get("planner_toc_headings") or metadata.get("toc_headings"),
        "chapter_titles": metadata.get("planner_chapter_titles") or metadata.get("chapter_titles"),
    }


def default_retrieval_adapter(
    question: str,
    top_k: int,
    book_ids: list[str] | None,
    *,
    planner_enabled: bool = False,
    force_planner: bool = False,
    case: EvalCase | None = None,
) -> list[dict[str, Any]]:
    from app.services.query_planner import plan_retrieval_query, should_use_query_planner
    from app.services.retrieval import search_parent_evidence

    query_plan = None
    planner_decision = should_use_query_planner(
        question,
        recent_history=case.recent_history if case is not None else None,
        planner_enabled=planner_enabled,
        force_planner=force_planner,
    )
    if planner_decision.should_use:
        query_plan = plan_retrieval_query(
            question=question,
            planner_enabled=True,
            **_planner_kwargs_from_case(case),
        )

    parent_evidence = search_parent_evidence(
        question=question,
        top_k=top_k,
        book_ids=book_ids,
        query_plan=query_plan,
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


def default_answer_adapter(
    question: str,
    top_k: int,
    book_ids: list[str] | None,
    *,
    planner_enabled: bool = False,
    force_planner: bool = False,
    case: EvalCase | None = None,
) -> dict[str, Any]:
    from app.services.qa import answer_question

    recent_history = case.recent_history if case is not None else None
    return answer_question(
        question=question,
        top_k=top_k,
        book_ids=book_ids,
        recent_history=recent_history,
        enable_query_reformulation=bool(recent_history),
        enable_query_planner=planner_enabled,
        force_query_planner=force_planner,
    )


def _invoke_eval_fn(
    func: Callable[..., Any],
    *,
    case: EvalCase,
    top_k: int,
    planner_enabled: bool,
    force_planner: bool,
) -> Any:
    kwargs: dict[str, Any] = {}
    if _supports_parameter(func, "planner_enabled"):
        kwargs["planner_enabled"] = planner_enabled
    if _supports_parameter(func, "force_planner"):
        kwargs["force_planner"] = force_planner
    if _supports_parameter(func, "case"):
        kwargs["case"] = case
    return func(case.question, top_k, case.query_book_ids, **kwargs)


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

    for test_type in sorted(set(KNOWN_TEST_TYPES).union(grouped.keys())):
        rows = grouped.get(test_type, [])
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


def _compare_metric_payloads(
    baseline_metric: dict[str, Any] | None,
    planner_metric: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_value = baseline_metric.get("value") if baseline_metric else None
    planner_value = planner_metric.get("value") if planner_metric else None
    delta = None
    if baseline_value is not None and planner_value is not None:
        delta = round(planner_value - baseline_value, 4)

    return {
        "baseline": baseline_value,
        "planner": planner_value,
        "delta": delta,
        "baseline_count": baseline_metric.get("count", 0) if baseline_metric else 0,
        "planner_count": planner_metric.get("count", 0) if planner_metric else 0,
    }


def _compare_summaries(
    baseline_summary: dict[str, Any],
    planner_summary: dict[str, Any],
    *,
    metric_keys: tuple[str, ...],
) -> dict[str, Any]:
    comparison = {
        "cases_total": baseline_summary.get("cases_total", 0),
        "metrics": {},
        "by_test_type": {},
    }

    for metric in metric_keys:
        comparison["metrics"][metric] = _compare_metric_payloads(
            baseline_summary.get("metrics", {}).get(metric),
            planner_summary.get("metrics", {}).get(metric),
        )

    baseline_types = baseline_summary.get("by_test_type", {})
    planner_types = planner_summary.get("by_test_type", {})
    all_types = sorted(set(KNOWN_TEST_TYPES).union(baseline_types.keys(), planner_types.keys()))
    for test_type in all_types:
        baseline_payload = baseline_types.get(test_type, {})
        planner_payload = planner_types.get(test_type, {})
        comparison["by_test_type"][test_type] = {
            "cases_total": max(
                baseline_payload.get("cases_total", 0),
                planner_payload.get("cases_total", 0),
            ),
            "metrics": {
                metric: _compare_metric_payloads(
                    baseline_payload.get("metrics", {}).get(metric),
                    planner_payload.get("metrics", {}).get(metric),
                )
                for metric in metric_keys
            },
        }

    return comparison


def _compare_case_results(
    baseline_results: list[dict[str, Any]],
    planner_results: list[dict[str, Any]],
    *,
    metric_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    baseline_by_case = {row["case_id"]: row for row in baseline_results}
    planner_by_case = {row["case_id"]: row for row in planner_results}
    changes: list[dict[str, Any]] = []

    for case_id in sorted(set(baseline_by_case).intersection(planner_by_case)):
        baseline_row = baseline_by_case[case_id]
        planner_row = planner_by_case[case_id]
        metric_deltas: dict[str, Any] = {}

        for metric in metric_keys:
            baseline_value = baseline_row.get("metrics", {}).get(metric)
            planner_value = planner_row.get("metrics", {}).get(metric)
            delta = None
            if baseline_value is not None and planner_value is not None:
                delta = round(planner_value - baseline_value, 4)
            if delta in {None, 0} and baseline_value == planner_value:
                continue
            metric_deltas[metric] = {
                "baseline": baseline_value,
                "planner": planner_value,
                "delta": delta,
            }

        if not metric_deltas:
            continue

        changes.append(
            {
                "case_id": case_id,
                "question": baseline_row.get("question"),
                "test_type": baseline_row.get("test_type"),
                "metric_deltas": metric_deltas,
            }
        )

    return changes


def _collect_regressions(summary_comparison: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    regressions: list[dict[str, Any]] = []

    for metric, payload in summary_comparison.get("metrics", {}).items():
        delta = payload.get("delta")
        if delta is not None and delta < 0:
            regressions.append(
                {
                    "scope": "overall",
                    "metric": metric,
                    "baseline": payload.get("baseline"),
                    "planner": payload.get("planner"),
                    "delta": delta,
                }
            )

    for test_type, payload in summary_comparison.get("by_test_type", {}).items():
        if payload.get("cases_total", 0) <= 0:
            continue
        for metric, metric_payload in payload.get("metrics", {}).items():
            delta = metric_payload.get("delta")
            if delta is None or delta >= 0:
                continue
            regressions.append(
                {
                    "scope": "by_test_type",
                    "test_type": test_type,
                    "metric": metric,
                    "baseline": metric_payload.get("baseline"),
                    "planner": metric_payload.get("planner"),
                    "delta": delta,
                }
            )

    regressions.sort(key=lambda item: (item.get("delta", 0), item.get("test_type") or "", item["metric"]))
    critical = [
        item
        for item in regressions
        if item.get("scope") == "by_test_type"
        and item.get("test_type") in CRITICAL_REGRESSION_BUCKETS
    ]
    return regressions, critical


def _build_comparison_report(
    baseline_report: dict[str, Any],
    planner_report: dict[str, Any],
    *,
    metric_keys: tuple[str, ...],
) -> dict[str, Any]:
    summary_comparison = _compare_summaries(
        baseline_report["summary"],
        planner_report["summary"],
        metric_keys=metric_keys,
    )
    regressions, critical_regressions = _collect_regressions(summary_comparison)
    focus_buckets = {
        test_type: summary_comparison["by_test_type"].get(test_type)
        for test_type in PLANNER_FOCUS_TEST_TYPES
    }

    return {
        "summary": summary_comparison,
        "focus_buckets": focus_buckets,
        "regressions": regressions,
        "critical_regressions": critical_regressions,
        "case_differences": _compare_case_results(
            baseline_report["results"],
            planner_report["results"],
            metric_keys=metric_keys,
        ),
    }


def evaluate_retrieval(
    cases: list[EvalCase],
    *,
    top_k: int = 8,
    retrieval_fn: RetrievalFn | None = None,
    planner_enabled: bool = False,
    force_planner: bool = False,
) -> dict[str, Any]:
    retrieval = retrieval_fn or default_retrieval_adapter
    results: list[dict[str, Any]] = []

    for case in cases:
        evidence = _invoke_eval_fn(
            retrieval,
            case=case,
            top_k=top_k,
            planner_enabled=planner_enabled,
            force_planner=force_planner,
        )
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
    planner_enabled: bool = False,
    force_planner: bool = False,
) -> dict[str, Any]:
    answer = answer_fn or default_answer_adapter
    results: list[dict[str, Any]] = []

    for case in cases:
        payload = _invoke_eval_fn(
            answer,
            case=case,
            top_k=top_k,
            planner_enabled=planner_enabled,
            force_planner=force_planner,
        ) or {}
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


def compare_retrieval(
    cases: list[EvalCase],
    *,
    top_k: int = 8,
    retrieval_fn: RetrievalFn | None = None,
    force_planner: bool = False,
) -> dict[str, Any]:
    baseline = evaluate_retrieval(
        cases,
        top_k=top_k,
        retrieval_fn=retrieval_fn,
        planner_enabled=False,
        force_planner=False,
    )
    planner = evaluate_retrieval(
        cases,
        top_k=top_k,
        retrieval_fn=retrieval_fn,
        planner_enabled=True,
        force_planner=force_planner,
    )
    return {
        "baseline": baseline,
        "planner": planner,
        "comparison": _build_comparison_report(
            baseline,
            planner,
            metric_keys=RETRIEVAL_METRIC_KEYS,
        ),
    }


def compare_answers(
    cases: list[EvalCase],
    *,
    top_k: int = 8,
    answer_fn: AnswerFn | None = None,
    force_planner: bool = False,
) -> dict[str, Any]:
    baseline = evaluate_answers(
        cases,
        top_k=top_k,
        answer_fn=answer_fn,
        planner_enabled=False,
        force_planner=False,
    )
    planner = evaluate_answers(
        cases,
        top_k=top_k,
        answer_fn=answer_fn,
        planner_enabled=True,
        force_planner=force_planner,
    )
    return {
        "baseline": baseline,
        "planner": planner,
        "comparison": _build_comparison_report(
            baseline,
            planner,
            metric_keys=ANSWER_METRIC_KEYS,
        ),
    }
