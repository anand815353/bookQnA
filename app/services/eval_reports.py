# app/services/eval_reports.py
"""Load JSON eval reports from disk for the local dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def reports_dir() -> Path:
    return Path("evals") / "reports"


def list_report_files(base: Path | None = None) -> list[Path]:
    directory = base or reports_dir()
    if not directory.is_dir():
        return []
    paths = [p for p in directory.glob("eval_report_*.json") if p.is_file()]
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def load_report(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_latest_report(base: Path | None = None) -> dict[str, Any] | None:
    files = list_report_files(base)
    if not files:
        return None
    return load_report(files[0])


def _results_list(block: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(block, dict):
        return []
    return list(block.get("results") or [])


def _bool_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [(r.get("eval_debug") or {}).get(key) for r in rows]
    known = [v for v in vals if v is not None]
    if not known:
        return None
    return round(sum(1 for v in known if v is True) / len(known), 4)


def _mean_int(rows: list[dict[str, Any]], key: str) -> float | None:
    vals: list[int] = []
    for r in rows:
        v = (r.get("eval_debug") or {}).get(key)
        if isinstance(v, (int, float)):
            vals.append(int(v))
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def _mean_stage_latencies(rows: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for r in rows:
        stage = (r.get("eval_debug") or {}).get("stage_latency_ms")
        if not isinstance(stage, dict):
            continue
        for sk, sv in stage.items():
            if isinstance(sv, (int, float)):
                buckets.setdefault(str(sk), []).append(float(sv))
    return {k: round(sum(v) / len(v), 2) for k, v in sorted(buckets.items()) if v}


def _aggregate_eval_debug(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "cases": len(rows),
        "retrieval_cache_hit_rate": _bool_rate(rows, "retrieval_cache_hit"),
        "generation_cache_hit_rate": _bool_rate(rows, "generation_cache_hit"),
        "reranker_enabled_rate": _bool_rate(rows, "reranker_enabled"),
        "grounded_rate": _bool_rate(rows, "grounded"),
        "avg_context_chars_before": _mean_int(rows, "context_chars_before"),
        "avg_context_chars_after": _mean_int(rows, "context_chars_after"),
        "avg_dense_result_count": _mean_int(rows, "dense_result_count"),
        "avg_lexical_result_count": _mean_int(rows, "lexical_result_count"),
        "avg_hybrid_result_count": _mean_int(rows, "hybrid_result_count"),
        "avg_reranker_latency_ms": _mean_int(rows, "reranker_latency_ms"),
        "avg_stage_latency_ms": _mean_stage_latencies(rows),
    }


def summarize_report_for_dashboard(report: dict[str, Any]) -> dict[str, Any]:
    """Pre-aggregate numbers for Jinja (HTML tables only)."""
    out: dict[str, Any] = {
        "is_comparison": False,
        "answer_rows": [],
        "answer_baseline_rows": [],
        "answer_planner_rows": [],
        "retrieval_rows": [],
        "retrieval_baseline_rows": [],
        "retrieval_planner_rows": [],
        "eval_debug_stats": {},
        "eval_debug_stats_baseline": {},
        "eval_debug_stats_planner": {},
    }

    answer_block = report.get("answer")
    if isinstance(answer_block, dict) and "baseline" in answer_block:
        out["is_comparison"] = True
        b_rows = _results_list(answer_block.get("baseline"))
        p_rows = _results_list(answer_block.get("planner"))
        out["answer_baseline_rows"] = b_rows
        out["answer_planner_rows"] = p_rows
        out["eval_debug_stats_baseline"] = _aggregate_eval_debug(b_rows)
        out["eval_debug_stats_planner"] = _aggregate_eval_debug(p_rows)
    else:
        rows = _results_list(answer_block if isinstance(answer_block, dict) else None)
        out["answer_rows"] = rows
        out["eval_debug_stats"] = _aggregate_eval_debug(rows)

    retr = report.get("retrieval")
    if isinstance(retr, dict) and "baseline" in retr:
        out["retrieval_baseline_rows"] = _results_list(retr.get("baseline"))
        out["retrieval_planner_rows"] = _results_list(retr.get("planner"))
    else:
        out["retrieval_rows"] = _results_list(retr if isinstance(retr, dict) else None)

    return out


def metrics_table_rows(summary_block: dict[str, Any] | None) -> list[tuple[str, Any, int]]:
    """(metric_name, mean_value, count) for Jinja tables."""
    if not summary_block or not isinstance(summary_block, dict):
        return []
    metrics = summary_block.get("metrics") or {}
    rows: list[tuple[str, Any, int]] = []
    for name in sorted(metrics.keys()):
        payload = metrics.get(name) or {}
        if not isinstance(payload, dict):
            continue
        rows.append((name, payload.get("value"), int(payload.get("count") or 0)))
    return rows


def case_table_rows_from_results(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Flatten answer result rows for the dashboard case table."""
    out: list[dict[str, Any]] = []
    for r in list(rows or []):
        ed = r.get("eval_debug") if isinstance(r.get("eval_debug"), dict) else {}
        stage = ed.get("stage_latency_ms")
        total_ms = None
        if isinstance(stage, dict):
            total_ms = stage.get("total")
        q = str(r.get("question") or "")
        out.append(
            {
                "case_id": r.get("case_id"),
                "test_type": r.get("test_type"),
                "grounded": ed.get("grounded"),
                "retrieval_cache_hit": ed.get("retrieval_cache_hit"),
                "generation_cache_hit": ed.get("generation_cache_hit"),
                "context_chars_before": ed.get("context_chars_before"),
                "context_chars_after": ed.get("context_chars_after"),
                "total_ms": total_ms,
                "question_trunc": (q[:120] + "...") if len(q) > 120 else q,
                "question_full": q,
            }
        )
    return out


def case_table_rows_for_dashboard(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if summary.get("is_comparison"):
        return case_table_rows_from_results(list(summary.get("answer_baseline_rows") or []))
    return case_table_rows_from_results(list(summary.get("answer_rows") or []))


def by_test_type_rows(summary_block: dict[str, Any] | None) -> list[tuple[str, int, list[tuple[str, Any, int]]]]:
    """(test_type, cases_total, [(metric, value, count), ...])."""
    if not summary_block or not isinstance(summary_block, dict):
        return []
    out: list[tuple[str, int, list[tuple[str, Any, int]]]] = []
    by_tt = summary_block.get("by_test_type") or {}
    if not isinstance(by_tt, dict):
        return []
    for test_type in sorted(by_tt.keys()):
        payload = by_tt.get(test_type) or {}
        if not isinstance(payload, dict):
            continue
        cases_total = int(payload.get("cases_total") or 0)
        if cases_total <= 0:
            continue
        mrows: list[tuple[str, Any, int]] = []
        for name in sorted((payload.get("metrics") or {}).keys()):
            cell = (payload.get("metrics") or {}).get(name) or {}
            if isinstance(cell, dict):
                mrows.append((name, cell.get("value"), int(cell.get("count") or 0)))
        out.append((test_type, cases_total, mrows))
    return out
