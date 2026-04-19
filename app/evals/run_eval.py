import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evals.dataset import load_eval_cases
from app.evals.runner import (
    compare_answers,
    compare_retrieval,
    evaluate_answers,
    evaluate_retrieval,
)
from app.settings import (
    EMBEDDING_PROVIDER,
    ENABLE_QUERY_PLANNER,
    HYBRID_RRF_K,
    LEXICAL_MAX_DOCS,
    QUERY_PLANNER_MODEL,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_MODE,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_output_path() -> Path:
    return Path("evals") / "reports" / f"eval_report_{_timestamp()}.json"


def _format_metric(value: float | None, count: int) -> str:
    if value is None:
        return f"n/a (n={count})"
    return f"{value:.4f} (n={count})"


def _format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}"


def _print_summary_block(title: str, summary: dict[str, Any]) -> None:
    print(f"\n[{title}]")
    print(f"cases_total: {summary.get('cases_total', 0)}")

    metrics = summary.get("metrics", {})
    for metric_name, metric in metrics.items():
        print(f"- {metric_name}: {_format_metric(metric.get('value'), metric.get('count', 0))}")

    by_type = summary.get("by_test_type", {})
    if not by_type:
        return

    print("by_test_type:")
    for test_type, payload in sorted(by_type.items()):
        if payload.get("cases_total", 0) <= 0:
            continue
        print(f"- {test_type}: cases={payload.get('cases_total', 0)}")
        for metric_name, metric in payload.get("metrics", {}).items():
            print(
                f"  {metric_name}: {_format_metric(metric.get('value'), metric.get('count', 0))}"
            )


def _print_comparison_block(title: str, comparison_payload: dict[str, Any]) -> None:
    comparison = comparison_payload.get("comparison", {})
    print(f"\n[{title} comparison]")
    print(f"cases_total: {comparison.get('summary', {}).get('cases_total', 0)}")

    print("overall:")
    for metric_name, metric in comparison.get("summary", {}).get("metrics", {}).items():
        print(
            f"- {metric_name}: "
            f"baseline={_format_metric(metric.get('baseline'), metric.get('baseline_count', 0))} "
            f"planner={_format_metric(metric.get('planner'), metric.get('planner_count', 0))} "
            f"delta={_format_delta(metric.get('delta'))}"
        )

    focus_buckets = comparison.get("focus_buckets", {})
    if focus_buckets:
        print("focus_buckets:")
        for test_type, payload in focus_buckets.items():
            if not payload or payload.get("cases_total", 0) <= 0:
                continue
            print(f"- {test_type}: cases={payload.get('cases_total', 0)}")
            for metric_name, metric in payload.get("metrics", {}).items():
                print(
                    f"  {metric_name}: "
                    f"baseline={_format_metric(metric.get('baseline'), metric.get('baseline_count', 0))} "
                    f"planner={_format_metric(metric.get('planner'), metric.get('planner_count', 0))} "
                    f"delta={_format_delta(metric.get('delta'))}"
                )

    critical_regressions = comparison.get("critical_regressions", [])
    if critical_regressions:
        print("critical_regressions:")
        for item in critical_regressions:
            print(
                f"- {item.get('test_type')}.{item.get('metric')}: "
                f"baseline={item.get('baseline')} planner={item.get('planner')} "
                f"delta={_format_delta(item.get('delta'))}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BookQnA retrieval/answer evaluation.")
    parser.add_argument(
        "--dataset",
        default=str(Path("evals") / "datasets" / "book_eval_cases.jsonl"),
        help="Path to eval dataset (.json or .jsonl).",
    )
    parser.add_argument(
        "--mode",
        choices=("retrieval", "answer", "both"),
        default="retrieval",
        help="Which evaluation suite to run.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Top-k evidence/citations to evaluate.")
    parser.add_argument("--label", default="", help="Optional run label for comparison tracking.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional report output path. Default: evals/reports/eval_report_<timestamp>.json",
    )
    parser.add_argument(
        "--planner-enabled",
        action="store_true",
        help="Run a single eval pass with planner-enabled retrieval/QA. Default remains planner-off.",
    )
    parser.add_argument(
        "--compare-planner",
        action="store_true",
        help="Run baseline and planner-enabled evals side by side on the same dataset.",
    )
    parser.add_argument(
        "--force-planner",
        action="store_true",
        help="Force planner usage even when gating heuristics would normally skip it.",
    )
    parser.add_argument(
        "--fail-on-answer-error",
        action="store_true",
        help="If set, fail the run when answer eval fails (e.g., missing API key).",
    )
    return parser


def _validate_planner_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.force_planner and not (args.compare_planner or args.planner_enabled):
        parser.error(
            "--force-planner requires --planner-enabled or --compare-planner. "
            "Baseline runs must remain planner-off."
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_planner_flags(parser, args)

    planner_mode = "compare" if args.compare_planner else ("enabled" if args.planner_enabled else "baseline")
    force_planner_effective = bool(args.force_planner and (args.compare_planner or args.planner_enabled))

    dataset_path = Path(args.dataset)
    cases = load_eval_cases(dataset_path)
    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_payload: dict[str, Any] = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "label": args.label or None,
            "dataset_path": str(dataset_path),
            "mode": args.mode,
            "top_k": args.top_k,
            "settings": {
                "embedding_provider": EMBEDDING_PROVIDER,
                "retrieval_mode": RETRIEVAL_MODE,
                "retrieval_fetch_k": RETRIEVAL_FETCH_K,
                "lexical_max_docs": LEXICAL_MAX_DOCS,
                "hybrid_rrf_k": HYBRID_RRF_K,
                "planner_default_enabled": ENABLE_QUERY_PLANNER,
                "planner_model": QUERY_PLANNER_MODEL,
            },
            "planner_enabled": bool(args.planner_enabled),
            "compare_planner": bool(args.compare_planner),
            "planner_mode": planner_mode,
            "force_planner": bool(args.force_planner),
            "force_planner_effective": force_planner_effective,
            "case_count": len(cases),
        }
    }

    print(f"Loaded {len(cases)} eval cases from: {dataset_path}")
    if args.label:
        print(f"Run label: {args.label}")
    if args.compare_planner:
        print("Planner comparison: baseline vs planner-enabled")
    elif args.planner_enabled:
        print("Planner mode: enabled")
    else:
        print("Planner mode: baseline")
    if force_planner_effective:
        print("Planner gating: forced on planner-enabled path")

    if args.mode in {"retrieval", "both"}:
        if args.compare_planner:
            retrieval_report = compare_retrieval(cases, top_k=args.top_k, force_planner=force_planner_effective)
            run_payload["retrieval"] = retrieval_report
            _print_comparison_block("retrieval", retrieval_report)
        else:
            retrieval_report = evaluate_retrieval(
                cases,
                top_k=args.top_k,
                planner_enabled=bool(args.planner_enabled),
                force_planner=force_planner_effective,
            )
            run_payload["retrieval"] = retrieval_report
            _print_summary_block("retrieval", retrieval_report["summary"])

    if args.mode in {"answer", "both"}:
        try:
            if args.compare_planner:
                answer_report = compare_answers(cases, top_k=args.top_k, force_planner=force_planner_effective)
                run_payload["answer"] = answer_report
                _print_comparison_block("answer", answer_report)
            else:
                answer_report = evaluate_answers(
                    cases,
                    top_k=args.top_k,
                    planner_enabled=bool(args.planner_enabled),
                    force_planner=force_planner_effective,
                )
                run_payload["answer"] = answer_report
                _print_summary_block("answer", answer_report["summary"])
        except Exception as exc:
            run_payload["answer"] = {"error": str(exc)}
            print(f"\n[answer]\nerror: {exc}")
            if args.fail_on_answer_error:
                raise

    output_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
