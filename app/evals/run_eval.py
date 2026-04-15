import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evals.dataset import load_eval_cases
from app.evals.runner import evaluate_answers, evaluate_retrieval
from app.settings import (
    EMBEDDING_PROVIDER,
    HYBRID_RRF_K,
    LEXICAL_MAX_DOCS,
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
        print(f"- {test_type}: cases={payload.get('cases_total', 0)}")
        for metric_name, metric in payload.get("metrics", {}).items():
            print(
                f"  {metric_name}: {_format_metric(metric.get('value'), metric.get('count', 0))}"
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
        "--fail-on-answer-error",
        action="store_true",
        help="If set, fail the run when answer eval fails (e.g., missing API key).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

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
            },
            "case_count": len(cases),
        }
    }

    print(f"Loaded {len(cases)} eval cases from: {dataset_path}")
    if args.label:
        print(f"Run label: {args.label}")

    if args.mode in {"retrieval", "both"}:
        retrieval_report = evaluate_retrieval(cases, top_k=args.top_k)
        run_payload["retrieval"] = retrieval_report
        _print_summary_block("retrieval", retrieval_report["summary"])

    if args.mode in {"answer", "both"}:
        try:
            answer_report = evaluate_answers(cases, top_k=args.top_k)
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
