"""
Run eval dataset cases with LangSmith tracing enabled so runs appear in your LangSmith project.

This is intentionally small: it reuses the same @traceable-wrapped retrieval path as the app
(`search_parent_evidence` via `app.evals.runner.default_retrieval_adapter`). It does **not**
upload LangSmith Datasets or run hosted evaluators; for scored metrics use `python -m app.evals.run_eval`.

Prerequisites (all required before a traced run):
  - ENABLE_LANGSMITH=true (same flag as app tracing in app/settings)
  - LANGSMITH_API_KEY or LANGCHAIN_API_KEY
  - LANGSMITH_PROJECT (non-empty in settings; set in app/.env or environment)
  - LANGSMITH_TRACING must not be explicitly set to a false value (e.g. ``false``); when validation
    passes, this runner sets LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 and project env vars for the
    process, matching ``app.main._configure_langsmith_env``.

Local JSON evals (`run_eval.py`) never require LangSmith.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

_APP_DIR = Path(__file__).resolve().parents[1]
load_dotenv(_APP_DIR / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run eval cases once each with LangSmith tracing (retrieval path only by default).",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path("evals") / "datasets" / "book_eval_cases.jsonl"),
        help="Path to eval dataset (.json or .jsonl), same as run_eval.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Top-k for retrieval.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max cases to run (0 = all).",
    )
    parser.add_argument(
        "--planner-enabled",
        action="store_true",
        help="Pass planner_enabled=True into the retrieval adapter (planner-gated unless --force-planner).",
    )
    parser.add_argument(
        "--force-planner",
        action="store_true",
        help="Force planner on the planner-enabled path (only used with --planner-enabled).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate environment and load cases; do not call retrieval.",
    )
    return parser


def _env_explicitly_disabled(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return False
    return str(raw).strip().lower() in {"0", "false", "no", "off"}


def validate_langsmith_eval_environment() -> tuple[bool, list[str]]:
    """Return (ok, error_messages). Does not mutate os.environ."""
    import app.settings as app_settings

    errors: list[str] = []

    try:
        import langsmith  # noqa: F401
    except ImportError:
        errors.append("Python package 'langsmith' is not installed (see requirements.txt).")

    if not app_settings.ENABLE_LANGSMITH:
        errors.append(
            "ENABLE_LANGSMITH is false in app settings. Set ENABLE_LANGSMITH=true in app/.env "
            "or the environment (same as enabling tracing for the web app)."
        )

    key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if not key:
        errors.append("Missing LANGSMITH_API_KEY (or LANGCHAIN_API_KEY).")

    project = (app_settings.LANGSMITH_PROJECT or "").strip()
    if not project:
        errors.append("LANGSMITH_PROJECT is empty in app settings.")

    if _env_explicitly_disabled("LANGSMITH_TRACING"):
        errors.append(
            "LANGSMITH_TRACING is set to a false value. Unset it or set LANGSMITH_TRACING=true "
            "so traces can export (this runner sets tracing env after validation)."
        )
    if _env_explicitly_disabled("LANGCHAIN_TRACING_V2"):
        errors.append(
            "LANGCHAIN_TRACING_V2 is set to a false value. Unset it or set LANGCHAIN_TRACING_V2=true."
        )

    return (len(errors) == 0, errors)


def apply_langsmith_tracing_env() -> None:
    """Mirror app.main._configure_langsmith_env so the LangSmith SDK exports traces."""
    import app.settings as app_settings

    api_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if not api_key:
        return
    project = (app_settings.LANGSMITH_PROJECT or "").strip()
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    ok, errors = validate_langsmith_eval_environment()
    if not ok:
        print("LangSmith eval runner cannot start — fix the following and retry:\n", file=sys.stderr)
        for line in errors:
            print(f"  - {line}", file=sys.stderr)
        print(
            "\nLocal metrics eval (no LangSmith): python -m app.evals.run_eval\n"
            "Tracing setup for the app: see README Observability and evals/README.md.",
            file=sys.stderr,
        )
        sys.exit(2)

    apply_langsmith_tracing_env()

    from app.evals.dataset import load_eval_cases
    from app.evals.runner import default_retrieval_adapter

    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(2)

    cases = load_eval_cases(dataset_path)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    print(
        f"LangSmith tracing: project={os.environ.get('LANGSMITH_PROJECT')} "
        f"cases={len(cases)} dataset={dataset_path} dry_run={args.dry_run}"
    )

    if args.dry_run:
        print("Dry run complete (no retrieval calls). Open LangSmith after a real run to inspect traces.")
        sys.exit(0)

    planner_enabled = bool(args.planner_enabled)
    force_planner = bool(args.force_planner)
    failures = 0
    for idx, case in enumerate(cases, start=1):
        label = f"[{idx}/{len(cases)}] case_id={case.case_id} test_type={case.test_type}"
        try:
            default_retrieval_adapter(
                case.question,
                args.top_k,
                case.query_book_ids,
                planner_enabled=planner_enabled,
                force_planner=force_planner,
                case=case,
            )
            print(f"{label} ok")
        except Exception as exc:
            failures += 1
            print(f"{label} FAILED: {exc}", file=sys.stderr)
            traceback.print_exc(limit=3, file=sys.stderr)

    print(
        f"Finished traced retrieval passes: total={len(cases)} failures={failures}. "
        f"View runs in LangSmith project {os.environ.get('LANGSMITH_PROJECT')} "
        "(look for search_parent_evidence spans)."
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
