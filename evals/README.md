# Evaluation Harness

This folder contains lightweight, repeatable evaluation assets for retrieval and answer quality.

**Working directory:** run all commands below from the **`bookQnA/`** project root (the same place you run `uvicorn app.main:app`) so `python -m app.evals…` imports resolve.

For optional reranking, FTS lexical retrieval, planner comparison, LangSmith tracing, and caches, see the **Advanced RAG features** section in the repo [`README.md`](../README.md).

## Dataset format

The default dataset file is [`evals/datasets/book_eval_cases.jsonl`](./datasets/book_eval_cases.jsonl).

Each JSONL row supports:

- `question` (required)
- `expected_book`
- `expected_pages`
- `expected_chapter`
- `expected_answer_keywords`
- `answerable`
- `test_type`
- `query_book_ids` (optional explicit book filter for the query call)
- `recent_history` (optional list of `{question, answer}` turns for follow-up evals)
- `citation_usefulness_score_manual` (optional placeholder/manual score)

Supported test buckets used by default:

- `follow_up`
- `ambiguous_short`
- `concept_lookup`
- `chapter_summary`
- `locate_section_page`
- `compare_concepts`
- `exact_phrase_lookup`
- `abstention`

## Run evals

Run retrieval-only evals (default):

```bash
python -m app.evals.run_eval
```

Run both retrieval and answer evals:

```bash
python -m app.evals.run_eval --mode both
```

Run a single planner-enabled pass:

```bash
python -m app.evals.run_eval --mode both --planner-enabled
```

Run baseline vs planner side by side on the same dataset:

```bash
python -m app.evals.run_eval --mode both --compare-planner
```

**Retrieval-only** (no Gemini answer generation; still compares retrieval metrics twice):

```bash
python -m app.evals.run_eval --mode retrieval --compare-planner
```

Default report path for `--compare-planner` is `evals/reports/planner_comparison_<UTC-timestamp>.json` (override with `--output ...`).

Force planner usage on the planner-enabled path, bypassing gating heuristics:

```bash
python -m app.evals.run_eval --mode both --compare-planner --force-planner
```

### Gemini `429` / free-tier quota

Answer evals call the **chat** model (`GEMINI_CHAT_MODEL` in `app/.env`, default `gemini-2.5-flash`) once per case, plus an extra call when a case uses `recent_history` and reformulation runs. The **query planner** uses `QUERY_PLANNER_MODEL` separately when planner flags are on.

If you see `ResourceExhausted: 429` with `generate_content_free_tier_requests` and a **per-day** limit, spacing alone will not help until the quota resets: use another model via `GEMINI_CHAT_MODEL`, enable billing on the Google AI project, or run **`python -m app.evals.run_eval --mode retrieval`** to skip answer generation while you tune retrieval.

### Planner comparison JSON layout

The report includes `run.report_type: "planner_comparison"` and, for each of `retrieval` / `answer` (when that mode ran):

- `baseline` / `planner` — full single-pass reports (`summary`, `results` with `latency_ms` per case).
- `comparison.summary.metrics` — each metric has `baseline`, `planner`, `delta`, `trend` (`improved` / `regressed` / `same` / `unknown`). Higher metric values are treated as better (including `abstention_correctness`).
- `comparison.latency_ms` — mean latency per arm and `delta_planner_minus_baseline_ms` (negative means planner path was faster on average).
- `comparison.per_case` — one row per case: `question`, `expected` (book/pages/chapter; `answer_keywords` when answer eval ran), per-metric `baseline` / `planner` / `delta` / `trend`, `baseline.evidence_top` vs `planner.evidence_top`, and `answer` + `eval_debug` when answer mode was included.
- `comparison.case_differences` — only cases where at least one metric changed (compact diff list).
- `comparison.critical_regressions` / `focus_buckets` — same buckets as before for rollout review.

The metric `top_k_evidence_hit` is the strict “evidence hit” over book/page/chapter expectations in top-k (see `comparison.summary_metric_notes` in the JSON).

Specify dataset/top-k/output:

```bash
python -m app.evals.run_eval --dataset evals/datasets/book_eval_cases.jsonl --top-k 10 --output evals/reports/run_a.json
```

## Compare retrieval settings

Use the same dataset, vary retrieval env settings or compare planner-off vs planner-on, and keep one JSON report per run.

Example:

```bash
$env:RETRIEVAL_MODE="dense_only"
python -m app.evals.run_eval --mode retrieval --label dense_only --output evals/reports/dense_only.json

$env:RETRIEVAL_MODE="hybrid"
python -m app.evals.run_eval --mode retrieval --label hybrid --output evals/reports/hybrid.json
```

Then compare `evals/reports/*.json` (or a single `planner_comparison_*.json`):

- overall `comparison.summary.metrics` (each metric: `baseline`, `planner`, `delta`, `trend`)
- `comparison.per_case` for side-by-side evidence and answers
- `comparison.latency_ms`
- `comparison.critical_regressions`
- `comparison.focus_buckets.follow_up`
- `comparison.focus_buckets.ambiguous_short`
- `comparison.focus_buckets.exact_phrase_lookup`
- `comparison.focus_buckets.abstention`
- `retrieval.comparison.summary.by_test_type` / `answer.comparison.summary.by_test_type` (inside a planner comparison report)

This keeps rollout decisions evidence-based: planner gains on follow-up and ambiguous queries should be weighed against regressions in exact-phrase lookup and abstention before enabling it by default.

## LangSmith traced retrieval passes (optional)

`python -m app.evals.run_eval` stays fully local: it never contacts LangSmith and does not need tracing env vars.

To push **one trace per eval case** through the same `@traceable` retrieval path as the web app (`search_parent_evidence` via `default_retrieval_adapter`), use:

```bash
python -m app.evals.langsmith_eval --dataset evals/datasets/book_eval_cases.jsonl
```

**Enable tracing** (same as the app): in `app/.env`, set `ENABLE_LANGSMITH=true`, `LANGSMITH_API_KEY` or `LANGCHAIN_API_KEY`, and `LANGSMITH_PROJECT`. Do not set `LANGSMITH_TRACING` or `LANGCHAIN_TRACING_V2` to false; the runner sets tracing-related process env to match `app.main._configure_langsmith_env` after checks pass.

**Dry run** (validate + load JSONL only; no Chroma retrieval):

```bash
python -m app.evals.langsmith_eval --dry-run
```

**CLI flags:** `--top-k`, `--limit`, `--planner-enabled`, `--force-planner` (same semantics as `run_eval` for the retrieval adapter), `--dataset`.

**What appears in LangSmith:** one run per process with spans for traced calls inside retrieval (notably `search_parent_evidence`). This script does **not** upload LangSmith datasets or run hosted evaluators; use `run_eval` for JSON metrics reports.
