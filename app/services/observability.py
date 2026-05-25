# app/services/observability.py
"""Optional LangSmith helpers: safe metadata, hashing, and conditional @traceable."""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Callable, TypeVar

from app.settings import (
    APP_ENV,
    APP_VERSION,
    ENABLE_LANGSMITH,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACE_FULL_CONTEXT,
    RETRIEVAL_MODE,
)


F = TypeVar("F", bound=Callable[..., Any])

_RE_WS = re.compile(r"\s+")


def stable_hash(value: str, *, prefix: str = "sha256") -> str:
    """Deterministic short correlation id from text (not reversible to full text)."""
    normalized = _RE_WS.sub(" ", (value or "").strip()).encode("utf-8", errors="ignore")
    digest = hashlib.sha256(normalized).hexdigest()
    return f"{prefix}:{digest[:24]}"


def langsmith_enabled() -> bool:
    """True when tracing is turned on and an API key is present (export will work)."""
    if not ENABLE_LANGSMITH:
        return False
    key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    return bool(key)


def _tracing_hooks_active() -> bool:
    """Attach @traceable when the feature flag is on (import-safe; does not require key yet)."""
    return bool(ENABLE_LANGSMITH)


def base_trace_metadata() -> dict[str, Any]:
    return {
        "app_env": APP_ENV,
        "app_version": APP_VERSION,
        "retrieval_mode": RETRIEVAL_MODE,
        "langsmith_project": LANGSMITH_PROJECT,
    }


def query_trace_metadata(
    *,
    question_len: int,
    question_hash: str,
    retrieval_question_len: int | None = None,
    retrieval_question_hash: str | None = None,
    effective_question_len: int | None = None,
    effective_question_hash: str | None = None,
    book_ids_count: int = 0,
    top_k: int = 0,
    include_debug: bool = False,
    recent_history_turns: int = 0,
) -> dict[str, Any]:
    """Safe query-side metadata (no raw question or history text)."""
    meta: dict[str, Any] = {
        "question_len": question_len,
        "question_hash": question_hash,
        "book_ids_count": book_ids_count,
        "top_k": top_k,
        "include_debug": include_debug,
        "recent_history_turns": recent_history_turns,
    }
    if retrieval_question_len is not None:
        meta["retrieval_question_len"] = retrieval_question_len
    if retrieval_question_hash is not None:
        meta["retrieval_question_hash"] = retrieval_question_hash
    if effective_question_len is not None:
        meta["effective_question_len"] = effective_question_len
    if effective_question_hash is not None:
        meta["effective_question_hash"] = effective_question_hash
    return meta


# Keys safe to forward from retrieval debug (no query strings, no full query_plan text).
_RETRIEVAL_DEBUG_SAFE_KEYS = frozenset(
    {
        "retrieval_mode",
        "top_k",
        "dense_result_count",
        "lexical_result_count",
        "hybrid_result_count",
        "reranker_enabled",
        "reranker_latency_ms",
        "retrieval_cache_hit",
        "retrieval_cache_latency_ms",
        "retrieval_cache_key_hash",
        "parent_sections_count",
        "candidate_children_count",
        "total_retrieval_latency_ms",
        "dense_candidates",
        "lexical_candidates",
    }
)


def retrieval_debug_trace_fields(debug: dict | None) -> dict[str, Any]:
    """Subset of retrieval debug safe for traces and LangSmith metadata."""
    if not debug:
        return {}
    out: dict[str, Any] = {}
    for key in _RETRIEVAL_DEBUG_SAFE_KEYS:
        if key in debug:
            out[key] = debug[key]
    stage = debug.get("stage_latency_ms")
    if isinstance(stage, dict):
        safe_stage: dict[str, Any] = {}
        for sk, sv in stage.items():
            if isinstance(sv, (int, float, bool)) or sv is None:
                safe_stage[str(sk)] = sv
        if safe_stage:
            out["stage_latency_ms"] = safe_stage
    qp = debug.get("query_plan")
    if isinstance(qp, dict) and qp.get("query_type") is not None:
        out["planner_query_type"] = qp.get("query_type")
    return out


def generation_invoke_metadata(
    *,
    prompt_version: str,
    model_name: str,
    retrieval_mode: str,
    planner_mode: str,
    context_chars: int,
    context_chars_before: int,
    context_chars_after: int,
    dense_result_count: int | None,
    lexical_result_count: int | None,
    hybrid_result_count: int | None,
    reranker_enabled: bool | None,
    reranker_latency_ms: int | None,
    retrieval_cache_hit: bool | None,
    generation_cache_hit: bool,
    citations_count: int,
    question_len: int,
    question_hash: str,
) -> dict[str, Any]:
    """RunnableConfig.metadata for the answer LLM invoke (scalars only)."""
    meta: dict[str, Any] = {
        **base_trace_metadata(),
        "prompt_version": prompt_version,
        "model_name": model_name,
        "retrieval_mode": retrieval_mode,
        "planner_mode": planner_mode,
        "context_chars": context_chars,
        "context_chars_before": context_chars_before,
        "context_chars_after": context_chars_after,
        "generation_cache_hit": generation_cache_hit,
        "citations_count": citations_count,
        "question_len": question_len,
        "question_hash": question_hash,
    }
    if dense_result_count is not None:
        meta["dense_result_count"] = dense_result_count
    if lexical_result_count is not None:
        meta["lexical_result_count"] = lexical_result_count
    if hybrid_result_count is not None:
        meta["hybrid_result_count"] = hybrid_result_count
    if reranker_enabled is not None:
        meta["reranker_enabled"] = reranker_enabled
    if reranker_latency_ms is not None:
        meta["reranker_latency_ms"] = reranker_latency_ms
    if retrieval_cache_hit is not None:
        meta["retrieval_cache_hit"] = retrieval_cache_hit
    if LANGSMITH_TRACE_FULL_CONTEXT:
        meta["trace_full_context_enabled"] = True
    return meta


def _try_traceable() -> Callable[..., Any] | None:
    try:
        from langsmith import traceable as traceable_fn

        return traceable_fn  # type: ignore[no-any-return]
    except ImportError:
        return None


def maybe_traceable(*args: Any, **kwargs: Any) -> Any:
    """
    LangSmith @traceable when ENABLE_LANGSMITH and package installed; otherwise identity.
    API key can be loaded from .env before first traced call; lifespan also sets tracing env.
    """
    if not _tracing_hooks_active():
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _noop_deco(fn: F) -> F:
            return fn

        return _noop_deco

    traceable_fn = _try_traceable()
    if traceable_fn is None:
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _noop_deco(fn: F) -> F:
            return fn

        return _noop_deco

    return traceable_fn(*args, **kwargs)


def trace_process_inputs_answer_question(inputs: dict[str, Any]) -> dict[str, Any]:
    q = inputs.get("question")
    rq = inputs.get("retrieval_question")  # not a param of answer_question — ignore
    out: dict[str, Any] = {}
    for k, v in inputs.items():
        if k == "question":
            out["question_len"] = len(str(q or ""))
            out["question_hash"] = stable_hash(str(q or ""))
        elif k == "recent_history":
            out["recent_history_turns"] = len(v) if isinstance(v, list) else 0
        elif k == "book_ids":
            out["book_ids_count"] = len(v) if isinstance(v, list) else 0
        else:
            out[k] = v
    return out


def trace_process_outputs_answer_question(outputs: dict[str, Any]) -> dict[str, Any]:
    if LANGSMITH_TRACE_FULL_CONTEXT:
        return dict(outputs)
    citations = outputs.get("citations") or []
    slim_cits: list[dict[str, Any]] = []
    if isinstance(citations, list):
        for c in citations:
            if not isinstance(c, dict):
                continue
            slim_cits.append(
                {
                    "book_id": c.get("book_id"),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                }
            )
    answer = str(outputs.get("answer") or "")
    return {
        "answer_chars": len(answer),
        "answer_hash": stable_hash(answer),
        "grounded": outputs.get("grounded"),
        "citations_count": len(citations) if isinstance(citations, list) else 0,
        "citations": slim_cits,
        "has_debug": "debug" in outputs,
    }


def trace_process_inputs_search_parent_evidence(inputs: dict[str, Any]) -> dict[str, Any]:
    q = inputs.get("question")
    out: dict[str, Any] = {}
    for k, v in inputs.items():
        if k == "question":
            out["question_len"] = len(str(q or ""))
            out["question_hash"] = stable_hash(str(q or ""))
        elif k == "book_ids":
            out["book_ids_count"] = len(v) if isinstance(v, list) else 0
        elif k == "debug_info" and v is not None:
            out["debug_info_attached"] = True
        elif k == "query_plan" and v is not None:
            qt = getattr(v, "query_type", None)
            if qt is None and isinstance(v, dict):
                qt = v.get("query_type")
            out["query_plan_query_type"] = qt
        else:
            out[k] = v
    return out


def trace_process_outputs_search_parent_evidence(outputs: Any) -> dict[str, Any]:
    if not isinstance(outputs, list):
        return {"evidence_type": type(outputs).__name__}
    n = len(outputs)
    fp_parts: list[str] = []
    for p in outputs[:20]:
        if not isinstance(p, dict):
            continue
        fp_parts.append(str(p.get("book_id", "")))
        fp_parts.append(str(p.get("parent_section_id", "")))
    fingerprint = stable_hash("|".join(fp_parts))
    return {"parent_sections_count": n, "evidence_fingerprint": fingerprint}


def trace_process_inputs_pack_context(inputs: dict[str, Any]) -> dict[str, Any]:
    pe = inputs.get("parent_evidence")
    if not isinstance(pe, list):
        return {"parent_evidence": "<non-list>"}
    child_n = 0
    for p in pe:
        if isinstance(p, dict):
            child_n += len(p.get("children") or [])
    return {
        "parent_sections_count": len(pe),
        "children_total": child_n,
        "max_chars": inputs.get("max_chars"),
        "include_parent_context": inputs.get("include_parent_context"),
    }


def trace_process_outputs_pack_context(outputs: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(outputs, dict):
        return {"output_type": type(outputs).__name__}
    ctx = outputs.get("context") or ""
    ctx_s = str(ctx)
    out: dict[str, Any] = {
        "context_chars": len(ctx_s),
        "citations_count": len(outputs.get("citations") or []),
        "context_chars_before": outputs.get("context_chars_before"),
        "context_chars_after": outputs.get("context_chars_after"),
    }
    if LANGSMITH_TRACE_FULL_CONTEXT:
        out["context_preview_hash"] = stable_hash(ctx_s[:2000])
    return out
