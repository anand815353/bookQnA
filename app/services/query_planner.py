import logging
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.logging_config import log_structured, sanitize_for_debug
from app.schemas import QueryPlannerPlan
from app.settings import (
    ENABLE_QUERY_PLANNER,
    LOG_DEBUG_SNIPPET_CHARS,
    MAX_HISTORY_CHARS,
    QUERY_PLANNER_ENABLE_GATING,
    QUERY_PLANNER_FORCE_IN_DEBUG,
    QUERY_PLANNER_LOW_TEMP,
    QUERY_PLANNER_MAX_ALT_QUERIES,
    QUERY_PLANNER_MAX_KEYWORDS,
    QUERY_PLANNER_MODEL,
    QUERY_PLANNER_SHORT_QUERY_WORDS,
    QUERY_PLANNER_USE_BOOK_METADATA,
)

logger = logging.getLogger(__name__)
PLANNER_HISTORY_QUESTION_CHARS = 180
PLANNER_HISTORY_ANSWER_CHARS = 220
PLANNER_HISTORY_MAX_CHARS = min(MAX_HISTORY_CHARS, 900)
PLANNER_METADATA_ITEM_CHARS = 90
PLANNER_MAX_METADATA_ITEMS = 8
PLANNER_STANDALONE_QUESTION_MAX_CHARS = 320
PLANNER_REASON_MAX_CHARS = 180
PLANNER_FOLLOW_UP_MARKERS = {
    "next",
    "previous",
    "earlier",
    "later",
    "above",
    "below",
    "again",
    "too",
}
PLANNER_PRONOUN_MARKERS = {
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "they",
    "them",
    "their",
    "there",
}
PLANNER_VAGUE_SHORT_QUERY_MARKERS = {
    "topic",
    "thing",
    "part",
    "concept",
    "section",
    "chapter",
    "one",
    "ones",
}
PLANNER_ALIAS_PATTERNS = (
    "also called",
    "known as",
    "sometimes called",
    "another name for",
    "another term for",
    "aka",
    "alias",
)
PLANNER_CHAPTER_LOCATOR_PATTERNS = (
    re.compile(r"\b(which|what|where)\s+(chapter|section|part|appendix|subchapter)\b"),
    re.compile(r"\b(next|previous|earlier|later|this|that)\s+(chapter|section|part|appendix|subchapter)\b"),
)
PLANNER_MISSING_SUBJECT_PATTERNS = (
    re.compile(r"^\s*(?:and|also|then)\b"),
    re.compile(r"^\s*(?:what about|how about)\b"),
)
_PLANNER_EXACT_PHRASE_PATTERNS_LEGACY = (
    re.compile(r"[\"“”`][^\"“”`]{2,}[\"“”`]"),
    re.compile(r"\bexact phrase\b"),
    re.compile(r"\bverbatim\b"),
    re.compile(r"\bword[- ]for[- ]word\b"),
)
_PLANNER_PAGE_CONSTRAINT_PATTERNS_LEGACY = (
    re.compile(r"\bpages?\s+\d+(?:\s*[-–]\s*\d+)?\b"),
    re.compile(r"\bp\.?\s*\d+\b"),
)
PLANNER_EXACT_PHRASE_PATTERNS = (
    re.compile(r"[\"`][^\"`]{2,}[\"`]"),
    re.compile(r"\bexact phrase\b"),
    re.compile(r"\bverbatim\b"),
    re.compile(r"\bword[- ]for[- ]word\b"),
)
PLANNER_PAGE_CONSTRAINT_PATTERNS = (
    re.compile(r"\bpages?\s+\d+(?:\s*-\s*\d+)?\b"),
    re.compile(r"\bp\.?\s*\d+\b"),
)
PLANNER_TERM_CONSTRAINED_PATTERNS = (
    re.compile(r"^\s*(?:define|definition of|meaning of)\s+.+\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:what is|what's)\s+.+\??\s*$", re.IGNORECASE),
    re.compile(
        r"^\s*where is\s+.+\s+(?:mentioned|defined|introduced|documented|explained)\s*\??\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:where|which page)\s+(?:is\s+)?.+\s+(?:mentioned|defined|introduced|documented)\s*\??\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:find|locate|show)\s+(?:the\s+)?(?:term|acronym|phrase|api|function|method|class|heading|title)\b.+$",
        re.IGNORECASE,
    ),
)
PLANNER_API_LOOKUP_PATTERNS = (
    re.compile(
        r"\b(?:api|function|method|class|endpoint|parameter|flag|argument)\s+[`\"']?[A-Za-z_][A-Za-z0-9_./:-]{1,80}[`\"']?",
        re.IGNORECASE,
    ),
    re.compile(r"\b[A-Z]{2,}[A-Z0-9_/-]{0,20}\b"),
)
PLANNER_REWRITE_SUPPRESSION_REASONS = frozenset(
    {
        "exact_phrase_lookup",
        "page_constraint",
        "term_constrained_lookup",
    }
)


@dataclass(frozen=True)
class QueryPlannerUsageDecision:
    should_use: bool
    forced: bool
    reason: str
    signals: tuple[str, ...] = ()
    suppressions: tuple[str, ...] = ()
    allow_legacy_rewrite: bool = True


def serialize_query_planner_usage_decision(decision: QueryPlannerUsageDecision | None) -> dict | None:
    if decision is None:
        return None
    payload = asdict(decision)
    payload["signals"] = list(decision.signals)
    payload["suppressions"] = list(decision.suppressions)
    return payload


def _tokenize_question(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _is_term_constrained_query(
    question: str,
    *,
    tokens: list[str],
    normalized_question: str,
    has_chapter_locator: bool,
) -> bool:
    if has_chapter_locator:
        return False

    token_count = len(tokens)
    raw_question = (question or "").strip()
    if not raw_question:
        return False

    if token_count <= 8 and any(pattern.match(raw_question) for pattern in PLANNER_TERM_CONSTRAINED_PATTERNS):
        return True

    if token_count <= 10 and any(pattern.search(raw_question) for pattern in PLANNER_API_LOOKUP_PATTERNS):
        return True

    constrained_markers = ("term", "acronym", "api", "function", "method", "class", "heading", "title")
    if token_count <= 10 and any(marker in normalized_question for marker in constrained_markers):
        return True
    return False


def _is_short_ambiguous_query(
    *,
    is_short_query: bool,
    has_recent_history: bool,
    pronoun_tokens: list[str],
    has_follow_up_marker: bool,
    has_chapter_locator: bool,
    has_missing_subject: bool,
    has_vague_short_reference: bool,
) -> bool:
    if not is_short_query:
        return False
    if has_chapter_locator:
        return True
    if has_follow_up_marker or has_missing_subject:
        return True
    if has_recent_history and (pronoun_tokens or has_vague_short_reference):
        return True
    if pronoun_tokens and has_follow_up_marker:
        return True
    return False


def should_use_query_planner(
    question: str,
    *,
    recent_history: list[dict[str, str]] | None = None,
    planner_enabled: bool | None = None,
    force_planner: bool = False,
) -> QueryPlannerUsageDecision:
    requested = ENABLE_QUERY_PLANNER if planner_enabled is None else planner_enabled
    tokens = _tokenize_question(question)
    token_count = len(tokens)
    normalized_question = " ".join(tokens)
    pronoun_tokens = [token for token in tokens if token in PLANNER_PRONOUN_MARKERS]
    has_recent_history = bool(recent_history)
    has_follow_up_marker = any(token in PLANNER_FOLLOW_UP_MARKERS for token in tokens)
    has_alias_language = any(pattern in normalized_question for pattern in PLANNER_ALIAS_PATTERNS)
    has_chapter_locator = any(pattern.search(normalized_question) for pattern in PLANNER_CHAPTER_LOCATOR_PATTERNS)
    has_missing_subject = any(pattern.search(question or "") for pattern in PLANNER_MISSING_SUBJECT_PATTERNS)
    has_vague_short_reference = any(token in PLANNER_VAGUE_SHORT_QUERY_MARKERS for token in tokens)
    has_exact_phrase_bias = any(pattern.search(question or "") for pattern in PLANNER_EXACT_PHRASE_PATTERNS)
    has_page_constraint = any(pattern.search(question or "") for pattern in PLANNER_PAGE_CONSTRAINT_PATTERNS)
    has_term_constraint = _is_term_constrained_query(
        question,
        tokens=tokens,
        normalized_question=normalized_question,
        has_chapter_locator=has_chapter_locator,
    )
    is_short_query = token_count > 0 and token_count <= QUERY_PLANNER_SHORT_QUERY_WORDS
    is_short_ambiguous_query = _is_short_ambiguous_query(
        is_short_query=is_short_query,
        has_recent_history=has_recent_history,
        pronoun_tokens=pronoun_tokens,
        has_follow_up_marker=has_follow_up_marker,
        has_chapter_locator=has_chapter_locator,
        has_missing_subject=has_missing_subject,
        has_vague_short_reference=has_vague_short_reference,
    )

    signals: list[str] = []
    suppressions: list[str] = []
    if has_recent_history:
        signals.append("recent_history")
    if pronoun_tokens:
        signals.append("pronoun_reference")
    if has_follow_up_marker:
        signals.append("follow_up_marker")
    if has_alias_language:
        signals.append("alias_language")
    if has_chapter_locator:
        signals.append("chapter_locator")
    if has_missing_subject:
        signals.append("missing_subject")
    if has_vague_short_reference:
        signals.append("vague_short_reference")
    if is_short_query:
        signals.append("short_query_candidate")
    if is_short_ambiguous_query:
        signals.append("short_query_ambiguous")

    if has_exact_phrase_bias:
        suppressions.append("exact_phrase_lookup")
    if has_page_constraint:
        suppressions.append("page_constraint")
    if has_term_constraint:
        suppressions.append("term_constrained_lookup")

    allow_legacy_rewrite = not any(reason in PLANNER_REWRITE_SUPPRESSION_REASONS for reason in suppressions)

    if force_planner:
        decision = QueryPlannerUsageDecision(
            should_use=True,
            forced=True,
            reason="forced_query_planner",
            signals=tuple(signals or ("forced",)),
            suppressions=tuple(suppressions),
            allow_legacy_rewrite=True,
        )
    elif not requested:
        decision = QueryPlannerUsageDecision(
            should_use=False,
            forced=False,
            reason="planner_disabled",
            signals=tuple(signals),
            suppressions=tuple(suppressions),
            allow_legacy_rewrite=allow_legacy_rewrite,
        )
    elif not QUERY_PLANNER_ENABLE_GATING:
        decision = QueryPlannerUsageDecision(
            should_use=True,
            forced=False,
            reason="planner_gating_disabled",
            signals=tuple(signals or ("feature_enabled",)),
            suppressions=tuple(suppressions),
            allow_legacy_rewrite=allow_legacy_rewrite,
        )
    elif suppressions:
        decision = QueryPlannerUsageDecision(
            should_use=False,
            forced=False,
            reason=suppressions[0],
            signals=tuple(signals),
            suppressions=tuple(suppressions),
            allow_legacy_rewrite=allow_legacy_rewrite,
        )
    elif has_alias_language:
        decision = QueryPlannerUsageDecision(
            should_use=True,
            forced=False,
            reason="alias_or_synonym_query",
            signals=tuple(signals),
            suppressions=tuple(),
            allow_legacy_rewrite=True,
        )
    elif has_chapter_locator:
        decision = QueryPlannerUsageDecision(
            should_use=True,
            forced=False,
            reason="chapter_locator_follow_up" if (has_recent_history or has_follow_up_marker or pronoun_tokens) else "chapter_locator_query",
            signals=tuple(signals),
            suppressions=tuple(),
            allow_legacy_rewrite=True,
        )
    elif has_recent_history and (pronoun_tokens or has_follow_up_marker):
        decision = QueryPlannerUsageDecision(
            should_use=True,
            forced=False,
            reason="follow_up_with_context",
            signals=tuple(signals),
            suppressions=tuple(),
            allow_legacy_rewrite=True,
        )
    elif is_short_ambiguous_query:
        decision = QueryPlannerUsageDecision(
            should_use=True,
            forced=False,
            reason="short_ambiguous_query",
            signals=tuple(signals),
            suppressions=tuple(),
            allow_legacy_rewrite=True,
        )
    elif pronoun_tokens and has_recent_history:
        decision = QueryPlannerUsageDecision(
            should_use=True,
            forced=False,
            reason="pronoun_heavy_follow_up",
            signals=tuple(signals),
            suppressions=tuple(),
            allow_legacy_rewrite=True,
        )
    else:
        decision = QueryPlannerUsageDecision(
            should_use=False,
            forced=False,
            reason="clear_specific_query",
            signals=tuple(signals),
            suppressions=tuple(),
            allow_legacy_rewrite=True,
        )

    log_structured(
        logger,
        "query_planner_gate_decision",
        planner_requested=requested,
        planner_enabled=requested or force_planner,
        planner_forced=decision.forced,
        planner_force_debug_enabled=QUERY_PLANNER_FORCE_IN_DEBUG,
        planner_gating_enabled=QUERY_PLANNER_ENABLE_GATING,
        should_use=decision.should_use,
        reason=decision.reason,
        signals=list(decision.signals),
        suppressions=list(decision.suppressions),
        allow_legacy_rewrite=decision.allow_legacy_rewrite,
        short_query_candidate=is_short_query,
        short_query_ambiguous=is_short_ambiguous_query,
        recent_history_turns=len(recent_history or []),
        question_len=len(question or ""),
        question_snippet=sanitize_for_debug(question, LOG_DEBUG_SNIPPET_CHARS),
    )
    return decision


def _log_planner_result(
    event: str,
    *,
    question: str,
    planner_enabled: bool,
    plan: QueryPlannerPlan,
    planner_latency_ms: int | None = None,
    level: int = logging.INFO,
    **extra_fields: object,
) -> None:
    fields: dict[str, object] = {
        "planner_enabled": planner_enabled,
        "original_question": sanitize_for_debug(question, LOG_DEBUG_SNIPPET_CHARS),
        "question_len": len(question or ""),
        "query_type": plan.query_type,
        "standalone_question": sanitize_for_debug(plan.standalone_question, LOG_DEBUG_SNIPPET_CHARS),
        "search_queries": plan.search_queries,
        "keywords": plan.keywords,
        "should_expand": plan.should_expand,
        "needs_exact_phrase_bias": plan.needs_exact_phrase_bias,
        "needs_chapter_lookup": plan.needs_chapter_lookup,
        "reason": plan.reason,
    }
    if planner_latency_ms is not None:
        fields["planner_latency_ms"] = planner_latency_ms
    fields.update(extra_fields)
    log_structured(
        logger,
        event,
        level=level,
        **fields,
    )


def _compact_text(text: str, max_chars: int) -> str:
    compacted = re.sub(r"\s+", " ", (text or "").strip())
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max(0, max_chars - 3)].rstrip() + "..."


def _bounded_unique_items(values: Sequence[str] | None, *, limit: int) -> list[str]:
    if not values or limit <= 0:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = (value or "").strip()
        if not item or item in seen:
            continue
        cleaned.append(item)
        seen.add(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def _resolve_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def _build_planner_llm(api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=QUERY_PLANNER_MODEL,
        temperature=QUERY_PLANNER_LOW_TEMP,
        google_api_key=api_key,
    )


def _build_recent_history_text(recent_history: list[dict[str, str]] | None) -> tuple[str, int]:
    if not recent_history:
        return "(none)", 0

    blocks: list[str] = []
    used_turns = 0
    total_chars = 0

    for idx, turn in enumerate(recent_history, start=1):
        question = _compact_text(turn.get("question", ""), PLANNER_HISTORY_QUESTION_CHARS)
        answer = _compact_text(turn.get("answer", ""), PLANNER_HISTORY_ANSWER_CHARS)
        if not question and not answer:
            continue
        block = (
            f"Turn {idx}\n"
            f"User: {question or '(none)'}\n"
            f"Assistant: {answer or '(none)'}"
        )
        projected = total_chars + (2 if blocks else 0) + len(block)
        if projected > PLANNER_HISTORY_MAX_CHARS:
            break
        blocks.append(block)
        total_chars = projected
        used_turns += 1

    if not blocks:
        return "(none)", 0
    return "\n\n".join(blocks), used_turns


def _build_book_metadata_text(
    *,
    book_title: str | None,
    toc_headings: Sequence[str] | None,
    chapter_titles: Sequence[str] | None,
) -> tuple[str, int]:
    if not QUERY_PLANNER_USE_BOOK_METADATA:
        return "(disabled)", 0

    blocks: list[str] = []
    heading_count = 0

    compact_title = _compact_text(book_title or "", PLANNER_METADATA_ITEM_CHARS)
    if compact_title:
        blocks.append(f"Book title: {compact_title}")

    heading_candidates = list(toc_headings or []) + list(chapter_titles or [])
    headings = [
        _compact_text(item, PLANNER_METADATA_ITEM_CHARS)
        for item in _bounded_unique_items(
            heading_candidates,
            limit=PLANNER_MAX_METADATA_ITEMS,
        )
    ]
    headings = [item for item in headings if item]
    if headings:
        heading_count = len(headings)
        blocks.append(
            "TOC/chapter hints:\n" + "\n".join(f"- {item}" for item in headings)
        )

    if not blocks:
        return "(none)", 0
    return "\n\n".join(blocks), heading_count


def _build_planner_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a retrieval query planner for a book question-answering system. "
                "You are not the answering model. "
                "Return only structured retrieval-planning output that matches the schema. "
                "Do not answer the user's question. "
                "Do not invent facts, pages, chapter names, section names, or book content. "
                "Only rewrite or restructure the user's request for retrieval. "
                "Use recent conversation only to resolve references in follow-up questions. "
                "Use provided book metadata only as light retrieval hints; if metadata is absent or uncertain, do not guess. "
                "If the original question is already clear and specific, keep the standalone question very close to it and avoid broadening. "
                "Search queries must be short retrieval queries, not answers. "
                "Keywords must be terse lexical search terms, not sentences. "
                "The reason must briefly explain the retrieval planning choice and must not answer the question."
            ),
            (
                "human",
                "Return a JSON object that matches the schema exactly.\n"
                "Limits:\n"
                f"- search_queries: at most {QUERY_PLANNER_MAX_ALT_QUERIES}\n"
                f"- keywords: at most {QUERY_PLANNER_MAX_KEYWORDS}\n\n"
                "Original question:\n{question}\n\n"
                "Recent conversation for reference resolution only:\n{recent_history}\n\n"
                "Book metadata for retrieval hints only:\n{book_metadata}"
            ),
        ]
    )


def _build_planner_chain(api_key: str):
    llm = _build_planner_llm(api_key)
    structured_llm = llm.with_structured_output(
        QueryPlannerPlan,
        method="json_schema",
        include_raw=True,
    )
    return _build_planner_prompt() | structured_llm


def _build_noop_plan(question: str, *, reason: str) -> QueryPlannerPlan:
    return QueryPlannerPlan(
        query_type="original",
        standalone_question=question,
        search_queries=[],
        keywords=[],
        should_expand=False,
        needs_exact_phrase_bias=False,
        needs_chapter_lookup=False,
        reason=reason,
    )


def _normalize_plan(
    plan: QueryPlannerPlan,
    *,
    original_question: str,
) -> QueryPlannerPlan:
    standalone_question = _compact_text(
        plan.standalone_question or original_question,
        PLANNER_STANDALONE_QUESTION_MAX_CHARS,
    )
    if not standalone_question:
        standalone_question = original_question

    search_queries = _bounded_unique_items(
        plan.search_queries,
        limit=QUERY_PLANNER_MAX_ALT_QUERIES,
    )
    search_queries = [
        _compact_text(item, PLANNER_STANDALONE_QUESTION_MAX_CHARS)
        for item in search_queries
        if item.strip() != standalone_question.strip()
    ]
    keywords = [
        _compact_text(item, PLANNER_METADATA_ITEM_CHARS)
        for item in _bounded_unique_items(
            plan.keywords,
            limit=QUERY_PLANNER_MAX_KEYWORDS,
        )
    ]
    reason = _compact_text(plan.reason or "retrieval_planning_applied", PLANNER_REASON_MAX_CHARS)

    return plan.model_copy(
        update={
            "standalone_question": standalone_question,
            "search_queries": search_queries,
            "keywords": keywords,
            "reason": reason,
        }
    )


def plan_retrieval_query(
    question: str,
    *,
    recent_history: list[dict[str, str]] | None = None,
    book_title: str | None = None,
    toc_headings: Sequence[str] | None = None,
    chapter_titles: Sequence[str] | None = None,
    planner_enabled: bool | None = None,
) -> QueryPlannerPlan:
    is_enabled = ENABLE_QUERY_PLANNER if planner_enabled is None else planner_enabled
    if not is_enabled:
        plan = _build_noop_plan(question, reason="planner_disabled_noop")
        _log_planner_result(
            "query_planner_skipped",
            question=question,
            planner_enabled=False,
            plan=plan,
        )
        return plan

    api_key = _resolve_api_key()
    if not api_key:
        plan = _build_noop_plan(question, reason="planner_missing_api_key_noop")
        _log_planner_result(
            "query_planner_skipped",
            question=question,
            planner_enabled=True,
            plan=plan,
        )
        return plan

    recent_history_text, history_turns_used = _build_recent_history_text(recent_history)
    book_metadata_text, metadata_headings_used = _build_book_metadata_text(
        book_title=book_title,
        toc_headings=toc_headings,
        chapter_titles=chapter_titles,
    )

    log_structured(
        logger,
        "query_planner_started",
        question_len=len(question or ""),
        question_snippet=sanitize_for_debug(question, LOG_DEBUG_SNIPPET_CHARS),
        recent_history_turns=history_turns_used,
        metadata_headings_used=metadata_headings_used,
        metadata_enabled=QUERY_PLANNER_USE_BOOK_METADATA,
    )

    try:
        chain = _build_planner_chain(api_key)
    except Exception:
        logger.exception("query_planner_chain_build_failed")
        plan = _build_noop_plan(question, reason="planner_chain_build_failed_noop")
        _log_planner_result(
            "query_planner_fallback",
            question=question,
            planner_enabled=True,
            plan=plan,
            level=logging.WARNING,
            failure_stage="chain_build",
        )
        return plan

    started = time.perf_counter()
    try:
        result = chain.invoke(
            {
                "question": question,
                "recent_history": recent_history_text,
                "book_metadata": book_metadata_text,
            }
        )
    except Exception:
        logger.exception("query_planner_invoke_failed")
        plan = _build_noop_plan(question, reason="planner_invoke_failed_noop")
        _log_planner_result(
            "query_planner_fallback",
            question=question,
            planner_enabled=True,
            plan=plan,
            level=logging.WARNING,
            failure_stage="invoke",
        )
        return plan

    planner_ms = int((time.perf_counter() - started) * 1000)
    parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
    parsed = result.get("parsed") if isinstance(result, dict) else result

    if parsing_error is not None or parsed is None:
        plan = _build_noop_plan(question, reason="planner_parse_failed_noop")
        _log_planner_result(
            "query_planner_fallback",
            question=question,
            planner_enabled=True,
            plan=plan,
            planner_latency_ms=planner_ms,
            level=logging.WARNING,
            failure_stage="parse",
            parsing_error=str(parsing_error) if parsing_error is not None else "missing_parsed_output",
        )
        return plan

    try:
        validated = QueryPlannerPlan.model_validate(parsed)
        normalized = _normalize_plan(validated, original_question=question)
    except Exception:
        logger.exception("query_planner_validation_failed")
        plan = _build_noop_plan(question, reason="planner_invalid_output_noop")
        _log_planner_result(
            "query_planner_fallback",
            question=question,
            planner_enabled=True,
            plan=plan,
            planner_latency_ms=planner_ms,
            level=logging.WARNING,
            failure_stage="validation",
        )
        return plan

    _log_planner_result(
        "query_planner_completed",
        question=question,
        planner_enabled=True,
        plan=normalized,
        planner_latency_ms=planner_ms,
    )
    return normalized
