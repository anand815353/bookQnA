# app/api/query.py
import logging
import time

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from app.logging_config import log_structured, sanitize_for_debug
from app.schemas import QueryRequest, QueryResponse
from app.services.qa import answer_question
from app.settings import LOG_DEBUG_SNIPPET_CHARS, QUERY_DEBUG_ENABLED, RETRIEVAL_MODE

router = APIRouter(tags=["query"])
logger = logging.getLogger(__name__)

@router.post("/query", response_model=QueryResponse)
def query_books(request: QueryRequest):
    started = time.perf_counter()
    debug_enabled = bool(request.debug and QUERY_DEBUG_ENABLED)
    log_structured(
        logger,
        "query_requested",
        question_len=len(request.question or ""),
        question_snippet=sanitize_for_debug(request.question, LOG_DEBUG_SNIPPET_CHARS),
        selected_books=request.book_ids or [],
        top_k=request.top_k,
        retrieval_mode=RETRIEVAL_MODE,
        debug_requested=bool(request.debug),
        debug_enabled=debug_enabled,
    )
    try:
        result = answer_question(
            question=request.question,
            book_ids=request.book_ids,
            top_k=request.top_k,
            **({"include_debug": True} if debug_enabled else {}),
        )
    except Exception:
        log_structured(
            logger,
            "query_failed",
            top_k=request.top_k,
            book_filter_count=len(request.book_ids or []),
            selected_books=request.book_ids or [],
            retrieval_mode=RETRIEVAL_MODE,
        )
        logger.exception("query_failed_exception")
        raise
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    citations = result.get("citations", []) or []
    chosen_citations = [
        f"{c.get('book_id', '')}:p{c.get('page_start', 0)}-{c.get('page_end', 0)}"
        for c in citations
    ]
    log_structured(
        logger,
        "query_completed",
        grounded=result.get("grounded"),
        citations=len(citations),
        chosen_citations=chosen_citations,
        duration_ms=elapsed_ms,
        debug_included=bool(result.get("debug")),
    )

    if request.debug and not QUERY_DEBUG_ENABLED:
        logger.info("query_debug_ignored reason=debug_not_enabled")

    if debug_enabled:
        return JSONResponse(content=jsonable_encoder(result))
    return QueryResponse(**result)
