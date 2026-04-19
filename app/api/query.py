# app/api/query.py
import inspect
import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from app.db import SessionLocal
from app.logging_config import log_structured, sanitize_for_debug
from app.schemas import QueryRequest, QueryResponse
from app.services.chat_history import get_chat_session, get_recent_session_context, save_query_exchange
from app.services.qa import answer_question
from app.settings import (
    ENABLE_QUERY_PLANNER,
    ENABLE_QUERY_REFORMULATION,
    LOG_DEBUG_SNIPPET_CHARS,
    MAX_HISTORY_TURNS,
    QUERY_DEBUG_ENABLED,
    QUERY_PLANNER_FORCE_IN_DEBUG,
    RETRIEVAL_MODE,
)

router = APIRouter(tags=["query"])
logger = logging.getLogger(__name__)

@router.post("/query", response_model=QueryResponse)
def query_books(request: QueryRequest):
    started = time.perf_counter()
    debug_enabled = bool(request.debug and QUERY_DEBUG_ENABLED)
    recent_history: list[dict[str, str]] = []

    db = SessionLocal()
    try:
        if request.session_id:
            if not get_chat_session(db, request.session_id):
                raise HTTPException(status_code=404, detail="Chat session not found.")
            recent_history = get_recent_session_context(
                db=db,
                session_id=request.session_id,
                limit_turns=MAX_HISTORY_TURNS,
            )
    finally:
        db.close()

    log_structured(
        logger,
        "query_requested",
        question_len=len(request.question or ""),
        question_snippet=sanitize_for_debug(request.question, LOG_DEBUG_SNIPPET_CHARS),
        selected_books=request.book_ids or [],
        session_id=request.session_id,
        recent_history_turns=len(recent_history),
        top_k=request.top_k,
        retrieval_mode=RETRIEVAL_MODE,
        debug_requested=bool(request.debug),
        debug_enabled=debug_enabled,
        planner_enabled=ENABLE_QUERY_PLANNER,
        planner_force_in_debug=QUERY_PLANNER_FORCE_IN_DEBUG,
    )
    answer_signature = inspect.signature(answer_question).parameters
    answer_kwargs = {
        "question": request.question,
        "book_ids": request.book_ids,
        "top_k": request.top_k,
    }
    if "include_debug" in answer_signature:
        answer_kwargs["include_debug"] = debug_enabled
    if recent_history and "recent_history" in answer_signature:
        answer_kwargs["recent_history"] = recent_history
    if request.session_id and "enable_query_reformulation" in answer_signature:
        answer_kwargs["enable_query_reformulation"] = ENABLE_QUERY_REFORMULATION
    if "enable_query_planner" in answer_signature:
        answer_kwargs["enable_query_planner"] = ENABLE_QUERY_PLANNER
    if "force_query_planner" in answer_signature:
        answer_kwargs["force_query_planner"] = bool(debug_enabled and QUERY_PLANNER_FORCE_IN_DEBUG)

    try:
        result = answer_question(**answer_kwargs)
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
        session_id=request.session_id,
        grounded=result.get("grounded"),
        citations=len(citations),
        chosen_citations=chosen_citations,
        duration_ms=elapsed_ms,
        debug_included=bool(result.get("debug")),
    )

    db = SessionLocal()
    try:
        session, _, _ = save_query_exchange(
            db=db,
            session_id=request.session_id,
            question_text=request.question,
            answer_text=result.get("answer", ""),
            grounded=bool(result.get("grounded")),
            selected_book_ids=request.book_ids or [],
            citations=citations,
        )
    finally:
        db.close()
    result["session_id"] = session.id

    if request.debug and not QUERY_DEBUG_ENABLED:
        logger.info("query_debug_ignored reason=debug_not_enabled")

    if debug_enabled:
        return JSONResponse(content=jsonable_encoder(result))
    return QueryResponse(**result)
