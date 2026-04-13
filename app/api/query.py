# app/api/query.py
import logging
import time

from fastapi import APIRouter
from app.schemas import QueryRequest, QueryResponse
from app.services.qa import answer_question

router = APIRouter(tags=["query"])
logger = logging.getLogger(__name__)

@router.post("/query", response_model=QueryResponse)
def query_books(request: QueryRequest):
    started = time.perf_counter()
    logger.info(
        "query_requested question_len=%s top_k=%s book_filter_count=%s",
        len(request.question or ""),
        request.top_k,
        len(request.book_ids or []),
    )
    try:
        result = answer_question(
            question=request.question,
            book_ids=request.book_ids,
            top_k=request.top_k
        )
    except Exception:
        logger.exception(
            "query_failed top_k=%s book_filter_count=%s",
            request.top_k,
            len(request.book_ids or []),
        )
        raise
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "query_completed grounded=%s citations=%s duration_ms=%s",
        result.get("grounded"),
        len(result.get("citations", [])),
        elapsed_ms,
    )
    return QueryResponse(**result)