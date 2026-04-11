# app/api/query.py
from fastapi import APIRouter
from app.schemas import QueryRequest, QueryResponse
from app.services.qa import answer_question

router = APIRouter(tags=["query"])

@router.post("/query", response_model=QueryResponse)
def query_books(request: QueryRequest):
    result = answer_question(
        question=request.question,
        book_ids=request.book_ids,
        top_k=request.top_k
    )
    return QueryResponse(**result)