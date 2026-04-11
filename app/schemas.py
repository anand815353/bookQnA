# app/schemas.py
from pydantic import BaseModel

class BookOut(BaseModel):
    id: str
    title: str
    file_name: str
    status: str
    total_pages: int | None = None
    error_message: str | None = None

class Citation(BaseModel):
    book_id: str
    book_title: str
    page_start: int
    page_end: int
    snippet: str

class QueryRequest(BaseModel):
    question: str
    book_ids: list[str] | None = None
    top_k: int = 4

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool