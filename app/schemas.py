# app/schemas.py
from datetime import datetime
from pydantic import BaseModel

class BookOut(BaseModel):
    id: str
    title: str
    file_name: str
    status: str
    total_pages: int | None = None
    total_sections: int | None = None
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    progress_percent: int = 0
    current_step: str | None = None
    last_processed_chunk_index: int | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime | None = None


class BookStatusOut(BaseModel):
    book_id: str
    status: str
    total_pages: int | None = None
    total_sections: int | None = None
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    progress_percent: int = 0
    current_step: str | None = None
    last_processed_chunk_index: int | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime | None = None


class Citation(BaseModel):
    book_id: str
    book_title: str
    page_start: int
    page_end: int
    snippet: str
    chapter_title: str | None = None
    subchapter_title: str | None = None
    section_title: str | None = None
    page_category: str | None = None
    structure_source: str | None = None
    confidence: float | None = None

class QueryRequest(BaseModel):
    question: str
    book_ids: list[str] | None = None
    top_k: int = 4

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool
