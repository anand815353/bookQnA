# app/schemas.py
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    session_id: str | None = None
    top_k: int = 4
    debug: bool = False

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool
    session_id: str | None = None


class QueryPlannerPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_type: Literal[
        "original",
        "follow_up",
        "ambiguous",
        "keyword_lookup",
        "chapter_lookup",
    ] = Field(
        default="original",
        description="Planner classification for retrieval handling only.",
    )
    standalone_question: str = Field(
        min_length=1,
        description="Standalone retrieval question. Must not answer the user.",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Short alternate retrieval queries for dense or lexical search.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="High-value lexical retrieval keywords only.",
    )
    should_expand: bool = Field(
        default=False,
        description="Whether retrieval should broaden slightly for ambiguity or follow-up context.",
    )
    needs_exact_phrase_bias: bool = Field(
        default=False,
        description="Whether retrieval should prefer exact phrase or quoted-term matches.",
    )
    needs_chapter_lookup: bool = Field(
        default=False,
        description="Whether the question likely needs chapter or section targeting.",
    )
    reason: str = Field(
        min_length=1,
        description="Brief retrieval-planning rationale, not an answer.",
    )


class ChatSessionCreateRequest(BaseModel):
    title: str | None = None


class ChatSessionOut(BaseModel):
    id: str
    title: str
    summary_text: str | None = None
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None = None
    preview_text: str | None = None


class ChatMessageOut(BaseModel):
    id: str
    session_id: str
    role: str
    question_text: str | None = None
    answer_text: str | None = None
    grounded: bool | None = None
    selected_book_ids: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime


class ChatSessionDetailOut(BaseModel):
    session: ChatSessionOut
    messages: list[ChatMessageOut] = Field(default_factory=list)


class ChatQueryRequest(QueryRequest):
    pass


class ChatQueryResponse(QueryResponse):
    session_id: str
