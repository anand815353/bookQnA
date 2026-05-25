# app/services/ingest.py
import logging
import time
from datetime import datetime
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm.exc import StaleDataError
from app.db import SessionLocal, engine
from app.models import Book, BookSection
from app.services.lexical_index import index_chunk_rows
from app.services.cache import invalidate_book
from app.services.retrieval import get_vectorstore, delete_book_vectors
from app.services.structure import (
    build_parent_sections,
    split_parent_sections_into_child_chunks,
)
from app.settings import (
    EMBEDDING_BACKOFF_BASE_SECONDS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_BATCH_SLEEP_SECONDS,
    EMBEDDING_MAX_RETRIES,
)

logger = logging.getLogger(__name__)


def _extract_retry_seconds(message: str, default: int = 60) -> int:
    import re

    match = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", message, flags=re.IGNORECASE)
    if match:
        return max(1, int(float(match.group(1))))
    return default


def _friendly_ingest_error(message: str) -> str:
    if "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in message or "PermissionDenied: 403" in message:
        return (
            "Embedding failed: API key/auth scope issue. "
            "Set GOOGLE_API_KEY (or GEMINI_API_KEY) to a valid Gemini API key."
        )
    if "API key required" in message:
        return "Embedding failed: missing API key. Set GOOGLE_API_KEY (or GEMINI_API_KEY)."
    if "quota" in message.lower() or "429" in message:
        return (
            "Embedding failed due to Gemini rate limit/quota. "
            "Wait a minute and retry ingestion, or use a smaller PDF/chunk settings."
        )
    return message


def _is_transient_embedding_error(message: str) -> bool:
    lowered = message.lower()
    transient_markers = [
        "quota",
        "429",
        "rate limit",
        "resource exhausted",
        "deadline exceeded",
        "timeout",
        "temporarily unavailable",
        "internal",
        "503",
    ]
    return any(marker in lowered for marker in transient_markers)


def _refresh_ingest_target(db, book_id: str) -> Book | None:
    book = db.get(Book, book_id)
    if not book:
        logger.info("ingestion_cancelled_missing_book book_id=%s", book_id)
        return None
    if book.status == "deleting":
        logger.info("ingestion_cancelled_delete_requested book_id=%s", book_id)
        return None
    return book


def _replace_book_sections(db, *, book_id: str, sections) -> None:
    db.query(BookSection).filter(BookSection.book_id == book_id).delete(synchronize_session=False)
    for section in sections:
        db.add(BookSection(**section.to_record()))


def update_ingestion_progress(
    db,
    book: Book,
    *,
    status: str | None = None,
    current_step: str | None = None,
    processed_chunks: int | None = None,
    failed_chunks: int | None = None,
    total_chunks: int | None = None,
    last_processed_chunk_index: int | None = None,
    error_message: str | None = None,
    finished: bool = False,
) -> bool:
    if status is not None:
        book.status = status
    if current_step is not None:
        book.current_step = current_step
    if processed_chunks is not None:
        book.processed_chunks = max(0, processed_chunks)
    if failed_chunks is not None:
        book.failed_chunks = max(0, failed_chunks)
    if total_chunks is not None:
        book.total_chunks = max(0, total_chunks)
    if last_processed_chunk_index is not None:
        book.last_processed_chunk_index = last_processed_chunk_index

    total = book.total_chunks or 0
    if total > 0:
        book.progress_percent = min(100, int((book.processed_chunks / total) * 100))
    else:
        book.progress_percent = 0

    book.error_message = error_message
    if finished:
        book.finished_at = datetime.utcnow()
    book.updated_at = datetime.utcnow()
    try:
        db.commit()
        return True
    except StaleDataError:
        db.rollback()
        logger.info("ingestion_progress_skip_stale_book book_id=%s", getattr(book, "id", "unknown"))
        return False


def embed_and_store_chunks_in_batches(
    *,
    db,
    book: Book,
    chunks,
    ids: list[str],
    vectorstore,
) -> None:
    total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    processed_chunks = 0
    failed_chunks = 0

    for batch_number, start in enumerate(range(0, len(chunks), EMBEDDING_BATCH_SIZE), start=1):
        current_book = _refresh_ingest_target(db, book.id)
        if not current_book:
            return
        book = current_book
        end = min(start + EMBEDDING_BATCH_SIZE, len(chunks))
        batch_docs = chunks[start:end]
        batch_ids = ids[start:end]
        attempts = 0

        while True:
            current_book = _refresh_ingest_target(db, book.id)
            if not current_book:
                return
            book = current_book
            attempts += 1
            step = f"embedding batch {batch_number}/{total_batches}"
            update_ingestion_progress(
                db,
                book,
                status="embedding",
                current_step=step,
                processed_chunks=processed_chunks,
                failed_chunks=failed_chunks,
                last_processed_chunk_index=max(processed_chunks - 1, -1),
            )
            try:
                vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
                processed_chunks += len(batch_docs)
                update_ingestion_progress(
                    db,
                    book,
                    status="embedding",
                    current_step=step,
                    processed_chunks=processed_chunks,
                    failed_chunks=failed_chunks,
                    last_processed_chunk_index=processed_chunks - 1,
                )
                logger.info(
                    "embedding_batch_success book_id=%s title=%s batch=%s/%s processed=%s total=%s",
                    book.id,
                    book.title,
                    batch_number,
                    total_batches,
                    processed_chunks,
                    len(chunks),
                )
                break
            except Exception as exc:
                message = str(exc)
                is_transient = _is_transient_embedding_error(message)
                logger.warning(
                    "embedding_batch_failure book_id=%s title=%s batch=%s/%s retry=%s/%s transient=%s error=%s",
                    book.id,
                    book.title,
                    batch_number,
                    total_batches,
                    attempts,
                    EMBEDDING_MAX_RETRIES,
                    is_transient,
                    message,
                )

                if not is_transient or attempts >= EMBEDDING_MAX_RETRIES:
                    failed_chunks += len(batch_docs)
                    update_ingestion_progress(
                        db,
                        book,
                        status="failed",
                        current_step=f"failed on embedding batch {batch_number}/{total_batches}",
                        processed_chunks=processed_chunks,
                        failed_chunks=failed_chunks,
                        error_message=_friendly_ingest_error(message),
                        finished=True,
                    )
                    raise

                retry_hint_seconds = _extract_retry_seconds(
                    message, default=int(EMBEDDING_BACKOFF_BASE_SECONDS)
                )
                backoff_seconds = max(
                    EMBEDDING_BACKOFF_BASE_SECONDS * (2 ** (attempts - 1)),
                    float(retry_hint_seconds),
                )
                logger.info(
                    "embedding_backoff_sleep book_id=%s title=%s batch=%s/%s retry=%s sleep_seconds=%.2f",
                    book.id,
                    book.title,
                    batch_number,
                    total_batches,
                    attempts,
                    backoff_seconds,
                )
                time.sleep(backoff_seconds)

        if batch_number < total_batches and EMBEDDING_BATCH_SLEEP_SECONDS > 0:
            logger.info(
                "embedding_batch_throttle book_id=%s title=%s batch=%s/%s sleep_seconds=%.2f",
                book.id,
                book.title,
                batch_number,
                total_batches,
                EMBEDDING_BATCH_SLEEP_SECONDS,
            )
            time.sleep(EMBEDDING_BATCH_SLEEP_SECONDS)


def ingest_book(book_id: str):
    db = SessionLocal()
    ingest_started = time.perf_counter()
    try:
        book = _refresh_ingest_target(db, book_id)
        if not book:
            return

        logger.info("ingestion_started book_id=%s title=%s", book.id, book.title)
        book.status = "parsing"
        book.current_step = "loading PDF pages"
        book.error_message = None
        book.progress_percent = 0
        book.failed_chunks = 0
        book.processed_chunks = 0
        book.total_chunks = 0
        book.total_sections = None
        book.last_processed_chunk_index = None
        book.started_at = datetime.utcnow()
        book.finished_at = None
        book.updated_at = datetime.utcnow()
        db.commit()

        book = _refresh_ingest_target(db, book_id)
        if not book:
            return

        # Reindex should replace stale vectors for the same book.
        delete_book_vectors(book.id)
        invalidate_book(book.id)
        db.query(BookSection).filter(BookSection.book_id == book.id).delete(synchronize_session=False)
        db.commit()

        book = _refresh_ingest_target(db, book_id)
        if not book:
            return

        file_path = Path(book.file_path)
        if not file_path.exists():
            update_ingestion_progress(
                db,
                book,
                status="failed",
                current_step="failed: source file missing",
                error_message="PDF file is missing; it may have been deleted during ingestion.",
                finished=True,
            )
            return

        parse_started = time.perf_counter()
        loader = PyMuPDFLoader(str(file_path))
        pages = loader.load()
        logger.info(
            "ingestion_parse_completed book_id=%s pages=%s duration_ms=%s",
            book.id,
            len(pages),
            int((time.perf_counter() - parse_started) * 1000),
        )

        if not pages:
            update_ingestion_progress(
                db,
                book,
                status="failed",
                current_step="failed: no pages extracted",
                error_message="No pages extracted from PDF.",
                finished=True,
            )
            return

        # enrich metadata
        update_ingestion_progress(
            db, book, status="structuring", current_step="structuring page metadata"
        )
        struct_started = time.perf_counter()
        for doc in pages:
            doc.metadata["book_id"] = book.id
            doc.metadata["book_title"] = book.title
            # page is often 0-indexed in loaders; normalize to 1-indexed for UI/citation
            page_no = int(doc.metadata.get("page", 0)) + 1
            doc.metadata["page_start"] = page_no
            doc.metadata["page_end"] = page_no
            doc.metadata["source_file"] = book.file_name

        update_ingestion_progress(
            db,
            book,
            status="structuring",
            current_step="building parent sections",
        )
        parent_sections = build_parent_sections(
            book_id=book.id,
            file_path=file_path,
            pages=pages,
        )
        _replace_book_sections(db, book_id=book.id, sections=parent_sections)

        book.total_pages = len(pages)
        book.total_sections = len(parent_sections)
        book.updated_at = datetime.utcnow()
        db.commit()
        logger.info(
            "ingestion_structuring_completed book_id=%s sections=%s duration_ms=%s",
            book.id,
            len(parent_sections),
            int((time.perf_counter() - struct_started) * 1000),
        )

        book = _refresh_ingest_target(db, book_id)
        if not book:
            return

        update_ingestion_progress(db, book, status="chunking", current_step="creating chunks")
        chunk_started = time.perf_counter()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
        chunks = split_parent_sections_into_child_chunks(
            pages=pages,
            parent_sections=parent_sections,
            splitter=splitter,
        )
        book.total_chunks = len(chunks)
        book.processed_chunks = 0
        book.failed_chunks = 0
        book.progress_percent = 0
        book.current_step = "chunking completed"
        book.updated_at = datetime.utcnow()
        db.commit()
        logger.info(
            "ingestion_chunking_completed book_id=%s chunks=%s duration_ms=%s",
            book.id,
            len(chunks),
            int((time.perf_counter() - chunk_started) * 1000),
        )

        book = _refresh_ingest_target(db, book_id)
        if not book:
            return

        ids = []
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["parent_section_id"] = chunk.metadata.get("section_id")
            ids.append(f"{book.id}::chunk::{i}")

        vectorstore = get_vectorstore()
        logger.info(
            "embedding_started book_id=%s title=%s total_chunks=%s batch_size=%s",
            book.id,
            book.title,
            len(chunks),
            EMBEDDING_BATCH_SIZE,
        )
        embed_and_store_chunks_in_batches(
            db=db,
            book=book,
            chunks=chunks,
            ids=ids,
            vectorstore=vectorstore,
        )

        def _fts_body_for_chunk(chunk) -> str:
            contextual = str(chunk.metadata.get("contextual_text", "") or "").strip()
            if contextual:
                return f"{contextual}\n{chunk.page_content or ''}"
            return chunk.page_content or ""

        fts_rows: list[dict] = []
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata
            page_start = int(meta.get("page_start", 0) or 0)
            raw_pe = meta.get("page_end")
            if raw_pe is None or str(raw_pe).strip() == "":
                page_end = page_start
            else:
                page_end = int(raw_pe)
            if page_end <= 0:
                page_end = page_start
            fts_rows.append(
                {
                    "chunk_id": ids[i],
                    "book_id": book.id,
                    "page_start": page_start,
                    "page_end": page_end,
                    "section_id": str(meta.get("section_id", "") or ""),
                    "chapter_title": str(meta.get("chapter_title", "") or ""),
                    "body": _fts_body_for_chunk(chunk),
                }
            )
        fts_indexed = index_chunk_rows(engine, book.id, fts_rows)
        logger.info(
            "ingestion_lexical_fts_indexed book_id=%s chunks_indexed=%s",
            book.id,
            fts_indexed,
        )

        book = _refresh_ingest_target(db, book_id)
        if not book:
            return

        book.status = "indexed"
        book.current_step = "indexing completed"
        book.progress_percent = 100
        book.error_message = None
        book.finished_at = datetime.utcnow()
        book.updated_at = datetime.utcnow()
        db.commit()
        logger.info(
            "ingestion_completed book_id=%s title=%s total_chunks=%s processed=%s failed=%s duration_ms=%s",
            book.id,
            book.title,
            book.total_chunks,
            book.processed_chunks,
            book.failed_chunks,
            int((time.perf_counter() - ingest_started) * 1000),
        )

    except Exception as e:
        book = db.get(Book, book_id)
        if book and book.status != "deleting":
            update_ingestion_progress(
                db,
                book,
                status="failed",
                current_step=book.current_step or "failed",
                error_message=_friendly_ingest_error(str(e)),
                finished=True,
            )
            logger.exception(
                "ingestion_failed book_id=%s title=%s error=%s",
                book.id,
                book.title,
                str(e),
            )
        return
    finally:
        db.close()
