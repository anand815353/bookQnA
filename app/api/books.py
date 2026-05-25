# app/api/books.py
import logging

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from app.db import SessionLocal
from app.models import Book, BookSection
from app.schemas import BookOut, BookStatusOut
from app.services.cache import invalidate_book
from app.services.storage import save_uploaded_pdf, delete_book_files
from app.services.ingest import ingest_book
from app.services.retrieval import delete_book_vectors

router = APIRouter(prefix="/books", tags=["books"])
logger = logging.getLogger(__name__)
INGEST_ACTIVE_STATUSES = {"parsing", "structuring", "chunking", "embedding"}


def _to_book_out(book: Book) -> BookOut:
    return BookOut(
        id=book.id,
        title=book.title,
        file_name=book.file_name,
        status=book.status,
        total_pages=book.total_pages,
        total_sections=book.total_sections,
        total_chunks=book.total_chunks,
        processed_chunks=book.processed_chunks,
        failed_chunks=book.failed_chunks,
        progress_percent=book.progress_percent,
        current_step=book.current_step,
        last_processed_chunk_index=book.last_processed_chunk_index,
        error_message=book.error_message,
        started_at=book.started_at,
        finished_at=book.finished_at,
        updated_at=book.updated_at,
    )


def _to_book_status_out(book: Book) -> BookStatusOut:
    return BookStatusOut(
        book_id=book.id,
        status=book.status,
        total_pages=book.total_pages,
        total_sections=book.total_sections,
        total_chunks=book.total_chunks,
        processed_chunks=book.processed_chunks,
        failed_chunks=book.failed_chunks,
        progress_percent=book.progress_percent,
        current_step=book.current_step,
        last_processed_chunk_index=book.last_processed_chunk_index,
        error_message=book.error_message,
        started_at=book.started_at,
        finished_at=book.finished_at,
        updated_at=book.updated_at,
    )


@router.post("/upload", response_model=BookOut)
def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    auto_ingest: bool = Form(True),
):
    logger.info("book_upload_requested file_name=%s auto_ingest=%s", file.filename, auto_ingest)
    if not file.filename.lower().endswith(".pdf"):
        logger.warning("book_upload_rejected reason=invalid_file_type file_name=%s", file.filename)
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    book_id, file_path = save_uploaded_pdf(file)

    db = SessionLocal()
    try:
        book = Book(
            id=book_id,
            title=title.strip(),
            file_name=file.filename,
            file_path=file_path,
            status="uploaded",
            current_step="uploaded",
        )
        db.add(book)
        db.commit()
        db.refresh(book)
        logger.info("book_upload_saved book_id=%s title=%s", book.id, book.title)

        if auto_ingest:
            background_tasks.add_task(ingest_book, book.id)
            logger.info("book_ingest_enqueued book_id=%s", book.id)

        return _to_book_out(book)
    finally:
        db.close()

@router.get("", response_model=list[BookOut])
def list_books():
    db = SessionLocal()
    try:
        books = db.query(Book).order_by(Book.created_at.desc()).all()
        logger.info("book_list_fetched count=%s", len(books))
        return [_to_book_out(b) for b in books]
    finally:
        db.close()

@router.get("/{book_id}", response_model=BookOut)
def get_book(book_id: str):
    db = SessionLocal()
    try:
        book = db.get(Book, book_id)
        if not book:
            logger.warning("book_get_missing book_id=%s", book_id)
            raise HTTPException(status_code=404, detail="Book not found.")
        logger.info("book_get_success book_id=%s status=%s", book.id, book.status)
        return _to_book_out(book)
    finally:
        db.close()


@router.post("/{book_id}/ingest", response_model=BookStatusOut)
def ingest_book_endpoint(book_id: str, background_tasks: BackgroundTasks):
    db = SessionLocal()
    try:
        book = db.get(Book, book_id)
        if not book:
            logger.warning("book_ingest_missing book_id=%s", book_id)
            raise HTTPException(status_code=404, detail="Book not found.")

        if book.status in INGEST_ACTIVE_STATUSES:
            logger.info("book_ingest_already_running book_id=%s status=%s", book.id, book.status)
            return _to_book_status_out(book)

        # Safe full restart reindex behavior.
        book.status = "uploaded"
        book.current_step = "reindex requested"
        book.error_message = None
        book.total_sections = None
        book.total_chunks = 0
        book.processed_chunks = 0
        book.failed_chunks = 0
        book.progress_percent = 0
        book.last_processed_chunk_index = None
        book.started_at = None
        book.finished_at = None
        db.commit()
        db.refresh(book)
        logger.info("book_ingest_reset_state book_id=%s", book.id)

        invalidate_book(book.id)

        background_tasks.add_task(ingest_book, book.id)
        logger.info("book_ingest_enqueued book_id=%s", book.id)
        return _to_book_status_out(book)
    finally:
        db.close()


@router.get("/{book_id}/status", response_model=BookStatusOut)
def get_book_status(book_id: str):
    db = SessionLocal()
    try:
        book = db.get(Book, book_id)
        if not book:
            logger.warning("book_status_missing book_id=%s", book_id)
            raise HTTPException(status_code=404, detail="Book not found.")
        logger.info("book_status_success book_id=%s status=%s progress=%s", book.id, book.status, book.progress_percent)
        return _to_book_status_out(book)
    finally:
        db.close()


@router.delete("/{book_id}")
def delete_book(book_id: str):
    db = SessionLocal()
    try:
        book = db.get(Book, book_id)
        if not book:
            logger.warning("book_delete_missing book_id=%s", book_id)
            raise HTTPException(status_code=404, detail="Book not found.")

        # Mark as deleting first so any in-flight ingestion can cooperatively stop.
        book.status = "deleting"
        book.current_step = "delete requested"
        book.error_message = None
        db.commit()
        db.refresh(book)

        deleted_vectors = delete_book_vectors(book_id)
        logger.info("book_delete_vectors_removed book_id=%s vectors=%s", book_id, deleted_vectors)
        invalidate_book(book_id)
        delete_book_files(book_id)
        db.query(BookSection).filter(BookSection.book_id == book_id).delete(synchronize_session=False)
        db.delete(book)
        db.commit()
        logger.info("book_delete_success book_id=%s", book_id)
        return {"ok": True}
    finally:
        db.close()
