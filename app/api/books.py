# app/api/books.py
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from app.db import SessionLocal
from app.models import Book
from app.schemas import BookOut
from app.services.storage import save_uploaded_pdf, delete_book_files
from app.services.ingest import ingest_book

router = APIRouter(prefix="/books", tags=["books"])

@router.post("/upload", response_model=BookOut)
def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    book_id, file_path = save_uploaded_pdf(file)

    db = SessionLocal()
    try:
        book = Book(
            id=book_id,
            title=title.strip(),
            file_name=file.filename,
            file_path=file_path,
            status="uploaded"
        )
        db.add(book)
        db.commit()
        db.refresh(book)

        background_tasks.add_task(ingest_book, book.id)

        return BookOut(
            id=book.id,
            title=book.title,
            file_name=book.file_name,
            status=book.status,
            total_pages=book.total_pages,
            error_message=book.error_message
        )
    finally:
        db.close()

@router.get("", response_model=list[BookOut])
def list_books():
    db = SessionLocal()
    try:
        books = db.query(Book).order_by(Book.created_at.desc()).all()
        return [
            BookOut(
                id=b.id,
                title=b.title,
                file_name=b.file_name,
                status=b.status,
                total_pages=b.total_pages,
                error_message=b.error_message
            )
            for b in books
        ]
    finally:
        db.close()

@router.get("/{book_id}", response_model=BookOut)
def get_book(book_id: str):
    db = SessionLocal()
    try:
        book = db.get(Book, book_id)
        if not book:
            raise HTTPException(status_code=404, detail="Book not found.")
        return BookOut(
            id=book.id,
            title=book.title,
            file_name=book.file_name,
            status=book.status,
            total_pages=book.total_pages,
            error_message=book.error_message
        )
    finally:
        db.close()

@router.delete("/{book_id}")
def delete_book(book_id: str):
    from app.services.retrieval import get_vectorstore

    db = SessionLocal()
    try:
        book = db.get(Book, book_id)
        if not book:
            raise HTTPException(status_code=404, detail="Book not found.")

        vectorstore = get_vectorstore()
        # delete by ids we know? in v1 easiest is metadata-aware cleanup later;
        # for now delete by where filter if supported by backend version or reindex strategy later.
        # Minimal v1: keep DB/files delete; vector cleanup can be improved in next chapter.
        db.delete(book)
        db.commit()
        delete_book_files(book_id)
        return {"ok": True}
    finally:
        db.close()