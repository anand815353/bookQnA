from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.db import SessionLocal
from app.models import Book, ChatSession
from app.services.ingest import ingest_book
from app.services.storage import save_uploaded_pdf
from app.settings import QUERY_DEBUG_ENABLED

router = APIRouter()
APP_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    db = SessionLocal()
    try:
        books = db.query(Book).order_by(Book.created_at.desc()).all()
        total_books = len(books)
        indexed_books = sum(1 for b in books if b.status == "indexed")
        ingesting_books = sum(
            1 for b in books if b.status in {"parsing", "structuring", "chunking", "embedding"}
        )

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "books": books,
                "total_books": total_books,
                "indexed_books": indexed_books,
                "ingesting_books": ingesting_books,
            },
        )
    finally:
        db.close()


@router.get("/library", response_class=HTMLResponse)
def books_page(request: Request):
    db = SessionLocal()
    try:
        books = db.query(Book).order_by(Book.created_at.desc()).all()
        return templates.TemplateResponse(
            request=request,
            name="books.html",
            context={"books": books},
        )
    finally:
        db.close()


@router.post("/upload-book")
def upload_book_from_ui(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    auto_ingest: bool = Form(True),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    clean_title = title.strip()
    if not clean_title:
        raise HTTPException(status_code=400, detail="Book title is required.")

    book_id, file_path = save_uploaded_pdf(file)
    db = SessionLocal()
    try:
        book = Book(
            id=book_id,
            title=clean_title,
            file_name=file.filename,
            file_path=file_path,
            status="uploaded",
        )
        db.add(book)
        db.commit()
    finally:
        db.close()

    if auto_ingest:
        background_tasks.add_task(ingest_book, book_id)

    return RedirectResponse(url="/library", status_code=303)


@router.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    db = SessionLocal()
    try:
        books = db.query(Book).order_by(Book.created_at.desc()).all()
        return templates.TemplateResponse(
            request=request,
            name="chat.html",
            context={
                "books": books,
                "result": None,
                "query_debug_enabled": QUERY_DEBUG_ENABLED,
                "selected_session_id": None,
            },
        )
    finally:
        db.close()


@router.get("/chat/{session_id}", response_class=HTMLResponse)
def chat_thread_page(request: Request, session_id: str):
    db = SessionLocal()
    try:
        books = db.query(Book).order_by(Book.created_at.desc()).all()
        session = db.get(ChatSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found.")
        return templates.TemplateResponse(
            request=request,
            name="chat.html",
            context={
                "books": books,
                "result": None,
                "query_debug_enabled": QUERY_DEBUG_ENABLED,
                "selected_session_id": session_id,
            },
        )
    finally:
        db.close()
