from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.db import SessionLocal
from app.models import Book, ChatSession
from app.services.ingest import ingest_book
from app.services.storage import save_uploaded_pdf
from app.services.eval_reports import (
    by_test_type_rows,
    case_table_rows_for_dashboard,
    case_table_rows_from_results,
    load_latest_report,
    metrics_table_rows,
    summarize_report_for_dashboard,
)
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


@router.get("/evals", response_class=HTMLResponse)
def eval_dashboard(request: Request):
    report = load_latest_report()
    if report is None:
        return templates.TemplateResponse(
            request=request,
            name="eval_dashboard.html",
            context={
                "title": "Evaluations",
                "has_report": False,
                "report": None,
                "summary": None,
                "case_table_rows": [],
                "retrieval_metric_rows": [],
                "answer_metric_rows": [],
                "answer_by_type_rows": [],
                "retrieval_baseline_metric_rows": [],
                "retrieval_planner_metric_rows": [],
                "answer_baseline_metric_rows": [],
                "answer_planner_metric_rows": [],
                "case_table_rows_planner": [],
            },
        )

    summary = summarize_report_for_dashboard(report)
    answer_block = report.get("answer") if isinstance(report.get("answer"), dict) else {}
    retrieval_block = report.get("retrieval") if isinstance(report.get("retrieval"), dict) else {}

    ctx: dict = {
        "title": "Evaluations",
        "has_report": True,
        "report": report,
        "summary": summary,
        "case_table_rows": case_table_rows_for_dashboard(summary),
        "case_table_rows_planner": [],
        "retrieval_metric_rows": [],
        "answer_metric_rows": [],
        "answer_by_type_rows": [],
        "retrieval_baseline_metric_rows": [],
        "retrieval_planner_metric_rows": [],
        "answer_baseline_metric_rows": [],
        "answer_planner_metric_rows": [],
    }

    if summary.get("is_comparison"):
        rb = retrieval_block.get("baseline") if isinstance(retrieval_block.get("baseline"), dict) else {}
        rp = retrieval_block.get("planner") if isinstance(retrieval_block.get("planner"), dict) else {}
        ab = answer_block.get("baseline") if isinstance(answer_block.get("baseline"), dict) else {}
        ap = answer_block.get("planner") if isinstance(answer_block.get("planner"), dict) else {}
        ctx["retrieval_baseline_metric_rows"] = metrics_table_rows(rb.get("summary"))
        ctx["retrieval_planner_metric_rows"] = metrics_table_rows(rp.get("summary"))
        ctx["answer_baseline_metric_rows"] = metrics_table_rows(ab.get("summary"))
        ctx["answer_planner_metric_rows"] = metrics_table_rows(ap.get("summary"))
        ctx["answer_by_type_rows"] = by_test_type_rows(ab.get("summary"))
        ctx["case_table_rows_planner"] = case_table_rows_from_results(list(summary.get("answer_planner_rows") or []))
    else:
        ctx["retrieval_metric_rows"] = metrics_table_rows(retrieval_block.get("summary"))
        ctx["answer_metric_rows"] = metrics_table_rows(answer_block.get("summary"))
        ctx["answer_by_type_rows"] = by_test_type_rows(answer_block.get("summary"))

    return templates.TemplateResponse(
        request=request,
        name="eval_dashboard.html",
        context=ctx,
    )


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
