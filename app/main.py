# app/main.py
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time
import uuid

from fastapi import FastAPI
from app.db import init_db
from fastapi.staticfiles import StaticFiles
from app.api.books import router as books_router
from app.api.query import router as query_router
from app.web.pages import router as pages_router
from app.logging_config import clear_request_id, set_request_id, setup_logging
from dotenv import load_dotenv

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app_startup")
    init_db()
    yield
    logger.info("app_shutdown")

app = FastAPI(title="Book Q&A", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

app.include_router(books_router)
app.include_router(query_router)
app.include_router(pages_router)


@app.middleware("http")
async def request_logging_middleware(request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    set_request_id(request_id)
    started = time.perf_counter()
    client = request.client.host if request.client else "-"
    logger.info("http_request_started method=%s path=%s client=%s", request.method, request.url.path, client)
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.exception(
            "http_request_failed method=%s path=%s client=%s duration_ms=%s",
            request.method,
            request.url.path,
            client,
            elapsed_ms,
        )
        clear_request_id()
        raise
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "http_request_finished method=%s path=%s status=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    clear_request_id()
    return response


@app.get("/health")
def health():
    return {"ok": True}