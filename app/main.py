# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db import init_db
from app.api.books import router as books_router
from app.api.query import router as query_router
from dotenv import load_dotenv
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Book Q&A", lifespan=lifespan)

app.include_router(books_router)
app.include_router(query_router)