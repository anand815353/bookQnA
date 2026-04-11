# app/services/storage.py
import shutil
import uuid
from pathlib import Path
from fastapi import UploadFile
from app.settings import BOOKS_DIR

def save_uploaded_pdf(file: UploadFile) -> tuple[str, str]:
    book_id = str(uuid.uuid4())
    book_dir = BOOKS_DIR / book_id
    book_dir.mkdir(parents=True, exist_ok=True)

    file_path = book_dir / "original.pdf"
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return book_id, str(file_path)

def delete_book_files(book_id: str) -> None:
    import shutil
    book_dir = BOOKS_DIR / book_id
    if book_dir.exists():
        shutil.rmtree(book_dir)