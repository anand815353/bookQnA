# app/services/ingest.py
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.db import SessionLocal
from app.models import Book
from app.services.retrieval import get_vectorstore

def ingest_book(book_id: str):
    db = SessionLocal()
    try:
        book = db.get(Book, book_id)
        if not book:
            return

        book.status = "ingesting"
        book.updated_at = datetime.utcnow()
        db.commit()

        loader = PyMuPDFLoader(book.file_path)
        pages = loader.load()

        if not pages:
            book.status = "failed"
            book.error_message = "No pages extracted from PDF."
            db.commit()
            return

        # enrich metadata
        for doc in pages:
            doc.metadata["book_id"] = book.id
            doc.metadata["book_title"] = book.title
            # page is often 0-indexed in loaders; normalize to 1-indexed for UI/citation
            page_no = int(doc.metadata.get("page", 0)) + 1
            doc.metadata["page_start"] = page_no
            doc.metadata["page_end"] = page_no
            doc.metadata["source_file"] = book.file_name

        book.total_pages = len(pages)
        db.commit()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(pages)

        ids = []
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            ids.append(f"{book.id}::chunk::{i}")

        vectorstore = get_vectorstore()
        vectorstore.add_documents(documents=chunks, ids=ids)

        book.status = "indexed"
        book.error_message = None
        book.updated_at = datetime.utcnow()
        db.commit()

    except Exception as e:
        book = db.get(Book, book_id)
        if book:
            book.status = "failed"
            book.error_message = str(e)
            book.updated_at = datetime.utcnow()
            db.commit()
        raise
    finally:
        db.close()