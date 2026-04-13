# app/services/retrieval.py
import logging
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.settings import (
    CHROMA_DIR,
    EMBEDDING_PROVIDER,
    GEMINI_EMBEDDING_MODEL,
    HF_EMBEDDING_MODEL,
    VECTORSTORE_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)

def get_embeddings():
    if EMBEDDING_PROVIDER == "huggingface":
        logger.info("embeddings_provider_selected provider=huggingface model=%s", HF_EMBEDDING_MODEL)
        return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)

    if EMBEDDING_PROVIDER == "gemini":
        # Prefer explicit API key over ambient ADC credentials to avoid scope issues.
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        logger.info("embeddings_provider_selected provider=gemini model=%s", GEMINI_EMBEDDING_MODEL)
        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=api_key,
        )

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER='{EMBEDDING_PROVIDER}'. "
        "Expected one of: huggingface, gemini."
    )

def get_vectorstore():
    logger.info(
        "vectorstore_init provider=%s collection=%s persist_directory=%s",
        EMBEDDING_PROVIDER,
        VECTORSTORE_COLLECTION_NAME,
        CHROMA_DIR,
    )
    return Chroma(
        collection_name=VECTORSTORE_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def delete_book_vectors(book_id: str) -> int:
    logger.info("vector_delete_requested book_id=%s", book_id)
    vectorstore = get_vectorstore()
    collection = vectorstore._collection

    try:
        existing = collection.get(where={"book_id": book_id}, include=[])
        ids = existing.get("ids", []) if existing else []
        if ids:
            vectorstore.delete(ids=ids)
        logger.info("vector_delete_completed book_id=%s deleted=%s", book_id, len(ids))
        return len(ids)
    except Exception:
        # Keep delete flow resilient even if vector cleanup support varies by backend.
        logger.exception("vector_delete_failed book_id=%s", book_id)
        return 0


def search_documents(question: str, top_k: int, book_ids: list[str] | None = None):
    logger.info(
        "vector_search_requested top_k=%s book_filter_count=%s question_len=%s",
        top_k,
        len(book_ids or []),
        len(question or ""),
    )
    vectorstore = get_vectorstore()
    if not book_ids:
        results = vectorstore.similarity_search(question, k=top_k)
        logger.info("vector_search_completed strategy=unfiltered results=%s", len(results))
        return results

    where_filter: dict = (
        {"book_id": book_ids[0]}
        if len(book_ids) == 1
        else {"book_id": {"$in": book_ids}}
    )
    try:
        results = vectorstore.similarity_search(question, k=top_k, filter=where_filter)
        logger.info("vector_search_completed strategy=filtered results=%s", len(results))
        return results
    except Exception:
        logger.exception("vector_search_filtered_failed using_fallback=true")
        retrieved = vectorstore.similarity_search(question, k=max(top_k * 3, 10))
        allowed = set(book_ids)
        fallback = [d for d in retrieved if d.metadata.get("book_id") in allowed][:top_k]
        logger.info("vector_search_completed strategy=fallback_filter results=%s", len(fallback))
        return fallback