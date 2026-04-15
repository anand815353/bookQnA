# app/settings.py
from pathlib import Path
import os


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


DATA_DIR = Path("data")
BOOKS_DIR = DATA_DIR / "books"
CHROMA_DIR = DATA_DIR / "chroma"
LOG_DIR = Path(os.getenv("LOG_DIR", str(DATA_DIR / "logs")))

BOOKS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "5"))
EMBEDDING_BATCH_SLEEP_SECONDS = float(os.getenv("EMBEDDING_BATCH_SLEEP_SECONDS", "5.0"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "6"))
EMBEDDING_BACKOFF_BASE_SECONDS = float(os.getenv("EMBEDDING_BACKOFF_BASE_SECONDS", "3.0"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower()
HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
VECTORSTORE_COLLECTION_PREFIX = os.getenv("VECTORSTORE_COLLECTION_PREFIX", "books").strip()
VECTORSTORE_COLLECTION_NAME = (
    f"{VECTORSTORE_COLLECTION_PREFIX}_{EMBEDDING_PROVIDER}"
    if VECTORSTORE_COLLECTION_PREFIX
    else EMBEDDING_PROVIDER
)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE_MAX_BYTES = int(os.getenv("LOG_FILE_MAX_BYTES", "2097152"))
LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))
LOG_DEBUG_SNIPPET_CHARS = int(os.getenv("LOG_DEBUG_SNIPPET_CHARS", "280"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()
RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "40"))
LEXICAL_MAX_DOCS = int(os.getenv("LEXICAL_MAX_DOCS", "5000"))
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
RETRIEVAL_TRACE_MAX_ITEMS = int(os.getenv("RETRIEVAL_TRACE_MAX_ITEMS", "30"))
QUERY_DEBUG_ENABLED = _env_bool("QUERY_DEBUG_ENABLED", default=False)
QUERY_DEBUG_CONTEXT_CHARS = int(os.getenv("QUERY_DEBUG_CONTEXT_CHARS", "1800"))
