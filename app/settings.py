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

# Gemini chat (answer + planner + reformulation): optional spacing to reduce free-tier 429 RPM bursts
GEMINI_CHAT_MIN_INTERVAL_SECONDS = float(os.getenv("GEMINI_CHAT_MIN_INTERVAL_SECONDS", "0"))
GEMINI_CHAT_MAX_RETRIES = max(1, int(os.getenv("GEMINI_CHAT_MAX_RETRIES", "6")))
# Q&A and query reformulation use this model (planner uses QUERY_PLANNER_MODEL separately).
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash").strip()
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
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "4"))
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "1400"))
ENABLE_QUERY_REFORMULATION = _env_bool("ENABLE_QUERY_REFORMULATION", default=True)
ENABLE_QUERY_PLANNER = _env_bool("ENABLE_QUERY_PLANNER", default=False)
QUERY_PLANNER_ENABLE_GATING = _env_bool("QUERY_PLANNER_ENABLE_GATING", default=True)
QUERY_PLANNER_FORCE_IN_DEBUG = _env_bool("QUERY_PLANNER_FORCE_IN_DEBUG", default=False)
QUERY_PLANNER_MODEL = os.getenv("QUERY_PLANNER_MODEL", "gemini-2.5-flash").strip()
QUERY_PLANNER_MAX_ALT_QUERIES = int(os.getenv("QUERY_PLANNER_MAX_ALT_QUERIES", "2"))
QUERY_PLANNER_MAX_KEYWORDS = int(os.getenv("QUERY_PLANNER_MAX_KEYWORDS", "6"))
QUERY_PLANNER_USE_BOOK_METADATA = _env_bool("QUERY_PLANNER_USE_BOOK_METADATA", default=True)
QUERY_PLANNER_LOW_TEMP = float(os.getenv("QUERY_PLANNER_LOW_TEMP", "0"))
QUERY_PLANNER_SHORT_QUERY_WORDS = int(os.getenv("QUERY_PLANNER_SHORT_QUERY_WORDS", "3"))

# Hybrid merge (defaults preserve prior dense*0.7 + lexical*1.0 behavior)
HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7"))
HYBRID_LEXICAL_WEIGHT = float(os.getenv("HYBRID_LEXICAL_WEIGHT", "1.0"))

# Lexical FTS index (optional; false = legacy Chroma corpus BM25 scan)
ENABLE_LEXICAL_RETRIEVAL = _env_bool("ENABLE_LEXICAL_RETRIEVAL", default=False)
LEXICAL_INDEX_VERSION = os.getenv("LEXICAL_INDEX_VERSION", "2").strip()

# Reranker (disabled by default). Prefer RERANKER_ENABLED; if unset, ENABLE_RERANKER is used.
if os.getenv("RERANKER_ENABLED") is not None:
    RERANKER_ENABLED = _env_bool("RERANKER_ENABLED", default=False)
else:
    RERANKER_ENABLED = _env_bool("ENABLE_RERANKER", default=False)
ENABLE_RERANKER = RERANKER_ENABLED
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "cross_encoder").strip().lower()
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip()
try:
    RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "8"))
except ValueError:
    RERANKER_TOP_N = 8

# Context packer
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "32000"))
INCLUDE_PARENT_CONTEXT = _env_bool("INCLUDE_PARENT_CONTEXT", default=True)

# Caches
ENABLE_RETRIEVAL_CACHE = _env_bool("ENABLE_RETRIEVAL_CACHE", default=False)
RETRIEVAL_CACHE_TTL_SECONDS = int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "300"))
ENABLE_GENERATION_CACHE = _env_bool("ENABLE_GENERATION_CACHE", default=False)
GENERATION_CACHE_TTL_SECONDS = int(os.getenv("GENERATION_CACHE_TTL_SECONDS", "86400"))
REDIS_URL = os.getenv("REDIS_URL", "").strip() or None
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "1").strip()

# Stable hash of cache-relevant settings subset
CACHE_CONFIG_VERSION = os.getenv(
    "CACHE_CONFIG_VERSION",
    "v1",
).strip()

# App identity (tracing / reports)
APP_ENV = os.getenv("APP_ENV", "local").strip()
APP_VERSION = os.getenv("APP_VERSION", "dev").strip()

# LangSmith (optional; requires API key in environment when enabled)
ENABLE_LANGSMITH = _env_bool("ENABLE_LANGSMITH", default=False)
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "bookqna-rag").strip()
LANGSMITH_TRACE_FULL_CONTEXT = _env_bool("LANGSMITH_TRACE_FULL_CONTEXT", default=False)
