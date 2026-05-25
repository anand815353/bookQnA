# app/services/cache.py
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# Key prefixes used by retrieval / generation caches (see build_*_cache_key).
RETRIEVAL_CACHE_KEY_PREFIX = "rq:"
GENERATION_CACHE_KEY_PREFIX = "gen:"


def _cache_backend_for_invalidation() -> CacheBackend:
    """Use the shared process backend (memory or Redis) so invalidation matches readers."""
    from app.settings import REDIS_URL

    return get_cache_backend(redis_url=REDIS_URL)


def invalidate_all_retrieval() -> int:
    """Remove all cached parent-evidence payloads. Returns approximate number of keys removed."""
    removed = _cache_backend_for_invalidation().delete_prefix(RETRIEVAL_CACHE_KEY_PREFIX)
    logger.info("cache_invalidate_all_retrieval removed=%s", removed)
    return removed


def invalidate_all_generation() -> int:
    """Remove all cached LLM answers. Returns approximate number of keys removed."""
    removed = _cache_backend_for_invalidation().delete_prefix(GENERATION_CACHE_KEY_PREFIX)
    logger.info("cache_invalidate_all_generation removed=%s", removed)
    return removed


def invalidate_generation_for_book(book_id: str) -> int:
    """
    Invalidate generation cache entries that may depend on this book.

    Generation keys are content-addressed (no book_id prefix), so this clears the entire
    generation cache — safe for typical sizes and TTL.
    """
    removed = invalidate_all_generation()
    logger.info(
        "cache_invalidate_generation_for_book book_id=%s generation_entries_removed=%s",
        book_id,
        removed,
    )
    return removed


def invalidate_book(book_id: str) -> tuple[int, int]:
    """
    Invalidate retrieval and generation caches after corpus changes for a book
    (delete, re-ingest, or vector rebuild). Uses prefix flush because keys are hashed.
    """
    r = invalidate_all_retrieval()
    g = invalidate_all_generation()
    logger.info(
        "cache_invalidate_book book_id=%s retrieval_removed=%s generation_removed=%s",
        book_id,
        r,
        g,
    )
    return r, g


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def key_hash_hex(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:24]


def build_retrieval_cache_key(
    *,
    normalized_query: str,
    book_ids: list[str] | None,
    retrieval_mode: str,
    top_k: int,
    planner_fingerprint: str,
    embedding_provider: str,
    embedding_model: str,
    lexical_index_version: str,
    config_version: str,
    enable_lexical_retrieval: bool,
    hybrid_dense_weight: float,
    hybrid_lexical_weight: float,
    hybrid_rrf_k: int,
    reranker_enabled: bool = False,
    reranker_provider: str = "cross_encoder",
    reranker_model: str = "",
    reranker_top_n: int = 8,
) -> str:
    payload: dict[str, Any] = {
        "type": "retrieval_parent_evidence",
        "q": normalized_query,
        "books": sorted(book_ids or []),
        "mode": retrieval_mode,
        "top_k": top_k,
        "planner_fp": planner_fingerprint,
        "emb_provider": embedding_provider,
        "emb_model": embedding_model,
        "lexical_ver": lexical_index_version,
        "cfg_ver": config_version,
        "lexical_on": enable_lexical_retrieval,
        "hybrid_dw": hybrid_dense_weight,
        "hybrid_lw": hybrid_lexical_weight,
        "hybrid_k": hybrid_rrf_k,
    }
    if reranker_enabled:
        payload["reranker"] = {
            "provider": reranker_provider,
            "model": reranker_model,
            "top_n": reranker_top_n,
        }
    return f"rq:{key_hash_hex(payload)}"


def build_generation_cache_key(
    *,
    normalized_question: str,
    book_ids: list[str] | None,
    model_name: str,
    prompt_version: str,
    context_hash: str,
    planner_mode: str,
    retrieval_mode: str,
    top_k: int,
) -> str:
    payload = {
        "type": "generation_answer",
        "q": normalized_question,
        "books": sorted(book_ids or []),
        "model": model_name,
        "prompt_ver": prompt_version,
        "ctx_hash": context_hash,
        "planner_mode": planner_mode,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
    }
    return f"gen:{key_hash_hex(payload)}"


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        pass

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Best-effort delete keys starting with prefix. Returns count removed (approx for Redis)."""
        pass


class NullCache(CacheBackend):
    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        return None

    def delete_prefix(self, prefix: str) -> int:
        return 0


class MemoryCache(CacheBackend):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        ttl = max(1, int(ttl_seconds))
        expires_at = time.monotonic() + float(ttl)
        with self._lock:
            self._store[key] = (expires_at, value)

    def delete_prefix(self, prefix: str) -> int:
        removed = 0
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
                removed += 1
        return removed


class RedisCache(CacheBackend):
    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> Any | None:
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            logger.exception("redis_cache_get_failed key_prefix=%s", key[:20])
            return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        try:
            ttl = max(1, int(ttl_seconds))
            self._client.setex(key, ttl, json.dumps(value, default=str))
        except Exception:
            logger.exception("redis_cache_set_failed key_prefix=%s", key[:20])

    def delete_prefix(self, prefix: str) -> int:
        try:
            cursor = 0
            removed = 0
            while True:
                cursor, keys = self._client.scan(cursor=cursor, match=f"{prefix}*", count=200)
                if keys:
                    removed += int(self._client.delete(*keys))
                if cursor == 0:
                    break
            return removed
        except Exception:
            logger.exception("redis_cache_delete_prefix_failed")
            return 0


_backend: CacheBackend | None = None
_backend_lock = threading.Lock()


def get_cache_backend(
    *,
    redis_url: str | None,
    prefer_memory: bool = False,
) -> CacheBackend:
    global _backend
    if prefer_memory:
        return MemoryCache()
    with _backend_lock:
        if _backend is not None:
            return _backend
        if redis_url:
            try:
                _backend = RedisCache(redis_url)
                logger.info("cache_backend_selected kind=redis")
            except ImportError:
                logger.warning("cache_backend_redis_import_failed falling_back=memory")
                _backend = MemoryCache()
            except Exception:
                logger.exception("cache_backend_redis_failed falling_back=memory")
                _backend = MemoryCache()
        else:
            _backend = MemoryCache()
            logger.info("cache_backend_selected kind=memory")
        return _backend


def reset_cache_backend_for_tests() -> None:
    global _backend
    with _backend_lock:
        _backend = None
