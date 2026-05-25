from pytest import approx

from app.services import cache as cache_mod
from app.services.cache import (
    MemoryCache,
    NullCache,
    build_generation_cache_key,
    build_retrieval_cache_key,
    invalidate_all_generation,
    invalidate_all_retrieval,
    invalidate_book,
    key_hash_hex,
)
from app.settings import HYBRID_DENSE_WEIGHT, HYBRID_LEXICAL_WEIGHT


def test_null_cache_never_returns_value():
    c = NullCache()
    assert c.get("any") is None
    c.set("any", {"a": 1}, 60)
    assert c.get("any") is None


def test_memory_cache_roundtrip():
    c = MemoryCache()
    c.set("k1", {"x": 1}, ttl_seconds=3600)
    assert c.get("k1") == {"x": 1}


def test_memory_cache_delete_prefix():
    c = MemoryCache()
    c.set("pre:a", 1, 3600)
    c.set("pre:b", 2, 3600)
    c.set("other", 3, 3600)
    assert c.delete_prefix("pre:") == 2
    assert c.get("pre:a") is None
    assert c.get("other") == 3


def test_retrieval_cache_key_includes_reranker_when_enabled():
    base = dict(
        normalized_query="hello world",
        book_ids=["b1", "b2"],
        retrieval_mode="hybrid",
        top_k=4,
        planner_fingerprint="abc",
        embedding_provider="huggingface",
        embedding_model="m1",
        lexical_index_version="1",
        config_version="v1",
        enable_lexical_retrieval=False,
        hybrid_dense_weight=0.7,
        hybrid_lexical_weight=1.0,
        hybrid_rrf_k=60,
    )
    k_off = build_retrieval_cache_key(**base, reranker_enabled=False)
    k_on = build_retrieval_cache_key(
        **base,
        reranker_enabled=True,
        reranker_provider="cross_encoder",
        reranker_model="BAAI/bge-reranker-base",
        reranker_top_n=8,
    )
    k_on_other = build_retrieval_cache_key(
        **base,
        reranker_enabled=True,
        reranker_provider="cross_encoder",
        reranker_model="other/model",
        reranker_top_n=8,
    )
    assert k_off == build_retrieval_cache_key(**base)
    assert k_on != k_off
    assert k_on_other != k_on


def test_retrieval_cache_key_stable():
    k1 = build_retrieval_cache_key(
        normalized_query="hello world",
        book_ids=["b2", "b1"],
        retrieval_mode="hybrid",
        top_k=4,
        planner_fingerprint="abc",
        embedding_provider="huggingface",
        embedding_model="m1",
        lexical_index_version="1",
        config_version="v1",
        enable_lexical_retrieval=False,
        hybrid_dense_weight=0.7,
        hybrid_lexical_weight=1.0,
        hybrid_rrf_k=60,
    )
    k2 = build_retrieval_cache_key(
        normalized_query="hello world",
        book_ids=["b1", "b2"],
        retrieval_mode="hybrid",
        top_k=4,
        planner_fingerprint="abc",
        embedding_provider="huggingface",
        embedding_model="m1",
        lexical_index_version="1",
        config_version="v1",
        enable_lexical_retrieval=False,
        hybrid_dense_weight=0.7,
        hybrid_lexical_weight=1.0,
        hybrid_rrf_k=60,
    )
    assert k1 == k2


def test_generation_cache_key_changes_with_context_hash():
    base = dict(
        normalized_question="q",
        book_ids=["b1"],
        model_name="gemini-2.5-flash",
        prompt_version="1",
        planner_mode="original",
        retrieval_mode="hybrid",
        top_k=3,
    )
    k1 = build_generation_cache_key(context_hash="h1", **base)
    k2 = build_generation_cache_key(context_hash="h2", **base)
    assert k1 != k2


def test_key_hash_hex_short():
    h = key_hash_hex({"a": 1, "b": 2})
    assert len(h) == 24


def test_retrieval_cache_key_changes_with_top_k():
    common = dict(
        normalized_query="q",
        book_ids=["b1"],
        retrieval_mode="hybrid",
        planner_fingerprint="none",
        embedding_provider="huggingface",
        embedding_model="m1",
        lexical_index_version="1",
        config_version="v1",
        enable_lexical_retrieval=False,
        hybrid_dense_weight=0.7,
        hybrid_lexical_weight=1.0,
        hybrid_rrf_k=60,
    )
    assert build_retrieval_cache_key(**common, top_k=3) != build_retrieval_cache_key(**common, top_k=4)


def test_retrieval_cache_key_changes_with_retrieval_mode():
    common = dict(
        normalized_query="q",
        book_ids=["b1"],
        top_k=4,
        planner_fingerprint="none",
        embedding_provider="huggingface",
        embedding_model="m1",
        lexical_index_version="1",
        config_version="v1",
        enable_lexical_retrieval=False,
        hybrid_dense_weight=0.7,
        hybrid_lexical_weight=1.0,
        hybrid_rrf_k=60,
    )
    assert build_retrieval_cache_key(**common, retrieval_mode="hybrid") != build_retrieval_cache_key(
        **common, retrieval_mode="dense_only"
    )


def test_generation_cache_key_changes_with_model_name():
    base = dict(
        normalized_question="q",
        book_ids=["b1"],
        prompt_version="1",
        context_hash="h",
        planner_mode="none",
        retrieval_mode="hybrid",
        top_k=4,
    )
    assert build_generation_cache_key(model_name="gemini-2.5-flash", **base) != build_generation_cache_key(
        model_name="gemini-2.0-flash", **base
    )


def test_invalidate_all_retrieval_leaves_generation_keys(monkeypatch):
    import app.settings as app_settings

    monkeypatch.setattr(app_settings, "REDIS_URL", None, raising=False)
    cache_mod.reset_cache_backend_for_tests()
    be = cache_mod.get_cache_backend(redis_url=None)
    be.set("rq:aaa", {"x": 1}, 3600)
    be.set("gen:bbb", {"answer": "y"}, 3600)
    invalidate_all_retrieval()
    assert be.get("rq:aaa") is None
    assert be.get("gen:bbb") == {"answer": "y"}
    invalidate_all_generation()
    assert be.get("gen:bbb") is None


def test_invalidate_book_clears_both_prefixes(monkeypatch):
    import app.settings as app_settings

    monkeypatch.setattr(app_settings, "REDIS_URL", None, raising=False)
    cache_mod.reset_cache_backend_for_tests()
    be = cache_mod.get_cache_backend(redis_url=None)
    be.set("rq:z", {}, 3600)
    be.set("gen:z", {}, 3600)
    invalidate_book("book-1")
    assert be.get("rq:z") is None
    assert be.get("gen:z") is None


def test_hybrid_weighted_merge_defaults_match_legacy_style():
    dense_scores = {"a": 1.0, "b": 0.5}
    lexical_scores = {"a": 0.2, "c": 1.0}
    keys = set(dense_scores) | set(lexical_scores)
    merged = {
        k: HYBRID_DENSE_WEIGHT * dense_scores.get(k, 0.0) + HYBRID_LEXICAL_WEIGHT * lexical_scores.get(k, 0.0)
        for k in keys
    }
    assert merged["a"] == approx(0.7 * 1.0 + 1.0 * 0.2)
    assert merged["b"] == approx(0.7 * 0.5)
    assert merged["c"] == approx(1.0 * 1.0)
