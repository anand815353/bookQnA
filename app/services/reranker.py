# app/services/reranker.py
from __future__ import annotations

import logging
from typing import Any, Protocol

from app import settings as app_settings

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        children: list[Any],
        scores: dict[str, float],
    ) -> tuple[list[Any], dict[str, float]]:
        ...


class NullReranker:
    def rerank(
        self,
        query: str,
        children: list[Any],
        scores: dict[str, float],
    ) -> tuple[list[Any], dict[str, float]]:
        return children, scores


class MockReranker:
    """Deterministic reranker for tests: sort the reranked head by `page_content` descending."""

    def __init__(self, top_n: int) -> None:
        self._top_n = top_n

    def rerank(
        self,
        query: str,
        children: list[Any],
        scores: dict[str, float],
    ) -> tuple[list[Any], dict[str, float]]:
        from app.services.retrieval import _doc_key

        if not children:
            return children, scores

        n = self._top_n if self._top_n > 0 else len(children)
        n = min(n, len(children))
        head = list(children[:n])
        tail = list(children[n:])
        head.sort(key=lambda d: str(getattr(d, "page_content", "") or ""), reverse=True)
        out = head + tail
        out_scores = dict(scores)
        for rank, doc in enumerate(out[:n], start=1):
            key = _doc_key(doc)
            out_scores[key] = float(n - rank + 1)
        return out, out_scores


class CrossEncoderReranker:
    def __init__(self, model_name: str, top_n: int) -> None:
        self._model_name = model_name
        self._top_n = top_n
        self._model: Any = None
        self._bypass = False
        self._warned_load = False
        self._warned_predict = False

    def _ensure_model(self) -> Any:
        if self._bypass:
            return None
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        except Exception:
            if not self._warned_load:
                logger.warning(
                    "reranker_cross_encoder_load_failed model=%s falling_back=null",
                    self._model_name,
                    exc_info=True,
                )
                self._warned_load = True
            self._bypass = True
            return None
        return self._model

    def rerank(
        self,
        query: str,
        children: list[Any],
        scores: dict[str, float],
    ) -> tuple[list[Any], dict[str, float]]:
        from app.services.retrieval import _doc_key, _doc_text_for_retrieval

        if not children:
            return children, scores

        if self._bypass:
            return children, scores

        model = self._ensure_model()
        if model is None:
            return children, scores

        n = self._top_n if self._top_n > 0 else len(children)
        n = min(n, len(children))
        head = list(children[:n])
        tail = list(children[n:])

        pairs = [(query, _doc_text_for_retrieval(doc)) for doc in head]
        try:
            raw = model.predict(pairs)
        except Exception:
            if not self._warned_predict:
                logger.warning(
                    "reranker_cross_encoder_predict_failed model=%s falling_back=null",
                    self._model_name,
                    exc_info=True,
                )
                self._warned_predict = True
            self._bypass = True
            return children, scores

        if hasattr(raw, "tolist"):
            ce_scores = raw.tolist()
        else:
            ce_scores = list(raw)
        if len(ce_scores) != len(head):
            logger.warning(
                "reranker_cross_encoder_score_len_mismatch expected=%s got=%s",
                len(head),
                len(ce_scores),
            )
            return children, scores

        order = sorted(range(len(head)), key=lambda i: float(ce_scores[i]), reverse=True)
        reranked_head = [head[i] for i in order]
        out = reranked_head + tail
        out_scores = dict(scores)
        for i in order:
            doc = head[i]
            out_scores[_doc_key(doc)] = float(ce_scores[i])
        return out, out_scores


_null = NullReranker()
_cached: Reranker | None = None
_cached_sig: tuple[Any, ...] | None = None


def _reranker_factory_signature() -> tuple[Any, ...]:
    return (
        app_settings.RERANKER_ENABLED,
        app_settings.RERANKER_PROVIDER,
        app_settings.RERANKER_MODEL,
        app_settings.RERANKER_TOP_N,
    )


def get_reranker() -> Reranker:
    global _cached, _cached_sig
    if not app_settings.RERANKER_ENABLED:
        return _null

    sig = _reranker_factory_signature()
    if _cached is not None and _cached_sig == sig:
        return _cached

    provider = app_settings.RERANKER_PROVIDER
    if provider == "mock":
        _cached = MockReranker(app_settings.RERANKER_TOP_N)
        _cached_sig = sig
        return _cached

    if provider == "cross_encoder":
        _cached = CrossEncoderReranker(app_settings.RERANKER_MODEL, app_settings.RERANKER_TOP_N)
        _cached_sig = sig
        return _cached

    logger.warning("reranker_unknown_provider provider=%s falling_back=null", provider)
    _cached = _null
    _cached_sig = sig
    return _cached
