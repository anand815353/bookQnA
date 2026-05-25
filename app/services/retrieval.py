# app/services/retrieval.py
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.db import engine
from app.logging_config import log_structured, sanitize_for_debug
from app.schemas import QueryPlannerPlan
from app.services import lexical_index
from app.services.cache import build_retrieval_cache_key, get_cache_backend, key_hash_hex
from app.services.observability import (
    maybe_traceable,
    trace_process_inputs_search_parent_evidence,
    trace_process_outputs_search_parent_evidence,
)
from app.services.reranker import get_reranker
from app.settings import (
    CACHE_CONFIG_VERSION,
    CHROMA_DIR,
    EMBEDDING_PROVIDER,
    ENABLE_LEXICAL_RETRIEVAL,
    ENABLE_RETRIEVAL_CACHE,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANKER_PROVIDER,
    RERANKER_TOP_N,
    GEMINI_EMBEDDING_MODEL,
    HF_EMBEDDING_MODEL,
    HYBRID_DENSE_WEIGHT,
    HYBRID_LEXICAL_WEIGHT,
    HYBRID_RRF_K,
    LEXICAL_INDEX_VERSION,
    LEXICAL_MAX_DOCS,
    LOG_DEBUG_SNIPPET_CHARS,
    REDIS_URL,
    RETRIEVAL_CACHE_TTL_SECONDS,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_MODE,
    RETRIEVAL_TRACE_MAX_ITEMS,
    VECTORSTORE_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)
VALID_RETRIEVAL_MODES = {"dense_only", "lexical_only", "hybrid"}

# SentenceTransformer on CUDA is not safe to initialize concurrently; parallel requests
# (e.g. two book deletes) otherwise hit: NotImplementedError: Cannot copy out of meta tensor.
_retrieval_resource_lock = threading.RLock()
_embeddings_singleton: Embeddings | None = None
_vectorstore_singleton: Chroma | None = None


def _embedding_model_for_cache() -> str:
    if EMBEDDING_PROVIDER == "gemini":
        return GEMINI_EMBEDDING_MODEL
    return HF_EMBEDDING_MODEL


def _planner_fingerprint(query_plan: QueryPlannerPlan | None) -> str:
    if query_plan is None:
        return "none"
    payload = query_plan.model_dump()
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _json_safe_meta(meta: dict) -> dict:
    out: dict = {}
    for k, v in (meta or {}).items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[str(k)] = v
        else:
            out[str(k)] = str(v)
    return out


def serialize_parent_evidence(evidence: list[dict]) -> list[dict]:
    serialized: list[dict] = []
    for parent in evidence:
        row = {k: v for k, v in parent.items() if k != "children"}
        row["children"] = []
        for child in parent.get("children") or []:
            doc = child.get("doc")
            if doc is None:
                continue
            row["children"].append({
                "page_content": doc.page_content or "",
                "metadata": _json_safe_meta(dict(doc.metadata or {})),
                "score": float(child.get("score", 0.0)),
            })
        serialized.append(row)
    return serialized


def deserialize_parent_evidence(data: list[dict]) -> list[dict]:
    restored: list[dict] = []
    for parent in data:
        row = {k: v for k, v in parent.items() if k != "children"}
        row["children"] = []
        for child in parent.get("children") or []:
            doc = Document(
                page_content=str(child.get("page_content", "")),
                metadata=dict(child.get("metadata") or {}),
            )
            row["children"].append({
                "doc": doc,
                "score": float(child.get("score", 0.0)),
            })
        restored.append(row)
    return restored


def _hydrate_docs_by_ids(vectorstore: Chroma, ids: list[str]) -> list[Document]:
    if not ids:
        return []
    try:
        payload = vectorstore._collection.get(ids=ids, include=["documents", "metadatas"])
    except Exception:
        logger.exception("hydrate_chunks_by_id_failed")
        return []
    if not payload:
        return []
    raw_ids = payload.get("ids") or []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    by_id: dict[str, Document] = {}
    for i, cid in enumerate(raw_ids):
        text = documents[i] if i < len(documents) else ""
        meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        by_id[str(cid)] = Document(page_content=text or "", metadata=dict(meta))
    return [by_id[i] for i in ids if i in by_id]
NOISY_PAGE_PENALTIES = {
    "toc": 0.20,
    "index": 0.15,
    "copyright": 0.20,
    "acknowledgments": 0.30,
    "front_matter": 0.45,
    "back_matter": 0.50,
    "content": 1.00,
}
MAX_CHILDREN_PER_PARENT_CONTEXT = 5
SNIPPET_CHARS = 420
PARENT_CONTEXT_CHARS = 2300


@dataclass(frozen=True)
class RetrievalQuerySpec:
    source: str
    text: str
    weight: float


def _query_plan_is_noop(query_plan: QueryPlannerPlan | None) -> bool:
    if query_plan is None:
        return True
    reason = str(query_plan.reason or "")
    return reason.startswith("planner_") and reason.endswith("_noop")


def _serialize_query_plan(query_plan: QueryPlannerPlan | None) -> dict | None:
    if query_plan is None:
        return None
    return query_plan.model_dump()


def _serialize_query_specs(specs: list[RetrievalQuerySpec]) -> list[dict[str, float | str]]:
    return [
        {
            "source": spec.source,
            "text": spec.text,
            "weight": spec.weight,
        }
        for spec in specs
    ]


def _normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _append_query_spec(
    specs: list[RetrievalQuerySpec],
    *,
    seen: set[str],
    source: str,
    text: str,
    weight: float,
) -> None:
    cleaned = _normalize_query_text(text)
    if not cleaned:
        return
    key = cleaned.lower()
    if key in seen:
        return
    specs.append(RetrievalQuerySpec(source=source, text=cleaned, weight=weight))
    seen.add(key)


def _build_dense_query_specs(
    question: str,
    query_plan: QueryPlannerPlan | None,
) -> list[RetrievalQuerySpec]:
    specs: list[RetrievalQuerySpec] = []
    seen: set[str] = set()
    _append_query_spec(
        specs,
        seen=seen,
        source="original_question",
        text=question,
        weight=1.0,
    )

    if _query_plan_is_noop(query_plan):
        return specs

    standalone_weight = 0.35 if query_plan.needs_exact_phrase_bias else 0.65
    if query_plan.should_expand or query_plan.query_type in {"follow_up", "ambiguous"}:
        standalone_weight = 0.75 if not query_plan.needs_exact_phrase_bias else standalone_weight
    _append_query_spec(
        specs,
        seen=seen,
        source="standalone_question",
        text=query_plan.standalone_question,
        weight=standalone_weight,
    )
    return specs[:2]


def _build_lexical_query_specs(
    question: str,
    query_plan: QueryPlannerPlan | None,
) -> list[RetrievalQuerySpec]:
    specs: list[RetrievalQuerySpec] = []
    seen: set[str] = set()
    original_weight = 1.25 if query_plan and not _query_plan_is_noop(query_plan) and query_plan.needs_exact_phrase_bias else 1.0
    _append_query_spec(
        specs,
        seen=seen,
        source="original_question",
        text=question,
        weight=original_weight,
    )

    if _query_plan_is_noop(query_plan):
        return specs

    standalone_weight = 0.55 if query_plan.needs_exact_phrase_bias else 0.85
    if not query_plan.needs_chapter_lookup and not query_plan.should_expand:
        standalone_weight = 0.75 if not query_plan.needs_exact_phrase_bias else standalone_weight
    _append_query_spec(
        specs,
        seen=seen,
        source="standalone_question",
        text=query_plan.standalone_question,
        weight=standalone_weight,
    )

    if not query_plan.needs_exact_phrase_bias:
        search_weight = 0.85 if query_plan.needs_chapter_lookup else 0.7
        for search_query in query_plan.search_queries:
            _append_query_spec(
                specs,
                seen=seen,
                source="planner_search_query",
                text=search_query,
                weight=search_weight,
            )

        keyword_query = " ".join(_normalize_query_text(keyword) for keyword in query_plan.keywords if keyword)
        keyword_weight = 0.55 if query_plan.needs_chapter_lookup else 0.45
        _append_query_spec(
            specs,
            seen=seen,
            source="planner_keywords",
            text=keyword_query,
            weight=keyword_weight,
        )
    return specs


def _accumulate_ranked_scores(
    docs_by_key: dict[str, Document],
    scores: dict[str, float],
    docs: list[Document],
    *,
    weight: float,
) -> None:
    for rank, doc in enumerate(docs, start=1):
        key = _doc_key(doc)
        docs_by_key.setdefault(key, doc)
        scores[key] = scores.get(key, 0.0) + (weight / (HYBRID_RRF_K + rank))


def get_embeddings() -> Embeddings:
    global _embeddings_singleton
    with _retrieval_resource_lock:
        if _embeddings_singleton is not None:
            return _embeddings_singleton

        if EMBEDDING_PROVIDER == "huggingface":
            logger.info("embeddings_provider_selected provider=huggingface model=%s", HF_EMBEDDING_MODEL)
            _embeddings_singleton = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
        elif EMBEDDING_PROVIDER == "gemini":
            # Prefer explicit API key over ambient ADC credentials to avoid scope issues.
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            logger.info("embeddings_provider_selected provider=gemini model=%s", GEMINI_EMBEDDING_MODEL)
            _embeddings_singleton = GoogleGenerativeAIEmbeddings(
                model=GEMINI_EMBEDDING_MODEL,
                google_api_key=api_key,
            )
        else:
            raise ValueError(
                f"Unsupported EMBEDDING_PROVIDER='{EMBEDDING_PROVIDER}'. "
                "Expected one of: huggingface, gemini."
            )
        return _embeddings_singleton


def get_vectorstore() -> Chroma:
    global _vectorstore_singleton
    with _retrieval_resource_lock:
        if _vectorstore_singleton is not None:
            return _vectorstore_singleton

        logger.info(
            "vectorstore_init provider=%s collection=%s persist_directory=%s",
            EMBEDDING_PROVIDER,
            VECTORSTORE_COLLECTION_NAME,
            CHROMA_DIR,
        )
        _vectorstore_singleton = Chroma(
            collection_name=VECTORSTORE_COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
        return _vectorstore_singleton


def _build_where_filter(book_ids: list[str] | None) -> dict | None:
    if not book_ids:
        return None
    if len(book_ids) == 1:
        return {"book_id": book_ids[0]}
    return {"book_id": {"$in": book_ids}}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _truncate_text(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 3)].rstrip() + "..."


def _doc_text_for_retrieval(doc: Document) -> str:
    contextual = str(doc.metadata.get("contextual_text", "") or "").strip()
    if contextual:
        return f"{contextual}\n{doc.page_content or ''}"
    return doc.page_content or ""


def _infer_page_category_from_doc(doc: Document) -> str:
    category = str(doc.metadata.get("page_category", "") or "").strip().lower()
    if category:
        return category

    text = (doc.page_content or "").strip().lower()
    first_line = ""
    for line in text.splitlines():
        if line.strip():
            first_line = line.strip()
            break

    if first_line in {"contents", "table of contents"}:
        return "toc"
    if any(marker in text for marker in ("copyright", "all rights reserved", "isbn")):
        return "copyright"
    if "acknowledgments" in text or "acknowledgements" in text:
        return "acknowledgments"
    if first_line == "index" or ("see also" in text and "index" in text):
        return "index"
    if any(marker in text for marker in ("appendix", "bibliography", "references", "glossary")):
        return "back_matter"
    if any(marker in text for marker in ("preface", "foreword", "dedication")):
        return "front_matter"
    return "content"


def _noise_penalty(doc: Document) -> float:
    category = _infer_page_category_from_doc(doc)
    penalty = NOISY_PAGE_PENALTIES.get(category, 1.0)
    if bool(doc.metadata.get("is_low_value_page", False)):
        penalty = min(penalty, 0.40)

    tokens = _tokenize(_doc_text_for_retrieval(doc))
    if len(tokens) < 20:
        penalty *= 0.80
    if tokens:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < 0.25:
            penalty *= 0.75
    return max(0.05, penalty)


def _doc_key(doc: Document) -> str:
    book_id = str(doc.metadata.get("book_id", ""))
    chunk_index = doc.metadata.get("chunk_index")
    if chunk_index is not None:
        return f"{book_id}::chunk::{chunk_index}"

    section_id = str(doc.metadata.get("section_id", ""))
    page_start = int(doc.metadata.get("page_start", 0) or 0)
    page_end = int(doc.metadata.get("page_end", page_start) or page_start)
    snippet = (doc.page_content or "")[:140].strip().lower()
    return f"{book_id}|{section_id}|{page_start}|{page_end}|{snippet}"


def _chunk_trace_id(doc: Document) -> str:
    book_id = str(doc.metadata.get("book_id", "") or "")
    chunk_index = doc.metadata.get("chunk_index")
    if chunk_index is not None:
        return f"{book_id}::chunk::{chunk_index}"
    return _doc_key(doc)


def _parent_key(doc: Document) -> str:
    book_id = str(doc.metadata.get("book_id", "") or "")
    parent_id = str(doc.metadata.get("parent_section_id") or doc.metadata.get("section_id") or "")
    if parent_id:
        return f"{book_id}::{parent_id}"
    page_start = int(doc.metadata.get("page_start", 0) or 0)
    page_end = int(doc.metadata.get("page_end", page_start) or page_start)
    return f"{book_id}::page::{page_start}-{page_end}"


def _metadata_quality(doc: Document) -> float:
    score = 0.35
    if doc.metadata.get("chapter_title"):
        score += 0.20
    if doc.metadata.get("subchapter_title"):
        score += 0.15
    if doc.metadata.get("section_title"):
        score += 0.20
    if _infer_page_category_from_doc(doc) == "content":
        score += 0.10
    if doc.metadata.get("structure_source") == "toc":
        score += 0.10
    return min(1.0, score)


def _dense_search(vectorstore: Chroma, question: str, fetch_k: int, book_ids: list[str] | None) -> list[Document]:
    where_filter = _build_where_filter(book_ids)
    if not where_filter:
        return vectorstore.similarity_search(question, k=fetch_k)

    try:
        return vectorstore.similarity_search(question, k=fetch_k, filter=where_filter)
    except Exception:
        logger.exception("vector_search_filtered_failed using_fallback=true")
        retrieved = vectorstore.similarity_search(question, k=max(fetch_k * 2, 20))
        allowed = set(book_ids or [])
        return [doc for doc in retrieved if doc.metadata.get("book_id") in allowed][:fetch_k]


def _load_lexical_corpus(vectorstore: Chroma, book_ids: list[str] | None) -> list[Document]:
    where_filter = _build_where_filter(book_ids)
    collection = vectorstore._collection
    kwargs = {"include": ["documents", "metadatas"], "limit": LEXICAL_MAX_DOCS}
    if where_filter:
        kwargs["where"] = where_filter

    try:
        payload = collection.get(**kwargs)
    except TypeError:
        kwargs.pop("limit", None)
        payload = collection.get(**kwargs)
    except Exception:
        logger.exception("lexical_corpus_fetch_failed filtered=%s", bool(where_filter))
        return []

    documents = payload.get("documents", []) if payload else []
    metadatas = payload.get("metadatas", []) if payload else []
    corpus = []
    for idx, text in enumerate(documents):
        metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
        corpus.append(Document(page_content=text or "", metadata=metadata))
    return corpus


def _bm25_rank(question: str, docs: list[Document], top_k: int) -> list[Document]:
    query_tokens = _tokenize(question)
    if not query_tokens or not docs:
        return []

    tokenized_docs = [_tokenize(_doc_text_for_retrieval(doc)) for doc in docs]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avgdl = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 1.0
    avgdl = max(avgdl, 1.0)

    query_terms = set(query_tokens)
    doc_freq = {term: 0 for term in query_terms}
    term_counts = []
    for tokens in tokenized_docs:
        counter = Counter(tokens)
        term_counts.append(counter)
        present = query_terms.intersection(counter.keys())
        for term in present:
            doc_freq[term] += 1

    n_docs = len(docs)
    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, int]] = []

    for idx, counter in enumerate(term_counts):
        dl = max(1, doc_lengths[idx])
        score = 0.0
        for term in query_terms:
            tf = counter.get(term, 0)
            if tf <= 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(((n_docs - df + 0.5) / (df + 0.5)) + 1.0)
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf * (k1 + 1) / max(denom, 1e-9))
        if score > 0:
            scored.append((score, idx))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [docs[idx] for _, idx in scored[:top_k]]


def _score_ranked_list(docs: list[Document], *, weight: float) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rank, doc in enumerate(docs, start=1):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + (weight / (HYBRID_RRF_K + rank))
    return scores


def _postprocess_ranked_docs(docs: list[Document], base_scores: dict[str, float], top_k: int) -> list[Document]:
    docs_by_key: dict[str, Document] = {}
    merged_scores: dict[str, float] = {}
    for doc in docs:
        key = _doc_key(doc)
        docs_by_key.setdefault(key, doc)
        merged_scores[key] = max(merged_scores.get(key, 0.0), base_scores.get(key, 0.0))

    ranked_keys = sorted(
        merged_scores.keys(),
        key=lambda key: merged_scores[key] * _noise_penalty(docs_by_key[key]),
        reverse=True,
    )

    selected: list[Document] = []
    seen_pages = set()
    parent_counts = defaultdict(int)
    noisy_count = 0
    max_noisy = max(1, top_k // 4)

    for key in ranked_keys:
        doc = docs_by_key[key]
        page_key = (
            doc.metadata.get("book_id"),
            int(doc.metadata.get("page_start", 0) or 0),
            int(doc.metadata.get("page_end", doc.metadata.get("page_start", 0)) or 0),
        )
        if page_key in seen_pages:
            continue

        parent_id = doc.metadata.get("parent_section_id") or doc.metadata.get("section_id")
        if parent_id and parent_counts[parent_id] >= 1:
            continue

        category = _infer_page_category_from_doc(doc)
        if category != "content" and noisy_count >= max_noisy:
            continue

        selected.append(doc)
        seen_pages.add(page_key)
        if parent_id:
            parent_counts[parent_id] += 1
        if category != "content":
            noisy_count += 1

        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for key in ranked_keys:
            doc = docs_by_key[key]
            if doc in selected:
                continue
            page_key = (
                doc.metadata.get("book_id"),
                int(doc.metadata.get("page_start", 0) or 0),
                int(doc.metadata.get("page_end", doc.metadata.get("page_start", 0)) or 0),
            )
            if page_key in seen_pages:
                continue
            selected.append(doc)
            seen_pages.add(page_key)
            if len(selected) >= top_k:
                break

    return selected


def _retrieve_child_rank_state(
    question: str,
    *,
    book_ids: list[str] | None,
    top_k: int,
    query_plan: QueryPlannerPlan | None = None,
) -> tuple[
    dict[str, Document],
    dict[str, float],
    str,
    int,
    int,
    dict[str, int],
    list[dict[str, float | str]],
    list[dict[str, float | str]],
]:
    started = time.perf_counter()
    mode = RETRIEVAL_MODE if RETRIEVAL_MODE in VALID_RETRIEVAL_MODES else "hybrid"
    fetch_k = max(top_k * 4, RETRIEVAL_FETCH_K)
    dense_query_specs = _build_dense_query_specs(question, query_plan) if mode in {"dense_only", "hybrid"} else []
    lexical_query_specs = _build_lexical_query_specs(question, query_plan) if mode in {"lexical_only", "hybrid"} else []
    log_structured(
        logger,
        "retrieval_requested",
        mode=mode,
        top_k=top_k,
        fetch_k=fetch_k,
        selected_books=book_ids or [],
        book_filter_count=len(book_ids or []),
        question_len=len(question or ""),
        question_snippet=sanitize_for_debug(question, LOG_DEBUG_SNIPPET_CHARS),
        planner_query_type=query_plan.query_type if query_plan is not None else None,
        dense_queries=[spec.text for spec in dense_query_specs],
        lexical_queries=[spec.text for spec in lexical_query_specs],
    )

    vectorstore = get_vectorstore()

    dense_docs_by_key: dict[str, Document] = {}
    lexical_docs_by_key: dict[str, Document] = {}
    dense_scores: dict[str, float] = {}
    lexical_scores: dict[str, float] = {}
    dense_ms = 0
    lexical_ms = 0
    if mode in {"dense_only", "hybrid"}:
        dense_started = time.perf_counter()
        for spec in dense_query_specs:
            dense_docs = _dense_search(vectorstore, spec.text, fetch_k, book_ids)
            _accumulate_ranked_scores(
                dense_docs_by_key,
                dense_scores,
                dense_docs,
                weight=spec.weight,
            )
        dense_ms = int((time.perf_counter() - dense_started) * 1000)
    if mode in {"lexical_only", "hybrid"}:
        lexical_started = time.perf_counter()
        if ENABLE_LEXICAL_RETRIEVAL:
            for spec in lexical_query_specs:
                chunk_ids = lexical_index.search_chunk_ids(
                    engine,
                    spec.text,
                    list(book_ids) if book_ids else None,
                    limit=fetch_k,
                )
                log_structured(
                    logger,
                    "lexical_fts_search",
                    query_snippet=sanitize_for_debug(spec.text, LOG_DEBUG_SNIPPET_CHARS),
                    fts_hit_count=len(chunk_ids),
                    fetch_k=fetch_k,
                    book_filter_count=len(book_ids or []),
                )
                lexical_docs = _hydrate_docs_by_ids(vectorstore, chunk_ids)
                if chunk_ids and len(lexical_docs) < len(chunk_ids):
                    logger.warning(
                        "lexical_fts_hydrate_partial requested=%s hydrated=%s",
                        len(chunk_ids),
                        len(lexical_docs),
                    )
                _accumulate_ranked_scores(
                    lexical_docs_by_key,
                    lexical_scores,
                    lexical_docs,
                    weight=spec.weight,
                )
            if not lexical_docs_by_key:
                log_structured(
                    logger,
                    "lexical_fts_empty_contribution",
                    reason="no_fts_hits_or_no_hydrated_docs",
                    mode=mode,
                    book_filter_count=len(book_ids or []),
                )
        else:
            lexical_corpus = _load_lexical_corpus(vectorstore, book_ids)
            for spec in lexical_query_specs:
                lexical_docs = _bm25_rank(spec.text, lexical_corpus, fetch_k)
                _accumulate_ranked_scores(
                    lexical_docs_by_key,
                    lexical_scores,
                    lexical_docs,
                    weight=spec.weight,
                )
        lexical_ms = int((time.perf_counter() - lexical_started) * 1000)

    merge_started = time.perf_counter()
    if mode == "dense_only":
        base_scores = dict(dense_scores)
        combined_docs = list(dense_docs_by_key.values())
    elif mode == "lexical_only":
        base_scores = dict(lexical_scores)
        combined_docs = list(lexical_docs_by_key.values())
    else:
        combined_docs = list(dense_docs_by_key.values()) + list(lexical_docs_by_key.values())
        all_keys = set(dense_scores) | set(lexical_scores)
        base_scores = {
            key: HYBRID_DENSE_WEIGHT * dense_scores.get(key, 0.0)
            + HYBRID_LEXICAL_WEIGHT * lexical_scores.get(key, 0.0)
            for key in all_keys
        }

    docs_by_key: dict[str, Document] = {}
    for doc in combined_docs:
        key = _doc_key(doc)
        docs_by_key.setdefault(key, doc)

    merge_ms = int((time.perf_counter() - merge_started) * 1000)
    total_ms = int((time.perf_counter() - started) * 1000)
    stage_latencies_ms = {
        "dense_search": dense_ms,
        "lexical_search": lexical_ms,
        "merge_rank": merge_ms,
        "total": total_ms,
    }

    log_structured(
        logger,
        "retrieval_child_candidates",
        mode=mode,
        dense=len(dense_docs_by_key),
        lexical=len(lexical_docs_by_key),
        unique=len(docs_by_key),
        dense_queries=[spec.text for spec in dense_query_specs],
        lexical_queries=[spec.text for spec in lexical_query_specs],
        stage_latency_ms=stage_latencies_ms,
    )
    return (
        docs_by_key,
        base_scores,
        mode,
        len(dense_docs_by_key),
        len(lexical_docs_by_key),
        stage_latencies_ms,
        _serialize_query_specs(dense_query_specs),
        _serialize_query_specs(lexical_query_specs),
    )


def _ranked_child_docs(docs_by_key: dict[str, Document], base_scores: dict[str, float]) -> list[Document]:
    return sorted(
        docs_by_key.values(),
        key=lambda doc: base_scores.get(_doc_key(doc), 0.0) * _noise_penalty(doc),
        reverse=True,
    )


def _build_parent_evidence(
    ranked_children: list[Document],
    *,
    base_scores: dict[str, float],
    top_k: int,
    max_children_per_parent: int = MAX_CHILDREN_PER_PARENT_CONTEXT,
) -> list[dict]:
    grouped: dict[str, dict] = {}
    for doc in ranked_children:
        doc_key = _doc_key(doc)
        child_score = base_scores.get(doc_key, 0.0) * _noise_penalty(doc)
        if child_score <= 0:
            continue

        parent_key = _parent_key(doc)
        group = grouped.setdefault(
            parent_key,
            {
                "parent_key": parent_key,
                "parent_section_id": doc.metadata.get("parent_section_id") or doc.metadata.get("section_id"),
                "book_id": doc.metadata.get("book_id", ""),
                "book_title": doc.metadata.get("book_title", "Unknown"),
                "chapter_title": doc.metadata.get("chapter_title"),
                "subchapter_title": doc.metadata.get("subchapter_title"),
                "section_title": doc.metadata.get("section_title"),
                "structure_source": doc.metadata.get("structure_source"),
                "children": [],
                "best_child_score": 0.0,
                "metadata_quality_sum": 0.0,
                "page_start": int(doc.metadata.get("parent_page_start", doc.metadata.get("page_start", 0)) or 0),
                "page_end": int(doc.metadata.get("parent_page_end", doc.metadata.get("page_end", 0)) or 0),
                "parent_excerpt": str(doc.metadata.get("parent_excerpt", "") or ""),
            },
        )

        group["children"].append({"doc": doc, "score": child_score})
        group["best_child_score"] = max(group["best_child_score"], child_score)
        group["metadata_quality_sum"] += _metadata_quality(doc)
        group["page_start"] = min(
            group["page_start"] or int(doc.metadata.get("page_start", 0) or 0),
            int(doc.metadata.get("parent_page_start", doc.metadata.get("page_start", 0)) or 0),
        )
        group["page_end"] = max(
            group["page_end"],
            int(doc.metadata.get("parent_page_end", doc.metadata.get("page_end", 0)) or 0),
        )

    parent_ranked: list[dict] = []
    for group in grouped.values():
        group["children"].sort(key=lambda item: item["score"], reverse=True)
        hit_count = len(group["children"])
        mean_quality = group["metadata_quality_sum"] / max(hit_count, 1)
        parent_score = (
            group["best_child_score"]
            + (0.08 * math.log1p(hit_count))
            + (0.06 * mean_quality)
        )
        confidence = max(0.0, min(1.0, parent_score * 20.0))
        top_children = group["children"][:max_children_per_parent]
        if not group["parent_excerpt"]:
            snippets = [
                _truncate_text(child["doc"].page_content, SNIPPET_CHARS)
                for child in top_children
            ]
            group["parent_excerpt"] = _truncate_text(" ".join(snippets), PARENT_CONTEXT_CHARS)
        else:
            group["parent_excerpt"] = _truncate_text(group["parent_excerpt"], PARENT_CONTEXT_CHARS)

        parent_ranked.append(
            {
                "parent_section_id": group["parent_section_id"],
                "parent_score": parent_score,
                "confidence": confidence,
                "book_id": group["book_id"],
                "book_title": group["book_title"],
                "chapter_title": group["chapter_title"],
                "subchapter_title": group["subchapter_title"],
                "section_title": group["section_title"],
                "structure_source": group["structure_source"],
                "page_start": group["page_start"],
                "page_end": group["page_end"],
                "children": top_children,
                "parent_excerpt": group["parent_excerpt"],
            }
        )

    parent_ranked.sort(key=lambda item: item["parent_score"], reverse=True)
    return parent_ranked[:top_k]


def delete_book_vectors(book_id: str) -> int:
    logger.info("vector_delete_requested book_id=%s", book_id)
    lexical_index.delete_book(engine, book_id)
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


def search_documents(
    question: str,
    top_k: int,
    book_ids: list[str] | None = None,
    *,
    query_plan: QueryPlannerPlan | None = None,
):
    docs_by_key, base_scores, mode, dense_count, lexical_count, stage_latencies_ms, _, _ = _retrieve_child_rank_state(
        question,
        book_ids=book_ids,
        top_k=top_k,
        query_plan=query_plan,
    )
    ranked_children = _ranked_child_docs(docs_by_key, base_scores)
    results = _postprocess_ranked_docs(ranked_children, base_scores, top_k)
    result_chunk_ids = [_chunk_trace_id(doc) for doc in results[:RETRIEVAL_TRACE_MAX_ITEMS]]
    log_structured(
        logger,
        "retrieval_completed",
        mode=mode,
        dense=dense_count,
        lexical=lexical_count,
        final=len(results),
        selected_chunk_ids=result_chunk_ids,
        stage_latency_ms=stage_latencies_ms,
    )
    return results


@maybe_traceable(
    run_type="retriever",
    name="search_parent_evidence",
    process_inputs=trace_process_inputs_search_parent_evidence,
    process_outputs=trace_process_outputs_search_parent_evidence,
)
def search_parent_evidence(
    question: str,
    top_k: int,
    book_ids: list[str] | None = None,
    *,
    query_plan: QueryPlannerPlan | None = None,
    debug_info: dict | None = None,
) -> list[dict]:
    started = time.perf_counter()
    retrieval_cache_hit = False
    retrieval_cache_latency_ms = 0
    retrieval_cache_key_hash: str | None = None
    cache_key_full: str | None = None

    norm_q = _normalize_query_text(question).lower()
    planner_fp = _planner_fingerprint(query_plan)
    if ENABLE_RETRIEVAL_CACHE:
        cache_key_full = build_retrieval_cache_key(
            normalized_query=norm_q,
            book_ids=list(book_ids) if book_ids else None,
            retrieval_mode=RETRIEVAL_MODE if RETRIEVAL_MODE in VALID_RETRIEVAL_MODES else "hybrid",
            top_k=top_k,
            planner_fingerprint=planner_fp,
            embedding_provider=EMBEDDING_PROVIDER,
            embedding_model=_embedding_model_for_cache(),
            lexical_index_version=LEXICAL_INDEX_VERSION,
            config_version=CACHE_CONFIG_VERSION,
            enable_lexical_retrieval=ENABLE_LEXICAL_RETRIEVAL,
            hybrid_dense_weight=HYBRID_DENSE_WEIGHT,
            hybrid_lexical_weight=HYBRID_LEXICAL_WEIGHT,
            hybrid_rrf_k=HYBRID_RRF_K,
            reranker_enabled=bool(RERANKER_ENABLED),
            reranker_provider=RERANKER_PROVIDER,
            reranker_model=RERANKER_MODEL,
            reranker_top_n=RERANKER_TOP_N,
        )
        retrieval_cache_key_hash = key_hash_hex(
            {"k": cache_key_full, "internal": "retrieval_parent_evidence_v1"},
        )
        cache_backend = get_cache_backend(redis_url=REDIS_URL)
        cache_started = time.perf_counter()
        try:
            cached = cache_backend.get(cache_key_full)
        except Exception:
            logger.exception("retrieval_cache_get_failed")
            cached = None
        retrieval_cache_latency_ms = int((time.perf_counter() - cache_started) * 1000)
        if cached is not None:
            retrieval_cache_hit = True
            parent_evidence = deserialize_parent_evidence(cached)
            total_ms = int((time.perf_counter() - started) * 1000)
            if debug_info is not None:
                debug_info.update(
                    {
                        "retrieval_mode": RETRIEVAL_MODE,
                        "top_k": top_k,
                        "selected_books": list(book_ids or []),
                        "dense_candidates": 0,
                        "lexical_candidates": 0,
                        "dense_queries": [],
                        "lexical_queries": [],
                        "candidate_children_count": 0,
                        "parent_sections_count": len(parent_evidence),
                        "retrieved_child_chunk_ids": [],
                        "grouped_section_ids": [],
                        "top_group_child_chunk_ids": [],
                        "query_plan": _serialize_query_plan(query_plan),
                        "stage_latency_ms": {"search_parent_total": total_ms, "total": total_ms},
                        "retrieval_cache_hit": True,
                        "retrieval_cache_key_hash": retrieval_cache_key_hash,
                        "retrieval_cache_latency_ms": retrieval_cache_latency_ms,
                        "dense_result_count": 0,
                        "lexical_result_count": 0,
                        "hybrid_result_count": 0,
                        "reranker_enabled": False,
                        "reranker_latency_ms": 0,
                        "total_retrieval_latency_ms": total_ms,
                    }
                )
            return parent_evidence

    docs_by_key, base_scores, mode, dense_count, lexical_count, stage_latencies_ms, dense_queries, lexical_queries = _retrieve_child_rank_state(
        question,
        book_ids=book_ids,
        top_k=max(top_k * 2, 8),
        query_plan=query_plan,
    )
    hybrid_result_count = len(docs_by_key)
    rank_started = time.perf_counter()
    ranked_children = _ranked_child_docs(docs_by_key, base_scores)
    child_rank_ms = int((time.perf_counter() - rank_started) * 1000)
    candidate_children = ranked_children[: max(top_k * 10, RETRIEVAL_FETCH_K)]

    reranker = get_reranker()
    rerank_started = time.perf_counter()
    reranked_children, reranked_scores = reranker.rerank(question, candidate_children, base_scores)
    rerank_ms = int((time.perf_counter() - rerank_started) * 1000)

    group_started = time.perf_counter()
    parent_evidence = _build_parent_evidence(
        reranked_children,
        base_scores=reranked_scores,
        top_k=top_k,
    )
    group_ms = int((time.perf_counter() - group_started) * 1000)
    total_ms = int((time.perf_counter() - started) * 1000)

    if ENABLE_RETRIEVAL_CACHE and cache_key_full:
        cache_backend = get_cache_backend(redis_url=REDIS_URL)
        try:
            cache_backend.set(
                cache_key_full,
                serialize_parent_evidence(parent_evidence),
                RETRIEVAL_CACHE_TTL_SECONDS,
            )
        except Exception:
            logger.exception("retrieval_cache_set_failed")

    child_chunk_ids = [
        _chunk_trace_id(doc) for doc in reranked_children[:RETRIEVAL_TRACE_MAX_ITEMS]
    ]
    grouped_section_ids = [
        str(item.get("parent_section_id", ""))
        for item in parent_evidence[:RETRIEVAL_TRACE_MAX_ITEMS]
    ]
    grouped_section_ids = [value for value in grouped_section_ids if value]
    top_group_child_chunk_ids = []
    for parent in parent_evidence[:RETRIEVAL_TRACE_MAX_ITEMS]:
        children = parent.get("children") or []
        if not children:
            continue
        best_child = children[0].get("doc")
        if best_child is None:
            continue
        top_group_child_chunk_ids.append(_chunk_trace_id(best_child))

    stage_trace = {
        **stage_latencies_ms,
        "child_rank": child_rank_ms,
        "rerank": rerank_ms,
        "parent_group": group_ms,
        "search_parent_total": total_ms,
    }
    log_structured(
        logger,
        "parent_retrieval_completed",
        mode=mode,
        dense=dense_count,
        lexical=lexical_count,
        planner_query_type=query_plan.query_type if query_plan is not None else None,
        dense_queries=dense_queries,
        lexical_queries=lexical_queries,
        parent_sections=len(parent_evidence),
        candidates=len(reranked_children),
        retrieved_child_chunk_ids=child_chunk_ids,
        grouped_section_ids=grouped_section_ids,
        top_group_child_chunk_ids=top_group_child_chunk_ids,
        stage_latency_ms=stage_trace,
    )

    if debug_info is not None:
        if retrieval_cache_key_hash is None and cache_key_full:
            retrieval_cache_key_hash = key_hash_hex(
                {"k": cache_key_full, "internal": "retrieval_parent_evidence_v1"},
            )
        debug_info.update(
            {
                "retrieval_mode": mode,
                "top_k": top_k,
                "selected_books": list(book_ids or []),
                "dense_candidates": dense_count,
                "lexical_candidates": lexical_count,
                "dense_queries": dense_queries,
                "lexical_queries": lexical_queries,
                "candidate_children_count": len(reranked_children),
                "parent_sections_count": len(parent_evidence),
                "retrieved_child_chunk_ids": child_chunk_ids,
                "grouped_section_ids": grouped_section_ids,
                "top_group_child_chunk_ids": top_group_child_chunk_ids,
                "query_plan": _serialize_query_plan(query_plan),
                "stage_latency_ms": stage_trace,
                "retrieval_cache_hit": retrieval_cache_hit,
                "retrieval_cache_key_hash": retrieval_cache_key_hash,
                "retrieval_cache_latency_ms": retrieval_cache_latency_ms,
                "dense_result_count": dense_count,
                "lexical_result_count": lexical_count,
                "hybrid_result_count": hybrid_result_count,
                "reranker_enabled": RERANKER_ENABLED,
                "reranker_latency_ms": rerank_ms,
                "total_retrieval_latency_ms": total_ms,
            }
        )
    return parent_evidence
