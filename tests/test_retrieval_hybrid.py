from langchain_core.documents import Document

from app.schemas import QueryPlannerPlan
from app.services.retrieval import (
    _bm25_rank,
    _build_dense_query_specs,
    _build_parent_evidence,
    _build_lexical_query_specs,
    _postprocess_ranked_docs,
    _score_ranked_list,
)


def test_bm25_rank_prefers_exact_phrase_overlap():
    docs = [
        Document(
            page_content="Neural networks and optimization overview",
            metadata={"book_id": "b1", "chunk_index": 0},
        ),
        Document(
            page_content="Chapter title: gradient descent in deep learning systems",
            metadata={"book_id": "b1", "chunk_index": 1},
        ),
        Document(
            page_content="Completely unrelated appendix material",
            metadata={"book_id": "b1", "chunk_index": 2},
        ),
    ]

    ranked = _bm25_rank("gradient descent deep learning", docs, top_k=2)

    assert ranked
    assert ranked[0].metadata["chunk_index"] == 1


def test_postprocess_suppresses_duplicate_pages_and_excess_noisy_docs():
    docs = [
        Document(
            page_content="Table of contents\nChapter 1 .... 10\nChapter 2 .... 20",
            metadata={
                "book_id": "b1",
                "chunk_index": 0,
                "page_start": 1,
                "page_end": 1,
                "page_category": "toc",
            },
        ),
        Document(
            page_content="Core explanation of retrieval quality and citations.",
            metadata={
                "book_id": "b1",
                "chunk_index": 1,
                "page_start": 25,
                "page_end": 25,
                "page_category": "content",
                "section_id": "s1",
                "parent_section_id": "s1",
            },
        ),
        Document(
            page_content="More core explanation from same page should be deduplicated.",
            metadata={
                "book_id": "b1",
                "chunk_index": 2,
                "page_start": 25,
                "page_end": 25,
                "page_category": "content",
                "section_id": "s1",
                "parent_section_id": "s1",
            },
        ),
    ]
    base_scores = _score_ranked_list(docs, weight=1.0)

    selected = _postprocess_ranked_docs(docs, base_scores, top_k=2)

    assert selected
    assert len(selected) == 2
    page_pairs = {(d.metadata.get("page_start"), d.metadata.get("page_end")) for d in selected}
    assert len(page_pairs) == 2


def test_parent_evidence_groups_child_hits_by_section():
    docs = [
        Document(
            page_content="Gradient descent is used for optimization.",
            metadata={
                "book_id": "b1",
                "book_title": "Book One",
                "chunk_index": 1,
                "section_id": "s1",
                "parent_section_id": "s1",
                "section_title": "Optimization Basics",
                "chapter_title": "Chapter 2",
                "page_start": 20,
                "page_end": 20,
                "parent_page_start": 19,
                "parent_page_end": 24,
                "parent_excerpt": "Optimization basics discussion.",
                "page_category": "content",
            },
        ),
        Document(
            page_content="Learning rate controls convergence speed.",
            metadata={
                "book_id": "b1",
                "book_title": "Book One",
                "chunk_index": 2,
                "section_id": "s1",
                "parent_section_id": "s1",
                "section_title": "Optimization Basics",
                "chapter_title": "Chapter 2",
                "page_start": 21,
                "page_end": 21,
                "parent_page_start": 19,
                "parent_page_end": 24,
                "page_category": "content",
            },
        ),
        Document(
            page_content="Unrelated appendix text.",
            metadata={
                "book_id": "b1",
                "book_title": "Book One",
                "chunk_index": 3,
                "section_id": "s2",
                "parent_section_id": "s2",
                "section_title": "Appendix",
                "chapter_title": "Appendix",
                "page_start": 120,
                "page_end": 120,
                "parent_page_start": 120,
                "parent_page_end": 130,
                "page_category": "back_matter",
            },
        ),
    ]
    base_scores = _score_ranked_list(docs, weight=1.0)
    ranked_docs = docs

    evidence = _build_parent_evidence(ranked_docs, base_scores=base_scores, top_k=2)

    assert evidence
    assert evidence[0]["parent_section_id"] == "s1"
    assert evidence[0]["page_start"] == 19
    assert evidence[0]["page_end"] == 24
    assert len(evidence[0]["children"]) == 2


def test_dense_query_specs_keep_original_question_as_primary_anchor():
    plan = QueryPlannerPlan(
        query_type="follow_up",
        standalone_question="What does the next chapter say about BM25 retrieval?",
        search_queries=["BM25 retrieval next chapter"],
        keywords=["BM25", "retrieval"],
        should_expand=True,
        needs_exact_phrase_bias=False,
        needs_chapter_lookup=False,
        reason="planner_applied",
    )

    specs = _build_dense_query_specs("What about the next chapter?", plan)

    assert [spec.text for spec in specs] == [
        "What about the next chapter?",
        "What does the next chapter say about BM25 retrieval?",
    ]
    assert specs[0].weight > specs[1].weight


def test_lexical_query_specs_use_planner_queries_and_keywords_cautiously():
    plan = QueryPlannerPlan(
        query_type="chapter_lookup",
        standalone_question="Which chapter covers BM25 ranking?",
        search_queries=[
            "chapter BM25 ranking",
            "BM25 ranking chapter title",
        ],
        keywords=["bm25", "ranking", "inverted index"],
        should_expand=True,
        needs_exact_phrase_bias=False,
        needs_chapter_lookup=True,
        reason="planner_applied",
    )

    specs = _build_lexical_query_specs("Which chapter covers BM25?", plan)

    assert [spec.text for spec in specs] == [
        "Which chapter covers BM25?",
        "Which chapter covers BM25 ranking?",
        "chapter BM25 ranking",
        "BM25 ranking chapter title",
        "bm25 ranking inverted index",
    ]


def test_lexical_query_specs_do_not_broaden_exact_phrase_queries():
    plan = QueryPlannerPlan(
        query_type="original",
        standalone_question="Where is gradient descent defined?",
        search_queries=["gradient descent definition chapter"],
        keywords=["gradient", "descent", "definition"],
        should_expand=True,
        needs_exact_phrase_bias=True,
        needs_chapter_lookup=False,
        reason="planner_applied",
    )

    specs = _build_lexical_query_specs('Where is "gradient descent" defined?', plan)

    assert [spec.source for spec in specs] == [
        "original_question",
        "standalone_question",
    ]
    assert "gradient descent definition chapter" not in [spec.text for spec in specs]


def test_noop_planner_plan_does_not_change_dense_or_lexical_queries():
    plan = QueryPlannerPlan(
        query_type="original",
        standalone_question="Resolved standalone question that should be ignored",
        search_queries=["unused planner query"],
        keywords=["unused", "keywords"],
        should_expand=True,
        needs_exact_phrase_bias=True,
        needs_chapter_lookup=True,
        reason="planner_parse_failed_noop",
    )

    dense_specs = _build_dense_query_specs("Original user question", plan)
    lexical_specs = _build_lexical_query_specs("Original user question", plan)

    assert [spec.text for spec in dense_specs] == ["Original user question"]
    assert [spec.text for spec in lexical_specs] == ["Original user question"]
