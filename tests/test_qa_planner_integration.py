from app.schemas import QueryPlannerPlan
from app.services.qa import ABSTAIN_MESSAGE, answer_question


def test_answer_question_uses_planner_standalone_question_and_passes_query_plan(monkeypatch):
    captured = {}

    def fake_plan_retrieval_query(question: str, **kwargs):
        captured["planner_question"] = question
        captured["planner_kwargs"] = kwargs
        return QueryPlannerPlan(
            query_type="follow_up",
            standalone_question="Resolved standalone retrieval question",
            search_queries=["resolved retrieval question"],
            keywords=["resolved", "retrieval"],
            should_expand=False,
            needs_exact_phrase_bias=False,
            needs_chapter_lookup=False,
            reason="planner_applied",
        )

    def fake_search_parent_evidence(question: str, top_k: int, book_ids=None, *, query_plan=None, debug_info=None):
        captured["retrieval_question"] = question
        captured["top_k"] = top_k
        captured["book_ids"] = list(book_ids or [])
        captured["query_plan"] = query_plan
        if debug_info is not None:
            debug_info["called"] = True
        return []

    monkeypatch.setattr("app.services.qa.plan_retrieval_query", fake_plan_retrieval_query)
    monkeypatch.setattr("app.services.qa.search_parent_evidence", fake_search_parent_evidence)

    response = answer_question(
        "What about the next section?",
        book_ids=["b1"],
        top_k=3,
        include_debug=True,
        recent_history=[{"question": "Summarize chapter one", "answer": "Summary"}],
        enable_query_planner=True,
        enable_query_reformulation=True,
    )

    assert response["answer"] == ABSTAIN_MESSAGE
    assert response["grounded"] is False
    assert captured["planner_question"] == "What about the next section?"
    assert captured["retrieval_question"] == "What about the next section?"
    assert response["debug"]["retrieval_question"] == "What about the next section?"
    assert captured["query_plan"].standalone_question == "Resolved standalone retrieval question"
    assert response["debug"]["planner"]["standalone_question"] == "Resolved standalone retrieval question"
    assert response["debug"]["planner_applied"] is True
    assert response["debug"]["stage_latency_ms"]["planner"] >= 0


def test_answer_question_falls_back_to_legacy_reformulation_when_planner_returns_noop(monkeypatch):
    captured = {}

    def fake_plan_retrieval_query(question: str, **kwargs):
        captured["planner_question"] = question
        return QueryPlannerPlan(
            query_type="original",
            standalone_question=question,
            search_queries=[],
            keywords=[],
            should_expand=False,
            needs_exact_phrase_bias=False,
            needs_chapter_lookup=False,
            reason="planner_parse_failed_noop",
        )

    def fake_build_standalone_question(question: str, recent_history_text: str):
        captured["recent_history_text"] = recent_history_text
        return "Legacy reformulated question", True, 7

    def fake_search_parent_evidence(question: str, top_k: int, book_ids=None, *, query_plan=None, debug_info=None):
        captured["retrieval_question"] = question
        captured["query_plan"] = query_plan
        return []

    monkeypatch.setattr("app.services.qa.plan_retrieval_query", fake_plan_retrieval_query)
    monkeypatch.setattr("app.services.qa._build_standalone_question", fake_build_standalone_question)
    monkeypatch.setattr("app.services.qa.search_parent_evidence", fake_search_parent_evidence)

    response = answer_question(
        "What about the next section?",
        top_k=2,
        include_debug=True,
        recent_history=[{"question": "Summarize chapter one", "answer": "Summary"}],
        enable_query_planner=True,
        enable_query_reformulation=True,
    )

    assert response["answer"] == ABSTAIN_MESSAGE
    assert captured["retrieval_question"] == "Legacy reformulated question"
    assert response["debug"]["retrieval_question"] == "Legacy reformulated question"
    assert captured["query_plan"].reason == "planner_parse_failed_noop"
    assert response["debug"]["planner"]["reason"] == "planner_parse_failed_noop"
    assert response["debug"]["reformulation_used"] is True
    assert response["debug"]["stage_latency_ms"]["reformulation"] == 7


def test_answer_question_skips_planner_for_exact_phrase_queries(monkeypatch):
    captured = {"planner_called": False}

    def fake_plan_retrieval_query(question: str, **kwargs):
        captured["planner_called"] = True
        return QueryPlannerPlan(
            query_type="original",
            standalone_question=question,
            search_queries=[],
            keywords=[],
            should_expand=False,
            needs_exact_phrase_bias=True,
            needs_chapter_lookup=False,
            reason="planner_applied",
        )

    def fake_build_standalone_question(question: str, recent_history_text: str):
        captured["recent_history_text"] = recent_history_text
        return "Legacy reformulated question", True, 5

    def fake_search_parent_evidence(question: str, top_k: int, book_ids=None, *, query_plan=None, debug_info=None):
        captured["retrieval_question"] = question
        captured["query_plan"] = query_plan
        return []

    monkeypatch.setattr("app.services.qa.plan_retrieval_query", fake_plan_retrieval_query)
    monkeypatch.setattr("app.services.qa._build_standalone_question", fake_build_standalone_question)
    monkeypatch.setattr("app.services.qa.search_parent_evidence", fake_search_parent_evidence)

    response = answer_question(
        'Find the exact phrase "take a step back"',
        top_k=2,
        include_debug=True,
        recent_history=[{"question": "Which quote was that?", "answer": "Summary"}],
        enable_query_planner=True,
        enable_query_reformulation=True,
    )

    assert response["answer"] == ABSTAIN_MESSAGE
    assert captured["planner_called"] is False
    assert captured["retrieval_question"] == 'Find the exact phrase "take a step back"'
    assert "recent_history_text" not in captured
    assert response["debug"]["planner_used"] is False
    assert response["debug"]["planner_decision"]["reason"] == "exact_phrase_lookup"
    assert response["debug"]["legacy_rewrite_allowed"] is False


def test_answer_question_skips_legacy_rewrite_for_term_constrained_queries(monkeypatch):
    captured = {"planner_called": False}

    def fake_plan_retrieval_query(question: str, **kwargs):
        captured["planner_called"] = True
        return QueryPlannerPlan(
            query_type="original",
            standalone_question=question,
            search_queries=[],
            keywords=[],
            should_expand=False,
            needs_exact_phrase_bias=False,
            needs_chapter_lookup=False,
            reason="planner_applied",
        )

    def fake_build_standalone_question(question: str, recent_history_text: str):
        captured["recent_history_text"] = recent_history_text
        return "Legacy reformulated question", True, 5

    def fake_search_parent_evidence(question: str, top_k: int, book_ids=None, *, query_plan=None, debug_info=None):
        captured["retrieval_question"] = question
        captured["query_plan"] = query_plan
        return []

    monkeypatch.setattr("app.services.qa.plan_retrieval_query", fake_plan_retrieval_query)
    monkeypatch.setattr("app.services.qa._build_standalone_question", fake_build_standalone_question)
    monkeypatch.setattr("app.services.qa.search_parent_evidence", fake_search_parent_evidence)

    response = answer_question(
        "Define BM25",
        top_k=2,
        include_debug=True,
        recent_history=[{"question": "What did we discuss?", "answer": "Summary"}],
        enable_query_planner=True,
        enable_query_reformulation=True,
    )

    assert response["answer"] == ABSTAIN_MESSAGE
    assert captured["planner_called"] is False
    assert captured["retrieval_question"] == "Define BM25"
    assert "recent_history_text" not in captured
    assert response["debug"]["planner_used"] is False
    assert response["debug"]["planner_decision"]["reason"] == "term_constrained_lookup"
    assert response["debug"]["legacy_rewrite_allowed"] is False


def test_answer_question_can_force_planner_for_experiments(monkeypatch):
    captured = {}

    def fake_plan_retrieval_query(question: str, **kwargs):
        captured["planner_question"] = question
        return QueryPlannerPlan(
            query_type="original",
            standalone_question="Forced planner question",
            search_queries=[],
            keywords=[],
            should_expand=False,
            needs_exact_phrase_bias=False,
            needs_chapter_lookup=False,
            reason="planner_applied",
        )

    def fake_search_parent_evidence(question: str, top_k: int, book_ids=None, *, query_plan=None, debug_info=None):
        captured["retrieval_question"] = question
        captured["query_plan"] = query_plan
        return []

    monkeypatch.setattr("app.services.qa.plan_retrieval_query", fake_plan_retrieval_query)
    monkeypatch.setattr("app.services.qa.search_parent_evidence", fake_search_parent_evidence)

    response = answer_question(
        'Find the exact phrase "take a step back"',
        top_k=2,
        include_debug=True,
        enable_query_planner=False,
        force_query_planner=True,
    )

    assert response["answer"] == ABSTAIN_MESSAGE
    assert captured["planner_question"] == 'Find the exact phrase "take a step back"'
    assert response["debug"]["planner_forced"] is True
    assert response["debug"]["planner_decision"]["reason"] == "forced_query_planner"
    assert response["debug"]["planner_used"] is True


def test_answer_question_passes_book_metadata_into_planner(monkeypatch):
    captured: dict = {}

    def fake_load_planner_book_hints(db, book_ids):
        captured["hint_book_ids"] = list(book_ids or [])
        return {
            "book_title": "Shelf Book",
            "toc_headings": ["Intro"],
            "chapter_titles": ["Chapter A"],
        }

    def fake_plan_retrieval_query(question: str, **kwargs):
        captured["planner_kwargs"] = kwargs
        return QueryPlannerPlan(
            query_type="original",
            standalone_question=question,
            search_queries=[],
            keywords=[],
            should_expand=False,
            needs_exact_phrase_bias=False,
            needs_chapter_lookup=False,
            reason="planner_applied",
        )

    def fake_search_parent_evidence(question: str, top_k: int, book_ids=None, *, query_plan=None, debug_info=None):
        return []

    monkeypatch.setattr("app.services.qa.load_planner_book_hints", fake_load_planner_book_hints)
    monkeypatch.setattr("app.services.qa.plan_retrieval_query", fake_plan_retrieval_query)
    monkeypatch.setattr("app.services.qa.search_parent_evidence", fake_search_parent_evidence)

    answer_question(
        "What is BM25?",
        book_ids=["bk1"],
        top_k=2,
        enable_query_planner=False,
        force_query_planner=True,
    )

    assert captured.get("hint_book_ids") == ["bk1"]
    assert captured.get("planner_kwargs", {}).get("book_title") == "Shelf Book"
    assert captured.get("planner_kwargs", {}).get("toc_headings") == ["Intro"]
