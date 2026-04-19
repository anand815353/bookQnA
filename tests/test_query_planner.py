import re
from pathlib import Path

from pydantic import ValidationError

from app import settings
from app.schemas import QueryPlannerPlan
from app.services.query_planner import plan_retrieval_query, should_use_query_planner


def test_query_planner_settings_exist_with_expected_types():
    assert isinstance(settings.ENABLE_QUERY_PLANNER, bool)
    assert isinstance(settings.QUERY_PLANNER_ENABLE_GATING, bool)
    assert isinstance(settings.QUERY_PLANNER_FORCE_IN_DEBUG, bool)
    assert isinstance(settings.QUERY_PLANNER_MODEL, str)
    assert isinstance(settings.QUERY_PLANNER_MAX_ALT_QUERIES, int)
    assert isinstance(settings.QUERY_PLANNER_MAX_KEYWORDS, int)
    assert isinstance(settings.QUERY_PLANNER_USE_BOOK_METADATA, bool)
    assert isinstance(settings.QUERY_PLANNER_LOW_TEMP, float)
    assert isinstance(settings.QUERY_PLANNER_SHORT_QUERY_WORDS, int)


def test_query_planner_plan_forbids_unknown_fields():
    try:
        QueryPlannerPlan(
            standalone_question="What is retrieval?",
            reason="test",
            unexpected_field=True,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("QueryPlannerPlan should reject unknown fields.")


def test_query_planner_source_has_single_authoritative_regex_constant_assignment():
    source = Path("app/services/query_planner.py").read_text(encoding="utf-8")

    exact_assignments = re.findall(r"^PLANNER_EXACT_PHRASE_PATTERNS\s*=", source, flags=re.MULTILINE)
    page_assignments = re.findall(r"^PLANNER_PAGE_CONSTRAINT_PATTERNS\s*=", source, flags=re.MULTILINE)
    assert len(exact_assignments) == 1
    assert len(page_assignments) == 1
    assert "Reassign these after the declaration block" not in source


def test_plan_retrieval_query_marks_disabled_noop_when_flag_is_off():
    plan = plan_retrieval_query("Explain BM25", planner_enabled=False)

    assert plan.standalone_question == "Explain BM25"
    assert plan.search_queries == []
    assert plan.keywords == []
    assert plan.reason == "planner_disabled_noop"


def test_plan_retrieval_query_returns_structured_plan_from_llm(monkeypatch):
    captured = {}

    class FakeChain:
        def invoke(self, payload):
            captured["payload"] = payload
            return {
                "parsed": QueryPlannerPlan(
                    query_type="follow_up",
                    standalone_question="What does the next chapter say about BM25 retrieval?",
                    search_queries=[
                        "next chapter BM25 retrieval",
                        "BM25 retrieval chapter",
                        "BM25 retrieval chapter",
                    ],
                    keywords=["BM25", "retrieval", "chapter", "retrieval"],
                    should_expand=True,
                    needs_exact_phrase_bias=False,
                    needs_chapter_lookup=True,
                    reason="Follow-up question needs chapter-aware retrieval reformulation.",
                ),
                "parsing_error": None,
                "raw": None,
            }

    monkeypatch.setattr("app.services.query_planner._resolve_api_key", lambda: "test-key")
    monkeypatch.setattr("app.services.query_planner._build_planner_chain", lambda _api_key: FakeChain())
    monkeypatch.setattr("app.services.query_planner.QUERY_PLANNER_USE_BOOK_METADATA", True)

    plan = plan_retrieval_query(
        "What about the next chapter?",
        recent_history=[{"question": "Summarize chapter one", "answer": "It introduces retrieval basics."}],
        book_title="Book One",
        toc_headings=["Chapter 1 Intro", "Chapter 2 BM25 Retrieval"],
        chapter_titles=["Chapter 1 Intro", "Chapter 2 BM25 Retrieval"],
        planner_enabled=True,
    )

    assert plan.query_type == "follow_up"
    assert plan.standalone_question == "What does the next chapter say about BM25 retrieval?"
    assert plan.search_queries == [
        "next chapter BM25 retrieval",
        "BM25 retrieval chapter",
    ]
    assert plan.keywords == ["BM25", "retrieval", "chapter"]
    assert plan.should_expand is True
    assert plan.needs_chapter_lookup is True
    assert "Summarize chapter one" in captured["payload"]["recent_history"]
    assert "Book One" in captured["payload"]["book_metadata"]


def test_plan_retrieval_query_falls_back_when_api_key_is_missing(monkeypatch):
    monkeypatch.setattr("app.services.query_planner._resolve_api_key", lambda: None)

    plan = plan_retrieval_query("What about the next chapter?", planner_enabled=True)

    assert plan.query_type == "original"
    assert plan.standalone_question == "What about the next chapter?"
    assert plan.search_queries == []
    assert plan.keywords == []
    assert plan.reason == "planner_missing_api_key_noop"


def test_plan_retrieval_query_falls_back_when_structured_parse_fails(monkeypatch):
    class FakeChain:
        def invoke(self, _payload):
            return {
                "parsed": None,
                "parsing_error": ValueError("bad json"),
                "raw": None,
            }

    monkeypatch.setattr("app.services.query_planner._resolve_api_key", lambda: "test-key")
    monkeypatch.setattr("app.services.query_planner._build_planner_chain", lambda _api_key: FakeChain())

    plan = plan_retrieval_query("Explain BM25", planner_enabled=True)

    assert plan.query_type == "original"
    assert plan.standalone_question == "Explain BM25"
    assert plan.search_queries == []
    assert plan.keywords == []
    assert plan.reason == "planner_parse_failed_noop"


def test_plan_retrieval_query_logs_structured_outcome(caplog):
    caplog.set_level("INFO")

    plan = plan_retrieval_query("Explain BM25", planner_enabled=False)

    assert plan.reason == "planner_disabled_noop"
    assert "query_planner_skipped" in caplog.text
    assert "planner_enabled=False" in caplog.text
    assert "original_question=Explain BM25" in caplog.text
    assert "standalone_question=Explain BM25" in caplog.text
    assert "query_type=original" in caplog.text


def test_should_use_query_planner_prefers_follow_up_questions_with_context():
    decision = should_use_query_planner(
        "What about the next section?",
        recent_history=[{"question": "Summarize chapter one", "answer": "Summary"}],
        planner_enabled=True,
    )

    assert decision.should_use is True
    assert decision.reason == "follow_up_with_context"
    assert "recent_history" in decision.signals
    assert "follow_up_marker" in decision.signals


def test_should_use_query_planner_skips_exact_phrase_queries():
    decision = should_use_query_planner(
        'Find the exact phrase "take a step back"',
        recent_history=[{"question": "Which quote was that?", "answer": "Unknown"}],
        planner_enabled=True,
    )

    assert decision.should_use is False
    assert decision.reason == "exact_phrase_lookup"
    assert "exact_phrase_lookup" in decision.suppressions
    assert decision.allow_legacy_rewrite is False


def test_should_use_query_planner_skips_term_constrained_queries():
    decision = should_use_query_planner(
        "Define BM25",
        planner_enabled=True,
    )

    assert decision.should_use is False
    assert decision.reason == "term_constrained_lookup"
    assert "term_constrained_lookup" in decision.suppressions
    assert decision.allow_legacy_rewrite is False


def test_should_use_query_planner_skips_clear_specific_questions():
    decision = should_use_query_planner(
        "Summarize chapter 1 foundations of AI engineering",
        planner_enabled=True,
    )

    assert decision.should_use is False
    assert decision.reason == "clear_specific_query"


def test_should_use_query_planner_skips_clear_short_queries():
    decision = should_use_query_planner(
        "Explain BM25",
        planner_enabled=True,
    )

    assert decision.should_use is False
    assert decision.reason == "clear_specific_query"


def test_should_use_query_planner_can_trigger_for_ambiguous_short_queries():
    decision = should_use_query_planner(
        "What about this?",
        planner_enabled=True,
    )

    assert decision.should_use is True
    assert decision.reason == "short_ambiguous_query"
    assert "short_query_ambiguous" in decision.signals


def test_should_use_query_planner_can_force_enable_for_experiments():
    decision = should_use_query_planner(
        "Explain BM25",
        planner_enabled=False,
        force_planner=True,
    )

    assert decision.should_use is True
    assert decision.forced is True
    assert decision.reason == "forced_query_planner"


def test_should_use_query_planner_for_chapter_locator_queries():
    decision = should_use_query_planner(
        "Which chapter covers BM25?",
        planner_enabled=True,
    )

    assert decision.should_use is True
    assert decision.reason == "chapter_locator_query"
