import json

from app.evals.dataset import load_eval_cases
from app.evals.models import EvalCase
from app.evals.runner import compare_answers, compare_retrieval, evaluate_answers, evaluate_retrieval


def test_load_eval_cases_jsonl_normalizes_fields(tmp_path):
    dataset_path = tmp_path / "cases.jsonl"
    rows = [
        {
            "id": "c1",
            "question": "What is RAG?",
            "expected_book": "Ai Engineering",
            "expected_pages": [10, {"start": 12, "end": 13}],
            "expected_answer_keywords": ["retrieval", "generation"],
            "answerable": "true",
            "test_type": "Concept Lookup",
            "recent_history": [{"question": "What is retrieval?", "answer": "Fetching context."}],
        }
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    cases = load_eval_cases(dataset_path)

    assert len(cases) == 1
    case = cases[0]
    assert case.case_id == "c1"
    assert case.expected_pages == [10, 12, 13]
    assert case.answerable is True
    assert case.test_type == "concept_lookup"
    assert case.recent_history == [{"question": "What is retrieval?", "answer": "Fetching context."}]


def test_evaluate_retrieval_tracks_chapter_summary_failures_separately():
    cases = [
        EvalCase(
            case_id="concept_hit",
            question="What does RAG stand for?",
            expected_book="Ai Engineering",
            expected_pages=[978],
            test_type="concept_lookup",
        ),
        EvalCase(
            case_id="chapter_miss",
            question="Summarize chapter 1",
            expected_book="Ai Engineering",
            expected_chapter="chapter 1",
            test_type="chapter_summary",
        ),
    ]

    def fake_retrieval(question: str, top_k: int, book_ids=None):
        if "rag stand for" in question.lower():
            return [
                {
                    "book_id": "b1",
                    "book_title": "Ai Engineering",
                    "page_start": 978,
                    "page_end": 978,
                    "chapter_title": "Chapter 11",
                }
            ]
        return [
            {
                "book_id": "b1",
                "book_title": "Ai Engineering",
                "page_start": 300,
                "page_end": 301,
                "chapter_title": "Chapter 5",
            }
        ]

    report = evaluate_retrieval(cases, top_k=4, retrieval_fn=fake_retrieval)
    by_type = report["summary"]["by_test_type"]

    assert by_type["concept_lookup"]["metrics"]["top_k_evidence_hit"]["value"] == 1.0
    assert by_type["chapter_summary"]["metrics"]["top_k_evidence_hit"]["value"] == 0.0


def test_evaluate_answers_computes_keyword_and_abstention_metrics():
    cases = [
        EvalCase(
            case_id="answerable_case",
            question="What does RAG stand for?",
            expected_book="Ai Engineering",
            expected_pages=[978],
            expected_answer_keywords=["retrieval-augmented generation", "rag"],
            answerable=True,
            test_type="concept_lookup",
        ),
        EvalCase(
            case_id="abstain_case",
            question="What is the capital of France?",
            expected_book="Ai Engineering",
            answerable=False,
            test_type="abstention",
        ),
    ]

    def fake_answer(question: str, top_k: int, book_ids=None):
        if "capital of france" in question.lower():
            return {
                "answer": "I do not know based on the indexed books.",
                "grounded": False,
                "citations": [],
            }
        return {
            "answer": "RAG means Retrieval-Augmented Generation.",
            "grounded": True,
            "citations": [
                {
                    "book_id": "b1",
                    "book_title": "Ai Engineering",
                    "page_start": 978,
                    "page_end": 978,
                    "chapter_title": "Chapter 11",
                }
            ],
        }

    report = evaluate_answers(cases, top_k=4, answer_fn=fake_answer)
    summary = report["summary"]["metrics"]

    assert summary["answer_keyword_coverage"]["value"] == 1.0
    assert summary["abstention_correctness"]["value"] == 1.0
    assert summary["correct_page_hit"]["value"] == 1.0


def test_retrieval_summary_contains_all_known_question_type_buckets():
    test_types = [
        "follow_up",
        "ambiguous_short",
        "concept_lookup",
        "chapter_summary",
        "locate_section_page",
        "compare_concepts",
        "exact_phrase_lookup",
        "abstention",
    ]
    cases = [
        EvalCase(
            case_id=f"case_{idx}",
            question=f"Question {idx}",
            test_type=test_type,
            answerable=test_type != "abstention",
        )
        for idx, test_type in enumerate(test_types, start=1)
    ]

    report = evaluate_retrieval(
        cases,
        top_k=4,
        retrieval_fn=lambda _q, _k, _ids: [],
    )
    by_type = report["summary"]["by_test_type"]

    for test_type in test_types:
        assert test_type in by_type


def test_compare_retrieval_surfaces_exact_phrase_regressions_and_follow_up_gains():
    cases = [
        EvalCase(
            case_id="follow_up_case",
            question="What about the next chapter?",
            expected_book="Ai Engineering",
            expected_chapter="chapter 7",
            test_type="follow_up",
            recent_history=[{"question": "Summarize chapter 6", "answer": "It covers RAG and agents."}],
        ),
        EvalCase(
            case_id="exact_phrase_case",
            question='Find the exact phrase "take a step back"',
            expected_book="Ai Engineering",
            expected_pages=[78],
            test_type="exact_phrase_lookup",
        ),
    ]

    def fake_retrieval(question: str, top_k: int, book_ids=None, *, planner_enabled: bool = False, case=None):
        if case and case.case_id == "follow_up_case":
            if planner_enabled:
                return [
                    {
                        "book_id": "b1",
                        "book_title": "Ai Engineering",
                        "page_start": 620,
                        "page_end": 621,
                        "chapter_title": "Chapter 7",
                    }
                ]
            return [
                {
                    "book_id": "b1",
                    "book_title": "Ai Engineering",
                    "page_start": 540,
                    "page_end": 541,
                    "chapter_title": "Chapter 6",
                }
            ]

        if planner_enabled:
            return [
                {
                    "book_id": "b1",
                    "book_title": "Ai Engineering",
                    "page_start": 90,
                    "page_end": 90,
                    "chapter_title": "Chapter 2",
                }
            ]
        return [
            {
                "book_id": "b1",
                "book_title": "Ai Engineering",
                "page_start": 78,
                "page_end": 78,
                "chapter_title": "Chapter 1",
            }
        ]

    report = compare_retrieval(cases, top_k=4, retrieval_fn=fake_retrieval)
    comparison = report["comparison"]

    assert comparison["focus_buckets"]["follow_up"]["metrics"]["chapter_hit"]["delta"] == 1.0
    assert comparison["focus_buckets"]["exact_phrase_lookup"]["metrics"]["correct_page_hit"]["delta"] == -1.0
    assert any(
        item.get("test_type") == "exact_phrase_lookup" and item.get("metric") == "correct_page_hit"
        for item in comparison["critical_regressions"]
    )


def test_compare_answers_surfaces_abstention_regressions():
    cases = [
        EvalCase(
            case_id="abstain_case",
            question="What is the capital of France?",
            expected_book="Ai Engineering",
            answerable=False,
            test_type="abstention",
        ),
        EvalCase(
            case_id="concept_case",
            question="What does RAG stand for?",
            expected_book="Ai Engineering",
            expected_pages=[978],
            expected_answer_keywords=["retrieval-augmented generation"],
            answerable=True,
            test_type="concept_lookup",
        ),
    ]

    def fake_answer(question: str, top_k: int, book_ids=None, *, planner_enabled: bool = False, case=None):
        if case and case.case_id == "abstain_case":
            if planner_enabled:
                return {
                    "answer": "France is a country in Europe.",
                    "grounded": False,
                    "citations": [],
                }
            return {
                "answer": "I do not know based on the indexed books.",
                "grounded": False,
                "citations": [],
            }

        return {
            "answer": "RAG means Retrieval-Augmented Generation.",
            "grounded": True,
            "citations": [
                {
                    "book_id": "b1",
                    "book_title": "Ai Engineering",
                    "page_start": 978,
                    "page_end": 978,
                    "chapter_title": "Chapter 11",
                }
            ],
        }

    report = compare_answers(cases, top_k=4, answer_fn=fake_answer)
    comparison = report["comparison"]

    assert comparison["focus_buckets"]["abstention"]["metrics"]["abstention_correctness"]["delta"] == -1.0
    assert any(
        item.get("test_type") == "abstention" and item.get("metric") == "abstention_correctness"
        for item in comparison["critical_regressions"]
    )


def test_compare_retrieval_can_force_planner_on_planner_path():
    cases = [
        EvalCase(
            case_id="follow_up_case",
            question="What about the next chapter?",
            expected_book="Ai Engineering",
            expected_chapter="chapter 7",
            test_type="follow_up",
        )
    ]
    captured = {"planner_calls": []}

    def fake_retrieval(question: str, top_k: int, book_ids=None, *, planner_enabled: bool = False, force_planner: bool = False, case=None):
        captured["planner_calls"].append((planner_enabled, force_planner))
        return []

    compare_retrieval(cases, top_k=4, retrieval_fn=fake_retrieval, force_planner=True)

    assert captured["planner_calls"] == [(False, False), (True, True)]


def test_compare_answers_keeps_baseline_planner_off_when_force_is_requested():
    cases = [
        EvalCase(
            case_id="follow_up_case",
            question="What about the next chapter?",
            expected_book="Ai Engineering",
            expected_chapter="chapter 7",
            test_type="follow_up",
        )
    ]
    captured = {"planner_calls": []}

    def fake_answer(question: str, top_k: int, book_ids=None, *, planner_enabled: bool = False, force_planner: bool = False, case=None):
        captured["planner_calls"].append((planner_enabled, force_planner))
        return {
            "answer": "I do not know based on the indexed books.",
            "grounded": False,
            "citations": [],
        }

    compare_answers(cases, top_k=4, answer_fn=fake_answer, force_planner=True)

    assert captured["planner_calls"] == [(False, False), (True, True)]
