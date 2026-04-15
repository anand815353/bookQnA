import json

from app.evals.dataset import load_eval_cases
from app.evals.models import EvalCase
from app.evals.runner import evaluate_answers, evaluate_retrieval


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
