from pathlib import Path
import sys

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app


def test_chat_endpoints_create_list_get_roundtrip():
    client = TestClient(app)

    create_response = client.post("/chats", json={"title": "Regression Session"})
    assert create_response.status_code == 200
    session_payload = create_response.json()
    assert session_payload["id"]
    assert session_payload["title"] == "Regression Session"

    list_response = client.get("/chats")
    assert list_response.status_code == 200
    sessions = list_response.json()
    match = [s for s in sessions if s["id"] == session_payload["id"]]
    assert match

    get_response = client.get(f"/chats/{session_payload['id']}")
    assert get_response.status_code == 200
    detail = get_response.json()
    assert detail["session"]["id"] == session_payload["id"]
    assert detail["messages"] == []


def test_query_multiturn_session_persists_citations_and_uses_followup_history(monkeypatch):
    client = TestClient(app)
    captures = {"history_lengths": []}

    def fake_answer_question(
        question: str,
        book_ids=None,
        top_k: int = 4,
        include_debug: bool = False,
        recent_history=None,
        enable_query_reformulation: bool = False,
    ):
        captures["history_lengths"].append(len(recent_history or []))
        return {
            "answer": f"Answer for {question}",
            "grounded": True,
            "citations": [
                {
                    "book_id": "b1",
                    "book_title": "Book One",
                    "page_start": 12,
                    "page_end": 12,
                    "snippet": f"Evidence for {question}",
                    "chapter_title": "Chapter 1",
                    "subchapter_title": None,
                    "section_title": None,
                    "page_category": None,
                    "structure_source": None,
                    "confidence": None,
                }
            ],
        }

    monkeypatch.setattr("app.api.query.answer_question", fake_answer_question)

    first = client.post("/query", json={"question": "Who is the author?", "top_k": 2})
    assert first.status_code == 200
    session_id = first.json()["session_id"]
    assert session_id

    second = client.post(
        "/query",
        json={"session_id": session_id, "question": "What about their background?", "top_k": 2},
    )
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id

    detail_response = client.get(f"/chats/{session_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert len(detail["messages"]) == 4
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][1]["role"] == "assistant"
    assert detail["messages"][2]["role"] == "user"
    assert detail["messages"][3]["role"] == "assistant"
    assert detail["messages"][1]["citations"][0]["snippet"].startswith("Evidence for Who is the author?")
    assert detail["messages"][3]["citations"][0]["snippet"].startswith("Evidence for What about their background?")

    assert captures["history_lengths"][0] == 0
    assert captures["history_lengths"][1] >= 1
