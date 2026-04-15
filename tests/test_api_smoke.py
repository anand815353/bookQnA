from fastapi.testclient import TestClient

from app.main import app


def _pdf_bytes() -> bytes:
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers.get("X-Request-ID")


def test_book_lifecycle_and_status(monkeypatch):
    client = TestClient(app)

    monkeypatch.setattr("app.api.books.ingest_book", lambda _book_id: None)
    monkeypatch.setattr("app.api.books.delete_book_vectors", lambda _book_id: 0)
    monkeypatch.setattr("app.api.books.delete_book_files", lambda _book_id: None)

    files = {"file": ("sample.pdf", _pdf_bytes(), "application/pdf")}
    data = {"title": "Sample Book", "auto_ingest": "false"}
    upload_response = client.post("/books/upload", files=files, data=data)
    assert upload_response.status_code == 200
    payload = upload_response.json()
    book_id = payload["id"]
    assert payload["status"] == "uploaded"

    status_response = client.get(f"/books/{book_id}/status")
    assert status_response.status_code == 200
    assert status_response.json()["status"] in {"uploaded", "ingesting", "indexed", "failed"}

    ingest_response = client.post(f"/books/{book_id}/ingest")
    assert ingest_response.status_code == 200
    assert ingest_response.json()["book_id"] == book_id

    list_response = client.get("/books")
    assert list_response.status_code == 200
    assert any(book["id"] == book_id for book in list_response.json())

    get_response = client.get(f"/books/{book_id}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == book_id

    delete_response = client.delete(f"/books/{book_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["ok"] is True


def test_query_endpoint_with_and_without_book_filter(monkeypatch):
    client = TestClient(app)

    def fake_answer_question(question: str, book_ids=None, top_k: int = 4, include_debug: bool = False):
        result = {
            "answer": f"Answer for: {question}",
            "grounded": True,
            "citations": [
                {
                    "book_id": "b1",
                    "book_title": "Book One",
                    "page_start": 1,
                    "page_end": 1,
                    "snippet": "Evidence",
                    "chapter_title": "Chapter 1",
                    "subchapter_title": None,
                    "confidence": None,
                }
            ],
        }
        if include_debug:
            result["debug"] = {"enabled": True, "stage_latency_ms": {"total": 1}}
        return result

    monkeypatch.setattr("app.api.query.answer_question", fake_answer_question)

    no_filter_response = client.post(
        "/query",
        json={"question": "What is this about?", "top_k": 3},
    )
    assert no_filter_response.status_code == 200
    assert "answer" in no_filter_response.json()

    filtered_response = client.post(
        "/query",
        json={"question": "What is this about?", "book_ids": ["b1"], "top_k": 2},
    )
    assert filtered_response.status_code == 200
    assert filtered_response.json()["citations"][0]["book_id"] == "b1"


def test_query_debug_payload_disabled_by_default(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr("app.api.query.QUERY_DEBUG_ENABLED", False)

    def fake_answer_question(question: str, book_ids=None, top_k: int = 4, include_debug: bool = False):
        result = {
            "answer": f"Answer for: {question}",
            "grounded": True,
            "citations": [],
        }
        if include_debug:
            result["debug"] = {"enabled": True}
        return result

    monkeypatch.setattr("app.api.query.answer_question", fake_answer_question)
    response = client.post("/query", json={"question": "debug please", "debug": True})
    assert response.status_code == 200
    assert "debug" not in response.json()


def test_query_debug_payload_returned_when_enabled(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr("app.api.query.QUERY_DEBUG_ENABLED", True)

    def fake_answer_question(question: str, book_ids=None, top_k: int = 4, include_debug: bool = False):
        result = {
            "answer": f"Answer for: {question}",
            "grounded": True,
            "citations": [],
        }
        if include_debug:
            result["debug"] = {"enabled": True, "retrieval": {"mode": "hybrid"}}
        return result

    monkeypatch.setattr("app.api.query.answer_question", fake_answer_question)
    response = client.post("/query", json={"question": "debug please", "debug": True})
    assert response.status_code == 200
    assert response.json()["debug"]["enabled"] is True


def test_request_id_header_is_propagated():
    client = TestClient(app)
    response = client.get("/health", headers={"X-Request-ID": "req-test-123"})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "req-test-123"


def test_query_logs_and_raises_on_failure(monkeypatch, caplog):
    client = TestClient(app, raise_server_exceptions=False)
    caplog.set_level("INFO")

    def fake_answer_question_fail(_question: str, book_ids=None, top_k: int = 4):
        raise RuntimeError("forced failure")

    monkeypatch.setattr("app.api.query.answer_question", fake_answer_question_fail)

    response = client.post(
        "/query",
        json={"question": "Will this fail?", "book_ids": ["b1"], "top_k": 2},
    )

    assert response.status_code == 500
    assert "query_failed" in caplog.text


def test_query_persists_exchange_in_session(monkeypatch):
    client = TestClient(app)

    def fake_answer_question(question: str, book_ids=None, top_k: int = 4, include_debug: bool = False):
        result = {
            "answer": f"Persisted answer for: {question}",
            "grounded": True,
            "citations": [
                {
                    "book_id": "b1",
                    "book_title": "Book One",
                    "page_start": 10,
                    "page_end": 11,
                    "snippet": "Stored evidence",
                    "chapter_title": "Chapter 2",
                    "subchapter_title": None,
                    "section_title": None,
                    "page_category": None,
                    "structure_source": None,
                    "confidence": None,
                }
            ],
        }
        if include_debug:
            result["debug"] = {"enabled": True}
        return result

    monkeypatch.setattr("app.api.query.answer_question", fake_answer_question)

    create_response = client.post("/chats", json={"title": "My Session"})
    assert create_response.status_code == 200
    session_id = create_response.json()["id"]

    query_response = client.post(
        "/query",
        json={
            "session_id": session_id,
            "question": "Where is the evidence?",
            "book_ids": ["b1"],
            "top_k": 2,
        },
    )
    assert query_response.status_code == 200
    payload = query_response.json()
    assert payload["session_id"] == session_id
    assert payload["grounded"] is True

    detail_response = client.get(f"/chats/{session_id}")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    messages = detail_payload["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["question_text"] == "Where is the evidence?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["answer_text"] == "Persisted answer for: Where is the evidence?"
    assert messages[1]["citations"][0]["book_id"] == "b1"


def test_query_creates_new_session_when_not_provided(monkeypatch):
    client = TestClient(app)

    def fake_answer_question(question: str, book_ids=None, top_k: int = 4, include_debug: bool = False):
        return {
            "answer": f"Answer for: {question}",
            "grounded": False,
            "citations": [],
        }

    monkeypatch.setattr("app.api.query.answer_question", fake_answer_question)

    query_response = client.post(
        "/query",
        json={"question": "Create a new session automatically", "top_k": 3},
    )
    assert query_response.status_code == 200
    payload = query_response.json()
    assert payload["session_id"]

    session_response = client.get(f"/chats/{payload['session_id']}")
    assert session_response.status_code == 200
    messages = session_response.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_chat_list_and_delete_session(monkeypatch):
    client = TestClient(app)

    def fake_answer_question(question: str, book_ids=None, top_k: int = 4, include_debug: bool = False):
        return {
            "answer": f"Answer for: {question}",
            "grounded": True,
            "citations": [],
        }

    monkeypatch.setattr("app.api.query.answer_question", fake_answer_question)

    query_response = client.post("/query", json={"question": "first question"})
    assert query_response.status_code == 200
    session_id = query_response.json()["session_id"]
    assert session_id

    list_response = client.get("/chats")
    assert list_response.status_code == 200
    sessions = list_response.json()
    matching = [s for s in sessions if s["id"] == session_id]
    assert matching
    assert "preview_text" in matching[0]

    delete_response = client.delete(f"/chats/{session_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["ok"] is True

    get_response = client.get(f"/chats/{session_id}")
    assert get_response.status_code == 404


def test_query_returns_404_for_unknown_session(monkeypatch):
    client = TestClient(app)

    def fake_answer_question(question: str, book_ids=None, top_k: int = 4, include_debug: bool = False):
        return {
            "answer": f"Answer for: {question}",
            "grounded": True,
            "citations": [],
        }

    monkeypatch.setattr("app.api.query.answer_question", fake_answer_question)

    response = client.post(
        "/query",
        json={"question": "continue unknown session", "session_id": "missing-session-id"},
    )
    assert response.status_code == 404


def test_chat_thread_page_renders_for_existing_session():
    client = TestClient(app)
    create_response = client.post("/chats", json={"title": "Thread Page Session"})
    assert create_response.status_code == 200
    session_id = create_response.json()["id"]

    response = client.get(f"/chat/{session_id}")
    assert response.status_code == 200
    assert "Ask Questions" in response.text


def test_chat_thread_page_returns_404_for_missing_session():
    client = TestClient(app)
    response = client.get("/chat/missing-session-id")
    assert response.status_code == 404


def test_query_passes_recent_history_for_follow_up(monkeypatch):
    client = TestClient(app)
    captured = {"recent_history_turns": None, "reformulation_enabled": None}

    def fake_answer_question(
        question: str,
        book_ids=None,
        top_k: int = 4,
        include_debug: bool = False,
        recent_history=None,
        enable_query_reformulation: bool = False,
    ):
        captured["recent_history_turns"] = len(recent_history or [])
        captured["reformulation_enabled"] = enable_query_reformulation
        return {
            "answer": f"Answer for: {question}",
            "grounded": True,
            "citations": [],
        }

    monkeypatch.setattr("app.api.query.answer_question", fake_answer_question)

    first_response = client.post("/query", json={"question": "Summarize chapter one", "top_k": 2})
    assert first_response.status_code == 200
    session_id = first_response.json()["session_id"]

    second_response = client.post(
        "/query",
        json={"question": "What about the next section?", "session_id": session_id, "top_k": 2},
    )
    assert second_response.status_code == 200
    assert captured["recent_history_turns"] >= 1
    assert isinstance(captured["reformulation_enabled"], bool)


def test_placeholder_session_title_is_upgraded_after_first_turn(monkeypatch):
    client = TestClient(app)

    def fake_answer_question(question: str, book_ids=None, top_k: int = 4, include_debug: bool = False, **kwargs):
        return {
            "answer": f"Answer for: {question}. Supporting details here.",
            "grounded": True,
            "citations": [],
        }

    monkeypatch.setattr("app.api.query.answer_question", fake_answer_question)

    create_response = client.post("/chats", json={})
    assert create_response.status_code == 200
    session_id = create_response.json()["id"]
    assert create_response.json()["title"] == "New chat"

    query_response = client.post(
        "/query",
        json={"session_id": session_id, "question": "Explain chapter three takeaways"},
    )
    assert query_response.status_code == 200

    session_response = client.get(f"/chats/{session_id}")
    assert session_response.status_code == 200
    session_payload = session_response.json()["session"]
    assert session_payload["title"] != "New chat"
    assert "chapter three" in session_payload["title"].lower()
    assert session_payload["summary_text"]


def test_chat_list_includes_summary_text(monkeypatch):
    client = TestClient(app)

    def fake_answer_question(question: str, book_ids=None, top_k: int = 4, include_debug: bool = False, **kwargs):
        return {
            "answer": "This answer gives a concise summary of the topic for list display.",
            "grounded": True,
            "citations": [],
        }

    monkeypatch.setattr("app.api.query.answer_question", fake_answer_question)
    query_response = client.post("/query", json={"question": "Generate session summary"})
    assert query_response.status_code == 200
    session_id = query_response.json()["session_id"]

    list_response = client.get("/chats")
    assert list_response.status_code == 200
    matching = [s for s in list_response.json() if s["id"] == session_id]
    assert matching
    assert "summary_text" in matching[0]
    assert matching[0]["summary_text"]
