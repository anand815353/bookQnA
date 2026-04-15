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
