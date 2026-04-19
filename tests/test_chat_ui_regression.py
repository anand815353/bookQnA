from pathlib import Path
import sys
from uuid import uuid4

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app


def _fake_answer_question(
    question: str,
    book_ids=None,
    top_k: int = 4,
    include_debug: bool = False,
    **kwargs,
):
    payload = {
        "answer": f"UI regression answer for: {question}",
        "grounded": True,
        "citations": [
            {
                "book_id": "book-ui-1",
                "book_title": "UI Regression Book",
                "page_start": 12,
                "page_end": 13,
                "snippet": f"Evidence snippet for {question}.",
                "chapter_title": "Chapter 1",
                "subchapter_title": "Section A",
                "section_title": None,
                "page_category": None,
                "structure_source": None,
                "confidence": None,
            }
        ],
    }
    if include_debug:
        payload["debug"] = {"enabled": True}
    return payload


def _fake_answer_question_many_citations(
    question: str,
    book_ids=None,
    top_k: int = 4,
    include_debug: bool = False,
    **kwargs,
):
    citations = []
    for idx in range(1, 6):
        citations.append(
            {
                "book_id": f"book-ui-{idx}",
                "book_title": f"Compact Book {idx}",
                "page_start": 10 + idx,
                "page_end": 10 + idx,
                "snippet": f"Longer evidence snippet {idx} for {question} that should be previewed in compact mode.",
                "chapter_title": f"Chapter {idx}",
                "subchapter_title": f"Section {idx}",
                "section_title": None,
                "page_category": None,
                "structure_source": None,
                "confidence": None,
            }
        )
    payload = {
        "answer": f"Compact citation answer for: {question}",
        "grounded": True,
        "citations": citations,
    }
    if include_debug:
        payload["debug"] = {"enabled": True}
    return payload


def test_chat_page_loads_with_core_ui_markers():
    client = TestClient(app)
    response = client.get("/chat")

    assert response.status_code == 200
    html = response.text
    assert 'id="sessions-box"' in html
    assert 'id="messages-box"' in html
    assert 'id="query-form"' in html
    assert 'id="thread-title"' in html
    assert 'id="session-state-pill"' in html


def test_chat_template_includes_thread_layout_and_turn_ux_markers():
    client = TestClient(app)
    response = client.get("/chat")

    assert response.status_code == 200
    html = response.text
    assert 'id="thread-book-context"' in html
    assert 'id="thread-updated-at"' in html
    assert 'id="thread-selected-books"' in html
    assert 'id="composer-title"' in html
    assert "turn-summary" in html
    assert "data-turn-toggle" in html
    assert "turn-latest-badge" in html
    assert "turn-evidence-summary" in html
    assert "citation-card compact" in html
    assert "citation-snippet-preview" in html
    assert "Expand evidence" in html


def test_chat_template_includes_compact_sidebar_row_markers():
    client = TestClient(app)
    response = client.get("/chat")

    assert response.status_code == 200
    html = response.text
    assert 'id="sessions-meta"' in html
    assert "class=\"session-title\"" in html
    assert "class=\"session-preview\"" in html
    assert "class=\"session-time\"" in html


def test_composer_options_are_available_but_collapsed_by_default():
    client = TestClient(app)
    response = client.get("/chat")

    assert response.status_code == 200
    html = response.text
    assert 'id="options-details"' in html
    assert 'id="options-details" open' not in html
    assert 'id="selected-books-summary"' in html
    assert 'id="top_k"' in html
    assert "visually-hidden" in html


def test_template_contains_empty_loading_and_error_state_markers():
    client = TestClient(app)
    response = client.get("/chat")

    assert response.status_code == 200
    html = response.text
    assert "Ask your first question below to start this chat." in html
    assert "thread-pending-state" in html
    assert "query-error-banner" in html
    assert 'resultSection.classList.add("is-highlight")' in html


def test_chat_template_debug_trace_includes_planner_markers():
    client = TestClient(app)
    response = client.get("/chat")

    assert response.status_code == 200
    html = response.text
    assert "Planner enabled:" in html
    assert "Planner gate reason:" in html
    assert "Planner standalone question:" in html
    assert "Planner search queries:" in html
    assert "planner output, retrieval trace, stage latency, and context preview for tuning" in html


def test_chat_page_renders_empty_state_guidance():
    client = TestClient(app)
    response = client.get("/chat")

    assert response.status_code == 200
    html = response.text
    assert "No chats yet" in html
    assert "No messages yet" in html
    assert "First question will create a new chat." in html


def test_no_chats_onboarding_state_renders():
    client = TestClient(app)
    response = client.get("/chat")

    assert response.status_code == 200
    html = response.text
    assert "No chats yet" in html
    assert "Start with New Chat and ask your first question." in html


def test_session_list_payload_supports_sidebar_render(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr("app.api.query.answer_question", _fake_answer_question)

    title = f"UI Sidebar {uuid4().hex[:8]}"
    create_response = client.post("/chats", json={"title": title})
    assert create_response.status_code == 200
    session_id = create_response.json()["id"]

    query_response = client.post(
        "/query",
        json={"session_id": session_id, "question": "Seed sidebar preview", "top_k": 2},
    )
    assert query_response.status_code == 200

    list_response = client.get("/chats")
    assert list_response.status_code == 200
    matching = [s for s in list_response.json() if s["id"] == session_id]
    assert matching
    session_payload = matching[0]
    assert session_payload["title"] == title
    assert "preview_text" in session_payload
    assert session_payload["preview_text"]
    assert session_payload["last_message_at"]


def test_empty_history_state_and_active_thread_bootstrap():
    client = TestClient(app)
    create_response = client.post("/chats", json={"title": "Thread Bootstrap"})
    assert create_response.status_code == 200
    session_id = create_response.json()["id"]

    detail_response = client.get(f"/chats/{session_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["messages"] == []

    page_response = client.get(f"/chat/{session_id}")
    assert page_response.status_code == 200
    html = page_response.text
    assert f'let currentSessionId = "{session_id}"' in html
    assert "No messages yet" in html


def test_active_thread_route_contains_collapsed_turn_bootstrap(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr("app.api.query.answer_question", _fake_answer_question)

    first = client.post("/query", json={"question": "First turn question", "top_k": 2})
    assert first.status_code == 200
    session_id = first.json()["session_id"]
    assert session_id

    second = client.post(
        "/query",
        json={"session_id": session_id, "question": "Second turn question", "top_k": 2},
    )
    assert second.status_code == 200

    page_response = client.get(f"/chat/{session_id}")
    assert page_response.status_code == 200
    html = page_response.text
    assert f'let currentSessionId = "{session_id}"' in html
    assert "const isCollapsedTurn = isOlderTurn;" in html
    assert 'turnClasses.push("is-collapsed")' in html
    assert "Latest answer" in html


def test_compact_citation_rendering_markers_and_payload_shape(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr("app.api.query.answer_question", _fake_answer_question_many_citations)

    query_response = client.post(
        "/query",
        json={"question": "Compact citation regression question", "top_k": 4},
    )
    assert query_response.status_code == 200
    payload = query_response.json()
    assert payload["session_id"]
    assert len(payload["citations"]) == 5
    assert payload["citations"][0]["chapter_title"] == "Chapter 1"
    assert payload["citations"][0]["subchapter_title"] == "Section 1"

    detail_response = client.get(f"/chats/{payload['session_id']}")
    assert detail_response.status_code == 200
    messages = detail_response.json()["messages"]
    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    assert assistant_messages
    assert len(assistant_messages[-1]["citations"]) == 5

    page_response = client.get(f"/chat/{payload['session_id']}")
    assert page_response.status_code == 200
    html = page_response.text
    assert "const collapsedCount = citationCount > 2 ? 2 : citationCount;" in html
    assert "Show all ${citationCount} sources" in html
    assert "previewChars: 150" in html
    assert "No evidence snippets returned for this answer." in html


def test_query_submit_flow_still_populates_thread_payload(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr("app.api.query.answer_question", _fake_answer_question)

    query_response = client.post(
        "/query",
        json={"question": "Does the UI submit flow still work?", "top_k": 3},
    )
    assert query_response.status_code == 200
    query_payload = query_response.json()
    assert query_payload["session_id"]
    assert query_payload["grounded"] is True
    assert query_payload["citations"]

    session_id = query_payload["session_id"]
    detail_response = client.get(f"/chats/{session_id}")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert len(detail_payload["messages"]) == 2
    assert detail_payload["messages"][0]["role"] == "user"
    assert detail_payload["messages"][1]["role"] == "assistant"
    assert detail_payload["messages"][1]["citations"][0]["book_title"] == "UI Regression Book"

    thread_page = client.get(f"/chat/{session_id}")
    assert thread_page.status_code == 200
    assert 'id="thread-title"' in thread_page.text
