from pathlib import Path
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import Base
from app.services.chat_history import (
    create_chat_session,
    delete_chat_session,
    get_chat_messages,
    get_recent_session_context,
    get_session_preview_text,
    list_chat_sessions,
    save_query_exchange,
)


def _build_db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return session_local()


def test_chat_history_service_roundtrip_and_delete():
    db = _build_db_session()
    try:
        session = create_chat_session(db, title=None)
        assert session.title == "New chat"

        sessions = list_chat_sessions(db)
        assert len(sessions) == 1
        assert sessions[0].id == session.id

        updated_session, user_msg, assistant_msg = save_query_exchange(
            db=db,
            session_id=session.id,
            question_text="Explain chapter one quickly",
            answer_text="Chapter one introduces the core concepts with examples.",
            grounded=True,
            selected_book_ids=["b1"],
            citations=[
                {
                    "book_id": "b1",
                    "book_title": "Book One",
                    "page_start": 3,
                    "page_end": 4,
                    "snippet": "Core concepts and definitions.",
                }
            ],
        )

        assert updated_session.id == session.id
        assert updated_session.title != "New chat"
        assert updated_session.summary_text
        assert user_msg.role == "user"
        assert assistant_msg.role == "assistant"

        messages = get_chat_messages(db, session.id)
        assert len(messages) == 2
        assert messages[0].question_text == "Explain chapter one quickly"
        assert messages[1].answer_text.startswith("Chapter one introduces")

        preview_text = get_session_preview_text(db, session.id)
        assert preview_text
        assert "Chapter one" in preview_text

        assert delete_chat_session(db, session.id) is True
        assert delete_chat_session(db, session.id) is False
    finally:
        db.close()


def test_recent_session_context_is_bounded_and_ordered():
    db = _build_db_session()
    try:
        session = create_chat_session(db, title="Topic thread")

        for idx in range(1, 7):
            save_query_exchange(
                db=db,
                session_id=session.id,
                question_text=f"Question {idx}",
                answer_text=f"Answer {idx}",
                grounded=True,
                selected_book_ids=[],
                citations=[],
            )

        recent_turns = get_recent_session_context(db, session.id, limit_turns=3)

        assert len(recent_turns) == 3
        assert [turn["question"] for turn in recent_turns] == [
            "Question 4",
            "Question 5",
            "Question 6",
        ]
        assert [turn["answer"] for turn in recent_turns] == [
            "Answer 4",
            "Answer 5",
            "Answer 6",
        ]
    finally:
        db.close()
