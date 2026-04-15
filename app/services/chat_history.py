import json
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models import ChatMessage, ChatSession

MAX_SESSION_TITLE_CHARS = 80
MAX_PREVIEW_CHARS = 120
HISTORY_QUESTION_CHARS = 220
HISTORY_ANSWER_CHARS = 420
MAX_SUMMARY_CHARS = 180
DEFAULT_SESSION_TITLE = "New chat"


def _utcnow() -> datetime:
    return datetime.utcnow()


def _to_json(value: Any, default: Any) -> str:
    return json.dumps(value if value is not None else default, ensure_ascii=True)


def _from_json(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def _derive_session_title(question_text: str | None) -> str:
    cleaned = " ".join((question_text or "").strip().split())
    if not cleaned:
        return DEFAULT_SESSION_TITLE
    if len(cleaned) <= MAX_SESSION_TITLE_CHARS:
        return cleaned
    return cleaned[: MAX_SESSION_TITLE_CHARS - 3].rstrip() + "..."


def _is_placeholder_title(title: str | None) -> bool:
    normalized = " ".join((title or "").strip().split()).lower()
    return normalized in {"", "new chat", "untitled"}


def _derive_session_summary(question_text: str | None, answer_text: str | None) -> str | None:
    source = " ".join((answer_text or question_text or "").strip().split())
    if not source:
        return None

    first_sentence = source.split(".")[0].strip()
    summary = first_sentence if len(first_sentence) >= 24 else source
    if len(summary) <= MAX_SUMMARY_CHARS:
        return summary
    return summary[: MAX_SUMMARY_CHARS - 3].rstrip() + "..."


def create_chat_session(db: Session, title: str | None = None) -> ChatSession:
    now = _utcnow()
    chat_session = ChatSession(
        id=str(uuid.uuid4()),
        title=(title or DEFAULT_SESSION_TITLE).strip() or DEFAULT_SESSION_TITLE,
        summary_text=None,
        created_at=now,
        updated_at=now,
        last_message_at=None,
    )
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return chat_session


def list_chat_sessions(db: Session, limit: int = 100) -> list[ChatSession]:
    return (
        db.query(ChatSession)
        .order_by(ChatSession.last_message_at.desc(), ChatSession.updated_at.desc())
        .limit(limit)
        .all()
    )


def get_latest_chat_message(db: Session, session_id: str) -> ChatMessage | None:
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .first()
    )


def _compact_preview(text: str | None) -> str | None:
    compact = " ".join((text or "").strip().split())
    if not compact:
        return None
    if len(compact) <= MAX_PREVIEW_CHARS:
        return compact
    return compact[: MAX_PREVIEW_CHARS - 3].rstrip() + "..."


def _compact_history_text(text: str | None, max_chars: int) -> str:
    compact = " ".join((text or "").strip().split())
    if not compact:
        return ""
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def get_session_preview_text(db: Session, session_id: str) -> str | None:
    latest = get_latest_chat_message(db, session_id)
    if latest is None:
        return None
    if latest.role == "assistant":
        return _compact_preview(latest.answer_text)
    return _compact_preview(latest.question_text)


def get_chat_session(db: Session, session_id: str) -> ChatSession | None:
    return db.get(ChatSession, session_id)


def get_chat_messages(db: Session, session_id: str) -> list[ChatMessage]:
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )


def get_recent_session_context(
    db: Session,
    session_id: str,
    limit_turns: int = 4,
) -> list[dict[str, str]]:
    if limit_turns <= 0:
        return []

    recent_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit((limit_turns * 4) + 4)
        .all()
    )
    recent_messages.reverse()

    turns: list[dict[str, str]] = []
    pending_user_question = ""

    for message in recent_messages:
        if message.role == "user":
            pending_user_question = _compact_history_text(
                message.question_text,
                HISTORY_QUESTION_CHARS,
            )
            continue

        if message.role != "assistant":
            continue

        question_text = _compact_history_text(
            message.question_text or pending_user_question,
            HISTORY_QUESTION_CHARS,
        )
        answer_text = _compact_history_text(
            message.answer_text,
            HISTORY_ANSWER_CHARS,
        )
        pending_user_question = ""
        if not question_text and not answer_text:
            continue
        turns.append(
            {
                "question": question_text,
                "answer": answer_text,
            }
        )

    return turns[-limit_turns:]


def delete_chat_session(db: Session, session_id: str) -> bool:
    session = get_chat_session(db, session_id)
    if session is None:
        return False

    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete(synchronize_session=False)
    db.delete(session)
    db.commit()
    return True


def append_user_message(
    db: Session,
    session_id: str,
    question_text: str,
    selected_book_ids: list[str] | None = None,
) -> ChatMessage:
    message = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        question_text=question_text,
        answer_text=None,
        grounded=None,
        selected_book_ids_json=_to_json(selected_book_ids or [], []),
        citations_json="[]",
        created_at=_utcnow(),
    )
    db.add(message)
    db.flush()
    return message


def append_assistant_message(
    db: Session,
    session_id: str,
    answer_text: str,
    grounded: bool,
    citations: list[dict[str, Any]] | None = None,
    *,
    question_text: str | None = None,
    selected_book_ids: list[str] | None = None,
) -> ChatMessage:
    message = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        question_text=question_text,
        answer_text=answer_text,
        grounded=grounded,
        selected_book_ids_json=_to_json(selected_book_ids or [], []),
        citations_json=_to_json(citations or [], []),
        created_at=_utcnow(),
    )
    db.add(message)
    db.flush()
    return message


def save_query_exchange(
    db: Session,
    session_id: str | None,
    question_text: str,
    answer_text: str,
    grounded: bool,
    selected_book_ids: list[str] | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> tuple[ChatSession, ChatMessage, ChatMessage]:
    chat_session = get_chat_session(db, session_id) if session_id else None
    if chat_session is None:
        chat_session = ChatSession(
            id=str(uuid.uuid4()),
            title=_derive_session_title(question_text),
            summary_text=None,
            created_at=_utcnow(),
            updated_at=_utcnow(),
            last_message_at=None,
        )
        db.add(chat_session)
        db.flush()

    user_message = append_user_message(
        db,
        session_id=chat_session.id,
        question_text=question_text,
        selected_book_ids=selected_book_ids,
    )
    assistant_message = append_assistant_message(
        db,
        session_id=chat_session.id,
        answer_text=answer_text,
        grounded=grounded,
        citations=citations,
        question_text=question_text,
        selected_book_ids=selected_book_ids,
    )

    now = _utcnow()
    if _is_placeholder_title(chat_session.title) and chat_session.last_message_at is None:
        chat_session.title = _derive_session_title(question_text)
    if not chat_session.summary_text:
        chat_session.summary_text = _derive_session_summary(question_text, answer_text)
    chat_session.updated_at = now
    chat_session.last_message_at = now
    db.commit()
    db.refresh(chat_session)
    db.refresh(user_message)
    db.refresh(assistant_message)
    return chat_session, user_message, assistant_message


def serialize_chat_session(chat_session: ChatSession, preview_text: str | None = None) -> dict[str, Any]:
    return {
        "id": chat_session.id,
        "title": chat_session.title,
        "summary_text": chat_session.summary_text,
        "created_at": chat_session.created_at,
        "updated_at": chat_session.updated_at,
        "last_message_at": chat_session.last_message_at,
        "preview_text": preview_text,
    }


def serialize_chat_message(message: ChatMessage) -> dict[str, Any]:
    selected_book_ids = _from_json(message.selected_book_ids_json, [])
    citations = _from_json(message.citations_json, [])

    if not isinstance(selected_book_ids, list):
        selected_book_ids = []
    if not isinstance(citations, list):
        citations = []

    return {
        "id": message.id,
        "session_id": message.session_id,
        "role": message.role,
        "question_text": message.question_text,
        "answer_text": message.answer_text,
        "grounded": message.grounded,
        "selected_book_ids": selected_book_ids,
        "citations": citations,
        "created_at": message.created_at,
    }
