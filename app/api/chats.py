from fastapi import APIRouter, HTTPException

from app.db import SessionLocal
from app.api.query import query_books
from app.schemas import (
    ChatQueryRequest,
    ChatQueryResponse,
    ChatSessionCreateRequest,
    ChatSessionDetailOut,
    ChatSessionOut,
)
from app.services.chat_history import (
    create_chat_session,
    delete_chat_session,
    get_chat_messages,
    get_chat_session,
    get_session_preview_text,
    list_chat_sessions,
    serialize_chat_message,
    serialize_chat_session,
)

router = APIRouter(prefix="/chats", tags=["chats"])


@router.post("", response_model=ChatSessionOut)
def create_session(payload: ChatSessionCreateRequest):
    db = SessionLocal()
    try:
        session = create_chat_session(db, title=payload.title)
        return ChatSessionOut(**serialize_chat_session(session))
    finally:
        db.close()


@router.get("", response_model=list[ChatSessionOut])
def list_sessions():
    db = SessionLocal()
    try:
        sessions = list_chat_sessions(db)
        return [
            ChatSessionOut(
                **serialize_chat_session(
                    session,
                    preview_text=get_session_preview_text(db, session.id),
                )
            )
            for session in sessions
        ]
    finally:
        db.close()


@router.get("/{session_id}", response_model=ChatSessionDetailOut)
def get_session(session_id: str):
    db = SessionLocal()
    try:
        session = get_chat_session(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found.")
        messages = get_chat_messages(db, session_id)
        return ChatSessionDetailOut(
            session=ChatSessionOut(
                **serialize_chat_session(
                    session,
                    preview_text=get_session_preview_text(db, session.id),
                )
            ),
            messages=[serialize_chat_message(message) for message in messages],
        )
    finally:
        db.close()


@router.delete("/{session_id}")
def delete_session(session_id: str):
    db = SessionLocal()
    try:
        deleted = delete_chat_session(db, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Chat session not found.")
        return {"ok": True}
    finally:
        db.close()


@router.post("/query", response_model=ChatQueryResponse)
def query_chat(payload: ChatQueryRequest):
    # Compatibility alias: canonical query flow is /query.
    return query_books(payload)
