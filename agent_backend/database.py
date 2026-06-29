import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

_client = None
_db = None


def get_db():
    global _client, _db
    if _client is None:
        _client = MongoClient(os.getenv("MONGO_URI"))
        _db = _client["enterprise-agent"]
    return _db


def _serialize(doc: dict) -> dict:
    """MongoDB uses _id internally — expose it as 'id' to the API."""
    if doc and "_id" in doc:
        doc["id"] = str(doc.pop("_id"))
    return doc


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def create_session(name: str, email: str) -> dict:
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    session = {
        "_id": str(uuid.uuid4()),
        "name": name,
        "email": email,
        "created_at": now,
        "modified_at": now,
    }
    db.sessions.insert_one(session)
    return _serialize(session)


def list_sessions(email: str) -> list:
    db = get_db()
    sessions = list(db.sessions.find({"email": email}).sort("modified_at", -1))
    return [_serialize(s) for s in sessions]


def get_session(session_id: str) -> dict | None:
    db = get_db()
    session = db.sessions.find_one({"_id": session_id})
    return _serialize(session) if session else None


def rename_session(session_id: str, name: str) -> dict | None:
    db = get_db()
    db.sessions.update_one(
        {"_id": session_id},
        {"$set": {"name": name, "modified_at": datetime.now(timezone.utc).isoformat()}},
    )
    return get_session(session_id)


def delete_session(session_id: str) -> None:
    db = get_db()
    db.sessions.delete_one({"_id": session_id})
    db.messages.delete_many({"session_id": session_id})


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def add_message(session_id: str, role: str, content: str, citations: list = None) -> dict:
    db = get_db()
    message = {
        "_id": str(uuid.uuid4()),
        "session_id": session_id,
        "role": role,
        "content": content,
        "citations": citations or [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    db.messages.insert_one(message)
    db.sessions.update_one(
        {"_id": session_id},
        {"$set": {"modified_at": datetime.now(timezone.utc).isoformat()}},
    )
    return _serialize(message)


def get_messages(session_id: str) -> list:
    db = get_db()
    messages = list(db.messages.find({"session_id": session_id}).sort("timestamp", 1))
    return [_serialize(m) for m in messages]
