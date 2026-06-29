import sys
from pathlib import Path

_backend_dir = str(Path(__file__).parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph.workflow import app as workflow
from audit_logger import log_query
import database as db

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Enterprise Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    name: str
    email: str

class RenameSessionRequest(BaseModel):
    name: str

class MessageRequest(BaseModel):
    message: str

class Source(BaseModel):
    source_file: str
    section_title: str
    score: float

class ChatResponse(BaseModel):
    reply: str
    sources: list[Source]
    reformulated_query: str | None = None

# ---------------------------------------------------------------------------
# Agent helper
# ---------------------------------------------------------------------------

_EMPTY_STATE = {
    "query": "",
    "reformulated_query": None,
    "retrieved_chunks": [],
    "retrieval_scores": [],
    "retry_count": 0,
    "answer": None,
    "citations": [],
    "can_answer": False,
}

def _run_agent(query: str) -> dict:
    result = workflow.invoke({**_EMPTY_STATE, "query": query})
    log_query(query, result)
    return result

def _build_sources(result: dict) -> list[Source]:
    score_map: dict[tuple, float] = {}
    for chunk, dist in zip(
        result.get("retrieved_chunks", []),
        result.get("retrieval_scores", []),
    ):
        meta = chunk.get("metadata", {})
        key = (meta.get("source_file"), meta.get("section_title"))
        score = round(1 - dist, 4)
        if key not in score_map or score > score_map[key]:
            score_map[key] = score

    return [
        Source(
            source_file=c.get("source_file", "Unknown"),
            section_title=c.get("section_title", ""),
            score=score_map.get((c.get("source_file"), c.get("section_title")), 0.0),
        )
        for c in result.get("citations", [])
    ]

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@app.post("/sessions", status_code=201)
def create_session(req: CreateSessionRequest):
    return db.create_session(name=req.name, email=req.email)


@app.get("/sessions")
def list_sessions(email: str):
    return db.list_sessions(email=email)


@app.patch("/sessions/{session_id}")
def rename_session(session_id: str, req: RenameSessionRequest):
    session = db.rename_session(session_id=session_id, name=req.name)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    db.delete_session(session_id=session_id)

# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return db.get_messages(session_id=session_id)


@app.post("/sessions/{session_id}/message")
def send_message(session_id: str, req: MessageRequest):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    db.add_message(session_id=session_id, role="user", content=req.message)

    result = _run_agent(req.message)
    answer = result.get("answer", "I couldn't find an answer.")
    sources = _build_sources(result)

    assistant_message = db.add_message(
        session_id=session_id,
        role="assistant",
        content=answer,
        citations=[s.model_dump() for s in sources],
    )

    return {
        "message": assistant_message,
        "reply": answer,
        "sources": sources,
        "reformulated_query": result.get("reformulated_query"),
    }
