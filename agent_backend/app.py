import sys
from pathlib import Path

_backend_dir = str(Path(__file__).parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph.workflow import app as workflow
from audit_logger import log_query

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

class ChatRequest(BaseModel):
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
# Routes
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = workflow.invoke({**_EMPTY_STATE, "query": req.message})
    log_query(req.message, result)

    # Build score lookup keyed by (source_file, section_title).
    # Cosine distance from ChromaDB: lower = better, so score = 1 - distance.
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

    sources = [
        Source(
            source_file=c.get("source_file", "Unknown"),
            section_title=c.get("section_title", ""),
            score=score_map.get((c.get("source_file"), c.get("section_title")), 0.0),
        )
        for c in result.get("citations", [])
    ]

    return ChatResponse(
        reply=result.get("answer", "I couldn't find an answer."),
        sources=sources,
        reformulated_query=result.get("reformulated_query"),
    )
