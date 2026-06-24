from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from agents.state import AgentState

CHROMA_PATH = Path(__file__).parent.parent / "data_ingestion" / "chroma_db"
COLLECTION_NAME = "enterprise_docs"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
POOR_RESULT_THRESHOLD = 0.5  # cosine distance; lower = better match

_model = None
_collection = None


def _get_resources():
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(COLLECTION_NAME)
    return _model, _collection


def retrieval_node(state: AgentState) -> AgentState:
    model, collection = _get_resources()

    query = state.get("reformulated_query") or state["query"]
    embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks, scores = [], []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta})
        scores.append(dist)

    return {**state, "retrieved_chunks": chunks, "retrieval_scores": scores}


def check_retrieval_quality(state: AgentState) -> str:
    scores = state.get("retrieval_scores", [])
    retry_count = state.get("retry_count", 0)

    if not scores:
        return "reformulate"

    best_score = min(scores)  # cosine distance — lower is better
    if best_score > POOR_RESULT_THRESHOLD and retry_count < 1:
        return "reformulate"

    return "reasoning"
