import chromadb
from sentence_transformers import SentenceTransformer

from agents.state import AgentState
from config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    RETRIEVAL_POOR_THRESHOLD,
    RETRIEVAL_TOP_K,
)

_model = None
_collection = None


def _get_resources():
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(CHROMA_COLLECTION)
    return _model, _collection


def retrieval_node(state: AgentState) -> AgentState:
    model, collection = _get_resources()

    query = state.get("reformulated_query") or state["query"]
    embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=RETRIEVAL_TOP_K,
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
    if best_score > RETRIEVAL_POOR_THRESHOLD and retry_count < 1:
        return "reformulate"

    return "reasoning"
