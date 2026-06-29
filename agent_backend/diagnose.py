"""
Run this from agent_backend/ to diagnose the pipeline step by step.
    python diagnose.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

QUERY = "How many sick leaves do I have?"

# ── Step 1: Retrieval ────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Retrieval")
print("=" * 60)

from sentence_transformers import SentenceTransformer
import chromadb
from config import CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL, RETRIEVAL_TOP_K

model = SentenceTransformer(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(CHROMA_COLLECTION)

print(f"Total vectors in ChromaDB: {collection.count()}")

embedding = model.encode(QUERY).tolist()
results = collection.query(
    query_embeddings=[embedding],
    n_results=RETRIEVAL_TOP_K,
    include=["documents", "metadatas", "distances"],
)

print(f"\nTop {RETRIEVAL_TOP_K} results for: '{QUERY}'")
for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
), 1):
    similarity = round(1 - dist, 4)
    print(f"\n[{i}] distance={round(dist,4)}  similarity={similarity}")
    print(f"    Source : {meta['source_file']} — {meta['section_title']}")
    print(f"    Preview: {doc[:120]}...")

# ── Step 2: Cohere API ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Cohere API (reasoning model)")
print("=" * 60)

import os
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from config import REASONING_MODEL

print(f"Model: {REASONING_MODEL}")
print(f"API key present: {'YES' if os.getenv('COHERE_API_KEY') else 'NO — KEY MISSING'}")

try:
    llm = ChatCohere(
        model=REASONING_MODEL,
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        max_tokens=512,
    )

    context = "\n\n---\n\n".join(
        f"[{meta['source_file']} — {meta['section_title']}]\n{doc}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    )

    prompt = f"""You are an enterprise knowledge assistant. Answer using ONLY these excerpts.
If not enough info, respond with exactly: CANNOT_ANSWER

Question: {QUERY}

Document Excerpts:
{context}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content

    print(f"\nRaw response type : {type(content)}")
    print(f"Raw response value: {content}")

except Exception as e:
    print(f"\nERROR calling Cohere: {type(e).__name__}: {e}")
