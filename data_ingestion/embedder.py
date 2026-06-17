"""
Embedder — Two outputs from one run
------------------------------------
1. embeddings_preview.json  — human-readable: every chunk with its full vector,
                               so you can see exactly what a 384-dim vector looks like.
2. chroma_db/               — ChromaDB collection on disk, ready for similarity search.

Embedding model: sentence-transformers/all-MiniLM-L6-v2
  - Free, runs fully local (no API key)
  - 384-dimensional vectors
  - Downloads once (~80 MB) to the HuggingFace cache, then works offline

Dependencies:
    pip install sentence-transformers chromadb
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHUNKS_FILE  = Path(__file__).parent / "chunks.json"
PREVIEW_FILE = Path(__file__).parent / "embeddings_preview.json"
CHROMA_DIR   = Path(__file__).parent / "chroma_db"
COLLECTION   = "enterprise_docs"
MODEL_NAME   = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality for English prose

# ---------------------------------------------------------------------------
# Load chunks
# ---------------------------------------------------------------------------

print("Loading chunks...")
chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
print(f"  {len(chunks)} chunks loaded from {CHUNKS_FILE.name}")

texts = [c["text"] for c in chunks]

# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

print(f"\nLoading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print("Embedding all chunks (this may take a moment)...")
vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
print(f"  Done. Each vector has {vectors.shape[1]} dimensions.")

# ---------------------------------------------------------------------------
# Output 1: Human-readable preview
# ---------------------------------------------------------------------------

print(f"\nWriting preview to {PREVIEW_FILE.name}...")

preview = []
for chunk, vector in zip(chunks, vectors):
    preview.append({
        # --- chunk identity ---
        "chunk_id":      chunk["chunk_id"],
        "source_file":   chunk["source_file"],
        "section_title": chunk["section_title"],
        "page_start":    chunk["page_start"],
        "page_end":      chunk["page_end"],
        "token_count":   chunk["token_count"],

        # --- the actual text ---
        "text": chunk["text"],

        # --- the embedding vector ---
        "embedding": {
            "model":       MODEL_NAME,
            "dimensions":  len(vector),
            # Full vector — all 384 floats, rounded to 6 decimal places for readability
            "vector":      [round(float(v), 6) for v in vector],
            # First 8 values shown separately so it's easy to see the "shape"
            "vector_preview_first_8": [round(float(v), 6) for v in vector[:8]],
        },
    })

PREVIEW_FILE.write_text(
    json.dumps(preview, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"  Saved {len(preview)} entries.")

# ---------------------------------------------------------------------------
# Output 2: ChromaDB vector store
# ---------------------------------------------------------------------------

print(f"\nInitialising ChromaDB at {CHROMA_DIR} ...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Drop and recreate so re-runs don't duplicate data
try:
    client.delete_collection(COLLECTION)
    print(f"  Existing collection '{COLLECTION}' cleared.")
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION,
    # Tell Chroma we're supplying our own vectors (don't use its default embedder)
    metadata={"hnsw:space": "cosine"},
)

print(f"  Ingesting {len(chunks)} chunks into collection '{COLLECTION}' ...")

# ChromaDB accepts batches; push everything in one shot (149 chunks is small)
collection.add(
    ids        = [c["chunk_id"] for c in chunks],
    embeddings = [v.tolist() for v in vectors],
    documents  = [c["text"] for c in chunks],
    metadatas  = [
        {
            "source_file":    c["source_file"],
            "section_title":  c["section_title"],
            "page_start":     c["page_start"],
            "page_end":       c["page_end"],
            "token_count":    c["token_count"],
            "sub_chunk_index": c["sub_chunk_index"],
        }
        for c in chunks
    ],
)

count = collection.count()
print(f"  ChromaDB collection '{COLLECTION}' now holds {count} vectors.")

# ---------------------------------------------------------------------------
# Quick sanity check — run one test query
# ---------------------------------------------------------------------------

print("\nRunning a quick test query: 'How many sick leaves are allowed?'")
test_query   = "How many sick leaves are allowed?"
query_vector = model.encode([test_query])[0].tolist()

results = collection.query(
    query_embeddings=[query_vector],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

print("\nTop 3 matching chunks:")
for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
), start=1):
    score = round(1 - dist, 4)   # cosine distance -> similarity score
    print(f"\n  [{i}] similarity={score}  |  {meta['source_file']} — {meta['section_title']}")
    print(f"       {doc[:200]}...")

print("\nDone. Both outputs are ready:")
print(f"  - {PREVIEW_FILE}")
print(f"  - {CHROMA_DIR}/")
