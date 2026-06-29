from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
CHROMA_PATH = BASE_DIR / "data_ingestion" / "chroma_db"

# ---------------------------------------------------------------------------
# Embedding & Vector Store
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # local model, no API key needed
CHROMA_COLLECTION = "enterprise_docs"

# ---------------------------------------------------------------------------
# Chunking  (used by data_ingestion/chunker.py)
# ---------------------------------------------------------------------------

CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

RETRIEVAL_TOP_K = 5                     # how many chunks to fetch per query
RETRIEVAL_POOR_THRESHOLD = 0.75         # cosine distance above this = poor match → reformulate

# ---------------------------------------------------------------------------
# LLM Models  (Cohere)
# ---------------------------------------------------------------------------

REFORMULATION_MODEL = "command-a-plus-05-2026"      # fast general model for query rewriting
REASONING_MODEL = "command-a-reasoning-08-2025"      # reasoning model for answer synthesis

REFORMULATION_MAX_TOKENS = 256
REASONING_MAX_TOKENS = 1024
