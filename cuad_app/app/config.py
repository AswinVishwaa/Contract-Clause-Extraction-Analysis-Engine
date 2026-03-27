import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHROMA_DIR    = ARTIFACTS_DIR / "chroma_db"
BM25_PATH     = ARTIFACTS_DIR / "bm25_index.pkl"
CHUNKS_PATH   = ARTIFACTS_DIR / "all_chunks.pkl"
FULLTEXT_PATH       = ARTIFACTS_DIR / "contracts_fulltext.pkl"


# ── Gemini ───────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-2.5-flash"

# ── Retrieval ────────────────────────────────────────────────────────
DENSE_TOP_K     = 100    # dense retrieval candidates
BM25_TOP_K      = 100    # sparse retrieval candidates
RRF_K           = 60    # RRF smoothing constant (standard)
RERANK_TOP_N    = 10     # final chunks passed to Gemini
MIN_SCORE       = -5.0  # below this → "clause not found"

# ── Embedding (must match Kaggle) ────────────────────────────────────
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"

# ── Reranker ─────────────────────────────────────────────────────────
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Chunking (for live PDF uploads) ─────────────────────────────────
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 50
CHAR_LIMIT      = CHUNK_SIZE * 4
OVERLAP_CHARS   = CHUNK_OVERLAP * 4
FULLTEXT_CHAR_LIMIT = 3_000_000  # fallback to RAG above this

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. Add it to your .env file:\n"
        "  GEMINI_API_KEY=your_key_here"
    )

print("✓ config loaded")