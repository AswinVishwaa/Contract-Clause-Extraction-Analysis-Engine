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

# ── Gemini ───────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-2.5-flash-preview-04-17"

# ── Retrieval ────────────────────────────────────────────────────────
DENSE_TOP_K     = 20    # dense retrieval candidates
BM25_TOP_K      = 20    # sparse retrieval candidates
RRF_K           = 60    # RRF smoothing constant (standard)
RERANK_TOP_N    = 5     # final chunks passed to Gemini
MIN_SCORE       = 0.30  # below this → "clause not found"

# ── Embedding (must match Kaggle) ────────────────────────────────────
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"

# ── Reranker ─────────────────────────────────────────────────────────
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Chunking (for live PDF uploads) ─────────────────────────────────
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 50
CHAR_LIMIT      = CHUNK_SIZE * 4
OVERLAP_CHARS   = CHUNK_OVERLAP * 4

print("✓ config loaded")