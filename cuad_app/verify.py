import sys
print(f"Python: {sys.version}")

# check all imports
try:
    import chromadb
    print(f"✓ chromadb {chromadb.__version__}")
except Exception as e:
    print(f"✗ chromadb: {e}")

try:
    import sentence_transformers
    print(f"✓ sentence-transformers {sentence_transformers.__version__}")
except Exception as e:
    print(f"✗ sentence-transformers: {e}")

try:
    import rank_bm25
    print(f"✓ rank-bm25 {rank_bm25.__version__}")
except Exception as e:
    print(f"✗ rank-bm25: {e}")

try:
    import fitz
    print(f"✓ pymupdf {fitz.__version__}")
except Exception as e:
    print(f"✗ pymupdf: {e}")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except Exception as e:
    print(f"✗ torch: {e}")

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except Exception as e:
    print(f"✗ transformers: {e}")

try:
    import google.generativeai as genai
    print(f"✓ google-generativeai {genai.__version__}")
except Exception as e:
    print(f"✗ google-generativeai: {e}")

try:
    import langchain
    print(f"✓ langchain {langchain.__version__}")
except Exception as e:
    print(f"✗ langchain: {e}")

try:
    import gradio
    print(f"✓ gradio {gradio.__version__}")
except Exception as e:
    print(f"✗ gradio: {e}")

# check artifacts
from pathlib import Path
import pickle

artifacts = Path("artifacts")
checks = [
    artifacts / "chroma_db",
    artifacts / "bm25_index.pkl",
    artifacts / "all_chunks.pkl",
]

print("\n=== Artifact check ===")
for p in checks:
    if p.exists():
        if p.is_file():
            size = p.stat().st_size / 1e6
            print(f"✓ {p.name}: {size:.1f} MB")
        else:
            files = list(p.rglob("*"))
            print(f"✓ {p.name}/: {len(files)} files")
    else:
        print(f"✗ MISSING: {p}")

# check env
from dotenv import load_dotenv
import os
load_dotenv()
key = os.getenv("GEMINI_API_KEY", "")
print(f"\n✓ GEMINI_API_KEY: {'set' if key else '✗ NOT SET'}")

print("\n✓ all checks done")