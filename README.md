# ⚖️ Legal Contract Analysis System

A RAG-based system for querying, extracting, and comparing clauses across 510 commercial contracts from the CUAD dataset. Built for the i2e Consulting Hireathon 2026 — Problem Statement 3.

---

## How to Run

### 1. Install dependencies

```bash
pip install chromadb sentence-transformers rank-bm25 pymupdf \
            google-generativeai gradio python-dotenv \
            transformers torch tqdm
```

### 2. Set your API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

### 3. Download artifacts

Run `chromadb.ipynb` on Kaggle (GPU recommended). Download these 3 files from Kaggle output and place them as shown:

```
artifacts/
├── chroma_db/        ← unzip chroma_db.zip here
├── bm25_index.pkl
├── all_chunks.pkl
└── contracts_fulltext.pkl
```

### 4. Start the app

```bash
python app/app.py
```

Open `http://localhost:7860` in your browser.

---

## Architecture

```
User Query
    │
    ├─────────────────────────────────────┐
    ▼                                     ▼
[Dense Retrieval]                  [BM25 Retrieval]
 ChromaDB + BGE-small-en-v1.5       rank-bm25 Okapi
 Top-100 candidates                 Top-100 candidates
    │                                     │
    └──────────────┬──────────────────────┘
                   ▼
           [RRF Merge]
      Reciprocal Rank Fusion k=60
                   │
                   ▼
      [Cross-Encoder Reranker]
     ms-marco-MiniLM-L-6-v2
       Top-10 chunks selected
                   │
                   ▼
        [Gemini 2.5 Flash]
      Answer + source citations
```

### Two retrieval strategies run in parallel

| Strategy | Triggered when | Approach |
|---|---|---|
| **A — Full Document** | Single contract selected + doc ≤ 3M chars | Entire contract text sent to Gemini in batches |
| **B — Hybrid RAG** | All Contracts mode, or very large doc | Dense + BM25 → RRF → Reranker → Gemini |
| **Dynamic Promotion** | Strategy B top rerank score > 2.0 | Automatically escalates to full-doc for high-confidence single-contract hits |

---

## Key Design Decisions

### 1. Hybrid retrieval (Dense + BM25) instead of dense-only

Dense embeddings (BGE-small) capture semantic similarity well but miss exact legal terms — a clause saying "construed under the laws of Delaware" scores lower than expected for the query "governing law" if the model hasn't seen that phrasing. BM25 catches exact keyword matches that dense retrieval misses. RRF fusion (k=60) merges both ranked lists without needing to tune score scales, since it only uses rank positions.

**Tradeoff:** Two retrieval passes + merge adds ~50ms latency. Acceptable for a legal QA tool where accuracy matters more than speed.

### 2. Cross-encoder reranker as a second-stage filter

The first-stage retrieval returns 100+100 candidates. A bi-encoder (used for dense retrieval) is fast but imprecise — it embeds query and document independently. The cross-encoder (ms-marco-MiniLM-L-6-v2) reads query and chunk together, giving much more accurate relevance scores. Running it only on the top 20 merged candidates keeps latency low.

**Tradeoff:** Reranker adds ~200ms on CPU. On the Kaggle T4 GPU this is negligible, but on a CPU-only deployment it is noticeable for complex queries.

### 3. Full-document mode (Strategy A) for single-contract queries

When a user selects one specific contract and it fits within Gemini's context window (≤3M chars), sending the entire document is strictly better than RAG — no retrieval errors, no missed clauses, no chunking artifacts. The RAG pipeline is a fallback for cross-contract search where full-doc is not feasible.

**Tradeoff:** Full-doc mode uses more Gemini tokens per query, which costs more at scale. For a demo with 510 contracts this is fine.

### 4. Regex-based clause tagging at ingest time (41 CUAD categories)

Each chunk is tagged with its likely clause type using CUAD-aligned regex patterns during ingestion. This tag is passed to Gemini as metadata context. Even when the tag is wrong (chunk labeled `unknown`), Gemini is explicitly instructed to read the actual text — so a mislabeled chunk still contributes if it contains the answer.

**Tradeoff:** Regex is brittle for unusual phrasing. A fine-tuned classifier would be more accurate but requires training data and adds inference overhead.

### 5. Section-aware chunking over fixed token windows

The ingestor splits on section headers (detected by regex for patterns like `Section 12`, `ARTICLE IV`, `1.2 Heading`) before applying the character limit. This keeps legal clauses intact — a termination clause that spans a full section stays in one chunk instead of being split mid-sentence.

**Tradeoff:** Section detection regex misses some header formats (e.g. headers in all-caps with no numbering), causing occasional oversized chunks that get split by the fallback character limit anyway.

---

## Known Limitations and Failure Modes

### 1. Absence queries fail in RAG mode

**Problem:** "Which contracts have NO cap on liability?" — RAG retrieves positive evidence. It cannot prove absence across 510 contracts from retrieved chunks alone. The system may confidently say a cap exists (because it found one) but cannot confirm absence without reading the whole contract.

**Mitigation:** The Clause Presence Matrix in the Analysis tab scans all indexed chunks and shows absence explicitly. Risk Flagging also surfaces missing protective clauses per contract.

### 2. Clause detection depends on regex — novel phrasing is missed

**Problem:** If a contract uses "maximum exposure" instead of "cap on liability", the regex won't tag it correctly. The chunk is still indexed but labeled `unknown`, which reduces its priority in filtered searches.

**Mitigation:** Gemini is instructed to read actual text regardless of label. Dense retrieval also catches semantic variants even without the correct tag.

### 3. Multi-page table extraction is not supported

**Problem:** CUAD contracts contain tables spanning multiple pages (e.g. payment schedules, milestone matrices). PyMuPDF extracts these as fragmented text rows without column context, making them hard to query accurately.

**Impact:** Queries like "what is the payment schedule" may return incomplete or garbled answers if the answer is in a table.

### 4. Chunking can split mid-clause on very long sections

**Problem:** When a single section exceeds `CHAR_LIMIT` (2048 chars), it is split with a 200-char overlap. If the key sentence is near the split boundary, one chunk may have the context and the other the answer — neither chunk alone is fully meaningful.

**Impact:** Rare but possible for extremely detailed indemnification or representation sections.

### 5. Dynamic Promotion can be wrong

**Problem:** If the top reranker score exceeds 2.0 for a chunk from Contract A, the system switches to full-document mode for Contract A even if the user's query spans multiple contracts. This can suppress relevant results from other contracts.

**Impact:** Affects "All Contracts" mode queries where one contract happens to strongly match the query keywords but is not the user's intended target.

### 6. Uploaded contracts are not persisted across restarts

**Problem:** PDFs uploaded through the UI are added to ChromaDB and BM25 in memory, and stitched into `fulltext_by_name` in memory. On app restart, these are lost. Only the pre-built Kaggle artifacts persist.

**Impact:** Uploaded contracts need to be re-uploaded after every restart. For a production deployment, `add_chunks` should also persist updated BM25 and fulltext to disk.

---

## Validation Results

Evaluated on 18 samples from CUAD_v1.json ground truth:

| Model | Exact Match | Recall (Partial) |
|---|---|---|
| Baseline (Gemini flash-lite, no RAG) | 2/18 — 11.1% | 12/18 — 66.7% |
| Full Pipeline (Hybrid RAG + Reranker + Gemini flash) | *run separately* | *run separately* |

The baseline uses raw Gemini on the CUAD context directly. The pipeline runs the full retriever → reranker → Gemini stack and is expected to improve exact match by surfacing precise clause text before generation.

---

## Dataset

**CUAD v1** — Contract Understanding Atticus Dataset
- 510 commercial contracts from SEC EDGAR
- 13,000+ expert-labelled clause annotations
- 41 clause categories
- License: CC BY 4.0
- Source: [The Atticus Project](https://www.atticusprojectai.org/cuad)
