import pickle
import re
import hashlib
from pathlib import Path
import numpy as np

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from app.config import (
    CHROMA_DIR, BM25_PATH, CHUNKS_PATH,
    EMBED_MODEL, DENSE_TOP_K, BM25_TOP_K, RRF_K
)


class HybridRetriever:

    def __init__(self):
        print("Loading ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection("contracts")
        print(f"  ✓ collection loaded | docs: {self.collection.count()}")

        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        print(f"  ✓ embedder ready")

        print("Loading BM25 index...")
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25          = data["bm25"]
        self.corpus_tokens = data["corpus_tokens"]

        print("Loading chunks...")
        with open(CHUNKS_PATH, "rb") as f:
            self.all_chunks = pickle.load(f)

        # contract name → id mapping for filtering
        self.contract_map = {
            c["metadata"]["contract_name"]: c["metadata"]["contract_id"]
            for c in self.all_chunks
        }
        self.contract_names = sorted(self.contract_map.keys())
        print(f"  ✓ {len(self.contract_names)} contracts indexed")
        print("✓ HybridRetriever ready\n")


    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())


    def _rrf_merge(self, dense_ids: list, bm25_ids: list) -> list[str]:
        """Reciprocal Rank Fusion — merge two ranked lists."""
        scores = {}
        for rank, doc_id in enumerate(dense_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + RRF_K)
        for rank, doc_id in enumerate(bm25_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + RRF_K)
        # sort by descending RRF score
        return sorted(scores, key=lambda x: -scores[x])


    def retrieve(self, query: str,
                 contract_name: str = None,
                 top_k: int = 20) -> list[dict]:
        """
        Hybrid retrieval — dense + BM25 merged via RRF.
        If contract_name provided, filters to that contract only.
        Returns list of chunk dicts with text + metadata.
        """
        # ── dense retrieval ──────────────────────────────────────────
        q_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        where_filter = None
        if contract_name and contract_name in self.contract_map:
            where_filter = {
                "contract_id": self.contract_map[contract_name]
            }

        dense_results = self.collection.query(
            query_embeddings = q_emb,
            n_results        = DENSE_TOP_K,
            where            = where_filter,
            include          = ["documents", "metadatas", "distances"]
        )

        dense_ids  = []
        dense_docs = {}
        for chunk_id, doc, meta, dist in zip(
            dense_results["ids"][0],
            dense_results["documents"][0],
            dense_results["metadatas"][0],
            dense_results["distances"][0]
        ):
            dense_ids.append(chunk_id)
            dense_docs[chunk_id] = {
                "text":     doc,
                "metadata": meta,
                "score":    1 - dist
            }

        # ── BM25 retrieval ───────────────────────────────────────────
        query_tokens = self._tokenize(query)
        bm25_scores  = self.bm25.get_scores(query_tokens)

        # if contract filter — zero out other contracts
        if contract_name and contract_name in self.contract_map:
            cid = self.contract_map[contract_name]
            for i, chunk in enumerate(self.all_chunks):
                if chunk["metadata"]["contract_id"] != cid:
                    bm25_scores[i] = 0.0

        # top BM25 indices
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:BM25_TOP_K]
        bm25_ids = []
        for idx in top_bm25_idx:
            if bm25_scores[idx] > 0:
                chunk   = self.all_chunks[idx]
                meta    = chunk["metadata"]
                cid     = f"{meta['contract_id']}_{meta['chunk_index']}"
                bm25_ids.append(cid)
                if cid not in dense_docs:
                    dense_docs[cid] = {
                        "text":     chunk["text"],
                        "metadata": meta,
                        "score":    float(bm25_scores[idx])
                    }

        # ── RRF merge ────────────────────────────────────────────────
        merged_ids = self._rrf_merge(dense_ids, bm25_ids)[:top_k]

        return [dense_docs[cid] for cid in merged_ids if cid in dense_docs]


    def add_chunks(self, new_chunks: list[dict]):
        """
        Add uploaded PDF chunks to ChromaDB + rebuild BM25.
        Called by ingestor after processing a new PDF.
        """
        if not new_chunks:
            return

        texts     = [c["text"]     for c in new_chunks]
        metadatas = [c["metadata"] for c in new_chunks]
        ids       = [
            f"{m['contract_id']}_{m['chunk_index']}"
            for m in metadatas
        ]

        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        self.collection.add(
            ids        = ids,
            embeddings = embeddings,
            documents  = texts,
            metadatas  = metadatas
        )

        # update in-memory chunks + rebuild BM25
        self.all_chunks.extend(new_chunks)
        new_tokens = [self._tokenize(c["text"]) for c in new_chunks]
        self.corpus_tokens.extend(new_tokens)

        self.bm25 = BM25Okapi(self.corpus_tokens)

        # update contract map
        for c in new_chunks:
            name = c["metadata"]["contract_name"]
            cid  = c["metadata"]["contract_id"]
            self.contract_map[name] = cid

        self.contract_names = sorted(self.contract_map.keys())
        print(f"✓ added {len(new_chunks)} chunks | total: {len(self.all_chunks)}")