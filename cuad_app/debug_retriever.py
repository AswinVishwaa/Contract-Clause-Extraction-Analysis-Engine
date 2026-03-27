from app.retriever import HybridRetriever

print("Loading Retriever...")
retriever = HybridRetriever()

query = "District of Columbia governing law"
print(f"\n--- QUERY: '{query}' ---")

# 1. TEST CHROMA (DENSE) DIRECTLY
print("\n🔍 1. CHROMA (DENSE) TOP 5 HITS:")
q_emb = retriever.embedder.encode([query], normalize_embeddings=True).tolist()
dense_results = retriever.collection.query(
    query_embeddings=q_emb,
    n_results=5,
    include=["documents", "metadatas", "distances"]
)
for i, (doc, meta, dist) in enumerate(zip(
    dense_results["documents"][0],
    dense_results["metadatas"][0],
    dense_results["distances"][0]
)):
    score = 1 - dist
    print(f"  [{i+1}] Score: {score:.3f} | Contract: {meta['contract_name']}")
    print(f"       Text: {doc[:80].strip()}...")

# 2. TEST BM25 (KEYWORD) DIRECTLY
print("\n🔍 2. BM25 (KEYWORD) TOP 5 HITS:")
import numpy as np
query_tokens = retriever._tokenize(query)
bm25_scores = retriever.bm25.get_scores(query_tokens)
top_bm25_idx = np.argsort(bm25_scores)[::-1][:5]

for i, idx in enumerate(top_bm25_idx):
    chunk = retriever.all_chunks[idx]
    score = bm25_scores[idx]
    print(f"  [{i+1}] Score: {score:.3f} | Contract: {chunk['metadata']['contract_name']}")
    print(f"       Text: {chunk['text'][:80].strip()}...")