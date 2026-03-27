from app.retriever import HybridRetriever
from app.reranker import Reranker
from app.config import MIN_SCORE

print("Loading backend services...")
retriever = HybridRetriever()
reranker  = Reranker()

# pick the first contract name
test_contract = retriever.contract_names[0]
print(f"\nTesting with: {test_contract}")

query = "what is the governing law of this agreement"
print(f"Query: '{query}'\n")

# step 1 — Hybrid Retrieval
retrieved = retriever.retrieve(query, contract_name=test_contract)
print(f"--- Step 1: Hybrid Retrieval ---")
print(f"Retrieved {len(retrieved)} chunks from Chroma/BM25.")
for i, c in enumerate(retrieved[:3]):
    print(f"  [{i+1}] Initial Score: {c['score']:.3f} | Clause: {c['metadata']['clause_type']}")
    print(f"       {c['text'][:80]}...")
print("\n")

# step 2 — Cross-Encoder Reranking
if retrieved:
    print(f"--- Step 2: Cross-Encoder Reranking ---")
    print(f"Applying reranker with MIN_SCORE threshold = {MIN_SCORE}")
    
    # THIS is the magic line that actually calls your fixed code!
    top_chunks = reranker.rerank(query, retrieved)
    
    print(f"Chunks surviving the {MIN_SCORE} filter: {len(top_chunks)}\n")
    
    for i, c in enumerate(top_chunks):
        # We can now format it as a percentage because of your sigmoid fix!
        prob_pct = c['rerank_score'] * 100
        print(f"  [{i+1}] Probability: {c['rerank_score']:.4f} ({prob_pct:.1f}%) | Clause: {c['metadata']['clause_type']}")
        print(f"       {c['text'][:80]}...")
else:
    print("No chunks retrieved to rerank.")