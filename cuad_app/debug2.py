from app.retriever import HybridRetriever
from app.reranker  import Reranker

retriever = HybridRetriever()
reranker  = Reranker()

# use the AFSALA contract
target = "AFSALABANCORPINC_08_01_1996-EX-1.1-AGENCY AGREEMENT"

# find exact name match
match = next((n for n in retriever.contract_names if "AFSALA" in n), None)
print(f"Contract found: {match}\n")

test_questions = [
    "what is the governing law of this agreement",
    "what are the termination conditions",
    "what fees does Capital Resources receive",
    "what happens if minimum shares are not sold",
]

for q in test_questions:
    retrieved  = retriever.retrieve(q, contract_name=match)
    top_chunks = reranker.rerank(q, retrieved)
    
    print(f"Q: {q}")
    print(f"   chunks after rerank: {len(top_chunks)}")
    if top_chunks:
        best = top_chunks[0]
        print(f"   best score : {best['rerank_score']:.4f}")
        print(f"   clause_type: {best['metadata']['clause_type']}")
        print(f"   text snippet: {best['text'][:120]}")
    else:
        print("   ✗ NO CHUNKS PASSED RERANKER")
    print()