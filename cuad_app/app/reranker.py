from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from app.config import RERANKER_MODEL, RERANK_TOP_N, MIN_SCORE


class Reranker:

    def __init__(self):
        print("Loading reranker model...")
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL
        )
        self.model.eval()
        print("✓ reranker ready")

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []

        pairs = [[query, c["text"]] for c in chunks]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding        = True,
                truncation     = True,
                max_length     = 512,
                return_tensors = "pt"
            )
            # raw logits — ms-marco cross encoder outputs raw scores
            # higher = more relevant, no sigmoid needed
            scores = self.model(**inputs).logits.squeeze(-1).tolist()

        if isinstance(scores, float):
            scores = [scores]

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        ranked = sorted(chunks, key=lambda x: -x["rerank_score"])

        # only filter if ALL scores are very negative
        # this prevents threshold from killing valid results
        top = ranked[:RERANK_TOP_N]
        meaningful = [c for c in top if c["rerank_score"] > MIN_SCORE]

        # if threshold kills everything, return top 1 anyway
        # let Gemini's prompt handle low confidence answers
        return meaningful if len(meaningful) >= 3 else ranked[:3]