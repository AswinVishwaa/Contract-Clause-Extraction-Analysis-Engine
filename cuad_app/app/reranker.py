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
        """
        Score each (query, chunk) pair with cross-encoder.
        Converts raw logits to probabilities via sigmoid.
        Returns top RERANK_TOP_N chunks sorted by score descending.
        Also filters below MIN_SCORE threshold.
        """
        if not chunks:
            return []

        # HF Tokenizers prefer separate lists for batched pairs
        queries = [query] * len(chunks)
        texts   = [c["text"] for c in chunks]

        with torch.no_grad():
            inputs = self.tokenizer(
                queries,
                texts,
                padding        = True,
                truncation     = True,
                max_length     = 512,
                return_tensors = "pt"
            )
            
            # 1. Get raw logits
            logits = self.model(**inputs).logits.squeeze(-1)
            
            # 2. Convert to 0.0 - 1.0 probability scale using sigmoid
            scores = torch.sigmoid(logits).tolist()

        # handle single result edge case
        if isinstance(scores, float):
            scores = [scores]

        # attach reranker probability score to each chunk
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = score

        # sort descending
        ranked = sorted(chunks, key=lambda x: -x["rerank_score"])

        # filter low confidence (now correctly comparing probability to 0.30)
        ranked = [c for c in ranked if c["rerank_score"] > MIN_SCORE]

        return ranked[:RERANK_TOP_N]