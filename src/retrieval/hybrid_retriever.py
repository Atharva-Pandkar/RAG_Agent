"""Hybrid retrieval: combine BM25 and dense rankings via Reciprocal Rank Fusion (RRF).

RRF score for a doc = sum over retrievers of 1 / (rrf_k + rank_in_that_retriever).
Avoids needing to normalize BM25 vs cosine scores onto a common scale.
"""
from __future__ import annotations
from typing import List, Dict
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever


class HybridRetriever:
    def __init__(self, chunks: List[Dict], model_name: str = "BAAI/bge-small-en-v1.5",
                 cache_key: str | None = None, rrf_k: int = 60, fetch_k: int = 20):
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks, model_name=model_name, cache_key=cache_key)
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        bm25_results = self.bm25.retrieve(query, k=self.fetch_k)
        dense_results = self.dense.retrieve(query, k=self.fetch_k)

        scores: dict[str, float] = {}
        for rank, r in enumerate(bm25_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, r in enumerate(dense_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (self.rrf_k + rank + 1)

        by_id = {c["id"]: c for c in self.chunks}
        ranked_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {"id": i, "doc": by_id[i]["doc"], "text": by_id[i]["text"], "score": scores[i]}
            for i in ranked_ids
        ]
