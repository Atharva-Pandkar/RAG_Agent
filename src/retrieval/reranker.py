"""Cross-encoder reranking: re-score a candidate set of retrieved chunks
against the query with a cross-encoder, then return the top-k.

Wraps any base retriever - fetches `fetch_k` candidates from it, reranks,
and truncates to `k`. See module docstring in dense_retriever.py for the
Windows pyarrow/torch import-order note (same fix applies here).
"""
from __future__ import annotations
import pyarrow  # noqa: F401  (import-order fix, see dense_retriever.py)
from typing import List, Dict
from sentence_transformers import CrossEncoder


class RerankRetriever:
    def __init__(self, base_retriever, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 fetch_k: int = 20):
        self.base_retriever = base_retriever
        self.model = CrossEncoder(model_name)
        self.fetch_k = fetch_k

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        candidates = self.base_retriever.retrieve(query, k=self.fetch_k)
        if not candidates:
            return candidates

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:k]
        return ranked
