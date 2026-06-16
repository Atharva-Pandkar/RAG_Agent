"""Hybrid retrieval: BM25 + FAISS dense rankings fused via Reciprocal Rank Fusion."""
from __future__ import annotations
import logging
import time
from typing import List, Dict

from .bm25_retriever import BM25Retriever
from .faiss_retriever import FaissRetriever

log = logging.getLogger("rag.retrieval.faiss_hybrid")


class FaissHybridRetriever:
    def __init__(self, chunks: List[Dict], model_name: str = "BAAI/bge-small-en-v1.5",
                 cache_key: str | None = None, rrf_k: int = 60, fetch_k: int = 20):
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k

        log.info("[faiss_hybrid] Initialising BM25 leg...")
        t0 = time.perf_counter()
        self.bm25 = BM25Retriever(chunks)
        log.info("[faiss_hybrid] BM25 leg ready (%.2fs)", time.perf_counter() - t0)

        log.info("[faiss_hybrid] Initialising FAISS dense leg...")
        t1 = time.perf_counter()
        self.dense = FaissRetriever(chunks, model_name=model_name, cache_key=cache_key)
        log.info("[faiss_hybrid] FAISS leg ready (%.2fs)", time.perf_counter() - t1)

        log.info("[faiss_hybrid] FaissHybridRetriever ready — rrf_k=%d  fetch_k=%d",
                 rrf_k, fetch_k)

    def add_chunks(self, new_chunks: List[Dict]) -> None:
        """Live-add chunks: rebuild BM25 and extend FAISS index."""
        self.chunks.extend(new_chunks)
        self.bm25.add_chunks(new_chunks)
        self.dense.add_chunks(new_chunks)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        log.debug("[faiss_hybrid] Retrieve — query: %r  k: %d  fetch_k: %d",
                  query[:80], k, self.fetch_k)

        # BM25 leg
        t0 = time.perf_counter()
        bm25_results = self.bm25.retrieve(query, k=self.fetch_k)
        log.debug("[faiss_hybrid] BM25 returned %d results (%.3fs)",
                  len(bm25_results), time.perf_counter() - t0)

        # Dense leg
        t1 = time.perf_counter()
        dense_results = self.dense.retrieve(query, k=self.fetch_k)
        log.debug("[faiss_hybrid] Dense returned %d results (%.3fs)",
                  len(dense_results), time.perf_counter() - t1)

        # RRF fusion
        scores: dict[str, float] = {}
        for rank, r in enumerate(bm25_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, r in enumerate(dense_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (self.rrf_k + rank + 1)

        by_id = {c["id"]: c for c in self.chunks}
        ranked_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:k]

        results = [
            {"id": i, "doc": by_id[i]["doc"], "text": by_id[i]["text"], "score": scores[i]}
            for i in ranked_ids
        ]
        log.debug("[faiss_hybrid] RRF fusion — %d candidates → top %d  best_score=%.4f",
                  len(scores), k, results[0]["score"] if results else 0)
        return results
