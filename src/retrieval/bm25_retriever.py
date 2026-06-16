"""BM25 sparse retrieval over a chunked corpus."""
from __future__ import annotations
import logging
import re
import time
from typing import List, Dict

from rank_bm25 import BM25Okapi

log = logging.getLogger("rag.retrieval.bm25")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: List[Dict]):
        log.info("[bm25] Tokenising %d chunks...", len(chunks))
        t0 = time.perf_counter()
        self.chunks = chunks
        self._corpus_tokens = [_tokenize(c["text"]) for c in chunks]
        log.info("[bm25] Building BM25 index...")
        self.bm25 = BM25Okapi(self._corpus_tokens)
        log.info("[bm25] Index ready — %d docs, %.2fs", len(chunks), time.perf_counter() - t0)

    def add_chunks(self, new_chunks: List[Dict]) -> None:
        """Append new chunks and rebuild the BM25 index (fast — < 1s for typical sizes)."""
        log.info("[bm25] Rebuilding index with %d additional chunks...", len(new_chunks))
        self.chunks.extend(new_chunks)
        self._corpus_tokens = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self._corpus_tokens)
        log.info("[bm25] Index rebuilt — %d docs total", len(self.chunks))

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        log.debug("[bm25] Scoring query: %r", query[:80])
        t0 = time.perf_counter()
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = [
            {"id": self.chunks[i]["id"], "doc": self.chunks[i]["doc"],
             "text": self.chunks[i]["text"], "score": float(scores[i])}
            for i in ranked
        ]
        log.debug("[bm25] Top-%d scored in %.3fs  top_score=%.4f",
                  k, time.perf_counter() - t0, results[0]["score"] if results else 0)
        return results
