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

    def retrieve(self, query: str, k: int = 5,
                filter_doc: str | None = None) -> List[Dict]:
        log.debug("[bm25] Scoring query: %r  filter_doc: %s", query[:80], filter_doc)
        t0 = time.perf_counter()

        if filter_doc:
            # Pre-filter: only score chunks whose doc matches the prefix
            indices = [i for i, c in enumerate(self.chunks)
                       if c.get("doc", "").startswith(filter_doc)]
            if not indices:
                log.warning("[bm25] filter_doc=%r matched 0 chunks — returning empty", filter_doc)
                return []
            filtered_tokens = [self._corpus_tokens[i] for i in indices]
            bm25_local = BM25Okapi(filtered_tokens)
            scores = bm25_local.get_scores(_tokenize(query))
            ranked_local = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            results = [
                {"id": self.chunks[indices[i]]["id"], "doc": self.chunks[indices[i]]["doc"],
                 "text": self.chunks[indices[i]]["text"], "score": float(scores[i]),
                 "section": self.chunks[indices[i]].get("section")}
                for i in ranked_local
            ]
        else:
            scores = self.bm25.get_scores(_tokenize(query))
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            results = [
                {"id": self.chunks[i]["id"], "doc": self.chunks[i]["doc"],
                 "text": self.chunks[i]["text"], "score": float(scores[i]),
                 "section": self.chunks[i].get("section")}
                for i in ranked
            ]

        log.debug("[bm25] Top-%d scored in %.3fs  top_score=%.4f",
                  k, time.perf_counter() - t0, results[0]["score"] if results else 0)
        return results
