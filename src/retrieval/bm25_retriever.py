"""BM25 sparse retrieval over a chunked corpus."""
from __future__ import annotations
import re
from typing import List, Dict
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self._corpus_tokens = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self._corpus_tokens)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {"id": self.chunks[i]["id"], "doc": self.chunks[i]["doc"],
             "text": self.chunks[i]["text"], "score": float(scores[i])}
            for i in ranked
        ]
