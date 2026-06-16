"""
RagPipeline - glue between a chunked corpus, a retriever, and a generator.
"""
from __future__ import annotations
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval.bm25_retriever import BM25Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("rag.pipeline")


class RagPipeline:
    def __init__(self, corpus_path: str, retrieval_strategy: str = "bm25",
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 rerank: bool = False,
                 rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 rerank_fetch_k: int = 20,
                 llm_fn: Optional[Callable[[str, List[Dict]], str]] = None):

        # ── 1. Load corpus ─────────────────────────────────────────────────
        abs_path = ROOT / corpus_path
        log.info("[pipeline] Loading corpus: %s", abs_path)
        if not abs_path.exists():
            log.error("[pipeline] Corpus file NOT FOUND: %s", abs_path)
            raise FileNotFoundError(f"Corpus not found: {abs_path}")

        t0 = time.perf_counter()
        with open(abs_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        self.chunks = corpus["chunks"]
        log.info("[pipeline] Corpus loaded — %d chunks (%.2fs)",
                 len(self.chunks), time.perf_counter() - t0)

        self.retrieval_strategy = retrieval_strategy
        self.llm_fn = llm_fn

        cache_key = f"{abs_path.stem}__{embedding_model.replace('/', '_')}"
        log.info("[pipeline] Cache key: %s", cache_key)

        # ── 2. Build retriever ─────────────────────────────────────────────
        log.info("[pipeline] Building retriever — strategy: %s", retrieval_strategy)
        t1 = time.perf_counter()

        if retrieval_strategy == "bm25":
            self.retriever = BM25Retriever(self.chunks)

        elif retrieval_strategy == "dense":
            from retrieval.dense_retriever import DenseRetriever
            self.retriever = DenseRetriever(self.chunks, model_name=embedding_model,
                                             cache_key=cache_key)

        elif retrieval_strategy == "hybrid":
            from retrieval.hybrid_retriever import HybridRetriever
            self.retriever = HybridRetriever(self.chunks, model_name=embedding_model,
                                              cache_key=cache_key)

        elif retrieval_strategy == "faiss":
            from retrieval.faiss_retriever import FaissRetriever
            self.retriever = FaissRetriever(self.chunks, model_name=embedding_model,
                                             cache_key=cache_key)

        elif retrieval_strategy == "faiss_hybrid":
            from retrieval.faiss_hybrid_retriever import FaissHybridRetriever
            self.retriever = FaissHybridRetriever(self.chunks, model_name=embedding_model,
                                                   cache_key=cache_key)
        else:
            log.error("[pipeline] Unknown retrieval_strategy: %s", retrieval_strategy)
            raise ValueError(f"Unknown retrieval_strategy: {retrieval_strategy}")

        log.info("[pipeline] Retriever ready — %s (%.2fs)",
                 type(self.retriever).__name__, time.perf_counter() - t1)

        # ── 3. Optional reranker ───────────────────────────────────────────
        if rerank:
            log.info("[pipeline] Adding reranker: %s (fetch_k=%d)", rerank_model, rerank_fetch_k)
            from retrieval.reranker import RerankRetriever
            self.retriever = RerankRetriever(self.retriever, model_name=rerank_model,
                                              fetch_k=rerank_fetch_k)
            log.info("[pipeline] Reranker ready")

        log.info("[pipeline] RagPipeline fully initialised (total %.2fs)",
                 time.perf_counter() - t0)

    def retrieve(self, question: str, k: int = 5) -> List[Dict]:
        log.debug("[pipeline] retrieve() — question: %r  k: %d", question[:80], k)
        t0 = time.perf_counter()
        results = self.retriever.retrieve(question, k=k)
        log.debug("[pipeline] retrieve() returned %d results (%.2fs)",
                  len(results), time.perf_counter() - t0)
        return results

    def add_chunks(self, new_chunks: List[Dict]) -> None:
        """Add new chunks to the in-memory retriever without restarting."""
        log.info("[pipeline] add_chunks() — %d new chunks", len(new_chunks))
        self.chunks.extend(new_chunks)
        if hasattr(self.retriever, "add_chunks"):
            self.retriever.add_chunks(new_chunks)
        else:
            log.warning("[pipeline] Retriever %s does not support add_chunks()",
                        type(self.retriever).__name__)

    def generate(self, question: str, contexts: List[Dict]) -> str:
        if self.llm_fn is not None:
            return self.llm_fn(question, contexts)
        return "[NO LLM CONFIGURED] Retrieved context only - see retrieved_ids."
