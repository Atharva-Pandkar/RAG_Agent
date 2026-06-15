"""
RagPipeline - glue between a chunked corpus, a retriever, and a generator.

Exposes the interface expected by eval/harness/run_eval.py:
    .retrieve(question, k) -> list[{"id","doc","text","score"}]
    .generate(question, contexts) -> str

Generation is pluggable via `llm_fn(question, contexts) -> str`. If no
llm_fn is provided (e.g. no API key configured yet), `.generate()` returns
a placeholder so the harness can still run end-to-end and retrieval metrics
can be computed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Callable, List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval.bm25_retriever import BM25Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class RagPipeline:
    def __init__(self, corpus_path: str, retrieval_strategy: str = "bm25",
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 llm_fn: Optional[Callable[[str, List[Dict]], str]] = None):
        corpus_path = ROOT / corpus_path
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        self.chunks = corpus["chunks"]
        self.retrieval_strategy = retrieval_strategy
        self.llm_fn = llm_fn

        # cache embeddings per (corpus, model) so repeated runs don't re-embed
        cache_key = f"{Path(corpus_path).stem}__{embedding_model.replace('/', '_')}"

        if retrieval_strategy == "bm25":
            self.retriever = BM25Retriever(self.chunks)
        elif retrieval_strategy == "dense":
            from retrieval.dense_retriever import DenseRetriever
            self.retriever = DenseRetriever(self.chunks, model_name=embedding_model, cache_key=cache_key)
        elif retrieval_strategy == "hybrid":
            from retrieval.hybrid_retriever import HybridRetriever
            self.retriever = HybridRetriever(self.chunks, model_name=embedding_model, cache_key=cache_key)
        else:
            raise ValueError(f"Unknown retrieval_strategy: {retrieval_strategy}")

    def retrieve(self, question: str, k: int = 5) -> List[Dict]:
        return self.retriever.retrieve(question, k=k)

    def generate(self, question: str, contexts: List[Dict]) -> str:
        if self.llm_fn is not None:
            return self.llm_fn(question, contexts)
        return "[NO LLM CONFIGURED] Retrieved context only - see retrieved_ids."
