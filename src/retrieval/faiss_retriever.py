"""Dense retrieval via a LangChain FAISS vector store.

Embedding backend
-----------------
Uses OpenAI embeddings (text-embedding-3-small) by default — no local model
to load, no PyTorch dependency, zero startup time.  The FAISS index is cached
to disk after the first build so OpenAI is only called once per corpus.

To use a local HuggingFace model instead, set:
    EMBED_PROVIDER=huggingface
    EMBED_MODEL=BAAI/bge-small-en-v1.5
in your .env (requires: pip install langchain-huggingface sentence-transformers)
"""
from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import List, Dict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[2]
FAISS_CACHE_DIR = ROOT / "Experiments" / "embeddings"

log = logging.getLogger("rag.retrieval.faiss")

# Module-level singleton so the embedding client is shared across all retrievers
_EMBEDDING_INSTANCES: dict[str, Embeddings] = {}


def _get_embeddings(model_name: str) -> Embeddings:
    """Return a cached embedding client, creating it if needed."""
    if model_name in _EMBEDDING_INSTANCES:
        log.debug("[faiss] Reusing embedding client: %s", model_name)
        return _EMBEDDING_INSTANCES[model_name]

    provider = os.environ.get("EMBED_PROVIDER", "openai").lower()
    log.info("[faiss] Creating embedding client — provider: %s  model: %s", provider, model_name)
    t0 = time.perf_counter()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        client = OpenAIEmbeddings(model=model_name)
        log.info("[faiss] OpenAI embedding client ready (%.2fs) — no local model to load",
                 time.perf_counter() - t0)

    elif provider == "huggingface":
        # Heavy: loads PyTorch + sentence-transformers (~15-20s on first call)
        log.info("[faiss] Loading HuggingFace model (this takes ~15-20s)...")
        import pyarrow  # noqa: F401  (Windows import-order fix before torch)
        from langchain_huggingface import HuggingFaceEmbeddings
        client = HuggingFaceEmbeddings(model_name=model_name)
        log.info("[faiss] HuggingFace model ready (%.2fs)", time.perf_counter() - t0)

    else:
        raise ValueError(f"Unknown EMBED_PROVIDER: {provider!r}. Use 'openai' or 'huggingface'.")

    _EMBEDDING_INSTANCES[model_name] = client
    return client


class FaissRetriever:
    """
    Initialisation paths
    --------------------
    CACHE HIT  → no embedding calls at init; FAISS index loaded lazily on
                 first retrieve() so startup is near-instant.
    CACHE MISS → embeds all chunks now (one OpenAI API batch call) and saves
                 the index to disk for future runs.
    """

    def __init__(self, chunks: List[Dict], model_name: str = "text-embedding-3-small",
                 cache_key: str | None = None):
        self.chunks      = chunks
        self._model_name = model_name
        self._store: FAISS | None = None
        self._index_dir: Path | None = None

        if cache_key:
            FAISS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._index_dir = FAISS_CACHE_DIR / f"faiss_{cache_key}"

        if self._index_dir and self._index_dir.exists():
            log.info("[faiss] Cache hit — index deferred to first query: %s", self._index_dir)
        else:
            log.info("[faiss] Cache miss — building index for %d chunks...", len(chunks))
            self._build_index()

    def _build_index(self) -> None:
        embeddings = _get_embeddings(self._model_name)
        docs = [
            Document(page_content=c["text"], metadata={"id": c["id"], "doc": c["doc"]})
            for c in self.chunks
        ]
        log.info("[faiss] Sending %d docs to embedding API...", len(docs))
        t0 = time.perf_counter()
        self._store = FAISS.from_documents(docs, embeddings)
        log.info("[faiss] Index built (%.2fs)", time.perf_counter() - t0)
        if self._index_dir:
            log.info("[faiss] Saving to cache: %s", self._index_dir)
            self._store.save_local(str(self._index_dir))
            log.info("[faiss] Cache saved — future startups will skip embedding")

    def _ensure_loaded(self) -> None:
        if self._store is not None:
            return
        log.info("[faiss] Loading index from cache (first query)...")
        t0 = time.perf_counter()
        embeddings = _get_embeddings(self._model_name)
        self._store = FAISS.load_local(
            str(self._index_dir), embeddings, allow_dangerous_deserialization=True
        )
        log.info("[faiss] Index loaded (%.2fs)", time.perf_counter() - t0)

    def add_chunks(self, new_chunks: List[Dict]) -> None:
        """Embed new chunks and merge into the existing FAISS index."""
        self._ensure_loaded()
        embeddings = _get_embeddings(self._model_name)
        new_docs = [
            Document(page_content=c["text"], metadata={"id": c["id"], "doc": c["doc"]})
            for c in new_chunks
        ]
        log.info("[faiss] Adding %d documents to existing index...", len(new_docs))
        t0 = time.perf_counter()
        self._store.add_documents(new_docs)
        self.chunks.extend(new_chunks)
        log.info("[faiss] Documents added (%.2fs)", time.perf_counter() - t0)
        if self._index_dir:
            log.info("[faiss] Saving updated index to cache...")
            self._store.save_local(str(self._index_dir))
            log.info("[faiss] Cache updated")

    def retrieve(self, query: str, k: int = 5,
                filter_doc: str | None = None) -> List[Dict]:
        self._ensure_loaded()
        log.debug("[faiss] Dense search — query: %r  k: %d  filter_doc: %s",
                  query[:80], k, filter_doc)
        t0 = time.perf_counter()

        if filter_doc:
            # Fetch a larger candidate set, then post-filter by doc prefix.
            # LangChain FAISS does not support prefix filtering natively, so
            # we over-fetch and trim.  The corpus is small enough that this is fast.
            fetch = min(k * 10, len(self.chunks))
            raw = self._store.similarity_search_with_relevance_scores(query, k=fetch)
            results = [
                {"id": doc.metadata["id"], "doc": doc.metadata["doc"],
                 "text": doc.page_content, "score": float(score)}
                for doc, score in raw
                if doc.metadata.get("doc", "").startswith(filter_doc)
            ][:k]
            log.debug("[faiss] %d results after filter (%.3fs)", len(results),
                      time.perf_counter() - t0)
        else:
            raw = self._store.similarity_search_with_relevance_scores(query, k=k)
            results = [
                {"id": doc.metadata["id"], "doc": doc.metadata["doc"],
                 "text": doc.page_content, "score": float(score)}
                for doc, score in raw
            ]
            log.debug("[faiss] %d results (%.3fs)", len(results), time.perf_counter() - t0)

        return results
