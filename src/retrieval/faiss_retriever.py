"""Dense retrieval via a LangChain FAISS vector store.

Builds (or loads, if cached) a FAISS index over the corpus chunks using
HuggingFaceEmbeddings, and persists it to
Experiments/embeddings/faiss_{cache_key}/ so repeated runs don't re-embed.

See dense_retriever.py for the Windows pyarrow/torch import-order note -
same fix applies here since langchain_community pulls in the same stack.
"""
from __future__ import annotations
import pyarrow  # noqa: F401  (import-order fix, see dense_retriever.py)
from pathlib import Path
from typing import List, Dict

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[2]
FAISS_CACHE_DIR = ROOT / "Experiments" / "embeddings"


class FaissRetriever:
    def __init__(self, chunks: List[Dict], model_name: str = "BAAI/bge-small-en-v1.5",
                 cache_key: str | None = None):
        self.chunks = chunks
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

        index_dir = None
        if cache_key:
            FAISS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            index_dir = FAISS_CACHE_DIR / f"faiss_{cache_key}"

        if index_dir and index_dir.exists():
            self.store = FAISS.load_local(
                str(index_dir), self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            docs = [
                Document(page_content=c["text"], metadata={"id": c["id"], "doc": c["doc"]})
                for c in chunks
            ]
            self.store = FAISS.from_documents(docs, self.embeddings)
            if index_dir:
                self.store.save_local(str(index_dir))

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        results = self.store.similarity_search_with_relevance_scores(query, k=k)
        return [
            {"id": doc.metadata["id"], "doc": doc.metadata["doc"],
             "text": doc.page_content, "score": float(score)}
            for doc, score in results
        ]
