"""Dense retrieval via sentence-transformers embeddings + cosine similarity.

NOTE: on Windows, importing pyarrow AFTER torch can cause an access
violation (DLL conflict between torch's and pyarrow's bundled runtimes).
`sentence_transformers` pulls in `datasets` -> `pyarrow`, so we import
pyarrow first here to force the safe load order regardless of import order
elsewhere in the process.
"""
from __future__ import annotations
import pyarrow  # noqa: F401  (import-order fix, see module docstring)
import numpy as np
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
EMBED_CACHE_DIR = ROOT / "Experiments" / "embeddings"


class DenseRetriever:
    def __init__(self, chunks: List[Dict], model_name: str = "BAAI/bge-small-en-v1.5",
                 cache_key: str | None = None):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        cache_path = None
        if cache_key:
            EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = EMBED_CACHE_DIR / f"{cache_key}.npy"

        if cache_path and cache_path.exists():
            self.embeddings = np.load(cache_path)
        else:
            texts = [c["text"] for c in chunks]
            self.embeddings = self.model.encode(
                texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
            ).astype(np.float32)
            if cache_path:
                np.save(cache_path, self.embeddings)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        q_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        scores = self.embeddings @ q_emb
        ranked = np.argsort(-scores)[:k]
        return [
            {"id": self.chunks[i]["id"], "doc": self.chunks[i]["doc"],
             "text": self.chunks[i]["text"], "score": float(scores[i])}
            for i in ranked
        ]
