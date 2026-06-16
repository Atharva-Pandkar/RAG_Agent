"""LangChain tool wrapping the experiment RagPipeline retriever.

Defaults to the strongest retrieval config found so far (hybrid retrieval
over fixed_size_1024_100 chunks - see Experiments/summarize_runs.py).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.tools import tool  # noqa: E402
from src.pipeline import RagPipeline  # noqa: E402

_PIPELINE: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = RagPipeline(
            corpus_path="Experiments/corpora/fixed_size_1024_100.json",
            retrieval_strategy="hybrid",
            embedding_model="BAAI/bge-small-en-v1.5",
        )
    return _PIPELINE


@tool
def search_10k_filings(query: str) -> str:
    """Search the 10-K filings of AAPL, CAT, JPM, KO, and WMT for passages
    relevant to the query. Use this for any question about financial figures,
    risk factors, business segments, or other content found in a company's
    annual report (10-K)."""
    pipeline = get_pipeline()
    results = pipeline.retrieve(query, k=5)
    if not results:
        return "No relevant passages found."

    parts = []
    for r in results:
        parts.append(f"[Source: {r['doc']} | chunk {r['id']}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)
