"""
RAG retrieval tool for the 10-K agent.

Uses the merged Unstructured+iXBRL corpus with FAISS-hybrid retrieval
(BM25 + dense via RRF, k=10) — the strongest config from run40.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.tools import tool  # noqa: E402
from src.pipeline import RagPipeline   # noqa: E402

_PIPELINE: RagPipeline | None = None

CORPUS_PATH  = "Experiments/corpora/merged_unstr_xbrl.json"
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
RETRIEVAL_K  = 10


def get_pipeline() -> RagPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = RagPipeline(
            corpus_path=CORPUS_PATH,
            retrieval_strategy="faiss_hybrid",
            embedding_model=EMBED_MODEL,
        )
    return _PIPELINE


@tool
def search_10k_filings(query: str) -> str:
    """Search the SEC 10-K annual report filings of five companies —
    Apple (AAPL), Caterpillar (CAT), JPMorgan Chase (JPM),
    Coca-Cola (KO), and Walmart (WMT) — for passages relevant to the query.

    Use this tool for ANY question about:
    - Financial figures (revenue, profit, assets, EPS, cash flow, etc.)
    - Risk factors and business risks
    - Business segments and operations
    - MD&A narrative and management commentary
    - Notes to financial statements
    - Corporate governance and legal proceedings

    You may call this tool multiple times with different or more specific queries
    if the first result is insufficient to fully answer the question.
    Always call this tool before answering — do not rely on prior knowledge.

    Args:
        query: A specific, focused search query. For multi-part questions,
               break them into separate calls (e.g. one for revenue, one for
               operating income).

    Returns:
        Up to 10 ranked passages with source document and chunk identifiers.
    """
    pipeline = get_pipeline()
    results = pipeline.retrieve(query, k=RETRIEVAL_K)

    if not results:
        return "No relevant passages found in the 10-K filings for this query."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        doc   = r.get("doc", "unknown")
        cid   = r.get("id", "")
        sec   = r.get("section") or "—"
        text  = r.get("text", "").strip()
        parts.append(
            f"[{i}] Source: {doc} | Section: {sec} | ID: {cid}\n{text}"
        )

    return "\n\n---\n\n".join(parts)
