"""
RAG tools for the document research agent.

Tools
-----
search_documents          — semantic + keyword search over the active corpus
list_available_documents  — return registry of what documents are loaded

Corpus
------
Uses active_corpus.json (seeded from merged_unstr_xbrl.json on first run,
then extended as users ingest new documents).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.tools import tool   # noqa: E402
from src.pipeline import RagPipeline    # noqa: E402
from .logger import get_logger          # noqa: E402

log = get_logger(__name__)

_PIPELINE: RagPipeline | None = None

CORPUS_PATH = "Experiments/corpora/active_corpus.json"
EMBED_MODEL = "text-embedding-3-small"
RETRIEVAL_K = 10

REGISTRY_PATH = ROOT / "Experiments" / "corpora" / "docs_registry.json"


def get_pipeline() -> RagPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        log.info("Initialising RAG pipeline — corpus: %s | strategy: faiss_hybrid | k: %d",
                 CORPUS_PATH, RETRIEVAL_K)
        t0 = time.perf_counter()
        _PIPELINE = RagPipeline(
            corpus_path=CORPUS_PATH,
            retrieval_strategy="faiss_hybrid",
            embedding_model=EMBED_MODEL,
        )
        log.info("RAG pipeline ready (%.1fs)", time.perf_counter() - t0)
    return _PIPELINE


# ── Tool 1: document search ──────────────────────────────────────────────────

@tool
def search_documents(query: str) -> str:
    """Search all documents loaded into the RAG system for passages relevant to the query.

    Use this tool for ANY factual question — financial figures, risk factors,
    business descriptions, MD&A commentary, footnotes, or any other content
    found in the ingested documents.

    You may call this tool multiple times with different queries if the first
    result does not fully answer the question.

    Args:
        query: The search query.
               - Direct question → pass verbatim.
               - Multi-part question → one call per sub-question.
               - Vague input → rephrase into a specific question first.

    Returns:
        Up to 10 ranked passages, each labelled with source document,
        section, and chunk ID.
    """
    log.info("TOOL CALL search_documents — query: %r", query)
    t0 = time.perf_counter()

    try:
        pipeline = get_pipeline()
        results  = pipeline.retrieve(query, k=RETRIEVAL_K)
    except Exception:
        log.exception("RAG pipeline retrieval failed for query: %r", query)
        return "Retrieval failed — please try again."

    elapsed = time.perf_counter() - t0
    log.info("TOOL RESULT — %d chunks retrieved in %.2fs", len(results), elapsed)

    if not results:
        log.warning("No chunks matched query: %r", query)
        return "No relevant passages found for this query."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        doc  = r.get("doc", "unknown")
        cid  = r.get("id", "")
        sec  = r.get("section") or "—"
        text = r.get("text", "").strip()
        log.debug("  [%d] %s | %s | %d chars", i, doc, sec, len(text))
        parts.append(f"[{i}] Source: {doc} | Section: {sec} | ID: {cid}\n{text}")

    return "\n\n---\n\n".join(parts)


# ── Tool 2: list documents ───────────────────────────────────────────────────

@tool
def list_available_documents() -> str:
    """Return a structured summary of all documents currently loaded in the RAG system.

    Call this tool when:
    - The user asks what documents or topics are available
    - The query is vague and you need to know the document scope before searching
    - You want to confirm which companies / filing years are present

    Returns:
        A formatted list of documents with their names, chunk counts, and sections.
    """
    log.info("TOOL CALL list_available_documents")
    if not REGISTRY_PATH.exists():
        return (
            "No document registry found yet. "
            "Documents may still be loading — try search_documents directly."
        )
    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Failed to read registry: %s", exc)
        return "Could not read the document registry."

    docs = registry.get("documents", [])
    if not docs:
        return "No documents are currently loaded in the system."

    lines = [f"{len(docs)} document(s) are loaded:\n"]
    for d in docs:
        name     = d.get("display_name") or d.get("id", "?")
        sections = d.get("sections", [])
        sec_str  = ", ".join(sections[:6])
        if len(sections) > 6:
            sec_str += f" … (+{len(sections) - 6} more)"
        lines.append(f"• {name}  [{d.get('chunk_count', '?')} chunks]")
        if sec_str:
            lines.append(f"  Sections: {sec_str}")

    updated = registry.get("updated_at", "")
    if updated:
        lines.append(f"\nRegistry last updated: {updated[:19].replace('T', ' ')} UTC")

    return "\n".join(lines)
