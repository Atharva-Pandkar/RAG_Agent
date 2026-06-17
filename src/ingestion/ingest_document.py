"""
Universal single-document ingestion pipeline.

Accepts PDF, HTML, DOCX, or plain text.  Parses with Unstructured-IO,
chunks with a fixed-size overlapping window (matching the existing corpus),
appends to active_corpus.json, updates the FAISS index, and refreshes the
document registry — without restarting the server.

Usage (standalone):
    python src/ingestion/ingest_document.py path/to/file.pdf

Usage (from FastAPI):
    from src.ingestion.ingest_document import ingest_file
    result = ingest_file(Path(path), pipeline=get_pipeline())
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

ROOT          = Path(__file__).resolve().parents[2]
CORPORA_DIR   = ROOT / "Experiments" / "corpora"
ACTIVE_CORPUS = CORPORA_DIR / "active_corpus.json"
REGISTRY_PATH = CORPORA_DIR / "docs_registry.json"

CHUNK_SIZE    = 4000   # characters — matches existing unstructured corpus
CHUNK_OVERLAP = 200

log = logging.getLogger("rag.ingestion")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _chunk_text(text: str, doc_id: str, section: str = "") -> List[Dict]:
    """Split text into overlapping fixed-size chunks."""
    chunks: List[Dict] = []
    start = idx = 0
    while start < len(text):
        end  = min(start + CHUNK_SIZE, len(text))
        body = text[start:end].strip()
        if len(body) > 80:          # skip near-empty tail fragments
            chunks.append({
                "id":          f"{doc_id}_upload_{idx}",
                "doc":         doc_id,
                "chunk_index": idx,
                "text":        body,
                "section":     section,
                "type":        "narrative",
            })
            idx += 1
        start = end - CHUNK_OVERLAP if end < len(text) else end
    return chunks


def _parse_file(path: Path, doc_id: str) -> List[Dict]:
    """Parse a document into chunks using Unstructured-IO with text fallback."""
    suffix = path.suffix.lower()
    log.info("[ingest] Parsing %s (%s)", path.name, suffix)

    # Plain text / markdown — no library needed
    if suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        return _chunk_text(text, doc_id)

    # All other formats → Unstructured-IO
    try:
        from unstructured.partition.auto import partition
        elements = partition(str(path))
        log.info("[ingest] Unstructured extracted %d elements", len(elements))

        # Group elements by their preceding Title element
        sections: List[tuple[str, str]] = []
        current_section = ""
        buf: List[str] = []

        for el in elements:
            el_type = type(el).__name__
            el_text = str(el).strip()
            if not el_text:
                continue
            if el_type == "Title":
                if buf:
                    sections.append((current_section, "\n".join(buf)))
                    buf = []
                current_section = el_text
            else:
                buf.append(el_text)
        if buf:
            sections.append((current_section, "\n".join(buf)))

        if not sections:
            log.warning("[ingest] No sections extracted — treating as single block")
            full_text = "\n".join(str(e) for e in elements)
            return _chunk_text(full_text, doc_id)

        chunks: List[Dict] = []
        for section, text in sections:
            chunks.extend(_chunk_text(text, doc_id, section))
        return chunks

    except Exception as exc:
        log.warning("[ingest] Unstructured-IO failed (%s) — raw text fallback", exc)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = path.read_bytes().decode("utf-8", errors="replace")
        return _chunk_text(text, doc_id)


# ── Registry ─────────────────────────────────────────────────────────────────

def _display_name(doc_id: str, filename: str) -> str:
    """Human-readable name for a document."""
    return filename or doc_id.replace("_", " ").title()


def load_registry() -> Dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"updated_at": None, "documents": []}


def save_registry(registry: Dict) -> None:
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False),
                             encoding="utf-8")


def build_registry_from_corpus(corpus_path: Path = ACTIVE_CORPUS) -> None:
    """Bootstrap the registry from an existing corpus JSON (called once at startup)."""
    log.info("[registry] Bootstrapping from corpus: %s", corpus_path.name)
    data   = json.loads(corpus_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])

    docs: Dict[str, Dict] = {}
    for c in chunks:
        doc_id = c.get("doc", "unknown")
        if doc_id not in docs:
            docs[doc_id] = {
                "id":           doc_id,
                "display_name": _pretty_doc_id(doc_id),
                "filename":     "",
                "ingested_at":  None,   # pre-existing docs have no ingest timestamp
                "chunk_count":  0,
                "sections":     set(),
            }
        docs[doc_id]["chunk_count"] += 1
        sec = c.get("section") or ""
        if sec and sec not in ("None", "—", ""):
            docs[doc_id]["sections"].add(sec)

    registry = {
        "documents": [
            {**{k: v for k, v in d.items() if k != "sections"},
             "sections": sorted(d["sections"])}
            for d in docs.values()
        ]
    }
    save_registry(registry)
    log.info("[registry] Built registry — %d documents", len(registry["documents"]))


def _pretty_doc_id(doc_id: str) -> str:
    """Convert 'aapl-20250927' → 'Apple 2025 Annual Report'."""
    _NAMES = {
        "aapl": "Apple", "cat": "Caterpillar", "jpm": "JPMorgan Chase",
        "ko": "Coca-Cola", "wmt": "Walmart",
    }
    parts = doc_id.split("-")
    company = _NAMES.get(parts[0].lower(), parts[0].upper())
    year    = parts[1][:4] if len(parts) > 1 and len(parts[1]) >= 4 else ""
    return f"{company} {year} Annual Report" if year else company


def _add_to_registry(doc_id: str, display_name: str, filename: str,
                     chunks: List[Dict]) -> None:
    registry = load_registry()
    # Remove any existing entry with this doc_id (re-ingest case)
    registry["documents"] = [d for d in registry["documents"] if d["id"] != doc_id]

    sections = sorted({
        c.get("section", "") or ""
        for c in chunks
        if c.get("section") and c["section"] not in ("", "None", "—")
    })
    registry["documents"].append({
        "id":           doc_id,
        "display_name": display_name,
        "filename":     filename,
        "ingested_at":  datetime.now(timezone.utc).isoformat(),
        "chunk_count":  len(chunks),
        "sections":     sections,
    })
    save_registry(registry)


# ── Active corpus ────────────────────────────────────────────────────────────

def ensure_active_corpus() -> None:
    """On first run, seed active_corpus.json from the merged baseline."""
    if ACTIVE_CORPUS.exists():
        return
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    baseline = CORPORA_DIR / "merged_unstr_xbrl.json"
    if baseline.exists():
        log.info("[corpus] Seeding active_corpus.json from merged baseline...")
        import shutil
        shutil.copy2(baseline, ACTIVE_CORPUS)
        log.info("[corpus] active_corpus.json ready (%d bytes)", ACTIVE_CORPUS.stat().st_size)
    else:
        log.warning("[corpus] No baseline found — starting with empty active_corpus.json")
        empty = {"strategy": "dynamic", "chunks": [], "sources": {}, "table_index": {}}
        ACTIVE_CORPUS.write_text(json.dumps(empty, indent=1), encoding="utf-8")


def _append_to_corpus(new_chunks: List[Dict]) -> None:
    data   = json.loads(ACTIVE_CORPUS.read_text(encoding="utf-8"))
    data.setdefault("chunks", [])
    # Remove any existing chunks for this doc (re-ingest)
    if new_chunks:
        doc_id = new_chunks[0]["doc"]
        data["chunks"] = [c for c in data["chunks"] if c.get("doc") != doc_id]
    data["chunks"].extend(new_chunks)
    ACTIVE_CORPUS.write_text(json.dumps(data, indent=1, ensure_ascii=False),
                             encoding="utf-8")
    log.info("[corpus] active_corpus.json updated — total %d chunks", len(data["chunks"]))


# ── Main ingestion entry point ───────────────────────────────────────────────

def ingest_file(
    path: Path,
    *,
    display_name: Optional[str] = None,
    pipeline=None,          # RagPipeline instance for live index update
) -> Dict:
    """
    Ingest a single document file into the RAG system.

    Returns:
        {"doc_id": str, "chunk_count": int, "sections": list[str]}
    """
    t0 = time.perf_counter()
    doc_id       = _slugify(path.stem)
    display_name = display_name or _display_name(doc_id, path.name)

    log.info("[ingest] Starting ingestion: %s → doc_id=%r", path.name, doc_id)

    # 1. Parse and chunk
    chunks = _parse_file(path, doc_id)
    if not chunks:
        raise ValueError(f"No text could be extracted from {path.name}")
    log.info("[ingest] Produced %d chunks", len(chunks))

    # 2. Persist to active_corpus.json
    _append_to_corpus(chunks)

    # 3. Update registry
    _add_to_registry(doc_id, display_name, path.name, chunks)

    # 4. Live-update the in-memory pipeline (no server restart)
    if pipeline is not None:
        log.info("[ingest] Updating in-memory pipeline...")
        pipeline.add_chunks(chunks)
        log.info("[ingest] Pipeline updated in-memory")

    sections = sorted({
        c.get("section", "") or ""
        for c in chunks
        if c.get("section") and c["section"] not in ("", "None", "—")
    })

    elapsed = time.perf_counter() - t0
    log.info("[ingest] Done — %d chunks, %d sections, %.1fs", len(chunks), len(sections), elapsed)
    return {"doc_id": doc_id, "chunk_count": len(chunks), "sections": sections,
            "display_name": display_name}


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to document to ingest")
    parser.add_argument("--name", help="Display name for the document")
    args = parser.parse_args()

    ensure_active_corpus()
    result = ingest_file(Path(args.file), display_name=args.name)
    print(json.dumps(result, indent=2))
