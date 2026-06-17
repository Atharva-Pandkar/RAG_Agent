"""
Build a chunked corpus using Unstructured-IO partitioning + chunk_by_title.

Each chunk carries the full section-aware metadata and cross-references
built in explore_unstructured.py:
  - section, has_table, table_markdown, element_types
  - section_chunk_ids, section_table_ids, section_narrative_ids

Output: Experiments/corpora/unstructured_{max_chars}_{overlap}.json

Usage (from project root):
    python src/ingestion/build_unstructured_corpus.py
    python src/ingestion/build_unstructured_corpus.py --max-characters 2000 --overlap 100
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ingestion.extract_structured import _table_to_markdown, SECTION_PATTERN  # noqa: E402

DATASET_DIR = ROOT / "Dataset"
CORPORA_DIR = ROOT / "Experiments" / "corpora"

COMPANY_FILES = {
    "aapl-20250927": "aapl-20250927.html",
    "cat-20251231":  "cat-20251231.html",
    "jpm-20251231":  "jpm-20251231.html",
    "ko-20251231":   "ko-20251231.html",
    "wmt-20260131":  "wmt-20260131.html",
}

_NOTE_PATTERN = re.compile(r"^Note\s+\d+", re.IGNORECASE)


def _html_table_to_md(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    tbl = soup.find("table")
    return _table_to_markdown(tbl) if tbl else html


def _detect_section(text: str) -> str | None:
    t = text.strip()
    if SECTION_PATTERN.match(t):
        return t[:80]
    if _NOTE_PATTERN.match(t) and len(t) < 80:
        return t
    return None


def _chunk_id(doc_id: str, i: int) -> str:
    return f"{doc_id}_unstr_{i}"


def _partition_file(html_path: Path, doc_id: str,
                    max_characters: int, overlap: int) -> list[dict]:
    from unstructured.partition.html import partition_html
    from unstructured.chunking.title import chunk_by_title

    elements = partition_html(filename=str(html_path))
    chunks = chunk_by_title(
        elements,
        max_characters=max_characters,
        overlap=overlap,
        include_orig_elements=True,
    )

    # ── First pass ─────────────────────────────────────────────────────────
    records: list[dict] = []
    current_section: str | None = None

    for i, chunk in enumerate(chunks):
        orig_els = []
        if chunk.metadata and hasattr(chunk.metadata, "orig_elements") and chunk.metadata.orig_elements:
            orig_els = chunk.metadata.orig_elements

        table_html = getattr(chunk.metadata, "text_as_html", None) if chunk.metadata else None
        orig_type_names = [type(e).__name__ for e in orig_els]
        has_table = "Table" in orig_type_names or bool(table_html)
        table_md = _html_table_to_md(table_html) if table_html else None

        text = str(chunk)

        # Detect section from chunk text (forward-propagating, order-stable)
        for line in text.splitlines():
            candidate = _detect_section(line.strip())
            if candidate:
                current_section = candidate
                break

        # Build the text field: for table chunks use markdown; otherwise plain text
        chunk_text = table_md if (has_table and table_md) else text

        records.append({
            "id": _chunk_id(doc_id, i),
            "doc": doc_id,
            "chunk_index": i,
            "text": chunk_text,
            "section": current_section,
            "element_types": orig_type_names,
            "has_table": has_table,
            "table_ids": [],          # filled below
            "section_chunk_ids": [],  # filled below
            "section_table_ids": [],
            "section_narrative_ids": [],
        })

    # ── Second pass: section cross-references ──────────────────────────────
    section_buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        section_buckets[rec["section"] or "__preamble__"].append(rec)

    for sec, recs in section_buckets.items():
        all_ids = [r["id"] for r in recs]
        table_ids = [r["id"] for r in recs if r["has_table"]]
        narrative_ids = [r["id"] for r in recs if not r["has_table"]]
        for r in recs:
            r["table_ids"] = table_ids if r["has_table"] else []
            r["section_chunk_ids"] = all_ids
            r["section_table_ids"] = table_ids
            r["section_narrative_ids"] = narrative_ids

    return records


def build(max_characters: int, overlap: int) -> Path:
    all_chunks: list[dict] = []

    for doc_id, fname in COMPANY_FILES.items():
        html_path = DATASET_DIR / fname
        print(f"  {doc_id}...", flush=True)
        chunks = _partition_file(html_path, doc_id, max_characters, overlap)
        all_chunks.extend(chunks)
        tables = sum(1 for c in chunks if c["has_table"])
        sections = len({c["section"] for c in chunks if c["section"]})
        print(f"    {len(chunks)} chunks  |  {tables} table chunks  |  {sections} sections")

    # Top-level table_index: table chunk_id → section
    table_index = {
        c["id"]: (c["section"] or "__preamble__")
        for c in all_chunks if c["has_table"]
    }

    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CORPORA_DIR / f"unstructured_{max_characters}_{overlap}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "strategy": "unstructured",
            "max_characters": max_characters,
            "overlap": overlap,
            "chunks": all_chunks,
            "table_index": table_index,
        }, f, indent=1, ensure_ascii=False)

    total_tables = sum(1 for c in all_chunks if c["has_table"])
    print(f"Wrote {len(all_chunks)} chunks ({total_tables} table chunks) -> {out_path}")
    print(f"  table_index: {len(table_index)} entries")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-characters", type=int, default=4000)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()
    build(args.max_characters, args.overlap)


if __name__ == "__main__":
    main()
