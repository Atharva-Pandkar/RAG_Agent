"""
Exploration script: parse 10-K HTML filings using Unstructured-IO and write
a compact JSON showing element breakdown, chunk output, and — crucially —
section-level cross-references linking every table chunk to its narrative
siblings and vice versa.

Pipeline:
  1. partition_html  → elements (Text, NarrativeText, Table, ListItem…)
  2. _assign_sections → walk elements in order, detect section headings via
     SECTION_PATTERN, stamp each element with a section label
  3. chunk_by_title  → CompositeElement / Table chunks
  4. _link_chunks    → use each chunk's orig_elements to recover its section,
     then build:
       chunk["section"]           – section label for this chunk
       chunk["section_chunk_ids"] – all chunk IDs in the same section
       chunk["section_table_ids"] – table chunk IDs in the same section
       chunk["section_narrative_ids"] – narrative chunk IDs in the same section
  5. section_index   – top-level map: section → {narrative_ids, table_ids}
  6. table_index     – top-level map: chunk_id → section (for quick reverse lookup)

Usage (from project root):
    python src/ingestion/explore_unstructured.py               # all 5 filings
    python src/ingestion/explore_unstructured.py --file aapl   # single company
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
OUT_DIR = ROOT / "Documents" / "unstructured_explore"

COMPANY_FILES = {
    "aapl": "aapl-20250927.html",
    "cat":  "cat-20251231.html",
    "jpm":  "jpm-20251231.html",
    "ko":   "ko-20251231.html",
    "wmt":  "wmt-20260131.html",
}

_NOTE_PATTERN = re.compile(r"^Note\s+\d+", re.IGNORECASE)


def _html_table_to_md(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    tbl = soup.find("table")
    return _table_to_markdown(tbl) if tbl else html


def _detect_section(text: str) -> str | None:
    """Return a section label if this text looks like a section heading."""
    t = text.strip()
    if SECTION_PATTERN.match(t):
        return t[:80]
    if _NOTE_PATTERN.match(t) and len(t) < 80:
        return t
    return None



def _chunk_id(doc_id: str, i: int) -> str:
    return f"{doc_id}_unstr_{i}"


def process_file(html_path: Path, max_characters: int, overlap: int) -> dict:
    from unstructured.partition.html import partition_html
    from unstructured.chunking.title import chunk_by_title

    doc_id = html_path.stem
    elements = partition_html(filename=str(html_path))
    element_type_counts = dict(Counter(type(e).__name__ for e in elements))

    chunks = chunk_by_title(
        elements,
        max_characters=max_characters,
        overlap=overlap,
        include_orig_elements=True,
    )

    # ── First pass: build chunk records, then assign sections ───────────────
    # Section assignment runs on the CHUNK stream (in document order) rather
    # than on orig_elements, because chunk_by_title copies Table elements so
    # id()-based matching against the original element list is unreliable.
    # We detect section headings from each chunk's own text and forward-
    # propagate, exactly as _assign_sections does on elements.
    chunk_records: list[dict] = []
    current_section: str | None = None

    for i, chunk in enumerate(chunks):
        cid = _chunk_id(doc_id, i)

        orig_els = []
        if chunk.metadata and hasattr(chunk.metadata, "orig_elements") and chunk.metadata.orig_elements:
            orig_els = chunk.metadata.orig_elements

        table_html = getattr(chunk.metadata, "text_as_html", None) if chunk.metadata else None
        orig_type_names = [type(e).__name__ for e in orig_els]
        has_table = "Table" in orig_type_names or bool(table_html)
        table_md = _html_table_to_md(table_html) if table_html else None

        text = str(chunk)

        # Check whether any line of this chunk is a section heading; if so,
        # update current_section BEFORE recording so the heading chunk itself
        # is tagged with the new section it opens.
        for line in text.splitlines():
            candidate = _detect_section(line.strip())
            if candidate:
                current_section = candidate
                break

        chunk_records.append({
            "chunk_id": cid,
            "chunk_index": i,
            "type": type(chunk).__name__,
            "section": current_section,
            "element_types": orig_type_names,
            "has_table": has_table,
            "table_markdown": table_md,
            # cross-ref placeholders — filled in second pass
            "section_chunk_ids": [],
            "section_table_ids": [],
            "section_narrative_ids": [],
            "char_len": len(text),
            "approx_tokens": len(text) // 4,
            "text_preview": text[:300].replace("\n", " ") + ("..." if len(text) > 300 else ""),
            "text": text,
        })

    # ── Second pass: group by section, write cross-references ──────────────
    # section → list of chunk records in that section
    section_buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in chunk_records:
        key = rec["section"] or "__preamble__"
        section_buckets[key].append(rec)

    for sec, recs in section_buckets.items():
        all_ids = [r["chunk_id"] for r in recs]
        table_ids = [r["chunk_id"] for r in recs if r["has_table"]]
        narrative_ids = [r["chunk_id"] for r in recs if not r["has_table"]]
        for r in recs:
            r["section_chunk_ids"] = all_ids
            r["section_table_ids"] = table_ids
            r["section_narrative_ids"] = narrative_ids

    # ── Top-level indexes ───────────────────────────────────────────────────
    section_index: dict[str, dict] = {}
    for sec, recs in section_buckets.items():
        section_index[sec] = {
            "chunk_count": len(recs),
            "narrative_ids": [r["chunk_id"] for r in recs if not r["has_table"]],
            "table_ids": [r["chunk_id"] for r in recs if r["has_table"]],
        }

    # table_id → section  (for reverse lookup without scanning all chunks)
    table_index: dict[str, str] = {
        r["chunk_id"]: (r["section"] or "__preamble__")
        for r in chunk_records if r["has_table"]
    }

    table_count = sum(1 for r in chunk_records if r["has_table"])
    return {
        "source": html_path.name,
        "doc_id": doc_id,
        "partitioner": "unstructured.partition.html",
        "chunker": "chunk_by_title",
        "settings": {"max_characters": max_characters, "overlap": overlap},
        "element_breakdown": element_type_counts,
        "stats": {
            "total_elements": sum(element_type_counts.values()),
            "total_chunks": len(chunk_records),
            "chunks_with_table": table_count,
            "sections_detected": len(section_index),
            "avg_approx_tokens": int(
                sum(r["approx_tokens"] for r in chunk_records) / max(len(chunk_records), 1)
            ),
        },
        "section_index": section_index,
        "table_index": table_index,
        "chunks": chunk_records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", choices=list(COMPANY_FILES), default=None)
    parser.add_argument("--max-characters", type=int, default=4000,
                        help="Max chars per chunk (default: 4000)")
    parser.add_argument("--overlap", type=int, default=200,
                        help="Char overlap between chunks (default: 200)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = {args.file: COMPANY_FILES[args.file]} if args.file else COMPANY_FILES

    for tag, fname in targets.items():
        html_path = DATASET_DIR / fname
        print(f"Processing {tag} ({fname})...", flush=True)
        result = process_file(html_path, args.max_characters, args.overlap)

        s = result["stats"]
        print(
            f"  elements: {s['total_elements']}  chunks: {s['total_chunks']}  "
            f"table_chunks: {s['chunks_with_table']}  "
            f"sections: {s['sections_detected']}  "
            f"avg_tokens: {s['avg_approx_tokens']}"
        )

        out_path = OUT_DIR / f"{tag}_unstructured_chunks.json"
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
