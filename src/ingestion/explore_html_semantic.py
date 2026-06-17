"""
Exploration script: parse 10-K HTML filings using LangChain's
HTMLSemanticPreservingSplitter and write a compact JSON for examination.

Custom table handler: reuses _table_to_markdown from extract_structured.py
so <table> elements become structured markdown rather than unstructured text.

Usage (from project root):
    python src/ingestion/explore_html_semantic.py                  # all 5 filings
    python src/ingestion/explore_html_semantic.py --file aapl      # single company
    python src/ingestion/explore_html_semantic.py --max-chunks 30  # limit output size
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATASET_DIR = ROOT / "Dataset"
OUT_DIR = ROOT / "Documents" / "semantic_explore"

# Reuse table→markdown converter from extract_structured.py
from src.ingestion.extract_structured import _table_to_markdown  # noqa: E402

# Map short company tag -> HTML file stem pattern
COMPANY_FILES = {
    "aapl": "aapl-20250927.html",
    "cat":  "cat-20251231.html",
    "jpm":  "jpm-20251231.html",
    "ko":   "ko-20251231.html",
    "wmt":  "wmt-20260131.html",
}

HEADERS_TO_SPLIT = [
    ("h1", "h1"), ("h2", "h2"), ("h3", "h3"),
    ("h4", "h4"), ("h5", "h5"), ("h6", "h6"),
]


def _tok_approx(text: str) -> int:
    """Rough token estimate (chars / 4) — avoids tiktoken dependency here."""
    return len(text) // 4


def process_file(html_path: Path, max_chunk_size: int, chunk_overlap: int,
                 max_chunks: int | None) -> dict:
    from langchain_text_splitters import HTMLSemanticPreservingSplitter

    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        # Reuse our existing table→markdown converter as a custom handler
        # so financial tables come out as structured markdown pipes rather
        # than unstructured text.
        custom_handlers={"table": _table_to_markdown},
        preserve_links=True,
        denylist_tags=["script", "style", "ix:nonfraction", "ix:nonnumeric",
                        "ix:header", "meta", "link"],
    )

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    docs = splitter.split_text(html)

    _TABLE_ROW_RE = re.compile(r"^\|.+\|", re.MULTILINE)

    chunks = []
    for i, doc in enumerate(docs):
        if max_chunks and i >= max_chunks:
            break
        text = doc.page_content
        # Detect markdown table rows produced by our custom handler
        has_table = bool(_TABLE_ROW_RE.search(text))
        chunks.append({
            "chunk_index": i,
            "metadata": doc.metadata,
            "has_table": has_table,
            "char_len": len(text),
            "approx_tokens": _tok_approx(text),
            "text_preview": text[:300].replace("\n", " ") + ("..." if len(text) > 300 else ""),
            "text": text,
        })

    total = len(docs)
    has_tables = sum(1 for c in chunks if c["has_table"])
    return {
        "source": html_path.name,
        "splitter": {
            "max_chunk_size": max_chunk_size,
            "chunk_overlap": chunk_overlap,
            "custom_handlers": ["table → markdown via _table_to_markdown"],
        },
        "stats": {
            "total_chunks": total,
            "shown_chunks": len(chunks),
            "chunks_with_table": has_tables,
            "avg_chars": int(sum(c["char_len"] for c in chunks) / max(len(chunks), 1)),
            "avg_approx_tokens": int(sum(c["approx_tokens"] for c in chunks) / max(len(chunks), 1)),
        },
        "chunks": chunks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", choices=list(COMPANY_FILES), default=None,
                        help="Single company to process (default: all)")
    parser.add_argument("--max-chunk-size", type=int, default=2000,
                        help="Max chars per chunk (default: 2000)")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                        help="Char overlap between chunks (default: 200)")
    parser.add_argument("--max-chunks", type=int, default=50,
                        help="Max chunks to include per file in output (default: 50)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = {args.file: COMPANY_FILES[args.file]} if args.file else COMPANY_FILES

    for tag, fname in targets.items():
        html_path = DATASET_DIR / fname
        print(f"Processing {tag} ({fname})...")
        result = process_file(html_path, args.max_chunk_size, args.chunk_overlap, args.max_chunks)

        stats = result["stats"]
        print(f"  total chunks: {stats['total_chunks']}  "
              f"table chunks: {stats['chunks_with_table']}  "
              f"avg tokens: {stats['avg_approx_tokens']}")

        out_path = OUT_DIR / f"{tag}_semantic_chunks.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
