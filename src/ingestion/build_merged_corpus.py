"""
Merge the Unstructured-IO corpus (strong narrative coverage) with the
iXBRL corpus (complete financial-table coverage) into a single retrieval
corpus.

Deduplication: an xbrl chunk is skipped if its first 120 characters of
normalised text already appear in any unstructured chunk.  This removes
tables that Unstructured-IO already captured via real <table> elements,
while keeping the ix:continuation-wrapped tables that it missed.

Output: Experiments/corpora/merged_unstr_xbrl.json

Usage
-----
    python src/ingestion/build_merged_corpus.py
    python src/ingestion/build_merged_corpus.py \\
        --unstructured Experiments/corpora/unstructured_4000_200.json \\
        --xbrl        Experiments/corpora/xbrl_merged.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CORPORA_DIR = ROOT / "Experiments" / "corpora"

_WS = re.compile(r"\s+")


def _norm(text: str, length: int = 120) -> str:
    return _WS.sub(" ", text).strip()[:length].lower()


def merge(unstr_path: Path, xbrl_path: Path, out_path: Path) -> Path:
    with open(unstr_path, encoding="utf-8") as f:
        unstr = json.load(f)
    with open(xbrl_path, encoding="utf-8") as f:
        xbrl = json.load(f)

    unstr_chunks: list[dict] = unstr["chunks"]
    xbrl_chunks: list[dict] = xbrl["chunks"]

    # Build fingerprint set from unstructured corpus for dedup
    unstr_fingerprints: set[str] = {_norm(c["text"]) for c in unstr_chunks}

    # Keep xbrl chunks whose content is not already in the unstr corpus
    added, skipped = 0, 0
    new_xbrl: list[dict] = []
    for c in xbrl_chunks:
        fp = _norm(c["text"])
        if fp in unstr_fingerprints:
            skipped += 1
        else:
            # Re-id to avoid collisions with unstructured ids
            c = dict(c)
            new_xbrl.append(c)
            added += 1

    merged_chunks = unstr_chunks + new_xbrl

    print(f"Unstructured chunks : {len(unstr_chunks)}")
    print(f"iXBRL chunks        : {len(xbrl_chunks)}")
    print(f"  Skipped (duplicate): {skipped}")
    print(f"  Added (new coverage): {added}")
    print(f"Merged total        : {len(merged_chunks)}")

    # Rebuild table_index over all chunks
    table_index = {
        c["id"]: (c["section"] or "__preamble__")
        for c in merged_chunks if c.get("has_table")
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "strategy": "merged_unstr_xbrl",
                "sources": {
                    "unstructured": str(unstr_path),
                    "xbrl": str(xbrl_path),
                },
                "chunks": merged_chunks,
                "table_index": table_index,
            },
            f,
            indent=1,
            ensure_ascii=False,
        )

    print(f"Wrote -> {out_path}")
    print(f"  table_index: {len(table_index)} entries")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unstructured",
        default=str(CORPORA_DIR / "unstructured_4000_200.json"),
    )
    parser.add_argument(
        "--xbrl",
        default=str(CORPORA_DIR / "xbrl_merged.json"),
    )
    parser.add_argument(
        "--output",
        default=str(CORPORA_DIR / "merged_unstr_xbrl.json"),
    )
    args = parser.parse_args()
    merge(Path(args.unstructured), Path(args.xbrl), Path(args.output))


if __name__ == "__main__":
    main()
