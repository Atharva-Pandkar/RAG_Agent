"""
Populate gold_chunk_ids in golden_set.json by matching each question's
evidence quote against chunks of a given corpus.

This makes the mapping corpus-specific: re-run whenever the chunking
config changes, writing output alongside the run's corpus file rather
than mutating the canonical golden_set.json.

Usage:
    python eval/golden_set/populate_gold_chunks.py --corpus Experiments/corpora/fixed_size_512_50.json
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SET_PATH = ROOT / "eval" / "golden_set" / "golden_set.json"

# map golden-set company tag -> doc id stem used in extracted text / corpus
COMPANY_TO_DOC = {
    "AAPL": "aapl-20250927",
    "CAT": "cat-20251231",
    "JPM": "jpm-20251231",
    "KO": "ko-20251231",
    "WMT": "wmt-20260131",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_quotes(evidence: str) -> list[str]:
    """Pull out double-quoted spans from an evidence string; if none, use whole string."""
    quotes = re.findall(r'"([^"]+)"', evidence)
    return quotes if quotes else [evidence]


def main(corpus_path: Path):
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    chunks = corpus["chunks"]

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)

    matched, unmatched = 0, 0
    for q in golden["questions"]:
        if q["type"] == "unanswerable":
            q["gold_chunk_ids"] = []
            continue

        doc_id = COMPANY_TO_DOC[q["company"]]
        doc_chunks = [c for c in chunks if c["doc"] == doc_id]

        quotes = [_normalize(s) for s in extract_quotes(q["evidence"]) if len(s) > 15]
        anchor_groups = [[_normalize(s) for s in g] for g in q.get("chunk_anchors", [])]
        gold_ids = []
        for c in doc_chunks:
            ctext = _normalize(c["text"])
            matched_chunk = False
            for quote in quotes:
                # use a shortened anchor (first ~60 chars) in case of minor
                # whitespace/formatting differences vs. extracted text
                anchor = quote[:60]
                if anchor in ctext:
                    matched_chunk = True
                    break
            if not matched_chunk:
                for group in anchor_groups:
                    if group and all(s in ctext for s in group):
                        matched_chunk = True
                        break
            if matched_chunk:
                gold_ids.append(c["id"])

        q["gold_chunk_ids"] = gold_ids
        if gold_ids:
            matched += 1
        else:
            unmatched += 1
            print(f"  WARNING: no chunk match for {q['id']} ({q['type']})")

    print(f"Matched: {matched}, Unmatched: {unmatched}")

    out_path = corpus_path.parent / f"golden_set_with_chunks_{corpus_path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)
    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    args = parser.parse_args()
    main(ROOT / args.corpus if not Path(args.corpus).is_absolute() else Path(args.corpus))
