"""
Build a chunked corpus from the extracted 10-K text files.

Usage:
    python src/ingestion/build_corpus.py --strategy fixed_size --chunk-size 512 --overlap 50

Output: Experiments/corpora/<strategy>_<chunk_size>_<overlap>.json
    {"chunks": [{"id", "doc", "text", "chunk_index"}, ...]}
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chunking.chunkers import STRATEGIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_DIR = ROOT / "Documents" / "extracted"
CORPORA_DIR = ROOT / "Experiments" / "corpora"


def build(strategy: str, chunk_size: int, overlap: int) -> Path:
    fn = STRATEGIES[strategy]
    all_chunks = []
    for txt_file in sorted(EXTRACTED_DIR.glob("*.txt")):
        doc_id = txt_file.stem
        text = txt_file.read_text(encoding="utf-8")
        chunks = fn(text, doc_id, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        print(f"  {doc_id}: {len(chunks)} chunks")

    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CORPORA_DIR / f"{strategy}_{chunk_size}_{overlap}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"strategy": strategy, "chunk_size": chunk_size,
                    "overlap": overlap, "chunks": all_chunks}, f, indent=1)
    print(f"Wrote {len(all_chunks)} chunks -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), default="fixed_size")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    build(args.strategy, args.chunk_size, args.overlap)
