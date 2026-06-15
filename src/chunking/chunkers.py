"""
Chunking strategies for 10-K text.

Phase 1 implements the two simplest baselines:
  - fixed_size: split on a sliding token window
  - recursive: split on paragraph/sentence boundaries, packing into a
    token budget (falls back to fixed-size split for oversized paragraphs)

Both return a list of dicts: {"id": str, "doc": str, "text": str, "chunk_index": int}
so they can be consumed uniformly by the retriever/pipeline.
"""
from __future__ import annotations
import re
from typing import List, Dict
import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def _tok_len(text: str) -> int:
    return len(_ENC.encode(text))


def fixed_size_chunks(text: str, doc_id: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
    tokens = _ENC.encode(text)
    chunks = []
    step = chunk_size - overlap
    idx = 0
    for i in range(0, len(tokens), step):
        window = tokens[i:i + chunk_size]
        if not window:
            break
        chunk_text = _ENC.decode(window)
        chunks.append({
            "id": f"{doc_id}_fixed_{idx}",
            "doc": doc_id,
            "text": chunk_text,
            "chunk_index": idx,
        })
        idx += 1
        if i + chunk_size >= len(tokens):
            break
    return chunks


def recursive_chunks(text: str, doc_id: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
    """Split on blank-line paragraphs, pack consecutive paragraphs into
    chunks up to chunk_size tokens. Paragraphs larger than chunk_size are
    further split via fixed_size_chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    idx = 0
    current: List[str] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens, idx
        if not current:
            return
        chunk_text = "\n\n".join(current)
        chunks.append({
            "id": f"{doc_id}_rec_{idx}",
            "doc": doc_id,
            "text": chunk_text,
            "chunk_index": idx,
        })
        idx += 1
        current = []
        current_tokens = 0

    for para in paragraphs:
        ptoks = _tok_len(para)
        if ptoks > chunk_size:
            flush()
            for sub in fixed_size_chunks(para, doc_id, chunk_size, overlap):
                sub["id"] = f"{doc_id}_rec_{idx}"
                sub["chunk_index"] = idx
                chunks.append(sub)
                idx += 1
            continue

        if current_tokens + ptoks > chunk_size:
            flush()

        current.append(para)
        current_tokens += ptoks

        if overlap == 0:
            continue

    flush()
    return chunks


STRATEGIES = {
    "fixed_size": fixed_size_chunks,
    "recursive": recursive_chunks,
}
