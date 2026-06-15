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


def section_based_chunks(structured_doc: dict, doc_id: str, chunk_size: int = 512,
                          overlap: int = 50) -> List[Dict]:
    """Chunk a structure-preserving extraction (see extract_structured.py).

    - Walks blocks in document order, packing consecutive "paragraph"
      blocks within the same section into chunks up to chunk_size tokens
      (paragraphs never merge across a section boundary).
    - "table" blocks (Markdown tables) are kept as their own standalone
      chunk(s) - never merged with narrative text - and split via
      fixed_size if a table alone exceeds chunk_size.
    - "heading" blocks are dropped as standalone chunks but their text is
      prepended as context to the first chunk of that section (cheap
      version of "contextual retrieval": every chunk knows what section
      it's from).

    Every chunk carries "section" metadata.
    """
    blocks = structured_doc["blocks"]

    chunks: List[Dict] = []
    idx = 0
    current: List[str] = []
    current_tokens = 0
    current_section = None
    section_heading_pending = None

    def flush():
        nonlocal current, current_tokens, idx
        if not current:
            return
        chunk_text = "\n\n".join(current)
        chunks.append({
            "id": f"{doc_id}_sec_{idx}",
            "doc": doc_id,
            "text": chunk_text,
            "chunk_index": idx,
            "section": current_section,
        })
        idx += 1
        current = []
        current_tokens = 0

    for b in blocks:
        if b["type"] == "heading":
            flush()
            current_section = b["section"]
            section_heading_pending = b["text"]
            continue

        if b["section"] != current_section:
            flush()
            current_section = b["section"]

        if b["type"] == "table":
            flush()
            table_text = b["text"]
            if section_heading_pending:
                table_text = f"[{section_heading_pending}]\n{table_text}"
            if _tok_len(table_text) > chunk_size:
                for sub in fixed_size_chunks(table_text, doc_id, chunk_size, overlap):
                    sub["id"] = f"{doc_id}_sec_{idx}"
                    sub["chunk_index"] = idx
                    sub["section"] = current_section
                    chunks.append(sub)
                    idx += 1
            else:
                chunks.append({
                    "id": f"{doc_id}_sec_{idx}",
                    "doc": doc_id,
                    "text": table_text,
                    "chunk_index": idx,
                    "section": current_section,
                })
                idx += 1
            section_heading_pending = None
            continue

        # paragraph block
        text = b["text"]
        if section_heading_pending:
            text = f"[{section_heading_pending}]\n{text}"
            section_heading_pending = None

        ptoks = _tok_len(text)
        if ptoks > chunk_size:
            flush()
            for sub in fixed_size_chunks(text, doc_id, chunk_size, overlap):
                sub["id"] = f"{doc_id}_sec_{idx}"
                sub["chunk_index"] = idx
                sub["section"] = current_section
                chunks.append(sub)
                idx += 1
            continue

        if current_tokens + ptoks > chunk_size:
            flush()

        current.append(text)
        current_tokens += ptoks

    flush()
    return chunks


STRATEGIES = {
    "fixed_size": fixed_size_chunks,
    "recursive": recursive_chunks,
    "section_based": section_based_chunks,
}
