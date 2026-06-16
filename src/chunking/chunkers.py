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

    Every chunk carries "section" metadata, plus table-awareness metadata:
      - "table_ids": list of block_ids of tables contained in this chunk
        (empty for pure-narrative chunks).
      - "has_table": bool, True if table_ids is non-empty.
      - "section_has_tables": bool, True if this chunk's section contains
        at least one table anywhere (even if this particular chunk doesn't).
    """
    blocks = structured_doc["blocks"]

    sections_with_tables = {b["section"] for b in blocks if b["type"] == "table"}

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
            "table_ids": [],
            "has_table": False,
            "section_has_tables": current_section in sections_with_tables,
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
                    sub["table_ids"] = [b["block_id"]]
                    sub["has_table"] = True
                    sub["section_has_tables"] = True
                    chunks.append(sub)
                    idx += 1
            else:
                chunks.append({
                    "id": f"{doc_id}_sec_{idx}",
                    "doc": doc_id,
                    "text": table_text,
                    "chunk_index": idx,
                    "section": current_section,
                    "table_ids": [b["block_id"]],
                    "has_table": True,
                    "section_has_tables": True,
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
                sub["table_ids"] = []
                sub["has_table"] = False
                sub["section_has_tables"] = current_section in sections_with_tables
                chunks.append(sub)
                idx += 1
            continue

        if current_tokens + ptoks > chunk_size:
            flush()

        current.append(text)
        current_tokens += ptoks

    flush()
    return chunks


def langchain_recursive_chunks(text: str, doc_id: str, chunk_size: int = 512,
                                overlap: int = 50) -> List[Dict]:
    """Split text using LangChain's RecursiveCharacterTextSplitter, with a
    tiktoken-based length function so chunk_size/overlap are in the same
    token units as the other strategies (for apples-to-apples comparison).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=_tok_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)

    chunks = []
    for idx, piece in enumerate(pieces):
        chunks.append({
            "id": f"{doc_id}_lc_{idx}",
            "doc": doc_id,
            "text": piece,
            "chunk_index": idx,
        })
    return chunks


def langchain_section_chunks(structured_doc: dict, doc_id: str, chunk_size: int = 512,
                              overlap: int = 50) -> List[Dict]:
    """Segment a structured extraction (see extract_structured.py) into
    sections using LangChain's MarkdownHeaderTextSplitter, then apply our
    own fixed_size_chunks (token sliding window with overlap) within each
    section.

    This is a hybrid of section_based_chunks and fixed_size_chunks: LangChain
    handles the "break into segments by heading" step (on a Markdown
    rendering of the doc), and our token-window logic handles the
    "pack/split a segment into fixed-size chunks" step. Unlike
    section_based_chunks, tables are not treated specially - they're part of
    the section's text and get swept up into the token windows.

    Table-awareness metadata (matches section_based_chunks):
      - "table_ids": list of block_ids of tables whose markdown text appears
        in this chunk (a table that straddles a chunk boundary - or whose
        overlap region repeats - can appear in more than one chunk).
      - "has_table": bool, True if table_ids is non-empty.
      - "section_has_tables": bool, True if this chunk's section contains
        at least one table anywhere.
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    table_marker_re = re.compile(r"<!--TABLE_ID:(\S+?)-->\n?")

    # Render the structured doc as Markdown with "#"-headings per section,
    # tagging each table with an invisible marker so we can recover which
    # table(s) ended up in which chunk after splitting.
    lines = []
    sections_with_tables = set()
    for b in structured_doc["blocks"]:
        if b["type"] == "heading":
            lines.append(f"# {b['text']}")
        elif b["type"] == "table":
            lines.append(f"<!--TABLE_ID:{b['block_id']}-->\n{b['text']}")
            sections_with_tables.add(b["section"])
        else:
            lines.append(b["text"])
    markdown_text = "\n\n".join(lines)

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "section")],
                                           strip_headers=False)
    segments = splitter.split_text(markdown_text)

    chunks: List[Dict] = []
    idx = 0
    for seg in segments:
        section = seg.metadata.get("section")
        text = seg.page_content
        for sub in fixed_size_chunks(text, doc_id, chunk_size=chunk_size, overlap=overlap):
            table_ids = sorted(set(table_marker_re.findall(sub["text"])))
            sub["text"] = table_marker_re.sub("", sub["text"])
            sub["table_ids"] = table_ids
            sub["has_table"] = bool(table_ids)
            sub["section_has_tables"] = section in sections_with_tables
            sub["id"] = f"{doc_id}_lcsec_{idx}"
            sub["chunk_index"] = idx
            sub["section"] = section
            chunks.append(sub)
            idx += 1

    return chunks


STRATEGIES = {
    "fixed_size": fixed_size_chunks,
    "recursive": recursive_chunks,
    "section_based": section_based_chunks,
    "langchain_recursive": langchain_recursive_chunks,
    "langchain_section": langchain_section_chunks,
}
