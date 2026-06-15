"""
Structure-preserving extraction for SEC 10-K HTML filings.

Goals (vs. the Phase 0 `extract_text.py` flatten-to-text approach):
  1. Preserve document hierarchy - tag every block of content with the
     PART / Item section it falls under (Item 1A Risk Factors, Item 7
     MD&A, Item 8 Financial Statements, etc.)
  2. Preserve tables - convert <table> elements to Markdown tables
     (row/column structure intact) instead of flattening cell text.
  3. Preserve the internal cross-reference graph - SEC 10-Ks are single
     HTML files where phrases like "See Note 5" or "as discussed in
     Item 7A" are <a href="...#some_id"> links to anchors elsewhere in
     the SAME document. We resolve these to block ids so a later
     multi-hop retriever can follow "this chunk references that chunk".

Output: one JSON file per filing with:
  {
    "doc_id": "aapl-20250927",
    "blocks": [
      {"block_id": "b123", "type": "heading"|"paragraph"|"table",
       "section": "Item 1A. Risk Factors", "text": "...",
       "anchors": ["i7193..._10", ...],
       "refs": [{"text": "Note 5", "target_block_id": "b456"}]}
    ],
    "section_index": {"Item 1A. Risk Factors": ["b40", "b41", ...]},
    "reference_graph": [{"from": "b123", "to": "b456", "text": "Note 5"}]
  }

This file is the new input for chunking (Phase 1.5): section-based and
parent-child chunkers can group blocks by `section`, and table blocks
can be kept whole or chunked separately from narrative text.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup, Tag

DATASET_DIR = Path(__file__).resolve().parents[2] / "Dataset"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "Documents" / "structured"

BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]

# Matches "PART I", "PART II", "Item 1.", "Item 1A.", "Item 7A." etc.
SECTION_PATTERN = re.compile(
    r"^\s*(PART\s+[IVX]+|Item\s+\d+[A-Za-z]?\.?)\s*[-—:.]?\s*(.*)$", re.IGNORECASE
)
# A heading candidate is short and matches the pattern; longer text after
# the marker (e.g. "Item 1A. Risk Factors") is kept as the section title.
MAX_HEADING_LEN = 120

# Footnote headings within Item 8, e.g. "Note 12 Commitments, Contingencies..."
NOTE_HEADING_PATTERN = re.compile(r"^\s*Note\s+(\d+[A-Za-z]?)\b", re.IGNORECASE)

# In-text cross-reference mentions, e.g. "see Note 5", "Item 7A", "Part II"
MENTION_PATTERN = re.compile(
    r"\b(Note|Item|Part)\s+(\d+[A-Za-z]?|[IVX]+)\b", re.IGNORECASE
)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_leaf_block(el: Tag) -> bool:
    """A block-level element with its own text and no block-level
    descendant that also carries text (avoids double-counting nested
    div > div > p structures), and not inside / containing a table."""
    if el.name not in BLOCK_TAGS:
        return False
    if el.find("table") is not None:
        return False
    for child in el.find_all(BLOCK_TAGS):
        if child.get_text(strip=True):
            return False
    return bool(el.get_text(strip=True))


def _table_to_markdown(table: Tag) -> str:
    """Convert an HTML table to a Markdown table, expanding colspan by
    repeating the cell value and rowspan by repeating into following rows."""
    rows = table.find_all("tr")
    grid: list[list[str]] = []
    rowspan_carry: dict[int, tuple[int, str]] = {}  # col_idx -> (remaining_rows, value)

    for tr in rows:
        cells = tr.find_all(["td", "th"])
        row: list[str] = []
        col_idx = 0

        def fill_carries():
            while col_idx_ref[0] in rowspan_carry:
                remaining, val = rowspan_carry[col_idx_ref[0]]
                row.append(val)
                if remaining > 1:
                    rowspan_carry[col_idx_ref[0]] = (remaining - 1, val)
                else:
                    del rowspan_carry[col_idx_ref[0]]
                col_idx_ref[0] += 1

        col_idx_ref = [0]
        fill_carries()

        for cell in cells:
            text = _normalize_ws(cell.get_text(" ", strip=True))
            colspan = int(cell.get("colspan", 1) or 1)
            rowspan = int(cell.get("rowspan", 1) or 1)
            for _ in range(colspan):
                row.append(text)
                if rowspan > 1:
                    rowspan_carry[col_idx_ref[0]] = (rowspan - 1, text)
                col_idx_ref[0] += 1
            fill_carries()

        if any(c for c in row):
            grid.append(row)

    if not grid:
        return ""

    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]

    # drop fully-empty columns (common in SEC filings for spacing)
    keep_cols = [c for c in range(width) if any(r[c] for r in grid)]
    grid = [[r[c] for c in keep_cols] for r in grid]
    if not grid or not grid[0]:
        return ""

    lines = []
    header = grid[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in grid[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def extract_structured(html_path: Path) -> dict:
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    body = soup.body or soup

    blocks: list[dict] = []
    anchor_to_block: dict[str, str] = {}
    pending_refs: list[tuple[str, str, str]] = []  # (from_block_id, target_frag, link_text)

    current_section = "Preamble"
    processed_tables = set()
    pending_anchor_ids: list[str] = []

    for el in body.find_all(True):
        # skip anything inside a table - its content/ids are handled when
        # the table itself is processed
        if el.find_parent("table") is not None:
            continue

        if el.get("id"):
            pending_anchor_ids.append(el["id"])

        if el.name == "table":
            if id(el) in processed_tables:
                continue
            processed_tables.add(id(el))

            # some filings (e.g. Walmart) render section headings as a
            # single-purpose table with repeated/duplicated cells, e.g.
            # "ITEM 1A. | ITEM 1A. | RISK FACTORS | RISK FACTORS".
            # Detect this by de-duplicating consecutive identical cells.
            raw_cells = [_normalize_ws(c.get_text(" ", strip=True))
                         for c in el.find_all(["td", "th"])]
            deduped = []
            for c in raw_cells:
                if c and (not deduped or deduped[-1] != c):
                    deduped.append(c)
            heading_candidate = " ".join(deduped)
            hm = SECTION_PATTERN.match(heading_candidate)
            if hm and len(heading_candidate) <= MAX_HEADING_LEN:
                block_id = f"b{len(blocks)}"
                current_section = heading_candidate
                blocks.append({
                    "block_id": block_id,
                    "type": "heading",
                    "section": current_section,
                    "text": heading_candidate,
                })
                for aid in pending_anchor_ids:
                    anchor_to_block.setdefault(aid, block_id)
                for anchor_el in el.find_all(id=True):
                    anchor_to_block.setdefault(anchor_el["id"], block_id)
                pending_anchor_ids = []
                continue

            md = _table_to_markdown(el)
            if not md:
                continue
            block_id = f"b{len(blocks)}"
            blocks.append({
                "block_id": block_id,
                "type": "table",
                "section": current_section,
                "text": md,
            })
            for aid in pending_anchor_ids:
                anchor_to_block.setdefault(aid, block_id)
            for anchor_el in el.find_all(id=True):
                anchor_to_block.setdefault(anchor_el["id"], block_id)
            pending_anchor_ids = []
            continue

        if not _is_leaf_block(el):
            continue

        text = _normalize_ws(el.get_text(" ", strip=True))
        if not text:
            continue

        block_id = f"b{len(blocks)}"

        block_type = "paragraph"
        m = SECTION_PATTERN.match(text)
        if m and len(text) <= MAX_HEADING_LEN:
            block_type = "heading"
            current_section = text

        blocks.append({
            "block_id": block_id,
            "type": block_type,
            "section": current_section,
            "text": text,
        })

        # register pending anchors (from this element and any skipped
        # ancestors/preceding empty elements) against this block
        for aid in pending_anchor_ids:
            anchor_to_block.setdefault(aid, block_id)
        pending_anchor_ids = []

        # collect outgoing internal references (links to fragments)
        for a in el.find_all("a", href=True):
            href = a["href"]
            if "#" not in href:
                continue
            frag = href.split("#", 1)[1]
            link_text = _normalize_ws(a.get_text(" ", strip=True))
            if frag and link_text:
                pending_refs.append((block_id, frag, link_text))

    # resolve references and attach to blocks
    by_id = {b["block_id"]: b for b in blocks}
    reference_graph = []
    for from_id, frag, link_text in pending_refs:
        target_id = anchor_to_block.get(frag)
        if not target_id or target_id == from_id:
            continue
        ref = {"text": link_text, "target_block_id": target_id}
        by_id[from_id].setdefault("refs", []).append(ref)
        reference_graph.append({"from": from_id, "to": target_id, "text": link_text})

    # merge bare "Item 1A." headings with the following title block
    # (e.g. "Item 1A." + "Risk Factors" -> "Item 1A. Risk Factors")
    for i, b in enumerate(blocks):
        if b["type"] != "heading":
            continue
        m = SECTION_PATTERN.match(b["text"])
        if not m or m.group(2).strip():
            continue
        if i + 1 >= len(blocks):
            continue
        nb = blocks[i + 1]
        if (nb["section"] == b["text"] and len(nb["text"]) <= MAX_HEADING_LEN
                and not SECTION_PATTERN.match(nb["text"])):
            merged = f"{b['text']} {nb['text']}"
            nb["type"] = "heading"
            old = b["text"]
            for bb in blocks[i:]:
                if bb["section"] == old:
                    bb["section"] = merged
                else:
                    break

    section_index: dict[str, list[str]] = {}
    for b in blocks:
        section_index.setdefault(b["section"], []).append(b["block_id"])

    # --- in-text cross-reference detection -------------------------------
    # SEC 10-Ks rarely hyperlink "see Note 5" / "Item 7A" mentions in body
    # text, so href-based resolution above mostly stays empty. Instead,
    # detect these mentions textually and resolve them to the heading
    # block of the referenced Note / Item / Part, giving a usable
    # cross-reference graph for multi-hop retrieval.

    # Note N -> block_id of the "Note N ..." heading
    note_index: dict[str, str] = {}
    for b in blocks:
        if b["type"] == "heading":
            m = NOTE_HEADING_PATTERN.match(b["text"])
            if m:
                note_index.setdefault(m.group(1).lower(), b["block_id"])

    # Item N -> block_id of its heading (from section_index keys)
    item_index: dict[str, str] = {}
    part_index: dict[str, str] = {}
    for section_name, block_ids in section_index.items():
        m = SECTION_PATTERN.match(section_name)
        if not m:
            continue
        marker = m.group(1).rstrip(".").lower()  # "item 7a" or "part ii"
        if marker.startswith("item"):
            item_index.setdefault(marker.split()[1], block_ids[0])
        elif marker.startswith("part"):
            part_index.setdefault(marker.split()[1], block_ids[0])

    for b in blocks:
        if b["type"] != "paragraph":
            continue
        for m in MENTION_PATTERN.finditer(b["text"]):
            kind, num = m.group(1).lower(), m.group(2).lower()
            target_id = None
            if kind == "note":
                target_id = note_index.get(num)
            elif kind == "item":
                target_id = item_index.get(num)
            elif kind == "part":
                target_id = part_index.get(num)
            if not target_id or target_id == b["block_id"]:
                continue
            ref = {"text": m.group(0), "target_block_id": target_id}
            b.setdefault("refs", []).append(ref)
            reference_graph.append({"from": b["block_id"], "to": target_id, "text": m.group(0)})

    return {
        "doc_id": html_path.stem,
        "blocks": blocks,
        "section_index": section_index,
        "reference_graph": reference_graph,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for html_file in sorted(DATASET_DIR.glob("*.html")):
        print(f"Extracting {html_file.name} ...")
        result = extract_structured(html_file)
        out_path = OUTPUT_DIR / (html_file.stem + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        n_tables = sum(1 for b in result["blocks"] if b["type"] == "table")
        n_headings = sum(1 for b in result["blocks"] if b["type"] == "heading")
        print(f"  -> {out_path}: {len(result['blocks'])} blocks "
              f"({n_headings} headings, {n_tables} tables), "
              f"{len(result['reference_graph'])} resolved cross-references, "
              f"{len(result['section_index'])} sections")


if __name__ == "__main__":
    main()
