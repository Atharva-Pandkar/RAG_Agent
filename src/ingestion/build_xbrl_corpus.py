"""
Build a fact-level chunked corpus by parsing SEC inline-XBRL (iXBRL) filings.

Why this exists
---------------
Unstructured-IO misses financial statement tables embedded inside
<ix:continuation> blocks — the SEC's mechanism for spanning a tagged
text-block across document sections.  Those blocks contain the actual
<table> elements that hold numeric financial data (e.g. income statement,
segment breakdowns, equity-method investee summaries).  Parsing the iXBRL
tags directly recovers every tagged numeric fact and groups them by the
HTML <table> they live in, producing rich, structured chunks that BM25
and dense retrieval can both index effectively.

Approach
--------
1.  Parse xbrli:context → period + segment label per context-id.
2.  Collect all visible ix:nonFraction facts (skip ix:hidden section).
3.  Group facts by their nearest <table> ancestor (each table → one chunk).
4.  Convert each fact-bearing table to a Markdown table for the chunk text.
5.  Forward-propagate section labels (same regex as build_unstructured_corpus).
6.  Write section cross-references (section_chunk_ids, section_table_ids,
    section_narrative_ids) so multi-hop retrieval can walk siblings.

Output: Experiments/corpora/xbrl_{doc_id}.json per filing, plus a merged
        Experiments/corpora/xbrl_merged.json for all five filings.

Usage
-----
    python src/ingestion/build_xbrl_corpus.py
    python src/ingestion/build_xbrl_corpus.py --companies aapl ko
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
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

# Common US-GAAP concept → human-readable label overrides.
# CamelCase splitting handles everything else.
_CONCEPT_LABELS: dict[str, str] = {
    "Revenues": "Net Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",
    "CostOfGoodsSold": "Cost of Goods Sold",
    "CostOfRevenue": "Cost of Revenue",
    "GrossProfit": "Gross Profit",
    "OperatingIncomeLoss": "Operating Income (Loss)",
    "NetIncomeLoss": "Net Income (Loss)",
    "NetIncomeLossAttributableToNoncontrollingInterest": "Net Income — Noncontrolling Interests",
    "NetIncomeLossAvailableToCommonStockholdersBasic": "Net Income Available to Common Shareholders",
    "EarningsPerShareBasic": "EPS Basic",
    "EarningsPerShareDiluted": "EPS Diluted",
    "Assets": "Total Assets",
    "Liabilities": "Total Liabilities",
    "StockholdersEquity": "Stockholders' Equity",
    "CashAndCashEquivalentsAtCarryingValue": "Cash and Cash Equivalents",
    "LongTermDebt": "Long-Term Debt",
    "ResearchAndDevelopmentExpense": "R&D Expense",
    "SellingGeneralAndAdministrativeExpense": "SG&A Expense",
    "DepreciationDepletionAndAmortization": "Depreciation & Amortization",
    "NetCashProvidedByUsedInOperatingActivities": "Operating Cash Flow",
    "NetCashProvidedByUsedInInvestingActivities": "Investing Cash Flow",
    "NetCashProvidedByUsedInFinancingActivities": "Financing Cash Flow",
    "CapitalExpendituresIncurredButNotYetPaid": "Capex",
    "PaymentsToAcquirePropertyPlantAndEquipment": "Capex (PPE)",
    "CommonStockSharesOutstanding": "Shares Outstanding",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "Pre-Tax Income",
    "IncomeTaxExpenseBenefit": "Income Tax Expense",
}


def _camel_to_label(name: str) -> str:
    """Convert 'OperatingIncomeLoss' → 'Operating Income Loss'."""
    local = name.split(":")[-1]
    if local in _CONCEPT_LABELS:
        return _CONCEPT_LABELS[local]
    # Insert spaces before uppercase letters that follow lowercase or digits
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", local)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    return spaced


def _scale_value(raw_text: str, decimals: str | None) -> str:
    """Apply scale factor from XBRL decimals attribute and format value."""
    text = raw_text.strip().replace(",", "").replace("(", "-").replace(")", "")
    try:
        dec = int(decimals or "0")
        val = float(text)
        if dec <= -6:
            # millions — keep as-is but add comma formatting
            formatted = f"{val:,.0f}" if val == int(val) else f"{val:,.3f}"
            return formatted
        elif dec <= -3:
            formatted = f"{val:,.0f}"
            return formatted
        else:
            return raw_text.strip()
    except (ValueError, TypeError):
        return raw_text.strip()


# ── iXBRL parsing helpers ──────────────────────────────────────────────────────

def _parse_contexts(soup) -> dict[str, dict]:
    """Build {context_id: {period, segment}} from xbrli:context elements."""
    ctx_map: dict[str, dict] = {}
    for ctx in soup.find_all("xbrli:context"):
        cid = ctx.get("id")
        if not cid:
            continue
        instant = ctx.find("xbrli:instant")
        start = ctx.find("xbrli:startdate")
        end = ctx.find("xbrli:enddate")
        segment = ctx.find("xbrli:segment")

        if instant:
            period = instant.get_text(strip=True)
        elif start and end:
            period = f"{start.get_text(strip=True)} -> {end.get_text(strip=True)}"
        else:
            period = "unknown"

        seg_text: str | None = None
        if segment:
            # Use the member name (last part after :) for readability
            members = segment.find_all(True)
            if members:
                raw = members[0].get_text(strip=True)
                seg_text = raw.split(":")[-1] if ":" in raw else raw
            else:
                seg_text = segment.get_text(strip=True)

        ctx_map[cid] = {"period": period, "segment": seg_text}
    return ctx_map


def _hidden_ids(soup) -> set[str]:
    """Return ids of all elements inside ix:hidden (should not be rendered)."""
    ids: set[str] = set()
    hidden = soup.find("ix:hidden")
    if hidden:
        for el in hidden.find_all(True):
            eid = el.get("id")
            if eid:
                ids.add(eid)
    return ids


def _detect_section(text: str) -> str | None:
    t = text.strip()
    if SECTION_PATTERN.match(t):
        return t[:80]
    if _NOTE_PATTERN.match(t) and len(t) < 80:
        return t
    return None


# ── Markdown cleanup ──────────────────────────────────────────────────────────

def _dedup_md_table(md: str) -> str:
    """Remove consecutive duplicate cells in each markdown table row.

    colspan expansion in _table_to_markdown produces rows like:
      | Label | Label | Label | 2025 | 2025 | 2025 |
    Deduplicate to:
      | Label | 2025 |
    while preserving the separator row (--- | --- | ...).
    """
    lines: list[str] = []
    for line in md.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            lines.append(line)
            continue
        cells = [c.strip() for c in stripped.split("|")]
        # cells[0] and cells[-1] are empty (leading/trailing |)
        inner = cells[1:-1]
        # Skip separator rows
        if all(re.fullmatch(r"-+", c) or c == "" for c in inner):
            lines.append(line)
            continue
        # Deduplicate consecutive identical cells
        deduped: list[str] = []
        for cell in inner:
            if not deduped or cell != deduped[-1]:
                deduped.append(cell)
        lines.append("| " + " | ".join(deduped) + " |")
    return "\n".join(lines)


# Segment names that indicate data belongs to equity-method investees,
# NOT the filing company's own consolidated financials.
_INVESTEE_SEGMENT_PATTERNS = re.compile(
    r"equitymethod|nonconsolidated|investee", re.IGNORECASE
)


def _concept_header(concept_names: list[str], periods: list[str], segments: list[str]) -> str:
    """Build a short natural-language header for a financial table chunk."""
    labels = [_camel_to_label(n) for n in concept_names[:6]]
    period_strs: list[str] = []
    for p in periods[:3]:
        # Extract year from period string like "2025-01-01 -> 2025-12-31"
        years = re.findall(r"\b(20\d{2})\b", p)
        if years:
            period_strs.append(years[-1])  # end year

    # Detect equity-method investee tables and surface the warning prominently
    # so the reranker sees it within the first 200 chars of the chunk.
    is_investee = segments and _INVESTEE_SEGMENT_PATTERNS.search(segments[0])
    if is_investee:
        warning = (
            "NOTE: This table shows the COMBINED FINANCIALS OF EQUITY METHOD "
            "INVESTEES — these are NOT the filing company's own consolidated "
            "results. Do not use these figures to answer questions about the "
            "filer's revenue, income, or operating performance.\n\n"
        )
    else:
        warning = ""

    header_parts: list[str] = []
    if labels:
        header_parts.append("Financial data: " + ", ".join(labels))
    if period_strs:
        header_parts.append("Period: " + " / ".join(sorted(set(period_strs))))
    if segments:
        header_parts.append("Segment: " + segments[0])
    meta = ". ".join(header_parts) + ".\n\n" if header_parts else ""
    return warning + meta


# ── Table-level fact extraction ────────────────────────────────────────────────

def _table_has_xbrl_facts(table_el) -> bool:
    return bool(table_el.find(re.compile(r"^ix:(non(fraction|numeric))$", re.I)))


def _extract_table_chunk(
    table_el,
    doc_id: str,
    chunk_idx: int,
    section: str | None,
    contexts: dict[str, dict],
    hidden: set[str],
) -> dict | None:
    """Convert one fact-bearing <table> to a corpus chunk dict."""
    from bs4 import Tag

    # Only process tables that contain at least one tagged fact
    facts_in_table = table_el.find_all(
        re.compile(r"^ix:nonfraction$", re.I)
    )
    visible_facts = [
        f for f in facts_in_table if f.get("id") not in hidden
    ]
    if not visible_facts:
        return None

    # Collect unique concepts and periods present in this table
    concept_names: list[str] = []
    periods: list[str] = []
    segments: list[str] = []
    for f in visible_facts:
        name = f.get("name", "")
        if name and name not in concept_names:
            concept_names.append(name)
        ctx = contexts.get(f.get("contextref", ""), {})
        p = ctx.get("period", "")
        if p and p not in periods:
            periods.append(p)
        seg = ctx.get("segment")
        if seg and seg not in segments:
            segments.append(seg)

    # Convert the table to markdown text, clean duplicate cells, add header
    md = _table_to_markdown(table_el)
    if not md.strip():
        return None
    md = _dedup_md_table(md)
    header = _concept_header(concept_names, periods, segments)
    text = header + md

    chunk_id = f"{doc_id}_xbrl_{chunk_idx}"
    return {
        "id": chunk_id,
        "doc": doc_id,
        "chunk_index": chunk_idx,
        "text": text,
        "section": section,
        "element_types": ["Table"],
        "has_table": True,
        "xbrl_concepts": concept_names[:20],   # capped to keep JSON compact
        "xbrl_periods": periods[:6],
        "xbrl_segments": segments[:10],
        "table_ids": [],          # filled in second pass
        "section_chunk_ids": [],
        "section_table_ids": [],
        "section_narrative_ids": [],
    }


# ── Per-filing processor ───────────────────────────────────────────────────────

def process_filing(html_path: Path, doc_id: str) -> list[dict]:
    from bs4 import BeautifulSoup

    print(f"  Parsing {doc_id}...", flush=True)
    # SEC filings declare windows-1252 charset; html.parser (not lxml) is used
    # because lxml's libxml2 re-encodes the input on charset detection and
    # drops namespace-qualified tags like ix:nonFraction in the process.
    with open(html_path, encoding="windows-1252", errors="replace") as f:
        soup = BeautifulSoup(f, "html.parser")

    contexts = _parse_contexts(soup)
    hidden = _hidden_ids(soup)

    print(f"    contexts: {len(contexts)}  hidden ids: {len(hidden)}", flush=True)

    # Walk the document in order; track current section via forward propagation.
    # We collect all <table> elements that contain iXBRL facts.
    current_section: str | None = None
    chunks: list[dict] = []
    seen_table_ids: set[int] = set()  # object ids to dedup same table visited twice

    # We process the full document tree in source order.
    # To handle ix:continuation correctly we walk the entire body.
    body = soup.find("body")
    if body is None:
        body = soup

    chunk_idx = 0
    for element in body.descendants:
        # Section detection on any text-bearing element
        if hasattr(element, "get_text"):
            for line in element.get_text().splitlines():
                candidate = _detect_section(line.strip())
                if candidate:
                    current_section = candidate
                    break

        # When we encounter a <table> we haven't processed yet
        if getattr(element, "name", None) == "table":
            eid = id(element)
            if eid in seen_table_ids:
                continue
            seen_table_ids.add(eid)

            chunk = _extract_table_chunk(
                element, doc_id, chunk_idx, current_section, contexts, hidden
            )
            if chunk:
                chunks.append(chunk)
                chunk_idx += 1

    print(f"    {len(chunks)} fact-bearing table chunks", flush=True)

    # ── Second pass: section cross-references ──────────────────────────────
    section_buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in chunks:
        section_buckets[rec["section"] or "__preamble__"].append(rec)

    for sec, recs in section_buckets.items():
        all_ids = [r["id"] for r in recs]
        table_ids = [r["id"] for r in recs if r["has_table"]]
        narrative_ids = [r["id"] for r in recs if not r["has_table"]]
        for r in recs:
            r["table_ids"] = table_ids
            r["section_chunk_ids"] = all_ids
            r["section_table_ids"] = table_ids
            r["section_narrative_ids"] = narrative_ids

    return chunks


# ── Corpus builder ─────────────────────────────────────────────────────────────

def build(companies: list[str] | None = None) -> Path:
    all_chunks: list[dict] = []
    selected = {k: v for k, v in COMPANY_FILES.items()
                if companies is None or any(c in k for c in companies)}

    for doc_id, fname in selected.items():
        html_path = DATASET_DIR / fname
        chunks = process_filing(html_path, doc_id)
        all_chunks.extend(chunks)
        sections = len({c["section"] for c in chunks if c["section"]})
        print(f"    {len(chunks)} chunks across {sections} sections")

    table_index = {
        c["id"]: (c["section"] or "__preamble__")
        for c in all_chunks if c["has_table"]
    }

    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CORPORA_DIR / "xbrl_merged.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "strategy": "xbrl",
                "chunks": all_chunks,
                "table_index": table_index,
            },
            f,
            indent=1,
            ensure_ascii=False,
        )

    print(f"\nWrote {len(all_chunks)} chunks -> {out_path}")
    print(f"  table_index: {len(table_index)} entries")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--companies", nargs="*", help="Filter by company prefix (e.g. aapl ko)")
    args = parser.parse_args()
    build(args.companies)


if __name__ == "__main__":
    main()
