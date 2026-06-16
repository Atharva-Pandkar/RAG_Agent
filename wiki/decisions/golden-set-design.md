> **[OUTDATED]** Content moved to [`decisions-log.md`](decisions-log.md).

## Golden Set Design (28 Q&A, 7 Types)
- Date: 2026-06-15
- Context: Need a fixed benchmark to compare retrieval and generation quality across experiments without re-authoring questions each time.
- Decision: Maintain a canonical `eval/golden_set/golden_set.json` with 28 hand-curated questions across 5 filings (AAPL, CAT, JPM, KO, WMT), tagged by 7 types: factual_lookup, table_numerical, multi_hop_comparative, risk_factor_synthesis, mda_interpretive, cross_section_fact, unanswerable.
- Alternatives Considered: Auto-generated Q&A from filings; smaller ad-hoc question sets; BEIR-style external benchmarks.
- Impact: Stratified analysis by question type is possible. Each entry has gold_answer + evidence quote for traceability. `gold_chunk_ids` are populated per-corpus via `populate_gold_chunks.py`, not stored in the canonical file.
