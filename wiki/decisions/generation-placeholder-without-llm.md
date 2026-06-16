> **[OUTDATED]** Content moved to [`decisions-log.md`](decisions-log.md).

## Generation Placeholder When No LLM Configured
- Date: 2026-06-15
- Context: Retrieval experiments should run before API keys and generation models are chosen.
- Decision: If `RagPipeline` receives no `llm_fn`, `.generate()` returns a fixed placeholder string. The harness still runs end-to-end and computes retrieval metrics where `gold_chunk_ids` exist.
- Alternatives Considered: Skip generation step entirely in retrieval-only mode; require LLM from the start.
- Impact: Run results include `generated_answer` placeholders. LLM-judge metrics (correctness, faithfulness) in `metrics.py` remain unused until generation is wired.
