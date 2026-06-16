> **[OUTDATED]** Content moved to [`decisions-log.md`](decisions-log.md).

## Phase 1 Baseline: BM25 + Fixed-Size Chunking
- Date: 2026-06-15
- Context: Phase 1 requires establishing the simplest end-to-end retrieval baseline before adding embeddings, hybrid search, or reranking.
- Decision: First experiment (`run01_bm25_fixed512`) uses fixed-size token windows (512 tokens, 50 overlap, cl100k_base via tiktoken) and BM25Okapi sparse retrieval with alphanumeric lowercase tokenization.
- Alternatives Considered: Dense embeddings first; section-aware chunking first; recursive chunking as the initial baseline.
- Impact: Sets the floor for recall@k/MRR comparisons. Run 01 completed retrieval-only (no LLM). Recursive and section-based chunkers are implemented but not yet benchmarked.
