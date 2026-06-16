# Open Issues

---

## rank-bm25 Missing from requirements.txt
- Severity: Medium
- Location: `requirements.txt`, `src/retrieval/bm25_retriever.py`
- Description: `BM25Retriever` imports `rank_bm25` but the dependency is commented out in `requirements.txt`. Fresh installs will fail at runtime.
- Suggested Fix: Uncomment `rank-bm25` in `requirements.txt` and pin a version.

---

## Gold Chunk Matching Failed for 12 Answerable Questions [RESOLVED — see updated entry below]
- Severity: High
- Location: `eval/golden_set/populate_gold_chunks.py`, `Experiments/corpora/golden_set_with_chunks_fixed_size_512_50.json`
- Description: Only 15 of 27 answerable questions received `gold_chunk_ids`. Evidence quotes (especially table_numerical and paraphrased evidence) don't appear verbatim in fixed-size chunks, so anchor matching fails. Retrieval metrics are null or incomplete for those questions.
- Suggested Fix: Improve matching (fuzzy match, multi-anchor, token overlap threshold); manually annotate hard cases; consider section-aware chunking for table questions.

---

## BM25 Retrieves Cross-Document Chunks
- Severity: High
- Location: `src/retrieval/bm25_retriever.py`, Run 01 results
- Description: Queries scoped to one company (e.g. AAPL-01) retrieve chunks from unrelated filings (e.g. WMT). No document filter is applied at retrieval time.
- Suggested Fix: Add optional `doc_filter` derived from question metadata (company tag) or use metadata-aware retrieval; at minimum filter by `doc` field when company is known from golden set.

---

## No Aggregate Metrics in Run Output
- Severity: Low
- Location: `eval/harness/run_eval.py`
- Description: Run JSON contains per-question metrics only. Comparing runs requires post-hoc aggregation.
- Suggested Fix: Add a `summary` block with mean recall@k, MRR, context precision (overall and by question type).

---

## baseline.yaml Not Wired to Pipeline [OUTDATED]
- Severity: Low
- Location: `Experiments/configs/baseline.yaml`
- Description: Template config from Phase 0 planning. Actual runs use per-run YAML files (run07+ for dense). `pipeline_class` still commented out; nested chunking/embedding sections unused.
- Suggested Fix: Mark as reference template or delete content in favor of run-specific configs.

---

## sys.path Manipulation for Imports
- Severity: Low
- Location: `src/pipeline.py`, `src/ingestion/build_corpus.py`
- Description: Modules insert project paths via `sys.path.insert` instead of using package-relative imports or installing the project as a package.
- Suggested Fix: Add `pyproject.toml` / editable install, or restructure `src/` as an installable package.

---

## Gold Chunk Matching Failed for 12 Answerable Questions [RESOLVED]
- Severity: High → resolved
- Location: `eval/golden_set/populate_gold_chunks.py`
- Description: Iteration 1 had only 15/27 answerable questions matched. Re-population against updated corpora now matches 26/27 on latest sidecars.
- Suggested Fix: Investigate the 1 remaining unmatched question; consider fuzzy matching for edge cases.

---

## sentence-transformers Missing from requirements.txt
- Severity: Medium
- Location: `requirements.txt`, `src/retrieval/dense_retriever.py`, `src/retrieval/hybrid_retriever.py`
- Description: Dense and hybrid retrievers depend on `sentence-transformers` (and transitively `torch`, `pyarrow`) but all Phase 1+ deps remain commented out in `requirements.txt`.
- Suggested Fix: Uncomment and pin `sentence-transformers`, `rank-bm25`, and note torch/pyarrow version constraints for Windows.

---

## HybridRetriever Double-Initializes Both Retrievers
- Severity: Low
- Location: `src/retrieval/hybrid_retriever.py`
- Description: Every hybrid pipeline init loads BM25 index + SentenceTransformer model + embedding matrix, even if only one retriever changed between configs.
- Suggested Fix: Share retriever instances or lazy-load; acceptable at current scale.

---

## Run 18 Not Yet Executed [RESOLVED]
- Severity: Low
- Location: `Experiments/configs/run18_hybrid_section1024_k10.yaml`
- Description: Config existed but no results JSON in `Experiments/runs/`.
- Suggested Fix: Run eval and append to experiment log.

---

## ms-marco Cross-Encoder Reranking Reduces 10-K Recall
- Severity: Medium
- Location: `src/retrieval/reranker.py`, Runs 19–20 vs 04/12
- Description: Reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2` (web-search tuned) drops recall@k on financial 10-K text. Run 19 (BM25+rerank): 0.394 vs Run 04: 0.416. Run 20 (hybrid+rerank): 0.348 vs Run 12: 0.440. Rerank truncates fetch_k=20 → k=5, discarding gold chunks that only appear at higher ranks.
- Suggested Fix: Try BGE reranker; rerank at k=10; increase fetch_k; evaluate domain-finetuned cross-encoder; test rerank on Run 16 (k=10) where recall is already high.

---

## Duplicate Run JSON Files in Experiments/runs/
- Severity: Low
- Location: `Experiments/runs/`
- Description: Multiple timestamped JSON files per run name (early + re-run batches). `summarize_runs.py` processes all files, producing duplicate rows.
- Suggested Fix: Deduplicate by run_name (keep latest) in summarize script, or archive old runs.
