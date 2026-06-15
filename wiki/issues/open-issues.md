# Open Issues

---

## rank-bm25 Missing from requirements.txt
- Severity: Medium
- Location: `requirements.txt`, `src/retrieval/bm25_retriever.py`
- Description: `BM25Retriever` imports `rank_bm25` but the dependency is commented out in `requirements.txt`. Fresh installs will fail at runtime.
- Suggested Fix: Uncomment `rank-bm25` in `requirements.txt` and pin a version.

---

## Gold Chunk Matching Failed for 12 Answerable Questions
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

## baseline.yaml Not Wired to Pipeline
- Severity: Low
- Location: `Experiments/configs/baseline.yaml`
- Description: The baseline config describes a full dense-embedding pipeline but `pipeline_class` is commented out and chunking/embedding sections are not consumed by any code path.
- Suggested Fix: Either wire config sections into a unified config loader or mark the file as `[OUTDATED]` template until dense pipeline exists.

---

## sys.path Manipulation for Imports
- Severity: Low
- Location: `src/pipeline.py`, `src/ingestion/build_corpus.py`
- Description: Modules insert project paths via `sys.path.insert` instead of using package-relative imports or installing the project as a package.
- Suggested Fix: Add `pyproject.toml` / editable install, or restructure `src/` as an installable package.
