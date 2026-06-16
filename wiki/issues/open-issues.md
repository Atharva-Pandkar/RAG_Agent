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
- Description: Iteration 1 had only 15/27 answerable questions matched. Merged corpus sidecar now matches **26/26** answerable questions. Unstructured sidecar: 25/26.
- Suggested Fix: Identify the 1 remaining unmatched question on unstructured corpus.

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

---

## Chatbot Uses Suboptimal Retrieval Config [OUTDATED — severity raised]
- Severity: **High** (was Medium)
- Location: `app/backend/rag_tool.py`
- Description: Demo app hardcodes hybrid (numpy) over `fixed_size_1024_100` at k=5 (recall ~0.440). Current best is Run 36: faiss_hybrid + unstructured 4000/200 at k=10 (recall **0.677**). Gap of ~0.24 recall vs production baseline.
- Suggested Fix: Update to Run 36 pipeline kwargs; expose via env/YAML; set k=10 for retrieval tool.

---

## Merged Corpus Underperforms Unstructured-Only Despite Full Gold Coverage
- Severity: Medium
- Location: `build_merged_corpus.py`, Runs 37–40 vs 32–36
- Description: Merged corpus has 26/26 gold-chunk matches (vs 25/26 unstructured) and 1,893 chunks vs 1,492, but recall@10 is 0.575 (Run 40) vs 0.677 (Run 36). Added iXBRL chunks may dilute BM25/FAISS ranking or dedup fingerprint may drop useful variants.
- Suggested Fix: Analyze which questions improve/regress on merged vs unstructured; tune dedup threshold; try iXBRL-only chunks for table_numerical question types.

---

## iXBRL / unstructured / merge deps Missing from requirements.txt
- Severity: Medium
- Location: `requirements.txt`
- Description: `build_unstructured_corpus.py` needs `unstructured[html]`; xbrl parser uses bs4/lxml (present) but no explicit ixbrl deps documented. Still no sentence-transformers, rank-bm25, faiss, langchain pinned.
- Suggested Fix: Pin full experiment stack in root requirements.

---

## Large Unstructured JSON Artifacts (JPM) [OUTDATED — production corpus built]
- Severity: Low
- Location: `Documents/unstructured_explore/jpm_unstructured_chunks.json`
- Description: JPM filing produces 890 chunks in explore output. Production corpus `unstructured_4000_200.json` consolidates all filings (1,492 chunks total).
- Suggested Fix: Explore artifacts optional; production corpus is the eval source of truth.

---

## Split Requirements Files and Missing Root Deps
- Severity: Medium
- Location: `requirements.txt`, `app/backend/requirements.txt`
- Description: Root `requirements.txt` still has Phase 1+ deps commented out. LangChain, FAISS, sentence-transformers, rank-bm25 needed for runs 21–31 but not pinned at root. App has separate deps (fastapi, langchain, langgraph) with no link to experiment stack.
- Suggested Fix: Consolidate or cross-reference; uncomment and pin experiment deps at root.

---

## FAISS Deserialization Flag
- Severity: Low
- Location: `src/retrieval/faiss_retriever.py`
- Description: `FAISS.load_local(..., allow_dangerous_deserialization=True)` required for cached indexes. Acceptable for local research artifacts; risky if index files are untrusted.
- Suggested Fix: Document trust boundary; consider safer serialization for production.

---

## CORS Wildcard on Chatbot API
- Severity: Low
- Location: `app/backend/main.py`
- Description: `allow_origins=["*"]` on FastAPI CORS middleware. Fine for local dev; should be restricted before deployment.
- Suggested Fix: Restrict to frontend origin in production.

---

## unstructured Package Missing from requirements.txt
- Severity: Medium
- Location: `requirements.txt`, `src/ingestion/explore_unstructured.py`
- Description: `explore_unstructured.py` imports `unstructured.partition.html` and `unstructured.chunking.title` but `unstructured` is not listed in root or app requirements. Script fails on fresh install.
- Suggested Fix: Add `unstructured[html]` (or equivalent extra) to `requirements.txt` with pinned version.

---

## Semantic Explore Table Detection Returns Zero
- Severity: Low
- Location: `src/ingestion/explore_html_semantic.py`, `Documents/semantic_explore/aapl_semantic_chunks.json`
- Description: AAPL semantic explore output reports `chunks_with_table: 0` despite custom table handler. Table markdown rows may not match the `|...|` regex detector, or handler output format differs from expectation.
- Suggested Fix: Inspect handler output; relax table detection regex; compare against unstructured explore (29 table chunks for same filing).

---

## Unstructured Explore Section Detection Is Heading-Heuristic Only
- Severity: Low
- Location: `src/ingestion/explore_unstructured.py`
- Description: Section labels derived from `SECTION_PATTERN` / Note-heading regex on chunk text lines. AAPL detects 12 sections; JPM detects 49. Granularity varies by filing structure; many chunks may share coarse labels or land in `__preamble__`.
- Suggested Fix: Compare section labels against `extract_structured.py` section tags; align patterns or import structured section map.

---

## Large Unstructured JSON Artifacts (JPM)
- Severity: Low
- Location: `Documents/unstructured_explore/jpm_unstructured_chunks.json`
- Description: JPM filing produces 890 chunks (486 table chunks, 2853 elements). Full JSON includes all chunk text — file is very large and slow to load in editors.
- Suggested Fix: Add `--max-chunks` preview flag (like semantic explore); store text externally or truncate previews for inspection-only runs.
