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

## KO-03 Unmatched on Unstructured Corpus
- Severity: Low
- Location: `Experiments/corpora/golden_set_with_chunks_unstructured_4000_200.json`, question KO-03
- Description: Despite `chunk_anchors` improvements (26/26 on fixed-size and merged), unstructured sidecar still misses KO-03. Evidence/anchors may not appear in any unstructured chunk for that KO filing section.
- Suggested Fix: Add KO-03-specific `chunk_anchors`; manually inspect KO chunks near evidence; or accept merged corpus for full coverage eval.

---

## merged_unstr_xbrl.json File Size (~24 MB)
- Severity: Medium
- Location: `Experiments/corpora/merged_unstr_xbrl.json`
- Description: Merged corpus JSON is ~749k lines / 24 MB with full chunk text inline. Slow to open in editors, heavy for git diffs, and loads entirely into memory at pipeline init alongside FAISS/BM25 indexes.
- Suggested Fix: Store chunk text in separate per-doc files; use compact JSON without pretty-print; add to `.gitattributes` LFS or document as generated artifact not for manual editing.

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

## Chatbot Uses Suboptimal Retrieval Config [RESOLVED]
- Severity: High → resolved (with tradeoff)
- Location: `app/backend/rag_tool.py`
- Description: Was hardcoded to hybrid fixed 1024 k=5. Now uses Run 40: faiss_hybrid + merged_unstr_xbrl k=10. Chose merged over Run 36 (best recall 0.677) for 26/26 gold coverage and iXBRL tables. Eval recall for chatbot config: 0.575.
- Suggested Fix: Consider env toggle between Run 36 (max recall) and Run 40 (max coverage).

---

## CORS Wildcard on Chatbot API [RESOLVED]
- Severity: Low → resolved
- Location: `app/backend/main.py`
- Description: Was `allow_origins=["*"]`. Now restricted to `http://localhost:5173` and `http://localhost:3000`.
- Suggested Fix: Add production origin when deploying.

---

## Frontend Ignores Source Citations from API
- Severity: Low
- Location: `app/frontend/src/App.jsx`, `app/backend/main.py`
- Description: Backend returns `sources: string[]` (chunk IDs) in `ChatResponse`, but frontend only displays `data.response`. Retrieved chunk IDs are discarded in the UI.
- Suggested Fix: Render sources list or inline citations below assistant messages.

---

## Section Metadata Not Passed Through Retrievers
- Severity: Low
- Location: `src/retrieval/faiss_hybrid_retriever.py`, `app/backend/rag_tool.py`
- Description: Merged/unstructured chunks have `section` field in corpus JSON, but retrievers only return `{id, doc, text, score}`. Tool output always shows `Section: —`.
- Suggested Fix: Include `section` (and `has_table`) from `by_id` chunk dict in retriever return payloads.

---

## app/README.md Outdated
- Severity: Low
- Location: `app/README.md`
- Description: Still describes `create_agent` ReAct loop and hybrid fixed 1024 corpus. Actual stack is deep agent + Run 40 merged faiss_hybrid k=10.
- Suggested Fix: Update setup docs, API response shape, and retrieval config description.

---

## Deep Agent Cold-Start Latency
- Severity: Medium
- Location: `app/backend/main.py`, `app/backend/rag_tool.py`
- Description: First `/chat` request triggers lazy init of deep agent + FAISS-hybrid pipeline over 1,893-chunk merged corpus (~20s noted in code comment). No warmup endpoint.
- Suggested Fix: Eager-init on startup or `/health` warmup; show loading state in frontend.

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

## Split Requirements Files and Missing Root Deps [OUTDATED — app deps updated]
- Severity: Medium
- Location: `requirements.txt`, `app/backend/requirements.txt`
- Description: Root `requirements.txt` still has Phase 1+ deps commented out. App now uses `deepagents` (not langgraph). Experiment deps (sentence-transformers, rank-bm25, faiss, unstructured) still not pinned at root.
- Suggested Fix: Pin full experiment + app stacks; cross-reference in README.

---

## CORS Wildcard on Chatbot API [RESOLVED — see updated entry above]
- Severity: Low
- Location: `app/backend/main.py`
- Description: `allow_origins=["*"]` on FastAPI CORS middleware. Fine for local dev; should be restricted before deployment.
- Suggested Fix: Restrict to frontend origin in production.

---

## FAISS Deserialization Flag
- Severity: Low
- Location: `src/retrieval/faiss_retriever.py`
- Description: `FAISS.load_local(..., allow_dangerous_deserialization=True)` required for cached indexes. Acceptable for local research artifacts; risky if index files are untrusted.
- Suggested Fix: Document trust boundary; consider safer serialization for production.

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
