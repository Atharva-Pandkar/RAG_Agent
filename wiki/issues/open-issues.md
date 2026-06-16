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

## BM25 Retrieves Cross-Document Chunks [OUTDATED — resolved Iteration 15; see entry below]
- Severity: High → resolved (chatbot path)
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

## Chatbot Uses Suboptimal Retrieval Config [RESOLVED — Iteration 11 evolution]
- Severity: High → resolved (with tradeoff)
- Location: `app/backend/rag_tool.py`
- Description: Was hardcoded to hybrid fixed 1024 k=5. Iteration 10 used Run 40 merged faiss_hybrid k=10. Iteration 11 uses `active_corpus.json` (seeded from merged baseline) with same faiss_hybrid + OpenAI embeddings + live ingest. Eval recall for static merged config remains ~0.575; active corpus may diverge after uploads.
- Suggested Fix: Env toggle between static eval corpus and active corpus; re-run eval after uploads.

---

## CORS Wildcard on Chatbot API [RESOLVED]
- Severity: Low → resolved
- Location: `app/backend/main.py`
- Description: Was `allow_origins=["*"]`. Now restricted to `http://localhost:5173` and `http://localhost:3000`.
- Suggested Fix: Add production origin when deploying.

---

## Frontend Ignores Source Citations from API [RESOLVED]
- Severity: Low → resolved
- Location: `app/frontend/src/App.jsx`, `app/backend/main.py`
- Description: Was discarding `sources` from API response. Iteration 10: `AssistantMessage` renders collapsible citation panel; `Citation` links to SEC EDGAR with ticker badge and section/doc label.
- Suggested Fix: Pass `section` through retrievers so citations show real section names instead of `"—"` / doc fallback.

---

## Source Metadata Parsed from Tool String (Fragile Coupling)
- Severity: Low
- Location: `app/backend/main.py`, `app/backend/rag_tool.py`
- Description: `SourceRef` fields are built by regex-parsing `search_documents` tool output (`Source: … | Section: … | ID: …`) and matching agent `[SOURCE:id]` markers in the final answer — not by joining retriever result dicts or corpus `by_id`. Any change to tool output format or marker convention breaks citation parsing silently.
- Suggested Fix: Return structured retrieval results from the tool (JSON) or build `SourceRef` from pipeline retrieve() metadata in `main.py` via chunk-ID lookup.

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

## Deep Agent Cold-Start Latency [RESOLVED]
- Severity: Medium → resolved
- Location: `app/backend/main.py`, `app/backend/rag_tool.py`
- Description: Was lazy init on first `/chat` (~20s). Now eager-init via FastAPI `lifespan`: pipeline warm-up query + agent compile at server start. `/health` exposes readiness; `/chat` returns 503 until ready.
- Suggested Fix: Show startup/loading state in frontend if server boot is slow; poll `/health` before enabling chat input. **[Partial — loading dots during chat turn only; no server-boot polling yet]**

---

## Chatbot Embedding Model Diverges from Eval Run 40
- Severity: Medium
- Location: `app/backend/rag_tool.py`, `src/retrieval/faiss_retriever.py`
- Description: Eval Run 40 used `BAAI/bge-small-en-v1.5` (local). Chatbot now uses `text-embedding-3-small` (OpenAI). Same corpus and faiss_hybrid strategy but different dense leg rankings — chatbot retrieval quality may not match benchmarked 0.575 recall@10.
- Suggested Fix: Document tradeoff; add env toggle for `EMBED_PROVIDER`/`EMBED_MODEL`; re-run eval with OpenAI embeddings for apples-to-apples comparison; or revert chatbot to HuggingFace for eval parity.

---

## app/backend/requirements.txt Missing RAG Pipeline Deps
- Severity: Medium
- Location: `app/backend/requirements.txt`, `src/retrieval/faiss_retriever.py`
- Description: Backend imports `RagPipeline` which needs `langchain_community`, `faiss-cpu`, `rank_bm25`, and (for HugFace path) `sentence-transformers`. Only LangChain/OpenAI packages listed in app requirements. Works only if root experiment deps installed globally.
- Suggested Fix: Add `faiss-cpu`, `langchain-community`, `rank-bm25` to `app/backend/requirements.txt`; document two-step install or unify requirements.

---

## EMBED_PROVIDER Not Documented in .env.example
- Severity: Low
- Location: `app/backend/.env.example`
- Description: `FaissRetriever` reads `EMBED_PROVIDER` (default `openai`) and `EMBED_MODEL` but `.env.example` only lists `OPENAI_API_KEY` and `CHAT_MODEL`.
- Suggested Fix: Add commented `EMBED_PROVIDER=openai` and `EMBED_MODEL=text-embedding-3-small` to `.env.example`.

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

---

## Frontend Ignores Source Citations from API [RESOLVED — Iteration 11]
- Severity: Low → resolved (different UX)
- Location: `app/frontend/src/App.jsx`
- Description: Iteration 10 collapsible EDGAR citation panel replaced by inline footnote superscripts `[N]` and a footnote list per assistant message. Sources still come from API `SourceRef[]`.
- Suggested Fix: Re-add optional EDGAR external links for known 10-K doc ids when `display` maps to a seeded filing.

---

## EDGAR Links Removed from Citation UI [NEW — Iteration 11 regression]
- Severity: Low
- Location: `app/backend/main.py`, `app/frontend/src/App.jsx`
- Description: Iteration 11 simplified `SourceRef` to `{id, doc, section, display}` for generic uploads. SEC browse URLs and ticker-colour badges from Iteration 10 were removed. 10-K answers no longer link to EDGAR.
- Suggested Fix: Restore optional `url` field when doc id matches known CIK in `_COMPANY_META`; render link in footnote row for seeded filings only.

---

## FAISS Index Not Purged on Re-Ingest
- Severity: Medium
- Location: `src/ingestion/ingest_document.py`, `src/retrieval/faiss_retriever.py`
- Description: Re-ingesting a document removes its chunks from `active_corpus.json` and replaces in-memory `pipeline.chunks`, but `FaissRetriever.add_chunks()` only appends embeddings. Old vectors for the same `doc_id` may remain searchable until full index rebuild.
- Suggested Fix: Delete FAISS vectors by doc metadata filter, or rebuild index from corpus on re-ingest; track doc→vector ids.

---

## unstructured Not in app/backend/requirements.txt
- Severity: Medium
- Location: `app/backend/requirements.txt`, `src/ingestion/ingest_document.py`
- Description: `POST /ingest` for PDF/DOCX/HTML depends on `unstructured.partition.auto`. App requirements list only FastAPI/LangChain stack. Plain `.txt`/`.md` uploads work without it; other formats fail or fall back to raw bytes decode.
- Suggested Fix: Add `unstructured[all-docs]` or pinned extras to `app/backend/requirements.txt`; document optional vs required formats in `app/README.md`.

---

## active_corpus.json Grows Unbounded
- Severity: Medium
- Location: `Experiments/corpora/active_corpus.json`, `src/ingestion/ingest_document.py`
- Description: Every upload appends chunks and rewrites the full JSON (~24 MB baseline from merged seed). No delete-document API. File size and startup load time increase with each upload.
- Suggested Fix: Add `DELETE /documents/{id}`; shard corpus by doc; compact JSON; exclude from git or use LFS.

---

## Agent Citation Markers Optional — Fallback Shows Top-3
- Severity: Low
- Location: `app/backend/main.py`, `app/backend/agent.py`
- Description: If the agent omits `[SOURCE:id]` markers, `_build_sources()` returns top-3 retrieval results regardless of whether they supported the answer. May mislead users about which passages were used.
- Suggested Fix: Log fallback rate; tighten prompt; return empty sources when no markers; or require markers in post-processing validation.

---

## Incremental Ingest Rebuilds Full BM25 Index
- Severity: Low
- Location: `src/retrieval/bm25_retriever.py`
- Description: `add_chunks()` retokenizes all chunks and rebuilds `BM25Okapi` on every upload. Acceptable at ~2k chunks; scales O(n) with total corpus size per ingest.
- Suggested Fix: Acceptable for demo scale; consider incremental BM25 or doc-scoped indexes if uploads become frequent.

---

## Test Suite JSON Not Yet Created [RESOLVED — Iteration 14]
- Severity: Low → resolved
- Location: `Experiments/10k_rag_eval_v2.json`, `Experiments/eval_suite_runner.py`
- Description: Stratified eval suite now committed in-repo as v2 (43 questions). Runner defaults to this path. v1 external file (`10k_rag_eval_suite.json`) still used as fallback only.
- Suggested Fix: Move to `eval/test_suite/` for consistency with golden set layout; run v2 baseline and log results.

---

## OpenAI Rate Limits During Parallel Eval Suite [PARTIALLY RESOLVED — Iteration 14]
- Severity: High → Medium
- Location: `Experiments/eval_suite_runner.py`
- Description: Default workers reduced from 3 to **1** to avoid TPM 429 during cross-company eval. First run (213013) failures remain historical; retry/backoff not yet implemented.
- Suggested Fix: Add exponential backoff on 429 in runner or agent; optional `--workers 3` with inter-query delay.

---

## BM25 Retrieves Cross-Document Chunks [RESOLVED — Iteration 15]
- Severity: High → resolved (chatbot path)
- Location: `src/retrieval/bm25_retriever.py`, `app/backend/rag_tool.py`, `app/backend/agent.py`
- Description: Was retrieving unrelated filing chunks for company-scoped questions. `doc_filter` on `search_documents` + agent rule 3b restricts retrieval to one doc prefix. Offline eval harness (`run_eval.py`) still unfiltered unless extended.
- Suggested Fix: Optional `filter_doc` in golden-set eval configs for company-tagged questions.

---

## Test Upload Documents Polluting active_corpus During Eval [RESOLVED — Iteration 15]
- Severity: Medium → resolved
- Location: `Experiments/corpora/active_corpus.json`, `docs_registry.json`
- Description: `tmpedd_eodk` upload removed. Active corpus back to 5 seeded filings, 1893 chunks. v2 Run 2 (`225849`) executed on clean corpus.
- Suggested Fix: Add `DELETE /documents/{id}` API to prevent recurrence; document eval reset procedure.

---

## Single-Fact Prose Regression After Guard Changes [RESOLVED — Iteration 15]
- Severity: Medium → resolved
- Location: `Experiments/runs/eval_suite_20260616_225849.json`
- Description: v2 Run 2 with `doc_filter` + hybrid judge: single-fact prose **24/24 (100%)**, table **20/20 (100%)**. Prior 50% regression (Run 220605 v1 / Run 224259 v2) addressed.
- Suggested Fix: Monitor cross-company category (still weakest at 43.75%).

---

## FaissHybrid RRF Results Still Omit section Metadata
- Severity: Low
- Location: `src/retrieval/faiss_hybrid_retriever.py`
- Description: `BM25Retriever` now returns `section` when filtering, but hybrid fusion builds results from `by_id` with only `{id, doc, text, score}` — `section` dropped before tool output. Citations still show `"—"`.
- Suggested Fix: Include `section: by_id[i].get("section")` in hybrid return payload; propagate through FAISS unfiltered path too.

---

## Cross-Company Comparison Still Weak on v2 Baseline (43.75%)
- Severity: Medium
- Location: `Experiments/runs/eval_suite_20260616_225849.json`, `app/backend/agent.py`
- Description: Best v2 run scores 7/16 (43.75%) on cross-company category despite 79.1% overall. Agent must run unfiltered multi-doc searches and synthesize; doc_filter not applicable.
- Suggested Fix: Prompt explicit per-company search + comparison template; dedicated multi-retrieval tool; evaluate CCC failures in Run 2 JSON.

---

## Eval Judge Methodology Changed Between v2 Runs
- Severity: Low
- Location: `Experiments/eval_suite_runner.py`, `Experiments/runs/eval_suite_20260616_224259.json`, `225849.json`
- Description: Run 1 (224259) used LLM-only judge; Run 2 (225849) adds hybrid exact-match/refusal pre-checks. Prose/table 100% on Run 2 reflects both `doc_filter` and judge changes — not isolated.
- Suggested Fix: Tag run JSON with judge version; re-score Run 1 responses with hybrid judge for apples-to-apples comparison.

---

## eval_ooc_quick Still Points to External v1 Suite Path
- Severity: Low
- Location: `Experiments/eval_ooc_quick.py`
- Description: Hardcoded `C:\Users\athar\Dropbox\PC\Downloads\10k_rag_eval_suite.json` while full runner now defaults to in-repo v2. OOC smoke test out of sync with canonical suite (v2 has 8 OOC questions vs v1's 10).
- Suggested Fix: Point `SUITE` to `Experiments/10k_rag_eval_v2.json`; share path constant with `eval_suite_runner.py`.

---

## XBRL Investee Warnings Require Corpus Rebuild to Take Effect
- Severity: Medium
- Location: `src/ingestion/build_xbrl_corpus.py`, `Experiments/corpora/active_corpus.json`
- Description: Equity-method investee `NOTE:` headers added to xbrl chunk builder but existing `xbrl_merged.json` / `active_corpus.json` / FAISS cache were built before this change. Chatbot may still retrieve unlabeled investee tables.
- Suggested Fix: Re-run `build_xbrl_corpus.py`, `build_merged_corpus.py`, re-seed or rebuild `active_corpus.json`, delete FAISS cache, restart server.

---

## Cross-Company Comparison Questions Fail at 12.5% (First Agent Eval) [OUTDATED — see v2 baseline]
- Severity: High → Medium (historical)
- Location: `Experiments/runs/eval_suite_20260616_213013.json`, `Experiments/runs/eval_suite_20260616_220605.json`
- Description: v1-era runs. v2 Run 2 (`225849`) cross-company category **43.75%** — improved but still weakest. See dedicated open issue above.
- Suggested Fix: Focus on CCC synthesis improvements (Iteration 16).

---

## Out-of-Corpus Refusal Rate Low (30% on First Run) [PARTIALLY RESOLVED — v2 Run 2: 75%]
- Severity: High → Medium
- Location: `Experiments/runs/eval_suite_20260616_225849.json`, `app/backend/agent.py`
- Description: v1 first run 30%; v2 Run 2 **75%** OOC category with entity guards + hybrid refusal judge. Remaining failures need per-question review.
- Suggested Fix: Run `eval_ooc_quick.py` on v2 path after prompt changes; registry-driven block for unknown entities.

---

## search_documents Tool Docstring Says 3 Passages but Returns 5
- Severity: Low
- Location: `app/backend/rag_tool.py`
- Description: Tool docstring claims "Up to 3 reranked passages" but `RERANK_K=5`. Agent and eval expectations may diverge.
- Suggested Fix: Align docstring with `RERANK_K` or reduce `RERANK_K` to 3 if context budget requires it.

---

## Out-of-Corpus Keyword List Is Hardcoded and Incomplete
- Severity: Low
- Location: `app/backend/rag_tool.py` (`_COMPANY_KEYWORDS`)
- Description: Mismatch guard only checks ~12 known out-of-corpus keywords (Tesla, NVIDIA, Amazon, etc.). Questions about other absent companies (e.g., PepsiCo, Deere) won't trigger retrieval warning.
- Suggested Fix: Derive guard from `docs_registry.json` — warn when query entity not in loaded doc ids; remove static keyword map.

---

## llm_reranker Not Listed in Requirements
- Severity: Low
- Location: `src/retrieval/llm_reranker.py`, `app/backend/requirements.txt`
- Description: Reranker imports `openai` at call time (already a backend dep). No separate dep issue, but module is chatbot-only and undocumented in setup docs.
- Suggested Fix: Document rerank step in `app/README.md`; note extra API cost per search call.

---

## Test Upload Documents Polluting active_corpus During Eval [OUTDATED — resolved Iteration 15; see entry above]
- Severity: Medium → resolved
- Location: `Experiments/corpora/active_corpus.json`, `Experiments/runs/eval_suite_20260616_213013.json`
- Description: First eval run cites chunks from uploaded doc `tmpedd_eodk` (e.g., SFP-002 net income, SFT questions) alongside legitimate 10-K chunks. Ad-hoc uploads merged into active corpus pollute retrieval and citation accuracy for benchmark runs.
- Suggested Fix: Reset active corpus to merged baseline before eval; add `DELETE /documents/{id}`; exclude `_upload_` doc ids from eval environment; document clean eval setup in experiment log.

---

## eval_suite_runner Missing Dependencies
- Severity: Low
- Location: `Experiments/eval_suite_runner.py`, `requirements.txt`
- Description: Runner imports `aiohttp` and uses `openai` + `python-dotenv` but is not listed in root or app requirements. Fresh env may fail on `import aiohttp`.
- Suggested Fix: Add `aiohttp` to root requirements or document in experiment log setup steps.
