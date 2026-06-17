# Changelog

Project-level record of what was added, changed, or removed each iteration.

---

## Iteration 1 — 2026-06-15

### Added
- `src/pipeline.py` — `RagPipeline` glue (corpus load, retriever dispatch, pluggable generation)
- `src/chunking/chunkers.py` — `fixed_size` and `recursive` chunking strategies (tiktoken cl100k_base)
- `src/retrieval/bm25_retriever.py` — BM25Okapi sparse retriever
- `src/ingestion/build_corpus.py` — CLI to build chunked corpora from extracted 10-K text
- `eval/golden_set/populate_gold_chunks.py` — corpus-specific gold chunk ID population
- `Experiments/configs/run01_bm25_fixed512.yaml` — first retrieval experiment config
- `Experiments/corpora/fixed_size_512_50.json` — 1,658 chunks across 5 filings
- `Experiments/corpora/golden_set_with_chunks_fixed_size_512_50.json` — golden set with chunk IDs for run 01 corpus
- `Experiments/runs/run01_bm25_fixed512_20260615_132621.json` — first completed eval run
- `/wiki/` — decisions, changelog, progression log, issues, performance notes, experiment log

### Changed
- `eval/harness/run_eval.py` — supports `golden_set_override` config key; loads pipeline dynamically via `pipeline_class`

### Removed
- (none)

---

## Iteration 2 — 2026-06-15

### Added
- `src/retrieval/dense_retriever.py` — BGE-small dense retrieval with `.npy` embedding cache
- `src/retrieval/hybrid_retriever.py` — BM25 + dense fusion via Reciprocal Rank Fusion (RRF)
- `src/ingestion/extract_structured.py` — structure-preserving 10-K extraction (sections, tables, cross-refs)
- `Documents/structured/` — JSON block graphs for all 5 filings
- `section_based` chunking strategy in `src/chunking/chunkers.py`
- `Experiments/embeddings/` — cached embedding matrices per corpus × model
- `Experiments/summarize_runs.py` — comparison table across run JSON files
- `Experiments/configs/run02`–`run18` — BM25/dense/hybrid grid over chunk sizes, section chunking, and top-k sweeps
- `Experiments/runs/` — completed results for runs 01–17
- `wiki/decisions/decisions-log.md` — single consolidated decisions log

### Changed
- `src/pipeline.py` — supports `dense` and `hybrid` retrieval strategies + `embedding_model` kwarg
- `src/ingestion/build_corpus.py` — section-based strategy reads from `Documents/structured/`
- Gold chunk sidecars — 26/27 answerable questions now matched (up from 15/27)
- Individual decision files marked `[OUTDATED]` → point to `decisions-log.md`

### Removed
- (none)

---

## Iteration 3 — 2026-06-15

### Added
- `src/retrieval/reranker.py` — `RerankRetriever` cross-encoder wrapper (fetch-k → rerank → top-k)
- `Experiments/configs/run19_bm25_fixed1024_rerank.yaml` — BM25 + ms-marco rerank on fixed 1024/100
- `Experiments/configs/run20_hybrid_fixed1024_rerank.yaml` — hybrid + ms-marco rerank on fixed 1024/100
- `Experiments/runs/run18_hybrid_section1024_k10_20260615_161152.json` — Run 18 results
- `Experiments/runs/run19_bm25_fixed1024_rerank_20260615_162232.json` — Run 19 results
- `Experiments/runs/run20_hybrid_fixed1024_rerank_20260615_162326.json` — Run 20 results

### Changed
- `src/pipeline.py` — `rerank`, `rerank_model`, `rerank_fetch_k` kwargs; wraps base retriever in `RerankRetriever` when enabled
- Phase 1 grid complete (Runs 01–18); Phase 2 reranking experiments started (Runs 19–20)

### Removed
- (none)

---

## Iteration 4 — 2026-06-15

### Added
- `src/retrieval/faiss_retriever.py` — LangChain FAISS + HuggingFaceEmbeddings dense retrieval with disk cache
- `src/retrieval/faiss_hybrid_retriever.py` — BM25 + FAISS RRF hybrid
- `langchain_recursive` and `langchain_section` chunking strategies in `src/chunking/chunkers.py`
- `src/ingestion/explore_html_semantic.py` — HTMLSemanticPreservingSplitter exploration script
- `Documents/semantic_explore/` — per-filing semantic chunk inspection JSON
- `Experiments/corpora/langchain_*.json` — LangChain chunker corpora (512/1024 variants)
- `Experiments/embeddings/faiss_*/` — persisted FAISS indexes per corpus × model
- `Experiments/configs/run21`–`run31` — LangChain chunking × BM25/FAISS/FAISS-hybrid grid
- `Experiments/runs/` — results for runs 21–31
- `app/backend/` — FastAPI chat API, LangChain ReAct agent, `search_10k_filings` RAG tool
- `app/frontend/` — React (Vite) chat UI skeleton
- `app/README.md` — full-stack setup instructions

### Changed
- `src/pipeline.py` — `faiss` and `faiss_hybrid` retrieval strategies
- `src/ingestion/build_corpus.py` — supports `langchain_section`; emits `table_index` in corpus JSON
- `section_based_chunks` — table metadata fields (`table_ids`, `has_table`, `section_has_tables`)
- Best recall nearly matched: Run 31 (FAISS hybrid + langchain section 1024, k=10) → **0.589** vs Run 16 → 0.598

### Removed
- (none)

---

## Iteration 5 — 2026-06-15

### Added
- `src/ingestion/explore_unstructured.py` — Unstructured-IO `partition_html` + `chunk_by_title` exploration with section cross-refs
- `Documents/unstructured_explore/` — inspection JSON for all 5 filings (aapl, cat, jpm, ko, wmt)
- Section-level indexes in unstructured output: `section_index`, `table_index`, per-chunk `section_*_ids` cross-refs

### Changed
- `Documents/semantic_explore/` — all 5 company semantic chunk files present (complete set)
- Chunking research now spans three exploration paths (structured, semantic, unstructured); eval pipeline unchanged

### Removed
- (none)

---

## Iteration 6 — 2026-06-16

### Added
- `src/ingestion/build_unstructured_corpus.py` — production Unstructured-IO corpus builder
- `src/ingestion/build_xbrl_corpus.py` — iXBRL fact-level table corpus from `ix:nonFraction` tags
- `src/ingestion/build_merged_corpus.py` — unstructured + iXBRL merge with 120-char fingerprint dedup
- `Experiments/corpora/unstructured_4000_200.json` — 1,492 chunks (698 table)
- `Experiments/corpora/xbrl_merged.json` — 401 iXBRL table chunks
- `Experiments/corpora/merged_unstr_xbrl.json` — 1,893 merged chunks (1,099 table)
- Golden sidecars for unstructured (25/26) and merged (26/26 answerable matched)
- `Experiments/configs/run32`–`run40` — unstructured + merged corpus eval grid
- `Experiments/runs/` — results for runs 32–40

### Changed
- **New best recall:** Run 34/36 → **0.677** (faiss_hybrid/hybrid unstructured k=10), up from Run 16 → 0.598
- Merged corpus achieves first **26/26** gold-chunk match rate but recall (0.575 at k=10) trails unstructured-only
- Unstructured path promoted from exploration to primary eval corpus

### Removed
- (none)

---

## Iteration 7 — 2026-06-16

### Added
- `chunk_anchors` field in `eval/golden_set/golden_set.json` — multi-substring gold chunk matching for 8 table/numerical questions
- `Experiments/configs/run41_hybrid_merged_k5.yaml` — hybrid merged corpus k=5 (mirrors Run 33 on merged)
- `Experiments/runs/run41_hybrid_merged_k5_20260616_031856.json`

### Changed
- `eval/golden_set/populate_gold_chunks.py` — supports `chunk_anchors` (AND matching within anchor groups)
- All golden sidecars regenerated; fixed-size corpora now **26/26** gold-chunk match
- Run 41: hybrid merged k=5 → recall 0.478, MRR 0.379 (same ballpark as Run 39 faiss_hybrid merged k=5)
- Production baseline unchanged: Run 36 recall **0.677**

### Removed
- (none)

---

## Iteration 8 — 2026-06-16

### Added
- `deepagents` integration — `create_deep_agent` with strict RAG grounding prompt
- `app/backend/.env.example` — `OPENAI_API_KEY`, `CHAT_MODEL` template
- `POST /chat` `sources` field — chunk IDs extracted from tool messages

### Changed
- `app/backend/rag_tool.py` — Run 40 config (faiss_hybrid + merged_unstr_xbrl, k=10); richer tool docstring and formatted output
- `app/backend/agent.py` — deep agent with 8-rule grounding system prompt
- `app/backend/main.py` — dotenv load, restricted CORS, source extraction, robust final-message parsing
- `app/backend/requirements.txt` — added `deepagents`, `python-dotenv`; removed `langgraph`

### Removed
- LangChain basic `create_agent` ReAct loop (replaced by deep agent)

---

## Iteration 9 — 2026-06-16

### Added
- `app/backend/logger.py` — centralized rotating file + console logging (`rag.*` hierarchy)
- FastAPI `lifespan` startup warmup — pipeline + dummy retrieve + deep agent compiled before first request
- HTTP request/response logging middleware in `main.py`
- OpenAI embedding path in `src/retrieval/faiss_retriever.py` — `EMBED_PROVIDER` env (`openai` | `huggingface`)
- `app/frontend/package-lock.json` — locked dependency tree after first `npm install`
- Vite dev proxy 120s timeout for long deep-agent turns (`vite.config.js`)

### Changed
- `app/backend/rag_tool.py` — embedding model switched to `text-embedding-3-small` (OpenAI); tool and pipeline init logging
- `app/backend/main.py` — eager startup init, async agent via `run_in_executor`, `/health` reports `initialising` vs `ok`, 503 if agent not ready
- `src/retrieval/faiss_retriever.py` — OpenAI default; lazy FAISS load on cache hit; module-level embedding client singleton
- `src/pipeline.py`, `faiss_hybrid_retriever.py` — structured init/retrieve logging
- Frontend build workflow — requires `npm install` before `npm run build` (Vite not globally installed)

### Removed
- Per-request lazy agent/pipeline init on first `/chat` (moved to startup lifespan)

---

## Iteration 10 — 2026-06-16

### Added
- `SourceRef` Pydantic model — structured citations with ticker, company, doc, section, EDGAR URL
- `_COMPANY_META` / `_edgar_url()` / `_doc_to_meta()` in `main.py` — company → SEC link mapping
- `_msg_text()` helper — extracts text blocks from list-shaped `AIMessage.content` (tool-use responses)
- Frontend `Citation` and `AssistantMessage` components — collapsible source panel with per-ticker colour badges
- `react-markdown` + `remark-gfm` — GFM markdown rendering for assistant answers (tables, lists, bold)
- Expanded `index.css` — markdown body styles, citation pills, loading dots, subtitle

### Changed
- `POST /chat` response — `sources` upgraded from `string[]` to `SourceRef[]`; regex parses tool output for doc/section/id
- `app/frontend/src/App.jsx` — stores `sources` per assistant message; sends only `{role, content}` to API
- `app/frontend/package.json` — added `react-markdown`, `remark-gfm`

### Removed
- Flat chunk-ID-only source extraction regex (`ID:\s*([\w\-_]+)`)

---

## Iteration 11 — 2026-06-16

### Added
- `src/ingestion/ingest_document.py` — universal ingest (PDF/HTML/DOCX/TXT/MD via Unstructured-IO); CLI + programmatic `ingest_file()`
- `Experiments/corpora/active_corpus.json` — mutable runtime corpus (seeded from `merged_unstr_xbrl.json` on first boot)
- `Experiments/corpora/docs_registry.json` — document metadata registry (display name, chunk count, sections, ingest timestamp)
- `Experiments/embeddings/faiss_active_corpus__text-embedding-3-small/` — FAISS cache for active corpus + OpenAI embeddings
- `POST /ingest` — multipart document upload; live corpus + index update
- `GET /documents` — registry JSON for frontend sidebar
- `list_available_documents` agent tool — scope discovery from registry
- `RagPipeline.add_chunks()` — in-memory retriever extension without restart
- `BM25Retriever.add_chunks()`, `FaissRetriever.add_chunks()`, `FaissHybridRetriever.add_chunks()` — incremental index updates
- Frontend document sidebar — upload button, doc list, section expanders, ingest status toasts
- Inline citation protocol — agent emits `[SOURCE:chunk_id]`; API returns footnote-numbered response + cited `SourceRef[]`
- `python-multipart` in `app/backend/requirements.txt`

### Changed
- `app/backend/rag_tool.py` — corpus → `active_corpus.json`; tool renamed `search_10k_filings` → `search_documents`
- `app/backend/agent.py` — document-agnostic system prompt; tools `[search_documents, list_available_documents]`
- `app/backend/main.py` — lifespan seeds active corpus + registry; `_build_sources()` footnote resolution; `SourceRef` → `{id, doc, section, display}` (no ticker/EDGAR)
- `app/frontend/src/App.jsx` — sidebar layout, footnote UI, superscript citation marks in markdown; removed EDGAR pill citations
- `app/frontend/src/index.css` — sidebar, upload, footnote, app-shell styles
- `app/backend/.env.example` — documents `EMBED_PROVIDER=openai`
- Retrievers — `section` still not returned in retrieve payloads (unchanged gap)

### Removed
- Hardcoded five-company scope in agent prompt
- `SourceRef.ticker`, `SourceRef.company`, `SourceRef.url` and EDGAR link UI from Iteration 10
- Chatbot dependency on static `merged_unstr_xbrl.json` path at runtime (eval artifact retained; active corpus is runtime source)
- Local BGE FAISS index files from working tree (replaced by OpenAI embedding caches for active/merged corpora)

---

## Iteration 12 — 2026-06-16

### Added
- `Experiments/eval_suite_runner.py` — end-to-end agent eval: async `/chat` queries + GPT-4o-mini LLM judge + category rubrics
- `Experiments/runs/eval_suite_20260616_213013.json` — first full suite run (47 questions, 57.4% score)
- Eval categories in runner: `single_fact_prose`, `single_fact_table`, `cross_company_comparison`, `out_of_corpus`, `adversarial` (0–2 rubrics each)
- CLI flags: `--suite`, `--url`, `--model`, `--workers`, `--out`, `--skip-ooc`

### Changed
- Benchmark scope expanded from retrieval-only (`run_eval.py`) to full agent + generation path
- Known failure modes documented from first run: cross-company 12.5%, out-of-corpus 30%, prose 92.9%, table 70%

### Removed
- (none)

---

## Iteration 13 — 2026-06-16

### Added
- `src/retrieval/llm_reranker.py` — GPT snippet-based rerank over retrieval candidates
- `Experiments/eval_ooc_quick.py` — fast 10-question out-of-corpus smoke test against live `/chat`
- `Experiments/runs/eval_suite_20260616_220605.json` — second full suite run (59.6% overall)
- Agent rules: entity verification via `list_available_documents`, source-doc matching, trap/ambiguity flagging
- Retrieval mismatch warning in `search_documents` for known out-of-corpus company keywords

### Changed
- `app/backend/rag_tool.py` — fetch k=10 → LLM rerank to k=5 before returning passages; mismatch guard appended to tool output
- `app/backend/agent.py` — expanded system prompt (10 rules): verify entity, verify passages, refuse templates, document-scope numbers
- Agent eval scores (Run 2 vs Run 1): cross-company **68.75%** (was 12.5%), out-of-corpus **70%** (was 30%), overall **59.6%** (was 57.4%)
- Regression: single-fact prose **50%** (was 92.9%) — likely over-refusal or judge strictness after guard changes

### Removed
- Passing all 10 raw retrieval chunks directly to agent (replaced by rerank-to-5 path)

---

## Iteration 14 — 2026-06-16

### Added
- `Experiments/10k_rag_eval_v2.json` — in-repo eval suite v2 (43 corpus-verified questions, cleaner OOC/adversarial design)
- XBRL equity-method investee warning header in `build_xbrl_corpus.py` (`_INVESTEE_SEGMENT_PATTERNS`)

### Changed
- `Experiments/eval_suite_runner.py` — default suite → `Experiments/10k_rag_eval_v2.json`; default `--workers` **3 → 1** (429 mitigation)
- Eval benchmark methodology: v2 replaces external v1 as canonical suite (v1 runs retained for history only)

### Removed
- (none — v1 suite file not deleted; runner fallbacks to Downloads v1 still exist)

---

## Iteration 15 — 2026-06-16

### Added
- `filter_doc` parameter on `RagPipeline.retrieve()`, BM25/FAISS/hybrid retrievers, and `search_documents` tool
- Agent prompt rule **3b** — use `doc_filter` for single-company questions
- `list_available_documents` output includes `(doc_id: …)` per document
- Hybrid eval judge in `eval_suite_runner.py` — numeric exact-match (±2.5%) for SFP/SFT; refusal regex for OOC
- `Experiments/runs/eval_suite_20260616_224259.json` — v2 Run 1 (52.3% overall)
- `Experiments/runs/eval_suite_20260616_225849.json` — v2 Run 2 (79.1% overall, production baseline)

### Changed
- `BM25Retriever.retrieve()` — optional doc prefix filter; returns `section` in result dict
- `FaissRetriever.retrieve()` — over-fetch + post-filter when `filter_doc` set
- `FaissHybridRetriever` — passes `filter_doc` to both legs before RRF
- `active_corpus.json` / `docs_registry.json` — cleaned to 5 baseline filings (1893 chunks); `tmpedd_eodk` removed

### Removed
- Test upload doc `tmpedd_eodk` from active corpus and registry

---

## Iteration 16 — 2026-06-17

### Added
- `Experiments/eval_ragas.py` — RAGAS-style four-metric eval (context recall/precision, faithfulness, answer correctness) on live `/chat` + v2 suite
- `Experiments/eval_reranker.py` — offline reranker isolation benchmark (Recall@10→5, MRR, drop rate) on SFP/SFT questions
- `Experiments/runs/ragas_eval_1781652528.json` — RAGAS pilot (n=18, composite 0.299; pre-corpus-lookup fix)
- `Experiments/runs/ragas_eval_1781653924.json` — RAGAS SFP+SFT only (n=22, composite 0.701)
- `Experiments/runs/ragas_eval_1781654954.json` — RAGAS full v2 suite (n=43, composite 0.613)
- `Experiments/runs/reranker_eval_1781652118.json` — reranker isolation (n=22, recall 72.7%→40.9%)

### Changed
- Eval tooling expanded beyond binary agent judge — stratified RAG quality dimensions now measurable on production stack

### Removed
- (none)
