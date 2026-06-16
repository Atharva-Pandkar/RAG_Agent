# Decisions Log

Single chronological record of architectural and technical decisions. Append new entries — never delete.

---

## Experiment-First Eval Harness
- Date: 2026-06-15
- Context: Benchmark RAG architectures on SEC 10-K filings before building production; need reproducible, comparable runs.
- Decision: YAML configs in `Experiments/configs/` + `eval/harness/run_eval.py` as the single entrypoint. Timestamped JSON per run in `Experiments/runs/`.
- Alternatives Considered: Ad-hoc scripts; notebook-driven eval; MLflow/W&B from day one.
- Impact: All experiments follow config → run → results. Pipelines expose `.retrieve()` and `.generate()`.

---

## Golden Set Design (28 Q&A, 7 Types)
- Date: 2026-06-15
- Context: Fixed benchmark to compare retrieval/generation without re-authoring questions.
- Decision: Canonical `eval/golden_set/golden_set.json` — 28 questions, 5 filings, 7 types (factual_lookup, table_numerical, multi_hop_comparative, risk_factor_synthesis, mda_interpretive, cross_section_fact, unanswerable).
- Alternatives Considered: Auto-generated Q&A; smaller ad-hoc sets; external benchmarks (BEIR).
- Impact: Stratified analysis by type. Gold answers + evidence quotes for traceability.

---

## Corpus-Specific Gold Chunk Mapping
- Date: 2026-06-15
- Context: Retrieval metrics need chunk IDs; chunk boundaries change with strategy/size.
- Decision: Never mutate `golden_set.json`. Run `populate_gold_chunks.py --corpus <path>` → sidecar file referenced via `golden_set_override` in configs.
- Alternatives Considered: Chunk IDs in canonical golden set; manual annotation.
- Impact: Re-chunking requires re-population. Matching uses 60-char anchor substring.

---

## Pipeline Interface (retrieve + generate)
- Date: 2026-06-15
- Context: Eval harness must work with any RAG stack without rewriting the runner.
- Decision: Python classes loaded via `pipeline_class` in config. Methods: `.retrieve(question, k)` and `.generate(question, contexts)`. `RagPipeline` is the reference implementation.
- Alternatives Considered: LangChain/LlamaIndex; function-based registration.
- Impact: New strategies extend or replace `RagPipeline`. Generation via optional `llm_fn`.

---

## Phase 1 Baseline: BM25 + Fixed-Size Chunking
- Date: 2026-06-15
- Context: Establish simplest end-to-end retrieval baseline before embeddings/hybrid/reranking.
- Decision: Run 01 uses fixed-size windows (512 tok, 50 overlap, cl100k_base) + BM25Okapi with alphanumeric lowercase tokenization.
- Alternatives Considered: Dense first; section-aware chunking first; recursive as initial baseline.
- Impact: Floor for recall@k/MRR comparisons. Recursive and section-based added subsequently.

---

## Generation Placeholder When No LLM Configured
- Date: 2026-06-15
- Context: Retrieval experiments should run before API keys/models are chosen.
- Decision: Without `llm_fn`, `.generate()` returns a fixed placeholder. Harness still runs end-to-end; retrieval metrics computed where `gold_chunk_ids` exist.
- Alternatives Considered: Skip generation in retrieval-only mode; require LLM from start.
- Impact: LLM-judge metrics in `metrics.py` remain unused until generation is wired.

---

## BGE-Small as Default Local Embedding Model
- Date: 2026-06-15
- Context: Need a local, no-API-key dense retrieval baseline for Phase 1 grid experiments.
- Decision: Default `embedding_model` in `RagPipeline` is `BAAI/bge-small-en-v1.5` via `sentence-transformers`. Cosine similarity on L2-normalized embeddings.
- Alternatives Considered: OpenAI text-embedding-3-large; Voyage; E5; BGE-large.
- Impact: Runs 07–18 use BGE-small. Swappable per config via `pipeline_kwargs.embedding_model`.

---

## Embedding Cache as NumPy Files
- Date: 2026-06-15
- Context: Re-running evals re-embeds the full corpus each time (~1–2k chunks × multiple configs).
- Decision: Cache embeddings to `Experiments/embeddings/<corpus_stem>__<model>.npy`. Key derived from corpus filename + model name in `RagPipeline`.
- Alternatives Considered: ChromaDB/FAISS index files; re-embed every run; pickle cache.
- Impact: First dense/hybrid run per corpus embeds once; subsequent runs load from disk. Cache must be invalidated manually if corpus or model changes.

---

## Hybrid Retrieval via Reciprocal Rank Fusion (RRF)
- Date: 2026-06-15
- Context: BM25 and dense scores are on incompatible scales; need a simple fusion without score normalization tuning.
- Decision: `HybridRetriever` fetches top-20 from BM25 and dense each, merges via RRF: `score += 1/(rrf_k + rank + 1)` with `rrf_k=60`.
- Alternatives Considered: Weighted linear combination; cross-encoder reranking (deferred to Phase 2); simple score normalization.
- Impact: Runs 08, 10, 12, 14, 16, 18 use hybrid. Best recall so far: Run 16 hybrid fixed 1024 k=10 → 0.598.

---

## Structured Extraction for Section-Aware Chunking
- Date: 2026-06-15
- Context: Flat text extraction loses 10-K structure (Item boundaries, tables, cross-references).
- Decision: `extract_structured.py` produces JSON per filing in `Documents/structured/` with typed blocks (heading, paragraph, table), section tags, Markdown tables, and an internal reference graph.
- Alternatives Considered: Regex section splitting on flat text; SEC EDGAR API structured data; manual section tags.
- Impact: Enables `section_based` chunking. Reference graph reserved for future multi-hop retrieval (Phase 3).

---

## Section-Based Chunking Rules
- Date: 2026-06-15
- Context: Financial Q&A often targets specific Items (Risk Factors, MD&A, Financial Statements) and table data.
- Decision: `section_based_chunks()` packs paragraphs within section boundaries; tables are standalone chunks (never merged with narrative); section headings prepended as `[Heading]` context to first chunk in section (cheap contextual retrieval).
- Alternatives Considered: Parent-child chunking; semantic chunking within sections; table-only corpus.
- Impact: Corpora `section_based_512_50` and `section_based_1024_100`. Runs 09–10, 13–14, 17–18.

---

## PyArrow Import Order Fix (Windows)
- Date: 2026-06-15
- Context: On Windows, importing pyarrow after torch causes DLL access violation via `sentence_transformers` → `datasets` → `pyarrow`.
- Decision: Force `import pyarrow` at top of `dense_retriever.py` before `sentence_transformers`.
- Alternatives Considered: Pin older torch/pyarrow versions; avoid sentence-transformers.
- Impact: Dense/hybrid retrieval works on Windows. Any new module importing both must follow same order.

---

## Cross-Encoder Reranking as Retriever Decorator
- Date: 2026-06-15
- Context: Phase 2 requires reranking without rewriting BM25/dense/hybrid retrievers. Need a composable pattern that fetches a wider candidate pool then re-scores.
- Decision: `RerankRetriever` wraps any base retriever. Fetches `fetch_k` candidates (default 20), scores query–passage pairs with a `CrossEncoder`, returns top-k by `rerank_score`. Enabled via `rerank: true` in `RagPipeline` kwargs.
- Alternatives Considered: Cohere rerank API; inline rerank logic in each retriever; rerank only in eval harness post-step.
- Impact: Runs 19–20. Pipeline wraps the chosen base retriever (BM25 or hybrid) transparently. `rerank_model` and `rerank_fetch_k` configurable per YAML.

---

## ms-marco-MiniLM as Default Cross-Encoder
- Date: 2026-06-15
- Context: Need a local, lightweight cross-encoder for Phase 2 reranking experiments without API keys.
- Decision: Default `rerank_model` is `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence_transformers.CrossEncoder`.
- Alternatives Considered: Cohere rerank v3; BGE reranker (`BAAI/bge-reranker-base`); larger ms-marco models.
- Impact: Runs 19–20. Initial results show reranking **reduces** recall@k on 10-K text vs non-reranked baselines (Run 19: 0.394 vs Run 04: 0.416; Run 20: 0.348 vs Run 12: 0.440). Model is web-search tuned, not financial-domain.

---

## LangChain Chunking Strategies for Comparison
- Date: 2026-06-15
- Context: Need to benchmark LangChain's standard splitters against hand-rolled chunkers on the same token budgets and corpora.
- Decision: Add `langchain_recursive` (RecursiveCharacterTextSplitter + tiktoken length fn) and `langchain_section` (MarkdownHeaderTextSplitter on structured doc → fixed_size token windows per section). Chunk IDs prefixed `_lc_` and `_lcsec_`.
- Alternatives Considered: Replace all custom chunkers with LangChain; use HTMLSemanticPreservingSplitter in production pipeline (deferred — explore script only).
- Impact: Runs 21–31. LangChain recursive underperforms hand-rolled recursive (Run 21: 0.331 vs Run 02: 0.411). LangChain section competitive at 1024 (Run 30: 0.418 recall).

---

## FAISS Vector Store via LangChain
- Date: 2026-06-15
- Context: NumPy dot-product dense retrieval doesn't scale; want persisted indexes and LangChain ecosystem compatibility.
- Decision: `FaissRetriever` uses `langchain_community.vectorstores.FAISS` + `HuggingFaceEmbeddings` (BGE-small). Indexes persisted to `Experiments/embeddings/faiss_{cache_key}/`. New pipeline strategies: `faiss` and `faiss_hybrid` (BM25 + FAISS RRF).
- Alternatives Considered: Keep numpy-only dense; ChromaDB; raw faiss without LangChain wrapper.
- Impact: Runs 22–27, 29, 31. FAISS hybrid matches best recall configs (Run 31: 0.589 vs Run 16: 0.598). Run 29 achieves best MRR (0.474).

---

## Chatbot App Skeleton (FastAPI + LangChain Agent)
- Date: 2026-06-15
- Context: Experiment retrieval stack needs a user-facing demo before production hardening.
- Decision: `app/backend/` — FastAPI with `POST /chat`, LangChain `create_agent` ReAct loop, single tool `search_10k_filings` wrapping `RagPipeline.retrieve()`. `app/frontend/` — React (Vite) with API proxy. Separate `app/backend/requirements.txt`.
- Alternatives Considered: Streamlit-only UI; wire generation through eval harness first; Gradio.
- Impact: End-to-end Q&A with LLM (gpt-4o-mini default via `CHAT_MODEL` env). Retrieval config hardcoded in `rag_tool.py` (hybrid fixed 1024 — not yet updated to Run 31 best config).

---

## HTML Semantic Splitter Exploration (Non-Production)
- Date: 2026-06-15
- Context: Evaluate LangChain's `HTMLSemanticPreservingSplitter` for direct HTML→chunk pipeline vs structured JSON approach.
- Decision: `explore_html_semantic.py` writes inspection JSON to `Documents/semantic_explore/`. Reuses `_table_to_markdown` from `extract_structured.py` as custom table handler. Not wired into `build_corpus.py` or eval runs.
- Alternatives Considered: Adopt as primary chunking path; merge with section_based chunker.
- Impact: Exploratory artifacts only (`aapl_semantic_chunks.json`, etc.). Informs future semantic/HTML chunking decision.

---

## Table Metadata and Reverse Index in Corpora
- Date: 2026-06-15
- Context: Table-heavy 10-K questions need traceability from table blocks to chunks.
- Decision: `section_based_chunks` and `langchain_section_chunks` attach `table_ids`, `has_table`, `section_has_tables` per chunk. `build_corpus.py` builds post-hoc `table_index` mapping table block_id → chunk ids.
- Alternatives Considered: Separate table-only corpus; no metadata (status quo in Phase 1 fixed chunks).
- Impact: Enables future table-targeted retrieval. Not yet used by retrievers or eval metrics.

---

## Unstructured-IO Exploration Pipeline
- Date: 2026-06-15
- Context: Evaluate Unstructured-IO's native HTML partitioning and title-based chunking as a third ingestion path alongside custom structured extraction and LangChain semantic splitting.
- Decision: `explore_unstructured.py` runs `partition_html` → section stamping → `chunk_by_title` (4000 chars, 200 overlap default). Output to `Documents/unstructured_explore/`. Reuses `SECTION_PATTERN` and `_table_to_markdown` from `extract_structured.py` for heading detection and table markdown.
- Alternatives Considered: Adopt as primary corpus builder immediately; merge with `extract_structured.py` block graph; use Unstructured hi_res PDF pipeline.
- Impact: Exploration artifacts for all 5 filings. Not wired into `build_corpus.py` or eval runs. Informs chunking strategy decision for Phase 1.5+.

---

## Section Assignment on Chunk Stream (Unstructured)
- Date: 2026-06-15
- Context: `chunk_by_title` copies Table elements, making `id()`-based matching against the original element list unreliable for section tagging.
- Decision: Detect section headings from each chunk's own text in document order and forward-propagate `current_section` — do not walk `orig_elements` for section assignment.
- Alternatives Considered: Fix upstream in Unstructured; map via element IDs; use only `orig_elements` metadata.
- Impact: Every unstructured chunk gets `section`, plus cross-refs: `section_chunk_ids`, `section_table_ids`, `section_narrative_ids`. Top-level `section_index` and `table_index` built post-hoc.

---

## Three-Way Chunking Exploration (Not Yet Consolidated) [OUTDATED — unstructured promoted Iteration 6]
- Date: 2026-06-15
- Context: Project now has three parallel HTML→chunk exploration paths with no production winner selected.
- Decision: Keep all three as research artifacts until eval-backed comparison:
  1. `extract_structured.py` → custom/section_based/langchain_section (in eval pipeline)
  2. `explore_html_semantic.py` → LangChain HTMLSemanticPreservingSplitter (`Documents/semantic_explore/`)
  3. `explore_unstructured.py` → Unstructured partition + chunk_by_title (`Documents/unstructured_explore/`)
- Alternatives Considered: Pick one path now; merge outputs into single corpus format.
- Impact: Next step is build eval corpora from unstructured/semantic paths and benchmark against Runs 16/31 baselines.

---

## Unstructured-IO Promoted to Eval Corpus
- Date: 2026-06-16
- Context: Iteration 5 exploration showed strong table/section coverage; needed eval-backed comparison against fixed-token and LangChain chunkers.
- Decision: `build_unstructured_corpus.py` produces `Experiments/corpora/unstructured_4000_200.json` (1,492 chunks, 698 table chunks). Table chunks use markdown text; section cross-refs preserved. Golden sidecar: 25/26 answerable matched.
- Alternatives Considered: Adopt semantic HTML path; stay with fixed 1024 only.
- Impact: Runs 32–36. **New best recall:** Run 34/36 (hybrid/faiss_hybrid unstructured k=10) → **0.677**, surpassing Run 16 (0.598).

---

## iXBRL Fact-Level Corpus for ix:continuation Tables
- Date: 2026-06-16
- Context: Unstructured-IO misses financial tables wrapped in `<ix:continuation>` blocks — SEC inline-XBRL mechanism for spanning tagged text across sections.
- Decision: `build_xbrl_corpus.py` parses `ix:nonFraction` facts, groups by HTML `<table>` ancestor, renders Markdown table chunks with GAAP concept labels. Output: per-filing `xbrl_{doc}.json` + `xbrl_merged.json` (401 table chunks).
- Alternatives Considered: Extend Unstructured custom handlers; manual table annotation; SEC API structured data.
- Impact: Recovers numeric financial tables invisible to Unstructured partition. Input to merged corpus.

---

## Merged Corpus: Unstructured + iXBRL with Fingerprint Dedup
- Date: 2026-06-16
- Context: Unstructured excels at narrative; iXBRL excels at continuation-wrapped tables. Need single retrieval index without duplicate table content.
- Decision: `build_merged_corpus.py` concatenates unstructured chunks + non-duplicate xbrl chunks. Dedup: skip xbrl chunk if first 120 chars of normalized text appear in any unstructured chunk. Output: `merged_unstr_xbrl.json` (1,893 chunks, 1,099 table chunks).
- Alternatives Considered: Union without dedup; iXBRL-only for tables; separate indexes per source.
- Impact: Runs 37–40. Gold sidecar achieves **26/26 answerable matched** (first full golden-set coverage). Recall at k=10 (0.575) below unstructured-only (0.677) — deduped xbrl adds coverage but may add noise or dilute ranking.

---

## Production Chunking Winner: Unstructured + FAISS Hybrid k=10
- Date: 2026-06-16
- Context: After Runs 01–40, unstructured corpus with faiss_hybrid at k=10 is the clear retrieval leader.
- Decision: Treat Run 36 (`faiss_hybrid` + `unstructured_4000_200` + k=10) as the production retrieval baseline until superseded. Run 40 (merged corpus) preferred when full gold-chunk coverage (26/26) is required for eval completeness.
- Alternatives Considered: Run 16 fixed 1024 hybrid; Run 31 langchain section; merged corpus as default.
- Impact: Chatbot `rag_tool.py` still on old hybrid fixed 1024 k=5 — needs update to Run 36 config. **[OUTDATED — updated to Run 40 in Iteration 8]**

---

## Multi-Anchor Gold Chunk Matching (`chunk_anchors`)
- Date: 2026-06-16
- Context: Single 60-char evidence anchor failed for table_numerical and multi-value questions where evidence is paraphrased or split across cells (e.g. operating income + total assets in separate table regions).
- Decision: Add optional `chunk_anchors` field to `golden_set.json` — list of substring groups; a chunk matches if **all** substrings in any group appear in normalized chunk text. `populate_gold_chunks.py` tries quote anchors first, then `chunk_anchors`. Sidecars regenerated for all corpora.
- Alternatives Considered: Manual chunk ID annotation; fuzzy token overlap; lower anchor length threshold only.
- Impact: Fixed-size corpora now **26/26** matched (up from 15/26 early iterations). Merged corpus remains 26/26. Unstructured still 25/26 (KO-03 unmatched). Enables fairer eval on table-heavy questions.

---

## Deep Agent for Production Chatbot
- Date: 2026-06-16
- Context: Basic LangChain ReAct loop lacked planning, context management, and strict grounding controls for financial Q&A.
- Decision: Replace `create_agent` with `deepagents.create_deep_agent` in `app/backend/agent.py`. System prompt enforces: search-before-answer, no hallucination, multi-search for complex questions, on-topic only (5 companies), source citation. Single tool: `search_10k_filings`.
- Alternatives Considered: Basic ReAct; custom LangGraph; RAG-only pipeline without agent layer.
- Impact: Agent can plan sub-queries and call retrieval multiple times per turn. Adds `deepagents` to `app/backend/requirements.txt`.

---

## Chatbot Retrieval: Run 40 (Merged FAISS-Hybrid k=10)
- Date: 2026-06-16
- Context: Eval best recall is Run 36 (unstructured, 0.677) but merged corpus has 26/26 gold-chunk coverage and includes iXBRL table recovery. Chatbot needs full filing coverage for demo Q&A.
- Decision: `rag_tool.py` uses Run 40 config: `faiss_hybrid` + `merged_unstr_xbrl.json` + k=10. Tool output includes ranked passages with doc, chunk ID, and section (when available in corpus metadata).
- Alternatives Considered: Run 36 (best recall); env-configurable corpus; dual-index routing (narrative vs table).
- Impact: Chatbot trades ~0.10 recall vs Run 36 for complete table+narrative coverage. First request ~20s (FAISS index load on 1,893-chunk merged corpus).

---

## API Source Extraction and Restricted CORS
- Date: 2026-06-16
- Context: Frontend needed citation traceability; wildcard CORS was flagged as unsafe.
- Decision: `POST /chat` returns `{response, sources[]}` — chunk IDs parsed from `search_10k_filings` ToolMessages via regex. CORS restricted to `localhost:5173` and `localhost:3000`. `python-dotenv` loads `app/backend/.env`.
- Alternatives Considered: Stream SSE with inline citations; keep wildcard CORS for dev simplicity.
- Impact: Superseded by structured `SourceRef[]` (Iteration 10). CORS issue resolved for local dev.

---

## OpenAI Embeddings for Chatbot FAISS (Production Path)
- Date: 2026-06-16
- Context: Local BGE-small via HuggingFace added ~15–20s startup (PyTorch + sentence-transformers) and Windows pyarrow import-order fragility. Chatbot shares the same OpenAI key as the LLM.
- Decision: `FaissRetriever` defaults to `text-embedding-3-small` via `langchain_openai.OpenAIEmbeddings`. Provider switchable via `EMBED_PROVIDER=huggingface` + `EMBED_MODEL`. `rag_tool.py` hardcodes `text-embedding-3-small`; FAISS index cached to `Experiments/embeddings/faiss_{cache_key}/`.
- Alternatives Considered: Keep BGE-small for parity with eval Runs 32–40; lazy HugFace load only; separate embedding API key service.
- Impact: Faster embedding client init (no local model). First cache miss embeds all 1,893 merged chunks via OpenAI API (one-time cost). **Retrieval rankings may differ from eval Run 40** (which used BGE-small). Cache key includes model name — BGE and OpenAI indexes coexist on disk.

---

## Eager Startup Warmup via FastAPI Lifespan
- Date: 2026-06-16
- Context: First `/chat` request previously triggered ~20s cold start (corpus load + FAISS + agent compile). Poor UX for demo.
- Decision: `main.py` uses FastAPI `lifespan` to eagerly call `get_pipeline()`, run a dummy `retrieve("warm-up", k=1)` to force FAISS load, then `build_agent()` before accepting requests. `/health` returns `initialising` until agent is ready; `/chat` returns 503 if not ready. Agent invocation runs in `run_in_executor` to avoid blocking the event loop.
- Alternatives Considered: Lazy init on first request; separate `/warmup` endpoint; background task after startup.
- Impact: Cold-start latency moved to server boot. Users wait at startup instead of first chat turn. Vite proxy timeout raised to 120s for multi-tool agent turns.

---

## Centralized Rotating Backend Logging
- Date: 2026-06-16
- Context: Debugging deep-agent tool calls and retrieval failures required ad-hoc prints; no persistent audit trail.
- Decision: `app/backend/logger.py` — shared `rag.*` logger hierarchy with console (INFO+) and rotating file handler (`logs/rag_backend.log`, 5 MB × 3). All backend modules use `get_logger(__name__)`. HTTP middleware logs request/response timing; pipeline and retrievers log init and retrieve at DEBUG.
- Alternatives Considered: stdlib logging per-module; structlog; LangSmith-only tracing.
- Impact: Full request lifecycle traceable in log file. DEBUG level on file may grow quickly under load — tune for production.

---

## Structured SourceRef with EDGAR Links
- Date: 2026-06-16
- Context: Flat chunk-ID `sources[]` (Iteration 8) was insufficient for a usable citation UI. Frontend needed ticker, company name, section label, and SEC EDGAR link.
- Decision: `SourceRef` Pydantic model on `ChatResponse` — `{id, ticker, company, doc, section, url}`. `_COMPANY_META` maps doc prefix → ticker/name/CIK; `_edgar_url()` builds SEC browse URL. Regex parses tool output lines `[N] Source: <doc> | Section: <section> | ID: <id>`. `_msg_text()` extracts text from list-shaped `AIMessage.content`. Deduped by chunk ID per turn.
- Alternatives Considered: Keep chunk-ID strings; corpus lookup by chunk ID; agent-generated inline markdown links; SSE streaming citations.
- Impact: API contract changed from `string[]` to structured objects. Frontend `Citation` component renders collapsible pills with per-ticker colour and EDGAR links. Section often `"—"` until retrievers pass corpus metadata. Parsing coupled to `rag_tool.py` output format.

---

## React Markdown for Assistant Responses
- Date: 2026-06-16
- Context: Agent answers include lists, bold figures, and table markdown from iXBRL chunks; plain `<p>` rendering lost structure.
- Decision: `react-markdown` + `remark-gfm` in `App.jsx` for assistant message bodies. `index.css` styles headings, lists, code blocks, and GFM tables. User messages remain plain text.
- Alternatives Considered: Raw HTML from agent; custom lightweight formatter; inline citations in markdown body.
- Impact: Richer financial answers in UI. Adds two frontend deps. Table CSS supports retrieved iXBRL markdown if agent echoes it.

---

## Collapsible Citation Panel in Frontend
- Date: 2026-06-16
- Context: k=10 retrieval can produce many sources; showing all inline would clutter the chat.
- Decision: `AssistantMessage` stores `sources` from API; collapsible "▸ N sources" toggle reveals `Citation` pills — ticker badge (colour per company), section/doc label, external link to EDGAR. Sources persisted in message state but stripped from API payload on resend.
- Alternatives Considered: Always-visible source list; inline footnote numbers; hover tooltips only.
- Impact: Source traceability without overwhelming the answer. Section label falls back to `doc` when section is `"—"`. **[OUTDATED — superseded by Iteration 11 inline footnotes; EDGAR pill UI removed.]**

---

## Active Corpus as Mutable Runtime Store
- Date: 2026-06-16
- Context: Chatbot was hardwired to static `merged_unstr_xbrl.json`. Users need to add documents at runtime without rebuilding corpora offline or restarting the server.
- Decision: Introduce `Experiments/corpora/active_corpus.json`. On first boot, `ensure_active_corpus()` copies `merged_unstr_xbrl.json` if the active file is missing. All chatbot retrieval reads/writes this file. Baseline merged corpus remains the eval artifact; active corpus is the live runtime store.
- Alternatives Considered: Env-switch between static corpora; sqlite chunk store; reload pipeline on each ingest (full restart).
- Impact: `rag_tool.py` corpus path is now `active_corpus.json`. FAISS cache key becomes `faiss_active_corpus__text-embedding-3-small`. Git may track a large seeded copy of the merged corpus under the active filename.

---

## Live Document Ingestion Without Server Restart
- Date: 2026-06-16
- Context: Demo and research workflows require uploading PDF/HTML/DOCX/TXT beyond the five seeded 10-K filings.
- Decision: `src/ingestion/ingest_document.py` parses uploads via Unstructured-IO (with text fallback), chunks at 4000/200 chars, appends to `active_corpus.json`, updates `docs_registry.json`, and calls `RagPipeline.add_chunks()` for in-memory BM25 + FAISS updates. FastAPI exposes `POST /ingest` (multipart upload) and `GET /documents` (registry for UI). Startup lifespan calls `ensure_active_corpus()` and bootstraps registry if missing.
- Alternatives Considered: CLI-only ingestion; full pipeline rebuild from disk on each upload; separate vector DB per document.
- Impact: Upload latency includes parse + chunk + JSON rewrite + BM25 rebuild + OpenAI embed for new chunks + FAISS cache save. Re-ingest of same `doc_id` replaces corpus chunks but FAISS may retain stale vectors until index rebuild (see open issue).

---

## Document-Agnostic Deep Agent
- Date: 2026-06-16
- Context: Agent system prompt and tool names referenced five fixed 10-K companies (`search_10k_filings`, on-topic-only list). Uploaded documents would be out of scope.
- Decision: Rename tool to `search_documents`; add `list_available_documents` reading `docs_registry.json`. System prompt requires search-before-answer for any loaded document, no fixed company list. Agent discovers scope via registry when queries are ambiguous.
- Alternatives Considered: Keep 10-K-specific naming; route uploads to a second agent; metadata filter by doc at retrieval time.
- Impact: Chatbot generalizes beyond SEC filings. Eval golden set still targets five 10-Ks; new benchmark needed for upload scenarios.

---

## Inline [SOURCE:id] Citation Protocol
- Date: 2026-06-16
- Context: Iteration 10 showed all k=10 retrieved sources or parsed tool output heuristically. Users need citations tied to claims actually used in the answer, not full retrieval dumps.
- Decision: Agent system prompt requires inline `[SOURCE:chunk_id]` markers after each sourced claim. `main.py` `_build_sources()` maps markers to `SourceRef`, replaces markers with numbered footnotes `[1]`, `[2]`, and returns only cited sources. Fallback: top-3 by retrieval order if agent omits markers.
- Alternatives Considered: Keep collapsible source panel; LLM-generated markdown footnotes only; structured JSON tool return (deferred).
- Impact: Cleaner answers with precise citations. Quality depends on agent compliance with marker format. Frontend renders superscript `[N]` in markdown and a footnote list below each answer.

---

## SourceRef Simplified for Generic Documents
- Date: 2026-06-16
- Context: Iteration 10 `SourceRef` included ticker, company, and SEC EDGAR URLs — 10-K-specific. Uploaded PDFs have no CIK or EDGAR link.
- Decision: `SourceRef` fields reduced to `{id, doc, section, display}` where `display` is `"Company / Section"` from `_doc_display()`. Known 10-K prefixes still map to friendly names (Apple, Walmart, etc.); unknown docs fall back to doc id. EDGAR URLs and ticker badges removed from API and UI.
- Alternatives Considered: Optional EDGAR URL when CIK known; dual citation components for 10-K vs upload.
- Impact: **[OUTDATED partial regression]** SEC deep links no longer in UI. Generic uploads display cleanly. `_doc_display()` hardcodes five ticker→name mappings.

---

## Frontend Document Sidebar and Upload Flow
- Date: 2026-06-16
- Context: Users could not see loaded documents or add new ones from the chat UI.
- Decision: `App.jsx` adds collapsible sidebar: document list from `GET /api/documents`, `+ Add` file picker (`.pdf,.html,.htm,.txt,.md,.docx`), ingest status toast, expandable section lists per doc. Vite proxy routes `/api/*` → backend. App title changed to "Document Research Assistant".
- Alternatives Considered: Separate admin page; drag-and-drop only; ingest via CLI only.
- Impact: `python-multipart` added to backend requirements. Upload uses same 120s proxy timeout as chat (large PDFs may need tuning).
