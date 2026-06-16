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

## Three-Way Chunking Exploration (Not Yet Consolidated)
- Date: 2026-06-15
- Context: Project now has three parallel HTML→chunk exploration paths with no production winner selected.
- Decision: Keep all three as research artifacts until eval-backed comparison:
  1. `extract_structured.py` → custom/section_based/langchain_section (in eval pipeline)
  2. `explore_html_semantic.py` → LangChain HTMLSemanticPreservingSplitter (`Documents/semantic_explore/`)
  3. `explore_unstructured.py` → Unstructured partition + chunk_by_title (`Documents/unstructured_explore/`)
- Alternatives Considered: Pick one path now; merge outputs into single corpus format.
- Impact: Next step is build eval corpora from unstructured/semantic paths and benchmark against Runs 16/31 baselines.
