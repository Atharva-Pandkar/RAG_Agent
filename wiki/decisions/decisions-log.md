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
