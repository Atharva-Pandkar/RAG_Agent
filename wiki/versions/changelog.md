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
