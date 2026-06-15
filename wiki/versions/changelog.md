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
