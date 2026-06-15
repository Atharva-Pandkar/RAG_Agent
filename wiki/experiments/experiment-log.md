# Experiment Log

Master registry of all RAG experiments. One row per run; append new entries — never delete.

**Legend:** Status = `planned` | `running` | `complete` | `failed` | `superseded`

---

## Run 01 — BM25 + Fixed-Size Chunking (512/50)

| Field | Value |
|-------|-------|
| **Run ID** | `run01_bm25_fixed512` |
| **Date** | 2026-06-15 |
| **Status** | complete |
| **Phase** | 1 |
| **Config** | `Experiments/configs/run01_bm25_fixed512.yaml` |
| **Results** | `Experiments/runs/run01_bm25_fixed512_20260615_132621.json` |

### Setup
| Component | Setting |
|-----------|---------|
| Chunking | `fixed_size`, 512 tokens, 50 overlap (cl100k_base) |
| Corpus | `Experiments/corpora/fixed_size_512_50.json` (1,658 chunks) |
| Retrieval | BM25Okapi, alphanumeric tokenization |
| top_k | 5 |
| Generation | None (placeholder) |
| Golden set | `golden_set_with_chunks_fixed_size_512_50.json` (15/27 answerable matched) |

### Corpus Breakdown
| Document | Chunks |
|----------|--------|
| aapl-20250927 | 110 |
| cat-20251231 | 304 |
| jpm-20251231 | 740 |
| ko-20251231 | 321 |
| wmt-20260131 | 183 |

### Results (retrieval only, n=15 with gold chunks)
| Metric | Value |
|--------|-------|
| Mean recall@5 | 0.444 |
| Mean MRR | 0.317 |
| Mean context precision | 0.107 |
| Perfect recall (recall=1.0) | 6 / 15 |
| Zero recall (recall=0.0) | 8 / 15 |

### Observations
- Cross-document retrieval pollution: many company-specific queries retrieve chunks from wrong filings (e.g. AAPL questions → WMT chunks).
- 13 questions had empty `gold_chunk_ids` (12 unmatched answerable + 1 unanswerable) — metrics not computed for those.
- BM25 alone is a weak baseline on financial 10-K text with fixed token windows; establishes floor for Phase 1 comparisons.

### Next Steps
- Add doc-level filtering by company
- Re-run after gold chunk matching improvements
- Compare against Run 02 (recursive chunking)

---

## Run 02 — BM25 + Recursive Chunking (512/50)

| Field | Value |
|-------|-------|
| **Run ID** | `run02_bm25_recursive512` |
| **Date** | — |
| **Status** | planned |
| **Phase** | 1 |
| **Config** | (not yet created) |

### Planned Setup
| Component | Setting |
|-----------|---------|
| Chunking | `recursive`, 512 tokens, 50 overlap |
| Retrieval | BM25 |
| top_k | 5 |

### Notes
Compare paragraph-aware splitting vs fixed windows on same retrieval backend.

---

## Run 03 — Dense Embeddings Baseline (TBD model)

| Field | Value |
|-------|-------|
| **Run ID** | `run03_dense_*` |
| **Date** | — |
| **Status** | planned |
| **Phase** | 1 |
| **Config** | (not yet created; see `baseline.yaml` template) |

### Planned Setup
| Component | Setting |
|-----------|---------|
| Chunking | fixed_size 512/50 (hold constant for fair comparison) |
| Embedding | TBD: OpenAI text-embedding-3-large / Voyage / BGE / E5 |
| Retrieval | dense (cosine similarity) |
| top_k | 5 |

---

## Experiment Matrix (Planned)

Track coverage across the Phase 1–3 grid. Mark cells when a run completes.

| Chunking ↓ / Retrieval → | BM25 | Dense | Hybrid |
|--------------------------|------|-------|--------|
| fixed_size 512/50 | ✅ Run 01 | planned | planned |
| recursive 512/50 | planned (Run 02) | planned | planned |
| section_based | planned | planned | planned |
| parent_child | planned | planned | planned |

**Reranking (Phase 2):** overlay Cohere / BGE reranker on top-k from best Phase 1 configs.

**Generation (Phase 1+):** wire LLM after retrieval baseline is stable; add correctness/faithfulness via LLM-judge.

---

## How to Add a New Experiment

1. Build corpus: `python src/ingestion/build_corpus.py --strategy <name> --chunk-size N --overlap M`
2. Populate gold chunks: `python eval/golden_set/populate_gold_chunks.py --corpus Experiments/corpora/<corpus>.json`
3. Create config: `Experiments/configs/<run_name>.yaml`
4. Run eval: `python eval/harness/run_eval.py --config Experiments/configs/<run_name>.yaml`
5. Append a new section to this file with setup, results, and observations.
