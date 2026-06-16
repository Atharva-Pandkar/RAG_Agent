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
| Golden set | `golden_set_with_chunks_fixed_size_512_50.json` (26/27 answerable matched on latest sidecar) |

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
- Compare against Run 02 (recursive chunking)

---

## Phase 1 Grid Summary (Runs 01–18, latest results, n=26)

All metrics: mean over questions with matched `gold_chunk_ids`. Generation: placeholder only.

| Run | Retrieval | Chunking | k | Recall@k | MRR | Ctx Prec |
|-----|-----------|----------|---|----------|-----|----------|
| 01 | bm25 | fixed 512/50 | 5 | 0.415 | 0.321 | 0.115 |
| 02 | bm25 | recursive 512/50 | 5 | 0.411 | 0.347 | 0.131 |
| 03 | bm25 | fixed 256/25 | 5 | 0.337 | 0.292 | 0.100 |
| 04 | bm25 | fixed 1024/100 | 5 | 0.416 | **0.397** | 0.154 |
| 05 | bm25 | fixed 512/50 | 3 | 0.338 | 0.301 | **0.167** |
| 06 | bm25 | fixed 512/50 | 10 | 0.572 | 0.347 | 0.085 |
| 07 | dense | fixed 512/50 | 5 | 0.322 | 0.277 | 0.146 |
| 08 | hybrid | fixed 512/50 | 5 | 0.488 | 0.274 | 0.154 |
| 09 | bm25 | section 512/50 | 5 | 0.340 | 0.280 | 0.092 |
| 10 | hybrid | section 512/50 | 5 | 0.367 | 0.292 | 0.100 |
| 11 | dense | fixed 1024/100 | 5 | 0.373 | 0.272 | **0.192** |
| 12 | hybrid | fixed 1024/100 | 5 | 0.440 | 0.360 | 0.177 |
| 13 | bm25 | section 1024/100 | 5 | 0.453 | 0.328 | 0.115 |
| 14 | hybrid | section 1024/100 | 5 | 0.405 | 0.328 | 0.115 |
| 15 | dense | fixed 1024/100 | 10 | 0.408 | 0.278 | 0.115 |
| 16 | hybrid | fixed 1024/100 | 10 | **0.598** | 0.379 | 0.131 |
| 17 | bm25 | section 1024/100 | 10 | 0.504 | 0.334 | 0.065 |
| 18 | hybrid | section 1024/100 | 10 | 0.528 | 0.348 | 0.073 |

**Key findings:**
- Hybrid + larger chunks + higher k wins on recall (Run 16: 0.598)
- Section hybrid at k=10 (Run 18: 0.528) underperforms fixed hybrid (Run 16)
- BM25 alone benefits strongly from k=10 (Run 06: 0.572) but context precision drops
- Dense alone underperforms BM25/hybrid on recall at this corpus size
- Section-based chunking did not beat fixed 1024 for BM25 (Run 13 vs 04)
- Cross-document retrieval pollution still affects all runs (no doc filter)

---

## Phase 2 Reranking Summary (Runs 19–20, n=26)

| Run | Base | vs Baseline | k | Recall@k | MRR | Ctx Prec | Δ Recall |
|-----|------|-------------|---|----------|-----|----------|----------|
| 19 | bm25 + rerank | Run 04 | 5 | 0.394 | 0.277 | 0.154 | −0.022 |
| 20 | hybrid + rerank | Run 12 | 5 | 0.348 | 0.310 | 0.169 | −0.092 |

Rerank config: `cross-encoder/ms-marco-MiniLM-L-6-v2`, fetch_k=20 → top-k.

**Key findings:**
- ms-marco reranking **hurts recall** on 10-K financial text at k=5
- Run 20 MRR slightly below Run 12 (0.310 vs 0.360); context precision comparable
- Run 16 (hybrid k=10, no rerank) remains overall best at 0.598 recall
- Next: try BGE reranker or rerank on Run 16's k=10 candidate pool

---

## Run 02 — BM25 + Recursive Chunking (512/50)

| Field | Value |
|-------|-------|
| **Run ID** | `run02_bm25_recursive512` |
| **Date** | 2026-06-15 |
| **Status** | complete |
| **Config** | `Experiments/configs/run02_bm25_recursive512.yaml` |
| **Results** | `Experiments/runs/run02_bm25_recursive512_20260615_153955.json` |

| Metric (n=26) | Value |
|--------|-------|
| Recall@5 | 0.411 |
| MRR | 0.347 |
| Context precision | 0.131 |

Slight MRR improvement over Run 01; recall comparable.

---

## Runs 03–06 — BM25 Chunk Size & Top-k Sweeps

| Run | Variation | Recall@k | MRR | Notes |
|-----|-----------|----------|-----|-------|
| 03 | 256/25 chunks | 0.337 | 0.292 | Smallest chunks hurt recall |
| 04 | 1024/100 chunks | 0.416 | 0.397 | Best BM25 MRR at k=5 |
| 05 | k=3 | 0.338 | 0.301 | Highest ctx precision (0.167) |
| 06 | k=10 | 0.572 | 0.347 | Highest BM25 recall |

---

## Runs 07–08 — Dense & Hybrid (Fixed 512/50)

| Run | Strategy | Recall@5 | MRR |
|-----|----------|----------|-----|
| 07 | dense (BGE-small) | 0.322 | 0.277 |
| 08 | hybrid (RRF) | 0.488 | 0.274 |

Hybrid clearly beats dense-only; still no doc filtering.

---

## Runs 09–10 — Section-Based Chunking (512/50)

| Run | Strategy | Recall@5 | MRR |
|-----|----------|----------|-----|
| 09 | bm25 + section | 0.340 | 0.280 |
| 10 | hybrid + section | 0.367 | 0.292 |

Section chunking at 512 did not improve over fixed 512 (Run 01/08).

---

## Runs 11–12 — Dense & Hybrid (Fixed 1024/100)

| Run | Strategy | Recall@5 | MRR | Ctx Prec |
|-----|----------|----------|-----|----------|
| 11 | dense | 0.373 | 0.272 | 0.192 |
| 12 | hybrid | 0.440 | 0.360 | 0.177 |

Larger chunks help both; hybrid wins on recall and MRR.

---

## Runs 13–14 — Section-Based Chunking (1024/100)

| Run | Strategy | Recall@5 | MRR |
|-----|----------|----------|-----|
| 13 | bm25 + section | 0.453 | 0.328 |
| 14 | hybrid + section | 0.405 | 0.328 |

Section 1024 BM25 (Run 13) is best section config; hybrid section underperforms hybrid fixed (Run 12).

---

## Runs 15–17 — Top-k=10 Sweeps (1024/100)

| Run | Strategy | Chunking | Recall@k | MRR |
|-----|----------|----------|----------|-----|
| 15 | dense | fixed | 0.408 | 0.278 |
| 16 | hybrid | fixed | **0.598** | 0.379 |
| 17 | bm25 | section | 0.504 | 0.334 |

**Run 16 is the current best overall config.**

---

## Run 18 — Hybrid + Section 1024, k=10

| Field | Value |
|-------|-------|
| **Run ID** | `run18_hybrid_section1024_k10` |
| **Date** | 2026-06-15 |
| **Status** | complete |
| **Config** | `Experiments/configs/run18_hybrid_section1024_k10.yaml` |
| **Results** | `Experiments/runs/run18_hybrid_section1024_k10_20260615_161152.json` |

| Metric (n=26) | Value |
|--------|-------|
| Recall@10 | 0.528 |
| MRR | 0.348 |
| Context precision | 0.073 |

Section hybrid at k=10 improves over Run 14 (k=5: 0.405) but does not beat Run 16 fixed hybrid (0.598).

---

## Run 19 — BM25 + Cross-Encoder Rerank (Fixed 1024/100)

| Field | Value |
|-------|-------|
| **Run ID** | `run19_bm25_fixed1024_rerank` |
| **Date** | 2026-06-15 |
| **Status** | complete |
| **Config** | `Experiments/configs/run19_bm25_fixed1024_rerank.yaml` |
| **Results** | `Experiments/runs/run19_bm25_fixed1024_rerank_20260615_162232.json` |

| Component | Setting |
|-----------|---------|
| Retrieval | BM25 → fetch 20 → ms-marco rerank → top 5 |
| Baseline | Run 04 (same corpus, no rerank) |

| Metric (n=26) | Run 19 | Run 04 (baseline) |
|--------|--------|-------------------|
| Recall@5 | 0.394 | 0.416 |
| MRR | 0.277 | 0.397 |

Reranking reduced recall and MRR vs BM25-only baseline.

---

## Run 20 — Hybrid + Cross-Encoder Rerank (Fixed 1024/100)

| Field | Value |
|-------|-------|
| **Run ID** | `run20_hybrid_fixed1024_rerank` |
| **Date** | 2026-06-15 |
| **Status** | complete |
| **Config** | `Experiments/configs/run20_hybrid_fixed1024_rerank.yaml` |
| **Results** | `Experiments/runs/run20_hybrid_fixed1024_rerank_20260615_162326.json` |

| Component | Setting |
|-----------|---------|
| Retrieval | Hybrid RRF → fetch 20 → ms-marco rerank → top 5 |
| Baseline | Run 12 (same corpus, no rerank) |

| Metric (n=26) | Run 20 | Run 12 (baseline) |
|--------|--------|-------------------|
| Recall@5 | 0.348 | 0.440 |
| MRR | 0.310 | 0.360 |
| Context precision | 0.169 | 0.177 |

Largest recall regression in Phase 2 so far. Hybrid+rerank at k=5 underperforms hybrid alone.

---

## Experiment Matrix (Updated)

| Chunking ↓ / Retrieval → | BM25 | Dense | Hybrid |
|--------------------------|------|-------|--------|
| fixed 256/25 | ✅ Run 03 | — | — |
| fixed 512/50 | ✅ Run 01, 05–06 | ✅ Run 07 | ✅ Run 08 |
| fixed 1024/100 | ✅ Run 04 | ✅ Run 11, 15 | ✅ Run 12, 16 |
| recursive 512/50 | ✅ Run 02 | — | — |
| section 512/50 | ✅ Run 09 | — | ✅ Run 10 |
| section 1024/100 | ✅ Run 13, 17 | — | ✅ Run 14, ✅ Run 18 |
| parent_child | — | — | — |

**Reranking overlay (Phase 2, fixed 1024/100, k=5):**

| Base | Run | Recall@5 | vs No-Rerank |
|------|-----|----------|--------------|
| BM25 (Run 04) | 19 + ms-marco | 0.394 | −0.022 |
| Hybrid (Run 12) | 20 + ms-marco | 0.348 | −0.092 |

**Overall best:** Run 16 — hybrid fixed 1024/100, k=10 → recall **0.598**. Run 31 (FAISS hybrid + langchain section 1024, k=10) → **0.589** (close second).

**Generation (Phase 1+):** eval harness still placeholder; chatbot app has LLM via LangChain agent.

---

## LangChain + FAISS Grid Summary (Runs 21–31, n=26)

| Run | Retrieval | Chunking | k | Recall@k | MRR | Ctx Prec |
|-----|-----------|----------|---|----------|-----|----------|
| 21 | bm25 | lc recursive 512 | 5 | 0.331 | 0.311 | 0.092 |
| 22 | faiss | lc recursive 512 | 5 | 0.308 | 0.299 | 0.146 |
| 23 | faiss_hybrid | lc recursive 512 | 5 | 0.415 | 0.279 | 0.138 |
| 24 | bm25 | lc recursive 1024 | 5 | 0.407 | 0.376 | 0.131 |
| 25 | faiss | lc recursive 1024 | 5 | 0.337 | 0.314 | 0.146 |
| 26 | faiss_hybrid | lc recursive 1024 | 5 | 0.436 | 0.346 | 0.146 |
| 27 | faiss_hybrid | lc recursive 1024 | 10 | 0.587 | 0.374 | 0.112 |
| 28 | bm25 | lc section 512 | 5 | 0.404 | 0.292 | 0.115 |
| 29 | faiss_hybrid | lc section 512 | 5 | 0.416 | **0.474** | 0.123 |
| 30 | bm25 | lc section 1024 | 5 | 0.418 | 0.385 | 0.138 |
| 31 | faiss_hybrid | lc section 1024 | 10 | **0.589** | 0.365 | 0.119 |

**Key findings:**
- FAISS hybrid + langchain section 1024 k=10 (Run 31) nearly matches Run 16 on recall
- Run 29 (faiss_hybrid lc section 512 k=5) achieves best MRR overall (0.474)
- LangChain recursive underperforms hand-rolled recursive at same token budget
- FAISS and numpy hybrid paths produce comparable top configs when chunking matches

---

## Experiment Matrix (Updated)

| Chunking ↓ / Retrieval → | BM25 | Dense/FAISS | Hybrid (numpy) | FAISS Hybrid |
|--------------------------|------|---------------|----------------|--------------|
| fixed 1024/100 | ✅ 04 | ✅ 11 (dense) | ✅ 12, **16** | — |
| hand recursive 512 | ✅ 02 | — | — | — |
| lc recursive 512 | ✅ 21 | ✅ 22 (faiss) | — | ✅ 23 |
| lc recursive 1024 | ✅ 24 | ✅ 25 (faiss) | — | ✅ 26, 27 |
| lc section 512 | ✅ 28 | — | — | ✅ 29 |
| lc section 1024 | ✅ 30 | — | — | ✅ **31** |
| section_based (custom) | ✅ 09, 13, 17 | — | ✅ 10, 14, 18 | — |

**Reranking overlay (Phase 2):** Runs 19–20 on fixed 1024; negative recall impact.

**App demo:** `app/backend/rag_tool.py` — hybrid fixed 1024 k=5 (suboptimal vs Run 31).

**Generation:** Chatbot uses LangChain agent + OpenAI; eval harness still retrieval-only.

---

## Chunking Exploration Comparison (Pre-Eval)

Side-by-side inspection outputs for AAPL (`aapl-20250927.html`). Not yet benchmarked on golden set.

| Path | Script | Chunks | Table Chunks | Sections | Avg Tokens | In Eval? |
|------|--------|--------|--------------|----------|------------|----------|
| Structured → fixed 1024 | `build_corpus.py` | ~110 (fixed_size) | — | — | 1024 window | ✅ Runs 04, 16 |
| LangChain semantic | `explore_html_semantic.py` | 112 (50 shown) | 0 detected | — | ~497 | ❌ |
| Unstructured-IO | `explore_unstructured.py` | 89 | 29 | 12 | ~453 | ❌ |

**All-filings unstructured stats:**

| Filing | Elements | Chunks | Table Chunks | Sections |
|--------|----------|--------|--------------|----------|
| AAPL | 540 | 89 | 29 | 12 |
| CAT | 955 | 145 | 55 | 12 |
| JPM | 2,853 | 890 | 486 | 49 |
| KO | 916 | 183 | 57 | 13 |
| WMT | 682 | 185 | 71 | 29 |

**Observations:**
- Unstructured produces fewer, larger chunks than fixed-token windows for AAPL (89 vs ~110)
- Unstructured explicitly links table chunks to narrative siblings within same section
- Semantic explore table detection appears broken (0 table chunks for AAPL)
- JPM unstructured granularity (890 chunks) may inflate retrieval index size vs fixed 1024 corpus

**Next:** Convert unstructured/semantic outputs to `Experiments/corpora/` format and run BM25/FAISS-hybrid eval (proposed Runs 32+).

---

## How to Add a New Experiment

1. Build corpus: `python src/ingestion/build_corpus.py --strategy <name> --chunk-size N --overlap M`
2. Populate gold chunks: `python eval/golden_set/populate_gold_chunks.py --corpus Experiments/corpora/<corpus>.json`
3. Create config: `Experiments/configs/<run_name>.yaml`
4. Run eval: `python eval/harness/run_eval.py --config Experiments/configs/<run_name>.yaml`
5. Append a new section to this file with setup, results, and observations.
