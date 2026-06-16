# Progression Log

High-level milestone tracker across project phases. Updated at the end of each iteration.

---

## Phase 0 — Foundation
**Status: Complete**

| Milestone | Status | Notes |
|-----------|--------|-------|
| Extract 10-K HTML → plain text | Done | 5 filings in `Documents/extracted/` |
| Golden set (28 Q&A, 7 types) | Done | `eval/golden_set/golden_set.json` |
| Eval harness scaffold | Done | `run_eval.py`, `metrics.py` |
| Dry-run validation | Done | `--dry-run` prints question type distribution |

---

## Phase 1 — Chunking × Retrieval Baselines
**Status: Complete**

| Milestone | Status | Notes |
|-----------|--------|-------|
| Fixed-size chunking | Done | 256/512/1024 token variants |
| Recursive chunking | Done | Run 02 benchmarked |
| Section-based chunking | Done | Structured extraction + `section_based` chunker |
| Corpus builder CLI | Done | Plain text + structured JSON inputs |
| Gold chunk population | Done | 26/27 answerable matched on latest corpora |
| BM25 retriever | Done | Runs 01–06, 09, 13, 17 |
| Dense embeddings (BGE-small) | Done | Runs 07, 11, 15; `.npy` cache |
| Hybrid retrieval (BM25 + dense RRF) | Done | Runs 08, 10, 12, 14, 16, 18 |
| Top-k sweep (k=3, 5, 10) | Done | Runs 05, 06, 15–18 |
| Run comparison helper | Done | `Experiments/summarize_runs.py` |
| Parent-child chunking | Not started | Planned |
| LLM generation wired | Not started | Placeholder answers only |
| Aggregate metrics in run output | Not started | Use `summarize_runs.py` post-hoc |
| Document-level retrieval filtering | Not started | Cross-doc pollution still present |

**Best configs so far (n=26, latest runs):**
| Run | Strategy | Chunking | k | Recall@k | MRR |
|-----|----------|----------|---|----------|-----|
| run16 | hybrid | fixed 1024/100 | 10 | **0.598** | 0.379 |
| run06 | bm25 | fixed 512/50 | 10 | 0.572 | 0.347 |
| run18 | hybrid | section 1024/100 | 10 | 0.528 | 0.348 |
| run04 | bm25 | fixed 1024/100 | 5 | 0.416 | **0.397** |

---

## Phase 2 — Reranking
**Status: In Progress (initial results negative on recall)**

| Milestone | Status | Notes |
|-----------|--------|-------|
| Cross-encoder reranker (ms-marco MiniLM) | Done | `RerankRetriever` wrapper; Runs 19–20 |
| Rerank on BM25 baseline (Run 04 → 19) | Done | Recall dropped 0.416 → 0.394 |
| Rerank on hybrid baseline (Run 12 → 20) | Done | Recall dropped 0.440 → 0.348 |
| Cohere reranker | Not started | — |
| BGE cross-encoder reranker | Not started | Try domain-finetuned model |
| Rerank on best config (Run 16, k=10) | Not started | Next candidate |

---

## Phase 3 — Advanced Retrieval
**Status: Partially Started**

| Milestone | Status | Notes |
|-----------|--------|-------|
| Structured cross-reference graph | Done (extracted) | Not yet used at retrieval time |
| Multi-query retrieval | Not started | — |
| Query expansion | Not started | — |
| Parent-document retrieval | Not started | — |
| Contextual retrieval (Anthropic-style) | Partial | Section heading prepended to chunks |

---

## Phase 4 — Context Construction
**Status: Partially Started**

| Milestone | Status | Notes |
|-----------|--------|-------|
| Top-k sweep | Done | k=3, 5, 10 in runs 05–06, 15–17 |
| Context compression | Not started | — |
| Long-context baseline | Not started | — |

---

## Phase 5 — Cost / Latency Optimization
**Status: Not Started**

---

## Next Up (Iteration 4 candidates)
1. Rerank on Run 16 baseline (hybrid fixed 1024, k=10) — current best without rerank
2. Try BGE reranker or domain-specific cross-encoder instead of ms-marco
3. Add document-level filtering by company to reduce cross-filing retrieval
4. Pin `rank-bm25` and `sentence-transformers` in `requirements.txt`
5. Wire LLM generation for end-to-end answer quality metrics
