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
| Fixed/recursive/section/LangChain chunkers | Done | Runs 01–31 |
| Unstructured-IO corpus in eval | Done | Runs 32–36 |
| iXBRL + merged corpora | Done | Runs 37–40 |
| BM25 / dense / hybrid / FAISS hybrid | Done | All strategies benchmarked |
| Gold chunk population | Done | **26/26** on merged corpus; 25/26 unstructured |
| Document-level retrieval filtering | Not started | Cross-doc pollution still present |

**Best configs (n=26, latest runs):**
| Run | Strategy | Corpus | k | Recall@k | MRR |
|-----|----------|--------|---|----------|-----|
| **36** | faiss_hybrid | unstructured 4000/200 | 10 | **0.677** | 0.396 |
| 34 | hybrid (numpy) | unstructured 4000/200 | 10 | 0.677 | 0.396 |
| 32 | bm25 | unstructured 4000/200 | 5 | 0.557 | 0.376 |
| 16 | hybrid (numpy) | fixed 1024/100 | 10 | 0.598 | 0.379 |
| 40 | faiss_hybrid | merged unstr+xbrl | 10 | 0.575 | 0.398 |

---

## Phase 2 — Reranking
**Status: In Progress (ms-marco negative; not retried on best configs)**

| Milestone | Status | Notes |
|-----------|--------|-------|
| Cross-encoder reranker (ms-marco MiniLM) | Done | Runs 19–20; hurts recall |
| Rerank on Run 36 baseline | Not started | — |
| BGE / domain cross-encoder | Not started | — |

---

## Phase 3 — Advanced Retrieval
**Status: Partially Started**

| Milestone | Status | Notes |
|-----------|--------|-------|
| iXBRL table recovery | Done | `build_xbrl_corpus.py` |
| Section/table cross-refs in chunks | Done | unstructured + merged corpora |
| Structured cross-reference graph | Done (extracted) | Not used at retrieval time |
| Multi-query / query expansion | Not started | — |

---

## Phase 4 — Context Construction
**Status: Partially Started**

| Milestone | Status | Notes |
|-----------|--------|-------|
| Top-k sweep | Done | k=3, 5, 10 benchmarked |
| Context compression | Not started | — |

---

## App / Demo Layer
**Status: Skeleton Complete (config outdated)**

| Milestone | Status | Notes |
|-----------|--------|-------|
| FastAPI + LangChain agent | Done | — |
| RAG tool | Done | **Still on hybrid fixed 1024 k=5** — should be Run 36 |
| React frontend | Done (skeleton) | — |
| Streaming / citations | Not started | — |

---

## Phase 5 — Cost / Latency Optimization
**Status: Not Started**

---

## Next Up (Iteration 7 candidates)
1. Update `rag_tool.py` to Run 36 (faiss_hybrid unstructured k=10)
2. Rerank on Run 36 baseline
3. Investigate why merged corpus underperforms unstructured-only despite 26/26 gold match
4. Semantic HTML corpus → eval (Runs 41+)
5. Document-level filtering by company
