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
**Status: Complete (extended with LangChain + FAISS variants)**

| Milestone | Status | Notes |
|-----------|--------|-------|
| Fixed-size chunking | Done | 256/512/1024 token variants |
| Recursive chunking | Done | Hand-rolled + LangChain variants |
| Section-based chunking | Done | Custom + LangChain section variants |
| LangChain chunking comparison | Done | Runs 21–31 |
| Corpus builder CLI | Done | Plain text + structured JSON; `table_index` |
| Gold chunk population | Done | 26/27 answerable matched |
| BM25 retriever | Done | Runs 01–06, 09, 13, 17, 21, 24, 28, 30 |
| Dense (numpy BGE-small) | Done | Runs 07, 11, 15 |
| Hybrid (numpy RRF) | Done | Runs 08, 10, 12, 14, 16, 18 |
| FAISS dense (LangChain) | Done | Runs 22, 25 |
| FAISS hybrid (LangChain RRF) | Done | Runs 23, 26, 27, 29, 31 |
| Top-k sweep | Done | k=3, 5, 10 across multiple configs |
| HTML semantic exploration | Done (explore) | LangChain HTMLSemanticPreservingSplitter; all 5 filings |
| Unstructured-IO exploration | Done (explore) | partition_html + chunk_by_title; all 5 filings |
| Consolidate chunking path for eval | Not started | 3 explore paths, 1 eval path |
| Parent-child chunking | Not started | — |
| Document-level retrieval filtering | Not started | Cross-doc pollution still present |

**Best configs so far (n=26, latest runs):**
| Run | Strategy | Chunking | k | Recall@k | MRR |
|-----|----------|----------|---|----------|-----|
| run16 | hybrid (numpy) | fixed 1024/100 | 10 | **0.598** | 0.379 |
| run31 | faiss_hybrid | langchain section 1024/100 | 10 | 0.589 | 0.365 |
| run27 | faiss_hybrid | langchain recursive 1024/100 | 10 | 0.587 | 0.374 |
| run29 | faiss_hybrid | langchain section 512/50 | 5 | 0.416 | **0.474** |

---

## Phase 2 — Reranking
**Status: In Progress (ms-marco negative; not retried on best configs)**

| Milestone | Status | Notes |
|-----------|--------|-------|
| Cross-encoder reranker (ms-marco MiniLM) | Done | Runs 19–20; hurts recall |
| Rerank on Run 16 / Run 31 baselines | Not started | — |
| BGE / domain cross-encoder | Not started | — |
| Cohere reranker | Not started | — |

---

## Phase 3 — Advanced Retrieval
**Status: Partially Started**

| Milestone | Status | Notes |
|-----------|--------|-------|
| Structured cross-reference graph | Done (extracted) | Not used at retrieval time |
| Table chunk metadata + reverse index | Done | In corpora; not used by retrievers |
| Multi-query retrieval | Not started | — |
| Query expansion | Not started | — |
| Parent-document retrieval | Not started | — |

---

## Phase 4 — Context Construction
**Status: Partially Started**

| Milestone | Status | Notes |
|-----------|--------|-------|
| Top-k sweep | Done | Multiple k values benchmarked |
| Context compression | Not started | — |
| Long-context baseline | Not started | — |

---

## App / Demo Layer
**Status: Skeleton Complete**

| Milestone | Status | Notes |
|-----------|--------|-------|
| FastAPI backend + `/chat` | Done | LangChain ReAct agent |
| RAG tool wrapping RagPipeline | Done | Hardcoded hybrid fixed 1024 (not Run 31) |
| React frontend | Done (skeleton) | Vite dev server + API proxy |
| Streaming responses | Not started | — |
| Source citations in UI | Not started | — |
| Session memory | Not started | Frontend resends full history |
| Eval harness LLM generation | Not started | Placeholder only in eval runs |

---

## Phase 5 — Cost / Latency Optimization
**Status: Not Started**

---

## Next Up (Iteration 6 candidates)
1. Build eval corpus from unstructured chunks; benchmark vs Run 16/31
2. Build eval corpus from semantic HTML chunks; compare table detection
3. Update `app/backend/rag_tool.py` to Run 31 config
4. Pin `unstructured`, langchain, faiss, sentence-transformers, rank-bm25 in requirements
5. Rerank on Run 16/31 baselines with BGE reranker
