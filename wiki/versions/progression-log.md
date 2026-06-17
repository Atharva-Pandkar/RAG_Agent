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
| Gold chunk population | Done | **26/26** merged + fixed-size; 25/26 unstructured (KO-03) |
| Document-level retrieval filtering | Done | `doc_filter` on chatbot path (Iteration 15); offline `run_eval.py` still unfiltered |

**Best configs (n=26, latest runs):**
| Run | Strategy | Corpus | k | Recall@k | MRR |
|-----|----------|--------|---|----------|-----|
| **36** | faiss_hybrid | unstructured 4000/200 | 10 | **0.677** | 0.396 |
| 34 | hybrid (numpy) | unstructured 4000/200 | 10 | 0.677 | 0.396 |
| 32 | bm25 | unstructured 4000/200 | 5 | 0.557 | 0.376 |
| 16 | hybrid (numpy) | fixed 1024/100 | 10 | 0.598 | 0.379 |
| 40 | faiss_hybrid | merged unstr+xbrl | 10 | 0.575 | 0.398 |
| 41 | hybrid (numpy) | merged unstr+xbrl | 5 | 0.478 | 0.379 |

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
**Status: Functional (MVP)**

| Milestone | Status | Notes |
|-----------|--------|-------|
| FastAPI + deep agent | Done | `deepagents.create_deep_agent` |
| RAG tool | Done | `active_corpus.json`; faiss_hybrid k=10; OpenAI embeddings |
| Source IDs in API | Done | `SourceRef[]` with footnote resolution; `{id, doc, section, display}` |
| Source citations in UI | Done | Inline `[N]` superscripts + footnote list (EDGAR links removed in Iter 11) |
| Live document ingest | Done | `POST /ingest`, `ingest_document.py`, incremental BM25/FAISS |
| Document registry + sidebar | Done | `docs_registry.json`, `GET /documents`, upload UI |
| Document-agnostic agent | Done | `search_documents` + `list_available_documents` |
| Markdown rendering | Done | `react-markdown` + GFM tables for assistant answers |
| React frontend | Done | Vite dev server + API proxy; `npm install` required |
| Production frontend build | Done | `npm run build` → `dist/` |
| Startup warmup | Done | Lifespan loads pipeline + agent before requests |
| Backend logging | Done | `logger.py` + `logs/rag_backend.log` |
| Streaming responses | Not started | — |
| CORS restricted | Done | localhost:5173/3000 only |

---

## Phase 5 — Cost / Latency Optimization
**Status: Partially Started**

| Milestone | Status | Notes |
|-----------|--------|-------|
| Eager startup warmup | Done | FastAPI lifespan; no per-request cold start |
| OpenAI embeddings (no local model) | Done | `text-embedding-3-small`; faster client init |
| FAISS index disk cache | Done | Rebuild only on cache miss or model change |
| Backend request logging | Done | Rotating file + HTTP middleware |
| Pre-built embedding cache for deploy | Not started | First boot still embeds ~1,893 chunks via API |
| Retrieval result caching per turn | Not started | — |
| Agent E2E eval suite | Done | v2 baseline Run `225849` — 79.1% overall |
| LLM rerank in chatbot | Done | 10 candidates → rerank to 5 via gpt-4o-mini |
| Hallucination / OOC guards | Done | Entity verify prompt + retrieval mismatch warning |
| OOC smoke test | Done | `eval_ooc_quick.py` |
| Eval suite v2 in repo | Done | `Experiments/10k_rag_eval_v2.json` (43 Q) |
| XBRL investee table warnings | Done | `build_xbrl_corpus.py` (rebuild pending) |
| Document-scoped retrieval | Done | `doc_filter` on search + retrievers |
| v2 agent eval baseline | Done | Run `225849` — 79.1% overall |
| BM25 cross-doc filter (chatbot) | Done | Addresses long-standing open issue |
| RAGAS-style diagnostic eval | Done | `eval_ragas.py` — 4-metric decomposition on live agent |
| Reranker isolation benchmark | Done | `eval_reranker.py` — recall drop 72.7%→40.9% flagged |

---

## Next Up (Iteration 17 candidates)
1. Fix or tune LLM reranker (7/22 answer-chunk drops; consider `RERANK_K` increase or bypass)
2. Improve cross-company synthesis (43.75% E2E; RAGAS faithfulness 0.38 on CCC)
3. Rebuild xbrl + merged + active corpus with investee warnings
4. Pass `section` through faiss_hybrid RRF results
5. Expose retrieved (not just cited) chunks in API for accurate RAGAS context recall
6. Sync `eval_ooc_quick.py` to v2 suite path
7. Add `DELETE /documents/{id}` API
8. Registry-driven mismatch guard (replace hardcoded keyword list)
9. Update `app/README.md` — doc_filter, eval tooling, reranker findings
10. Pin `aiohttp`, `unstructured`, full RAG deps in requirements
