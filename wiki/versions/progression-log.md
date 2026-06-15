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
**Status: In Progress**

| Milestone | Status | Notes |
|-----------|--------|-------|
| Fixed-size chunking | Done | 512 tok / 50 overlap |
| Recursive chunking | Done (code) | Not yet benchmarked |
| Corpus builder CLI | Done | `build_corpus.py` |
| Gold chunk population | Done | 15/27 answerable questions matched for fixed_512 corpus |
| BM25 retriever | Done | First end-to-end retrieval run complete |
| Dense embeddings (OpenAI/Voyage/BGE/E5) | Not started | `requirements.txt` entries commented out |
| Hybrid retrieval (dense + BM25) | Not started | — |
| Section-based chunking | Not started | Planned for Phase 1 |
| Parent-child chunking | Not started | Planned for Phase 1 |
| LLM generation wired | Not started | Placeholder answers only |
| Aggregate metrics in run output | Not started | Per-question metrics only |

**Current baseline (Run 01):** BM25 + fixed 512/50 → mean recall@5 **0.444**, MRR **0.317**, context precision **0.107** (15 questions with gold chunks).

---

## Phase 2 — Reranking
**Status: Not Started**

| Milestone | Status |
|-----------|--------|
| Cohere reranker | Not started |
| BGE cross-encoder reranker | Not started |

---

## Phase 3 — Advanced Retrieval
**Status: Not Started**

| Milestone | Status |
|-----------|--------|
| Multi-query retrieval | Not started |
| Query expansion | Not started |
| Parent-document retrieval | Not started |
| Contextual retrieval (Anthropic-style) | Not started |

---

## Phase 4 — Context Construction
**Status: Not Started**

| Milestone | Status |
|-----------|--------|
| Top-k sweep | Not started |
| Context compression | Not started |
| Long-context baseline | Not started |

---

## Phase 5 — Cost / Latency Optimization
**Status: Not Started**

---

## Next Up (Iteration 2 candidates)
1. Fix gold chunk matching for the 12 unmatched answerable questions
2. Add document-level filtering so BM25 doesn't retrieve cross-company chunks
3. Run recursive chunking baseline (Run 02)
4. Uncomment and pin `rank-bm25` in `requirements.txt`
5. Add aggregate summary metrics to run output JSON
