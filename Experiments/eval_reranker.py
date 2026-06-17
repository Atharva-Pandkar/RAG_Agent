"""
Reranker isolation evaluation.

Measures how well the LLM reranker performs independently of the agent:
  - Recall@10  : was the answer chunk in the top-10 retrieved candidates?
  - Recall@K   : is it still in the top-K after reranking?
  - MRR@10/K   : Mean Reciprocal Rank before and after reranking
  - Promotion  : did the reranker move the answer chunk up in rank?
  - Drop rate  : did the reranker cut the answer chunk out of top-K?

Uses only SFP + SFT questions from the v2 eval suite (unambiguous numeric answers).

Usage:
    python Experiments/eval_reranker.py
    python Experiments/eval_reranker.py --rerank-k 3   # test smaller top-K
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(ROOT / "app" / "backend" / ".env")

from src.pipeline import RagPipeline
from src.retrieval.llm_reranker import llm_rerank

# ── Number extraction (same logic as hybrid judge) ─────────────────────────────

_NUM_RE = re.compile(
    r"""
    \$\s*([\d,]+\.?\d*)\s*(?:billion|B)\b
    | \$\s*([\d,]+\.?\d*)\s*(?:million|M)\b
    | \$\s*([\d,]+\.?\d*)
    | ([\d,]+\.?\d*)\s*(?:billion|B)\b
    | ([\d,]+\.?\d*)\s*(?:million|M)\b
    | \b(\d+\.\d{2})\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SCALE = {0: 1000, 1: 1, 2: 1, 3: 1000, 4: 1, 5: 1}


def _extract_numbers(text: str) -> set[float]:
    values: set[float] = set()
    for m in _NUM_RE.finditer(text):
        for i, grp in enumerate(m.groups()):
            if grp:
                try:
                    values.add(round(float(grp.replace(",", "")) * _SCALE.get(i, 1), 4))
                except ValueError:
                    pass
    return values


def _chunk_has_answer(chunk: dict, expected_nums: set[float], tol: float = 0.025) -> bool:
    chunk_nums = _extract_numbers(chunk.get("text", ""))
    for ev in expected_nums:
        if ev < 0.01:
            continue
        for cv in chunk_nums:
            if abs(ev - cv) / max(abs(ev), 1e-9) <= tol:
                return True
    return False


def _first_hit_rank(chunks: list[dict], expected_nums: set[float]) -> int | None:
    """Return 1-based rank of first chunk containing the answer, or None."""
    for i, c in enumerate(chunks, 1):
        if _chunk_has_answer(c, expected_nums):
            return i
    return None


# ── Main evaluation ────────────────────────────────────────────────────────────

def run(suite_path: Path, rerank_k: int, retrieval_k: int, rerank_model: str) -> None:
    suite  = json.loads(suite_path.read_text(encoding="utf-8"))
    # Only numeric-answer categories
    qs = [q for q in suite["questions"]
          if q["category"] in ("single_fact_prose", "single_fact_table")]

    print(f"\n{'='*65}")
    print(f"  Reranker Isolation Eval")
    print(f"  retrieval_k={retrieval_k}  rerank_k={rerank_k}  model={rerank_model}")
    print(f"  {len(qs)} questions (SFP + SFT)")
    print(f"{'='*65}\n")

    print("  Loading RAG pipeline...", flush=True)
    pipeline = RagPipeline(
        corpus_path="Experiments/corpora/active_corpus.json",
        retrieval_strategy="faiss_hybrid",
        embedding_model="text-embedding-3-small",
    )

    # Doc-id mapping (same as rag_tool.py infers from list_available_documents)
    # For single-company questions we apply metadata filtering to mirror prod behaviour
    _DOC_KEYWORDS = {
        "apple": "aapl-", "aapl": "aapl-",
        "caterpillar": "cat-", "cat": "cat-",
        "jpmorgan": "jpm-", "jpm": "jpm-", "jp morgan": "jpm-",
        "coca-cola": "ko-", "coca cola": "ko-", "coke": "ko-",
        "walmart": "wmt-",
    }

    def _detect_doc(question: str) -> str | None:
        q = question.lower()
        for kw, prefix in _DOC_KEYWORDS.items():
            if kw in q:
                return prefix
        return None

    results = []
    for q in qs:
        expected_nums = _extract_numbers(q["answer"])
        if not expected_nums:
            print(f"  [SKIP] {q['id']} — no numeric answer to look for")
            continue

        doc_filter = _detect_doc(q["question"])
        t0 = time.perf_counter()

        # ── Step 1: raw retrieval ──────────────────────────────────────────
        candidates = pipeline.retrieve(q["question"], k=retrieval_k, filter_doc=doc_filter)
        retrieval_time = time.perf_counter() - t0

        rank_before = _first_hit_rank(candidates, expected_nums)
        recall_at_k_pre = rank_before is not None  # Recall@retrieval_k

        # ── Step 2: reranker ───────────────────────────────────────────────
        t1 = time.perf_counter()
        reranked = llm_rerank(q["question"], candidates, return_k=rerank_k, model=rerank_model)
        rerank_time = time.perf_counter() - t1

        rank_after = _first_hit_rank(reranked, expected_nums)
        recall_at_k_post = rank_after is not None  # Recall@rerank_k

        promoted = (
            rank_before is not None and rank_after is not None
            and rank_after < rank_before
        )
        dropped = recall_at_k_pre and not recall_at_k_post  # was found, now lost

        result = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"][:70],
            "doc_filter": doc_filter,
            "expected_nums": sorted(expected_nums)[:3],
            "recall_pre": recall_at_k_pre,
            "recall_post": recall_at_k_post,
            "rank_before": rank_before,
            "rank_after":  rank_after,
            "promoted": promoted,
            "dropped": dropped,
            "retrieval_s": round(retrieval_time, 2),
            "rerank_s":    round(rerank_time, 2),
        }
        results.append(result)

        icon = "✓" if recall_at_k_post else ("✗" if dropped else "—")
        rb = f"rank {rank_before}" if rank_before else "not found"
        ra = f"rank {rank_after}" if rank_after else "not found"
        mv = "↑" if promoted else ("↓DROP" if dropped else ("=" if rank_after == rank_before else "→"))
        print(f"  [{icon}] {q['id']:10s}  {rb:>10} → {ra:<10}  {mv}  "
              f"({retrieval_time:.2f}s + {rerank_time:.2f}s rerank)")

    # ── Summary stats ──────────────────────────────────────────────────────────
    n = len(results)
    recall_pre  = sum(1 for r in results if r["recall_pre"])  / n
    recall_post = sum(1 for r in results if r["recall_post"]) / n
    drops       = sum(1 for r in results if r["dropped"])
    promotions  = sum(1 for r in results if r["promoted"])

    mrr_pre = sum(
        1 / r["rank_before"] for r in results if r["rank_before"]
    ) / n
    mrr_post = sum(
        1 / r["rank_after"] for r in results if r["rank_after"]
    ) / n

    avg_rank_before = sum(
        r["rank_before"] for r in results if r["rank_before"]
    ) / max(recall_pre * n, 1)
    avg_rank_after = sum(
        r["rank_after"] for r in results if r["rank_after"]
    ) / max(recall_post * n, 1)

    print(f"\n{'='*65}")
    print(f"  RERANKER RESULTS  (n={n})")
    print(f"{'='*65}")
    print(f"  {'Metric':40s} {'Before':>10} {'After':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Recall (answer in top-k)':40s} {recall_pre*100:>9.1f}% {recall_post*100:>9.1f}%")
    print(f"  {'MRR (Mean Reciprocal Rank)':40s} {mrr_pre:>10.3f} {mrr_post:>10.3f}")
    print(f"  {'Avg rank of answer chunk':40s} {avg_rank_before:>10.1f} {avg_rank_after:>10.1f}")
    print(f"  {'-'*60}")
    print(f"  Promotions (reranker moved answer up): {promotions}/{n} ({promotions/n*100:.0f}%)")
    print(f"  Drops (answer in top-{retrieval_k}, lost after rerank): {drops}/{n} ({drops/n*100:.0f}%)")
    print(f"\n  Interpretation:")
    delta_recall = (recall_post - recall_pre) * 100
    delta_mrr    = mrr_post - mrr_pre
    if delta_recall >= 0 and delta_mrr >= 0:
        print(f"  Reranker IMPROVED recall by {delta_recall:+.1f}pp and MRR by {delta_mrr:+.3f}")
    elif drops > 0:
        print(f"  Reranker dropped {drops} answer chunk(s) — consider increasing rerank_k")
    print(f"{'='*65}\n")

    out_path = ROOT / "Experiments" / "runs" / f"reranker_eval_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "retrieval_k": retrieval_k,
        "rerank_k": rerank_k,
        "n": n,
        "recall_pre": recall_pre,
        "recall_post": recall_post,
        "mrr_pre": mrr_pre,
        "mrr_post": mrr_post,
        "avg_rank_before": avg_rank_before,
        "avg_rank_after": avg_rank_after,
        "drops": drops,
        "promotions": promotions,
        "per_question": results,
    }, indent=2), encoding="utf-8")
    print(f"  Results saved → {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",       default=str(ROOT / "Experiments" / "10k_rag_eval_v2.json"))
    parser.add_argument("--rerank-k",   type=int, default=5)
    parser.add_argument("--retrieval-k",type=int, default=10)
    parser.add_argument("--model",       default="gpt-4o-mini")
    args = parser.parse_args()
    run(Path(args.suite), args.rerank_k, args.retrieval_k, args.model)
