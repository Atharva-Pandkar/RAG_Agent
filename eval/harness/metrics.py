"""
Metric implementations for the 10-K RAG eval harness.

Two tiers:
  - Retrieval metrics: compare retrieved chunk ids/text against gold evidence.
  - Generation metrics: compare generated answer against gold answer
    (correctness, faithfulness) - LLM-judge based, model-agnostic.

Kept dependency-light for Phase 0; LLM-judge calls are stubbed behind
a single `llm_judge` function so the backing model can be swapped.
"""
from __future__ import annotations
from typing import Callable, Sequence


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Fraction of gold evidence chunks present in the top-k retrieved chunks."""
    if not gold_ids:
        return float("nan")
    topk = set(retrieved_ids[:k])
    hit = sum(1 for g in gold_ids if g in topk)
    return hit / len(gold_ids)


def mrr(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Mean reciprocal rank of the first gold chunk in the retrieved list."""
    gold_set = set(gold_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def context_precision(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Fraction of the top-k retrieved chunks that are relevant (in gold set)."""
    topk = retrieved_ids[:k]
    if not topk:
        return float("nan")
    gold_set = set(gold_ids)
    return sum(1 for d in topk if d in gold_set) / len(topk)


# ---------------------------------------------------------------------------
# Generation-quality metrics (LLM-judge based)
# ---------------------------------------------------------------------------

JudgeFn = Callable[[str, str, str], dict]
"""Signature: judge(question, gold_answer, generated_answer) -> dict with
   keys like {"correctness": 0-1, "faithfulness": 0-1, "rationale": str}"""


def answer_correctness(question: str, gold_answer: str, generated_answer: str,
                        judge: JudgeFn) -> float:
    result = judge(question, gold_answer, generated_answer)
    return result["correctness"]


def faithfulness(question: str, retrieved_context: str, generated_answer: str,
                  judge: JudgeFn) -> float:
    """Checks whether claims in generated_answer are supported by retrieved_context.
    `judge` here is reused with retrieved_context passed as the 'gold_answer' slot,
    so the judge prompt should be written generically (claim vs. source text)."""
    result = judge(question, retrieved_context, generated_answer)
    return result["faithfulness"]
