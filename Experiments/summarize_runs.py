"""Summarize all run JSON files in Experiments/runs into a comparison table."""
import json
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "Experiments" / "runs"

rows = []
for f in sorted(RUNS.glob("*.json")):
    data = json.load(open(f))
    qs = data["per_question"]
    with_gold = [q for q in qs if q["gold_chunk_ids"]]
    n = len(with_gold)
    if n == 0:
        continue
    avg_recall = sum(q["recall_at_k"] for q in with_gold) / n
    avg_mrr = sum(q["mrr"] for q in with_gold) / n
    avg_prec = sum(q["context_precision"] for q in with_gold) / n
    rows.append({
        "run": data["run_name"],
        "top_k": data["config"]["top_k"],
        "n_scored": n,
        "recall_at_k": avg_recall,
        "mrr": avg_mrr,
        "context_precision": avg_prec,
    })

print(f"{'run':30} {'k':3} {'n':3} {'recall@k':10} {'mrr':8} {'ctx_prec':10}")
for r in rows:
    print(f"{r['run']:30} {r['top_k']:<3} {r['n_scored']:<3} "
          f"{r['recall_at_k']:<10.3f} {r['mrr']:<8.3f} {r['context_precision']:<10.3f}")
