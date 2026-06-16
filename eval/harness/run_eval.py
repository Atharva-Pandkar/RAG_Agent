"""
Eval harness entrypoint.

Usage:
    python run_eval.py --config Experiments/configs/baseline.yaml

A "pipeline" is any object exposing:
    .retrieve(question: str, k: int) -> list[dict]   # each dict: {id, text, score}
    .generate(question: str, contexts: list[dict]) -> str

For Phase 0, no pipeline exists yet - this script can be run with
`--dry-run` to validate the golden set loads correctly and to print
the metric matrix that WILL be populated once Phase 1 pipelines exist.
"""
from __future__ import annotations
import argparse
import json
import importlib
import sys
from pathlib import Path
from datetime import datetime, timezone

import yaml

from metrics import recall_at_k, mrr, context_precision

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SET_PATH = ROOT / "eval" / "golden_set" / "golden_set.json"

sys.path.insert(0, str(ROOT))


def load_golden_set(override: str | None = None) -> dict:
    path = (ROOT / override) if override else GOLDEN_SET_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pipeline(config: dict):
    """Dynamically import a pipeline class given dotted path in config['pipeline_class']."""
    module_path, class_name = config["pipeline_class"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**config.get("pipeline_kwargs", {}))


def run(config: dict, dry_run: bool = False):
    golden = load_golden_set(config.get("golden_set_override"))
    questions = golden["questions"]
    k = config.get("top_k", 5)

    results = {
        "run_name": config.get("run_name", "unnamed"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "n_questions": len(questions),
        "per_question": [],
    }

    if dry_run:
        print(f"Loaded golden set: {len(questions)} questions")
        by_type = {}
        for q in questions:
            by_type[q["type"]] = by_type.get(q["type"], 0) + 1
        print("Question type distribution:")
        for t, c in sorted(by_type.items()):
            print(f"  {t}: {c}")
        print("\nNo pipeline configured / --dry-run set. "
              "Once a Phase 1 pipeline exists, set 'pipeline_class' in the "
              "config and re-run without --dry-run to populate metrics.")
        return results

    pipeline = load_pipeline(config)

    for q in questions:
        retrieved = pipeline.retrieve(q["question"], k=k)
        retrieved_ids = [r["id"] for r in retrieved]
        gold_ids = q.get("gold_chunk_ids", [])  # populate once chunk-level gold mapping exists

        answer = pipeline.generate(q["question"], retrieved)

        entry = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "generated_answer": answer,
            "retrieved_ids": retrieved_ids,
        }
        entry["gold_chunk_ids"] = gold_ids
        if gold_ids:
            entry["recall_at_k"] = recall_at_k(retrieved_ids, gold_ids, k)
            entry["mrr"] = mrr(retrieved_ids, gold_ids)
            entry["context_precision"] = context_precision(retrieved_ids, gold_ids, k)
        else:
            entry["recall_at_k"] = None
            entry["mrr"] = None
            entry["context_precision"] = None

        results["per_question"].append(entry)

    out_dir = ROOT / "Experiments" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{results['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run(cfg, dry_run=args.dry_run)
