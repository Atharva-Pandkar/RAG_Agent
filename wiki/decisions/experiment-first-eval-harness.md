## Experiment-First Eval Harness
- Date: 2026-06-15
- Context: The project goal is to benchmark RAG architectures on SEC 10-K filings before building production. We need reproducible, comparable runs across chunking/embedding/retrieval combinations.
- Decision: Use YAML configs in `Experiments/configs/` paired with `eval/harness/run_eval.py` as the single entrypoint. Each run writes timestamped JSON to `Experiments/runs/`.
- Alternatives Considered: Ad-hoc scripts per experiment; notebook-driven evaluation; integrated MLflow/W&B from day one.
- Impact: All future experiments follow the same config → run → results pattern. Pipelines must expose `.retrieve()` and `.generate()` to plug into the harness.
