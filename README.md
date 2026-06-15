# 10-K Filing RAG — Experimentation Framework

Research/benchmarking project to identify the best RAG architecture for
answering questions about SEC 10-K filings, before building a production system.

## Structure

```
Dataset/                 Raw 10-K HTML filings (AAPL, CAT, JPM, KO, WMT)
Documents/
  extracted/             Plain-text extractions of each filing (Phase 0 output)
src/
  ingestion/             extract_text.py - HTML -> clean text (Phase 1: section-aware chunking lives here)
eval/
  golden_set/
    golden_set.json      28 hand-curated Q&A pairs grounded in the 5 filings
  harness/
    metrics.py           recall@k, MRR, context precision, correctness/faithfulness (LLM-judge)
    run_eval.py          Eval entrypoint - runs any pipeline against the golden set
Experiments/
  configs/               YAML configs per experiment (chunking/embedding/retrieval/reranker combos)
  runs/                  JSON results per run, timestamped
```

## Golden Set

28 questions across 5 filings (AAPL, CAT, JPM, KO, WMT), covering 7 types:
factual lookup, table-based numerical, multi-hop comparative, risk factor
synthesis, MD&A interpretive, cross-section facts, and unanswerable
(hallucination probes). Each entry has a gold answer and a traceable
evidence quote/section.

To extend: add entries to `eval/golden_set/golden_set.json` following the
existing schema. `gold_chunk_ids` can be populated once a chunking scheme
is fixed, to enable retrieval-level metrics (recall@k, MRR, context precision).

## Running the eval harness

```bash
pip install -r requirements.txt
python eval/harness/run_eval.py --config Experiments/configs/baseline.yaml --dry-run
```

`--dry-run` validates the golden set and prints the question-type
distribution. Once a pipeline is implemented (Phase 1+), point
`pipeline_class` in the config at a class exposing `.retrieve()` and
`.generate()`, and drop `--dry-run` to get full metrics written to
`Experiments/runs/`.

## Roadmap

- **Phase 0** (done): extraction, golden set, eval harness scaffold
- **Phase 1**: chunking strategies (fixed/recursive/section-based/parent-child) x embeddings (OpenAI/Voyage/BGE/E5) x retrieval (dense/BM25/hybrid)
- **Phase 2**: reranking (Cohere, BGE cross-encoder)
- **Phase 3**: advanced retrieval (multi-query, query expansion, parent-document, contextual retrieval)
- **Phase 4**: context construction (top-k sweep, compression, long-context baseline)
- **Phase 5**: end-to-end cost/latency optimization

Each phase = one or more YAML configs in `Experiments/configs/`, run via
`run_eval.py`, with results compared in `Experiments/runs/`.
