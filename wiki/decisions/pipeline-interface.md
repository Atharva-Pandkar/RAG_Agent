> **[OUTDATED]** Content moved to [`decisions-log.md`](decisions-log.md).

## Pipeline Interface (retrieve + generate)
- Date: 2026-06-15
- Context: The eval harness must work with any RAG stack (BM25, dense, hybrid, reranked) without rewriting the runner.
- Decision: Pipelines are plain Python classes loaded via `pipeline_class` in config. Required methods: `.retrieve(question, k) -> list[{id, doc, text, score}]` and `.generate(question, contexts) -> str`. `RagPipeline` in `src/pipeline.py` is the first implementation.
- Alternatives Considered: LangChain/LlamaIndex abstractions; function-based pipeline registration.
- Impact: New retrieval strategies extend `RagPipeline` or replace it. Generation is injected via optional `llm_fn` callback.
