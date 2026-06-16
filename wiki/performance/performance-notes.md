# Performance Notes

---

## BM25Retriever — Full-Corpus Scoring
- Concern: Every query calls `BM25Okapi.get_scores()` over all 1,658 chunks, then sorts the full score vector. Complexity is O(n) per query with no index pruning.
- Location: `src/retrieval/bm25_retriever.py`
- Priority: Medium (acceptable at ~1.7k chunks; will matter at scale)
- Notes: Fine for current 5-document corpus. Consider document-level pre-filtering before scoring, or migrating to a search engine (Elasticsearch, Lucene) if corpus grows.

---

## RagPipeline — Full Corpus in Memory
- Concern: Entire corpus JSON (all chunk text) loaded into memory at pipeline init. BM25 tokenizes all chunks on construction.
- Location: `src/pipeline.py`, `src/retrieval/bm25_retriever.py`
- Priority: Low (current corpus ~1,658 chunks)
- Notes: Dense embedding index will add significant memory once vectors are stored. Plan for lazy loading or vector DB when moving to Phase 1 dense retrieval.

---

## fixed_size_chunks — Full-Document Tokenization
- Concern: Each document is fully encoded to tokens before sliding-window chunking. Large filings (JPM: 740 chunks) encode the entire text upfront.
- Location: `src/chunking/chunkers.py`
- Priority: Low
- Notes: Acceptable for offline corpus build. Streaming chunking would help if rebuilding frequently on much larger document sets.

---

## populate_gold_chunks — Linear Scan
- Concern: For each question, iterates all chunks for the target document with substring matching. O(questions × chunks_per_doc).
- Location: `eval/golden_set/populate_gold_chunks.py`
- Priority: Low
- Notes: Fast enough at current scale. Could index normalized text if this becomes a bottleneck.

---

## Eval Harness — Sequential Query Loop
- Concern: 28 questions processed sequentially with no batching or parallelism.
- Location: `eval/harness/run_eval.py`
- Priority: Low
- Notes: Run 01 completes quickly. Parallelism becomes relevant when adding embedding API calls or LLM generation.

---

## DenseRetriever — First-Run Embedding Cost
- Concern: Initial run per corpus embeds all chunks via SentenceTransformer (batch_size=32). JPM alone contributes ~740 chunks; full corpus ~1,658–2,000+ depending on strategy.
- Location: `src/retrieval/dense_retriever.py`
- Priority: Medium
- Notes: Subsequent runs load from `Experiments/embeddings/*.npy`. First hybrid run also triggers dense embed even if BM25-only run preceded it.

---

## HybridRetriever — Dual Index Memory
- Concern: Holds BM25 tokenized corpus + full embedding matrix + loaded SentenceTransformer model simultaneously. Fetches top-20 from each retriever per query (40 candidates before RRF merge).
- Location: `src/retrieval/hybrid_retriever.py`
- Priority: Medium
- Notes: Acceptable for 5-document research corpus. Memory scales linearly with chunk count × embedding dim (384 for BGE-small).

---

## DenseRetriever — Full-Corpus Dot Product
- Concern: Query embedding dotted against entire embedding matrix (`embeddings @ q_emb`) with no approximate nearest-neighbor index.
- Location: `src/retrieval/dense_retriever.py`
- Priority: Low (current scale)
- Notes: Fine at ~2k chunks. FAISS or vector DB needed if corpus grows beyond research set.

---

## extract_structured — Full HTML Parse Per Filing
- Concern: BeautifulSoup walks entire 10-K HTML DOM per filing; JPM filing is largest (~740 fixed-size chunks worth of content).
- Location: `src/ingestion/extract_structured.py`
- Priority: Low
- Notes: One-time offline step. Output JSON cached in `Documents/structured/`.

---

## RerankRetriever — Cross-Encoder Inference Per Query
- Concern: Every query runs `CrossEncoder.predict()` over `fetch_k` query–passage pairs (default 20). Model loaded at pipeline init alongside base retriever (BM25 + dense + cross-encoder for hybrid+rerank configs).
- Location: `src/retrieval/reranker.py`, `src/pipeline.py`
- Priority: Medium
- Notes: Acceptable for 28-question eval set. Latency scales with fetch_k × corpus complexity. Hybrid+rerank (Run 20) is the heaviest pipeline init (BM25 + bi-encoder + cross-encoder).

---

## FaissRetriever — Index Build and Dual Embedding Stack
- Concern: First run per corpus builds FAISS index via LangChain embeddings. Chatbot defaults to OpenAI (`text-embedding-3-small`); eval runs used HuggingFace BGE-small. Cache miss triggers full-corpus API embed at startup warmup.
- Location: `src/retrieval/faiss_retriever.py`, `src/retrieval/faiss_hybrid_retriever.py`
- Priority: Medium
- Notes: Index persisted to disk avoids rebuild. FAISS hybrid init loads BM25 + embedding client + FAISS index. HugFace path still loads PyTorch (~15–20s) if `EMBED_PROVIDER=huggingface`.

---

## LangChain Agent — LLM Call Per Chat Turn [OUTDATED]
- Concern: `create_agent` ReAct loop may invoke `search_10k_filings` multiple times per user message, each triggering full retrieval (BM25 + FAISS RRF in best configs).
- Location: `app/backend/agent.py`, `app/backend/rag_tool.py`
- Priority: Medium
- Notes: **[OUTDATED]** Replaced by deep agent. See "Deep Agent — Multi-Tool-Call Per Turn" below.

---

## explore_html_semantic — Full HTML Split Per Filing
- Concern: `HTMLSemanticPreservingSplitter` processes entire HTML files; default output capped at 50 chunks per file in JSON but full split runs regardless.
- Location: `src/ingestion/explore_html_semantic.py`
- Priority: Low
- Notes: Exploration-only script. JPM filing produces largest chunk counts.

---

## explore_unstructured — partition_html on Full 10-K HTML
- Concern: `partition_html` parses entire HTML DOM per filing. JPM: 2,853 elements → 890 chunks (486 table chunks). Single-threaded, memory-heavy.
- Location: `src/ingestion/explore_unstructured.py`
- Priority: Medium (JPM scale)
- Notes: One-time exploration step. Output JSON for JPM is very large (full chunk text retained). Consider preview/truncation flags.

---

## Chunking Exploration — No Eval Benchmark Yet [PARTIALLY RESOLVED]
- Concern: Three HTML→chunk paths existed but only structured-derived strategies were in eval grid.
- Location: `Documents/semantic_explore/`, `build_unstructured_corpus.py`
- Priority: Medium → Low for unstructured (now benchmarked Runs 32–36)
- Notes: Semantic HTML path still not in eval. Unstructured is now production winner.

---

## Unstructured Corpus — Large Index (1,492 chunks)
- Concern: Production unstructured corpus is ~90% larger than fixed 1024 corpus (~1,658 chunks across 5 docs but different granularity). FAISS index + BM25 tokenization at init; JPM dominates chunk count.
- Location: `Experiments/corpora/unstructured_4000_200.json`, `faiss_hybrid` retriever
- Priority: Medium
- Notes: Run 36 init loads BM25 + HuggingFaceEmbeddings + FAISS over 1,492 chunks. Acceptable for research; monitor latency in chatbot.

---

## build_xbrl_corpus — Full iXBRL DOM Parse
- Concern: Parses all `ix:nonFraction` facts and walks HTML table ancestry per filing. JPM filing is largest; runs once offline.
- Location: `src/ingestion/build_xbrl_corpus.py`
- Priority: Low
- Notes: Produces 401 table-only chunks merged into 1,893-chunk index.

---

## build_merged_corpus — O(n×m) Fingerprint Dedup
- Concern: For each of 401 xbrl chunks, checks 120-char normalized prefix against set of 1,492 unstructured fingerprints. Linear but small at current scale.
- Location: `src/ingestion/build_merged_corpus.py`
- Priority: Low
- Notes: Dedup may collapse distinct tables with similar headers — could explain merged underperformance vs unstructured-only.

---

## merged_unstr_xbrl.json — 24 MB Inline Text
- Concern: Single JSON file holds all 1,893 chunks with full text (~24 MB). `RagPipeline` loads entire file into memory; git operations and IDE indexing suffer.
- Location: `Experiments/corpora/merged_unstr_xbrl.json`, `src/pipeline.py`
- Priority: Medium
- Notes: Unstructured corpus (`unstructured_4000_200.json`) is smaller but same pattern. Consider sharded storage or sqlite for production.

---

## Chatbot Pipeline — Merged Corpus Cold Start (~20s) [OUTDATED]
- Concern: Lazy init loads 24 MB merged JSON + BM25 index + HuggingFaceEmbeddings + FAISS index (1,893 chunks). Comment in `main.py` notes ~20s first request.
- Location: `app/backend/rag_tool.py`, `app/backend/main.py`
- Priority: High (user-facing)
- Notes: **[OUTDATED — resolved Iteration 9]** Eager lifespan init at startup. See "Server Startup Blocks Until Pipeline + Agent Ready" below. Deep agent may still invoke `search_10k_filings` 2–4× per complex question.

---

## Deep Agent — Multi-Tool-Call Per Turn
- Concern: System prompt instructs multi-search for complex questions; deep agent adds planning overhead (write_todos, subagents) on top of each retrieval call.
- Location: `app/backend/agent.py`
- Priority: Medium
- Notes: Each search retrieves k=10 passages from 1,893-chunk index. Monitor token usage and latency for multi-part financial questions.

---

## OpenAI Embedding — First FAISS Cache Miss (~1,893 chunks)
- Concern: On cache miss, `FAISS.from_documents` sends all merged-corpus chunks to OpenAI embedding API in one batch. Network latency + API rate limits block startup warmup.
- Location: `src/retrieval/faiss_retriever.py`, `app/backend/main.py` lifespan
- Priority: High (user-facing at first deploy)
- Notes: Subsequent runs load cached index from `Experiments/embeddings/faiss_merged_unstr_xbrl__text-embedding-3-small/`. Pre-build index offline or ship pre-built cache for demos. OpenAI cost scales with corpus size × re-embeds on model change.

---

## Server Startup Blocks Until Pipeline + Agent Ready
- Concern: FastAPI lifespan runs corpus load, BM25 init, FAISS warm-up query, and deep agent compile synchronously before `yield`. Uvicorn does not accept requests until complete.
- Location: `app/backend/main.py`
- Priority: Medium
- Notes: Eliminates per-request cold start but shifts latency to boot. With OpenAI cache hit, startup is faster than old BGE path (~15–20s model load). With cache miss, startup may take minutes. `/health` returns `initialising` during boot.

---

## Rotating Log File at DEBUG Level
- Concern: File handler logs DEBUG for all `rag.*` modules including full message previews and chunk retrieval details. High chat volume fills 5 MB log quickly (3 rotations).
- Location: `app/backend/logger.py`
- Priority: Low
- Notes: Set file handler to INFO in production; keep DEBUG for troubleshooting sessions only.

---

## SourceRef Regex Parsing on Every Chat Turn
- Concern: `main.py` iterates all agent messages and regex-parses tool output strings to build `SourceRef[]`. O(messages × chunks) per request; coupled to string format.
- Location: `app/backend/main.py`
- Priority: Low
- Notes: Typical turn has 1–4 tool calls × 10 chunks — negligible. Prefer structured tool return or corpus lookup if format changes frequently.

---

## ReactMarkdown Client-Side Render
- Concern: Each assistant message re-parsed and rendered as GFM markdown on every React re-render (message list scroll, loading state toggle).
- Location: `app/frontend/src/App.jsx`
- Priority: Low
- Notes: Fine for short financial answers. Memoize `AssistantMessage` or cap message history if conversations grow long.

---

## Live Ingest — Full active_corpus.json Rewrite
- Concern: `_append_to_corpus()` loads entire active corpus JSON, modifies chunk list, and writes back with `indent=1`. Baseline file is ~24 MB; each upload blocks on serialize + disk I/O.
- Location: `src/ingestion/ingest_document.py`
- Priority: Medium
- Notes: Runs in `run_in_executor` on `/ingest` but still holds GIL during JSON dump. Large uploads or frequent ingests will slow the endpoint and grow startup load time.

---

## Live Ingest — BM25 Full Rebuild
- Concern: Every upload calls `BM25Retriever.add_chunks()`, which retokenizes all chunks and instantiates a new `BM25Okapi` over the full corpus.
- Location: `src/retrieval/bm25_retriever.py`, `src/ingestion/ingest_document.py`
- Priority: Low (demo scale)
- Notes: ~2k chunks rebuilds in under a second locally. Cost grows linearly with total chunks × upload frequency.

---

## Live Ingest — FAISS Incremental Embed + Cache Save
- Concern: Each upload embeds new chunks via OpenAI API (`add_documents`) and re-saves the full FAISS index to disk. Network latency + API cost per upload; cache file rewrite on every ingest.
- Location: `src/retrieval/faiss_retriever.py`, `src/ingestion/ingest_document.py`
- Priority: Medium
- Notes: Large PDFs producing hundreds of chunks trigger a sizable embed batch. Re-ingest without vector purge may inflate index size (see open issue).

---

## Unstructured-IO Parse on Upload
- Concern: `partition()` on PDF/DOCX/HTML is CPU- and memory-heavy compared to plain text chunking. Runs synchronously inside ingest executor.
- Location: `src/ingestion/ingest_document.py`
- Priority: Medium
- Notes: First use may download models depending on Unstructured config. Timeout risk if Vite proxy limit (120s) shared with chat.

---

## Startup — active_corpus Seed Copy
- Concern: First boot copies `merged_unstr_xbrl.json` → `active_corpus.json` (~24 MB disk copy) before pipeline warm-up.
- Location: `src/ingestion/ingest_document.py`, `app/backend/main.py` lifespan
- Priority: Low
- Notes: One-time per environment unless active file deleted. Subsequent boots load the active file directly.

---

## Citation Resolution — Regex Over All Agent Messages
- Concern: `_build_sources()` scans every `search_documents` ToolMessage plus final answer for `[SOURCE:id]` markers. Typical cost negligible; grows with multi-tool deep-agent turns.
- Location: `app/backend/main.py`
- Priority: Low
- Notes: Prefer structured tool JSON return if agent turns become very long.

---

## Eval Suite Runner — Parallel /chat Requests
- Concern: Default 3 workers send concurrent deep-agent turns, each invoking multiple LLM + retrieval calls. First full run (~47 questions) completed in ~few minutes wall time but triggered gpt-4o-mini TPM rate limits (429 → HTTP 500 on several questions).
- Location: `Experiments/eval_suite_runner.py`, `app/backend/main.py`
- Priority: High (benchmark validity)
- Notes: Cross-company category scores may be artificially low until throttling/retry added. Recommend `--workers 1` for reliable baseline runs.

---

## Eval Suite Runner — Sequential LLM Judge Pass
- Concern: After parallel agent queries, judge calls GPT-4o-mini once per question sequentially (47 API calls). Adds latency and cost on top of agent inference already consumed during query phase.
- Location: `Experiments/eval_suite_runner.py`
- Priority: Low
- Notes: Judge uses `response_format=json_object` and caps system response at 3000 chars. Re-judging cached responses would avoid re-querying agent.

---

## Eval Suite — Full Agent Turn Per Question
- Concern: Each suite question is an isolated `/chat` with single user message — no conversation reuse. Typical latency 6–30s per question (search + generation + citations). 47 questions × agent turn dominates eval wall time vs judge phase.
- Location: `Experiments/eval_suite_runner.py`, `app/backend/agent.py`
- Priority: Medium
- Notes: Complex cross-company questions may invoke 2–4 tool calls. Timeout set to 180s per request in runner.

---

## LLM Reranker — Extra OpenAI Call Per Search
- Concern: Every `search_documents` call now runs `llm_rerank()` after hybrid retrieval: ~500 input tokens (10 × 200-char previews + query) + ~20 output tokens on gpt-4o-mini, before the main deep-agent LLM turn.
- Location: `src/retrieval/llm_reranker.py`, `app/backend/rag_tool.py`
- Priority: Medium
- Notes: Multi-search questions (cross-company, comparisons) multiply rerank cost. Falls back to top-5 unranked on API failure. Consider caching rerank results within a single agent turn.

---

## Entity Verification — Extra list_available_documents Tool Call
- Concern: Updated agent prompt instructs calling `list_available_documents` before searching any named entity. Adds one LLM tool round-trip + registry JSON read per question, increasing latency and TPM usage vs Run 1 eval.
- Location: `app/backend/agent.py`, `app/backend/rag_tool.py`
- Priority: Medium
- Notes: May contribute to single-fact prose regression (50% on Run 2). Registry is small (~5 docs) — could inject doc list into system prompt at startup instead.

---

## eval_ooc_quick — Sequential /chat Without Throttling
- Concern: Smoke test fires 10 OOC questions one at a time but each still runs full agent + optional list + search + rerank + judge. ~2 LLM calls per question (agent + judge).
- Location: `Experiments/eval_ooc_quick.py`
- Priority: Low
- Notes: Safer than parallel full suite for rate limits; good for iterative prompt tuning.

---

## Eval Suite v2 — Sequential 43-Question Baseline (Not Yet Run)
- Concern: New v2 suite (43 questions) has no logged agent eval run yet. Wall time estimate: ~6–30s × 43 ≈ 5–20 min sequential at workers=1, plus 43 judge calls.
- Location: `Experiments/10k_rag_eval_v2.json`, `Experiments/eval_suite_runner.py`
- Priority: Medium
- Notes: Run after cleaning `active_corpus.json` and rebuilding xbrl chunks with investee warnings for apples-to-apples benchmark.

---

## XBRL Corpus Rebuild — Full Pipeline Re-run Required
- Concern: Investee table warnings only exist in newly built xbrl chunks. Rebuild chain: `build_xbrl_corpus.py` → `build_merged_corpus.py` → re-seed `active_corpus.json` → FAISS cache invalidation → server restart.
- Location: `src/ingestion/build_xbrl_corpus.py`, `Experiments/corpora/`, `Experiments/embeddings/`
- Priority: Medium
- Notes: Warning text placed in first ~200 chars intentionally for LLM reranker snippet visibility.
