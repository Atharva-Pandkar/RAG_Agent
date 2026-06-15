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
