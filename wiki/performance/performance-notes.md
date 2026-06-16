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
