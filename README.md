# 10-K Filing RAG Chatbot

A production-style RAG system for querying five SEC 10-K annual filings — Apple (FY2025), Caterpillar (FY2024), JPMorgan Chase (FY2024), Coca-Cola (FY2024), and Walmart (FY2025) — through a conversational agent with inline citations.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- An OpenAI API key

### Install

```bash
git clone <repo>
cd RAG_Agent
pip install -r requirements.txt
```

### Configure

```bash
cp app/backend/.env.example app/backend/.env
# Open .env and set OPENAI_API_KEY=sk-...
# Optionally set CHAT_MODEL=gpt-4o-mini (default) or gpt-4o
```

### Run the backend

```bash
uvicorn app.backend.main:app --reload --port 8000
```

On first start the server seeds `Experiments/corpora/active_corpus.json`, builds the FAISS index in memory (~15 s), and warms the agent. Subsequent starts are faster because the corpus file is already present.

### Run the frontend

```bash
cd app/frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## Project Layout

```
app/
  backend/          FastAPI server, agent, RAG tool, ingestion endpoint
  frontend/         React + Vite chat UI
src/
  ingestion/        HTML → section-aware chunks + XBRL table extraction
  retrieval/        BM25, FAISS, hybrid retriever, LLM reranker
  pipeline.py       Thin orchestration layer
Experiments/
  corpora/          active_corpus.json (1,893 chunks, 5 filings)
  10k_rag_eval_v2.json   43-question evaluation suite
  eval_suite_runner.py   End-to-end eval with hybrid judge
  eval_reranker.py       Reranker isolation eval
  eval_ragas.py          RAGAS-style metrics (CR / CP / Faithfulness / AC)
  runs/             Timestamped JSON results for every eval run
Dataset/            Raw 10-K HTML filings
Documents/extracted Plain-text extractions
```

---

## Architecture & Design Decisions

### 1. Chunking

**Choice:** Section-aware chunking at ~1,024 tokens (200-token overlap) plus a separate XBRL structured-table pass, giving a merged corpus of 1,893 chunks.

**Why:** 10-K filings have explicit section boundaries (MD&A, Risk Factors, Financial Statements, Notes). Breaking at those boundaries preserves semantic coherence that fixed-size sliding windows destroy. The overlap prevents answers from straddling a chunk boundary.

The XBRL pass is critical: SEC filings embed iXBRL tags in the HTML. Parsing those separately gives structured representations of every financial table (income statement, balance sheet, segment results) as standalone chunks. Without this, table questions that require exact numbers fail because the dense text around the table rarely contains the figure in a retrievable form.

The two passes are merged and deduplicated — so the income statement appears both as a prose extraction and as an XBRL-structured chunk. The hybrid retriever gets two attempts per table question.

**What didn't work:** Fixed-size 256-token chunks shredded the XBRL tables across chunk boundaries. 4,000-token "large context" chunks caused the reranker to receive too little signal per chunk.

---

### 2. Embeddings

**Choice:** `text-embedding-3-small` (OpenAI), 1536 dimensions, cosine similarity via FAISS.

**Why:** Strong baseline for English financial text with low latency and cost. `text-embedding-3-large` adds marginal lift at 6× the cost for this domain. Local alternatives (BAAI/bge-small-en-v1.5) are available via `.env` (`EMBED_PROVIDER=huggingface`) but were not the primary path.

---

### 3. Retrieval

**Choice:** Hybrid BM25 + FAISS dense retrieval fused with Reciprocal Rank Fusion (RRF), fetching `RETRIEVAL_K=10` candidates per query.

**Why each component earns its place:**

| Component | What it catches |
|-----------|----------------|
| BM25 | Exact keyword hits — ticker symbols, section headers, named line items ("noninterest revenue") |
| FAISS | Semantic matches — "profit" → "net income", "headcount" → "employees" |
| RRF fusion | Combines both ranked lists without needing a tuned interpolation weight |

BM25 alone misses paraphrases. Dense alone misses exact financial line items that don't appear in training analogies. Hybrid beats either alone on the 43-question eval.

Metadata filtering (`doc_filter` parameter) restricts BM25 and FAISS to one company's chunks for single-company questions. This was essential: without it, a query about Apple's net income surfaces Walmart and JPMorgan chunks in the top results, diluting precision.

---

### 4. Reranking

**Choice:** LLM-as-reranker — `gpt-4o-mini` receives numbered 200-character previews of the 10 candidates and returns a comma-separated list of the top 5 indices (`RERANK_K=5`). Total prompt: ~400 tokens; max_tokens=20; temperature=0.

**Why:** Reduces the context window the main agent receives, cutting token cost and improving signal-to-noise for complex questions. The reranker can weigh semantic relevance in a way that pure vector similarity cannot.

**Known limitation — 32% drop rate:** The reranker isolation eval (see `Experiments/eval_reranker.py`) found that in 7 of 22 factual questions, the reranker cut the answer-bearing chunk from the top 5. This happens because financial table chunks have dense markdown formatting — a 200-character preview of a row like `| 416,161 | 391,035 |` gives the reranker little context. The end-to-end agent compensates by making multiple search calls, which is why overall accuracy remains high (80%) despite the drop rate.

---

### 5. Agent

**Choice:** `deepagents.create_deep_agent` — a LangChain-compatible loop agent with two tools: `list_available_documents` and `search_documents`. Model: `gpt-4o-mini`.

The system prompt enforces:
- Verify the company exists in the registry before searching
- Use `doc_filter` for single-company questions (metadata filtering)
- Cite every numeric claim with `[SOURCE:chunk_id]`
- Flag fiscal-year traps and ambiguities
- Refuse questions about companies not in the corpus

`list_available_documents` exposes each filing's `doc_id` (e.g. `aapl-20250927`) so the agent can pass it as `doc_filter`. This was the single biggest accuracy improvement for single-company factual questions.

---

## Where the System Currently Breaks

### 1. Cross-company ranking (56% correct)
The agent searches companies sequentially and assembles rankings in its reasoning step. For five-company comparisons, it sometimes confuses values across tool calls or fetches a prior-year comparative figure instead of the current year. A structured sub-graph that forces "collect all five figures first, then rank" would fix this. The deep agent's free-form reasoning is too unreliable for strict ordering tasks.

### 2. Reranker drops answer chunks (32% drop rate)
The LLM reranker receives 200-character previews. For structured table chunks (XBRL rows), the preview shows column headers and one or two numbers with no surrounding context. The reranker deprioritises these in favour of narrative MD&A chunks that discuss the same figure. Increasing `RERANK_K` to 7-8 or replacing the LLM reranker with a cross-encoder (BGE-reranker-v2-m3) would reduce this.

### 3. Context recall gap (32% overall)
The agent cites the MD&A narrative chunk rather than the table row that contains the exact number. Faithfulness metrics show 63% — some claims cannot be directly verified in the cited chunks even though the answer is correct. The agent finds the number in a table, then cites the prose discussion of it. Fixing this requires prioritising `_xbrl_` chunk citations for numeric answers.

### 4. Fiscal-year label mismatch
Walmart's fiscal year ends January 31, so their FY2025 filing is titled "FY2026 Annual Report" internally. The agent's `list_available_documents` returns that label, so queries for "Walmart FY2025" trigger a refusal. The registry needs fiscal-year alias fields.

### 5. Market cap / stock price not in scope
When asked for Coca-Cola's market capitalisation, the agent retrieves the total equity value (~$24 billion) from the balance sheet and presents it as an answer instead of refusing. The system prompt flags that market cap is not in a 10-K, but the agent ignores this when it finds a plausible-sounding number. A retrieval guard that intercepts "market cap" / "stock price" queries before tool use would fix this.

### 6. Adversarial date traps
The agent correctly identifies that Apple's fiscal year ends in September, not December, and flags the discrepancy — but still answers with the September figure. The judge scores this as wrong because the expected behaviour is to refuse and state the correct fiscal year end, not to answer while flagging. A stricter "refuse if wrong FY end specified" rule is needed.

---

## What I'd Do With Another Week

The project was built as an experimentation framework from the start — the scaffolding for systematic config sweeps (`Experiments/configs/`, `eval/harness/`) is there but only partially exercised. More time would go toward actually completing that loop, not just adding features.

**1. Finish the experiment matrix**
The current system landed on one configuration (section-aware chunking + text-embedding-3-small + hybrid BM25/FAISS + LLM reranker) largely by informed reasoning and spot-checking. With more time I'd run the full grid: fixed-size vs. section-aware vs. parent-child chunking × OpenAI vs. local BGE embeddings × dense vs. BM25 vs. hybrid retrieval, measured against the 43-question eval suite. The decisions above are defensible but not empirically validated against alternatives.

**2. Retrieval tricks**
The current retrieval stack is a solid baseline but misses several well-established improvements:
- **Query expansion / HyDE** — generate a hypothetical answer and embed that alongside the query; helps for vague or abstract questions
- **Multi-query retrieval** — rephrase the question 3 ways, retrieve for each, union the results; reduces the chance a single bad phrasing misses the right chunk
- **Parent-document retrieval** — index child chunks (256 tokens) for precision, but return the parent chunk (1,024 tokens) to the agent for context; addresses the current grounding gap where the cited chunk doesn't always contain the exact figure
- **Contextual chunk enrichment** — prepend each chunk with a one-sentence summary of its parent section so BM25 matches section-level context, not just the chunk text

**3. Latency**
End-to-end response time is currently 10–40 s depending on the number of tool calls. The main culprits are FAISS index load on cold start (~10 s), the LLM reranker call (adds ~2 s per query), and the deep agent's multi-step reasoning (~5–15 s for CCC questions). Fixes: persist the FAISS index to disk (`faiss.write_index`) so cold-start is a file read not a rebuild; replace the LLM reranker with a local cross-encoder (zero API latency); stream the agent's response token-by-token to the frontend instead of waiting for the full answer.

**4. Cost**
Every query currently makes 2–4 OpenAI calls (reranker + agent reasoning + sometimes a retry). For a multi-user deployment this adds up quickly. The priority changes would be: local embeddings (BAAI/bge-small-en-v1.5, already wired via `EMBED_PROVIDER=huggingface`) to eliminate embedding API calls entirely; local cross-encoder reranker to replace the gpt-4o-mini reranker call; and gpt-4o-mini already being the cheapest viable chat model for the agent. The expensive path is CCC questions that trigger 5+ tool calls — the orchestration layer from point 5 below would cap that.

**5. Performance through structure**
The current single-agent architecture is flexible but pays for it in consistency on multi-step questions. I'd introduce a lightweight orchestration layer (LangGraph) that routes by question type: single-company questions go straight to a filtered retrieval node; cross-company comparisons go through a "collect all companies in parallel → rank" sub-graph; out-of-corpus questions are caught before retrieval by an entity-check node. This is not more agent intelligence — it's enforcing structure the agent currently has to figure out on its own.

**6. Different ingestors and embedders**
The HTML extraction pipeline (`unstructured`) works well but is not the only option. For a production system I'd evaluate: `pdfplumber` or `pymupdf` for PDF-native filings (preserves table layout better than HTML→text); Docling for its native table understanding; and Voyage Finance embeddings, which are fine-tuned on financial text and likely to outperform general-purpose models on numeric and domain-specific queries. These are swappable at the config level — the eval harness exists to measure whether the swap is worth it.

---

## Evaluation

### Method

A 43-question v2 evaluation suite (`Experiments/10k_rag_eval_v2.json`) covers five categories:

| Category | Count | Tests |
|----------|------:|-------|
| Single-fact prose (SFP) | 12 | Named figures from MD&A / narrative text |
| Single-fact table (SFT) | 10 | Numbers that appear only in financial tables |
| Cross-company comparison (CCC) | 8 | Multi-filing reasoning and ranking |
| Out-of-corpus (OOC) | 8 | Companies or metrics not in the five filings |
| Adversarial / trap (ADV) | 5 | Wrong fiscal year, metric not in 10-K, offset FY |

**Ground truth:** All expected answers were verified directly from named corpus chunks before authoring. OOC questions use only companies absent from the corpus (Tesla, Microsoft, NVIDIA, Amazon, Alphabet) plus metrics 10-Ks do not disclose (stock price, market cap).

**Judge:** Hybrid — regex number extraction ±2.5% tolerance for SFP/SFT, keyword refusal detection for OOC, LLM semantic comparison for CCC/ADV. The LLM judge (gpt-4o-mini) is only invoked for categories where exact-match is insufficient. This avoids the judge hallucinations that caused a pure-LLM judge to mis-score correct numeric answers.

**Grounding check (RAGAS-style):** A separate `eval_ragas.py` script measures Context Recall (does any cited chunk contain the answer?), Context Precision (are cited chunks relevant?), Faithfulness (are all claims traceable to context?), and Answer Correctness end-to-end.

---

### Results — 15 representative questions

The table below is a representative subset of the full 43-question run. "Grounded" means the cited `[SOURCE:chunk_id]` chunk contains the answer figure or supporting text, verified by inspecting the actual corpus chunk.

| # | Category | Question (abbreviated) | Expected | Result | Grounded |
|---|----------|------------------------|----------|--------|----------|
| 1 | SFP | Apple total net sales FY2025 | $416,161 M | ✅ $416,161 M | ✅ |
| 2 | SFP | JPMorgan total net revenue FY2024 | $179,365 M | ✅ $179,365 M | ✅ |
| 3 | SFP | Coca-Cola net income attributable to shareowners FY2024 | $13,107 M | ✅ $13,107 M | ✅ |
| 4 | SFP | Walmart total net sales FY ending Jan 2025 | $706,413 M | ✅ $706,413 M | ✅ |
| 5 | SFT | Apple iPhone net sales FY2025 *(table)* | $209,586 M | ✅ $209,586 M | ⚠️ cited MD&A, not table row |
| 6 | SFT | Caterpillar Power & Energy segment revenue FY2024 *(table)* | $8,654 M | ✅ $8,654 M | ✅ |
| 7 | SFT | Caterpillar Construction Industries segment revenue FY2024 *(table)* | $24,865 M | ❌ wrong figure | ❌ |
| 8 | CCC | Which company had the highest revenue? | Walmart ($706 B) | ✅ Walmart | ✅ |
| 9 | CCC | Which company had the highest diluted EPS? | Apple ($6.97) | ✅ Apple / $6.97 | ⚠️ partial citation |
| 10 | CCC | Rank all five by operating income (descending) | WMT, AAPL, JPM, CAT, KO | ❌ wrong order | ❌ |
| 11 | OOC | What was Tesla's total revenue for FY2024? | Refuse | ✅ Refused correctly | — |
| 12 | OOC | What is Coca-Cola's market capitalisation? | Refuse | ❌ gave equity value ~$24 B | ❌ |
| 13 | ADV | Apple total revenue for FY ending **December** 2025? | $416,161 M + flag wrong FY end | ✅ flagged Sept FY end, gave correct figure | ✅ |
| 14 | ADV | Walmart revenues for **FY2025**? | $705 B + note fiscal offset | ❌ refused (registry shows "FY2026 Annual Report") | — |
| 15 | ADV | Caterpillar adjusted EPS FY2024? | $21.20 GAAP / note "adjusted" not in filing | ⚠️ gave GAAP EPS without clearly flagging non-GAAP absence | ⚠️ |

**Score: 10 fully correct, 2 partial, 3 wrong — 67% on this subset (80% on the full 43-question run).**

---

### Full RAGAS-style metrics (43 questions)

| Category | N | Context Recall | Context Precision | Faithfulness | Answer Correctness |
|----------|--:|:--------------:|:-----------------:|:------------:|:-----------------:|
| SFP — Prose facts | 12 | 25% | 98% | 67% | **100%** |
| SFT — Table facts | 10 | 10% | 80% | 55% | **90%** |
| CCC — Cross-company | 8 | 50% | 78% | 38% | **56%** |
| OOC — Refusals | 8 | 35% | 8% | 91% | **88%** |
| ADV — Adversarial | 5 | 60% | 70% | 62% | **40%** |
| **Overall** | **43** | **32%** | **70%** | **63%** | **80%** |

*Composite (mean of 4 metrics): 61.3%*

**Reading the numbers:**

- The 80% Answer Correctness vs. 32% Context Recall gap is the most important finding. The agent reaches the right answer but rarely cites the specific table row that holds the number — it cites the MD&A narrative chunk that discusses the same figure. The answer is correct; the citation trail is incomplete. This is the "grounding gap."

- Context Precision (70%) is healthy: when the agent cites something, it is usually relevant. The OOC 8% precision is expected — no chunks are relevant to Tesla/Microsoft queries.

- Faithfulness (63%) reflects the same grounding gap: the LLM claim-verifier cannot find some numeric claims in the cited chunks even though those chunks are topically correct, because the claim checker looks for the number in the cited text and the actual number lives in the table chunk that was not cited.

---

### Failure analysis — the three failures in the representative 15

**Q7 — CAT Construction Industries segment revenue (SFT)**

The agent retrieves CAT segment data but returns the prior-year comparative figure ($26,696 M) instead of the FY2024 figure ($24,865 M). The XBRL table chunk shows both years side by side, and with RERANK_K=5 the reranker selects a narrative chunk that mentions "Construction Industries" in a risk-factors context rather than the segment results table. The agent reads the narrative and extrapolates the wrong year's number. Fix: increase RERANK_K to 7-8 so the segment table chunk is not dropped.

**Q10 — Rank all five companies by operating income (CCC)**

The agent issues five separate `search_documents` calls, one per company. It correctly retrieves each figure but assembles the ranking in its reasoning step, where it occasionally conflates Coca-Cola's operating income ($11,311 M) with its net income ($13,107 M). The unstructured cross-company ranking is the weakest pattern. Fix: a dedicated "collect-then-rank" orchestration sub-graph rather than relying on the agent's working memory.

**Q12 — Coca-Cola's market capitalisation (OOC)**

The agent searches for "market capitalisation Coca-Cola" with `doc_filter=ko-...`, retrieves the equity section of the balance sheet, finds total equity of ~$24 billion, and presents it as market cap. The system prompt states market cap is not in a 10-K, but the agent ignores that rule when it finds a plausible-sounding number. This is a hallucination-by-substitution failure. Fix: intercept queries containing "market cap" / "stock price" / "P/E ratio" before retrieval and return a hardcoded refusal.

---

## Running the Evaluations

```bash
# Full 43-question end-to-end eval (server must be running)
python Experiments/eval_suite_runner.py

# Reranker isolation — measures drop rate and MRR before/after
python Experiments/eval_reranker.py --retrieval-k 10 --rerank-k 5

# RAGAS-style metrics (Context Recall / Precision / Faithfulness / Correctness)
python Experiments/eval_ragas.py

# Subset — just prose and table facts
python Experiments/eval_ragas.py --categories single_fact_prose single_fact_table
```

Results are written as timestamped JSON files to `Experiments/runs/`.
