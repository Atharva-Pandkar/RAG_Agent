"""
RAGAS-style evaluation metrics for the RAG pipeline.

Computes four standard RAG quality dimensions without the ragas library:

  Context Recall      — did the retrieved chunks contain the answer?
                        Numeric: does any chunk hold the expected number (±2.5%)
                        Qualitative: LLM checks if ground truth can be derived from context

  Context Precision   — are the retrieved chunks actually about the right thing?
                        % of top-k chunks that are relevant to the query (LLM judge)

  Faithfulness        — is the generated answer grounded in the context?
                        Decomposes answer into atomic claims; checks each against context

  Answer Correctness  — is the answer right?
                        Numeric: exact-match of key numbers (±2.5%)
                        Semantic: LLM similarity score for qualitative questions

Usage (server must be running):
    python Experiments/eval_ragas.py
    python Experiments/eval_ragas.py --categories single_fact_prose single_fact_table
    python Experiments/eval_ragas.py --n 10   # quick sample
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / "app" / "backend" / ".env")

import aiohttp
from openai import OpenAI

# ── Corpus lookup (chunk_id → text) ───────────────────────────────────────────
_CORPUS_TEXT: dict[str, str] = {}

def _load_corpus(corpus_path: Path) -> None:
    global _CORPUS_TEXT
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    chunks = data["chunks"] if isinstance(data, dict) else data
    _CORPUS_TEXT = {c["id"]: c.get("text", "") for c in chunks}
    print(f"  Corpus loaded: {len(_CORPUS_TEXT):,} chunks indexed by ID")

# ── Number helpers (shared with other eval scripts) ────────────────────────────

_NUM_RE = re.compile(
    r"""
    \$\s*([\d,]+\.?\d*)\s*(?:billion|B)\b
    | \$\s*([\d,]+\.?\d*)\s*(?:million|M)\b
    | \$\s*([\d,]+\.?\d*)
    | ([\d,]+\.?\d*)\s*(?:billion|B)\b
    | ([\d,]+\.?\d*)\s*(?:million|M)\b
    | \b(\d+\.\d{2})\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SCALE = {0: 1000, 1: 1, 2: 1, 3: 1000, 4: 1, 5: 1}


def _nums(text: str) -> set[float]:
    out: set[float] = set()
    for m in _NUM_RE.finditer(text):
        for i, g in enumerate(m.groups()):
            if g:
                try:
                    out.add(round(float(g.replace(",", "")) * _SCALE.get(i, 1), 4))
                except ValueError:
                    pass
    return out


def _numbers_match(expected: str, response: str, tol: float = 0.025) -> bool:
    en, rn = _nums(expected), _nums(response)
    if not en:
        return False
    matched = sum(
        1 for ev in en if ev >= 0.01 and any(
            abs(ev - rv) / max(abs(ev), 1e-9) <= tol for rv in rn
        )
    )
    return matched >= max(1, len(en) // 2)


# ── LLM helpers ────────────────────────────────────────────────────────────────

_CLIENT: OpenAI | None = None


def _llm(prompt: str, system: str = "You are a strict evaluation judge. Return only valid JSON.",
          max_tokens: int = 300) -> str:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = _CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return r.choices[0].message.content.strip()


# ── Metric 1: Context Recall ───────────────────────────────────────────────────

def context_recall(question: str, expected: str, contexts: list[str],
                   category: str) -> dict:
    """Can the ground-truth answer be derived from the retrieved contexts?"""
    joined = "\n\n---\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts))

    # Fast numeric path
    if category in ("single_fact_prose", "single_fact_table"):
        found = any(_numbers_match(expected, c) for c in contexts)
        return {
            "score": 1.0 if found else 0.0,
            "method": "numeric",
            "detail": "Key number found in context" if found else "Key number NOT in context",
        }

    # LLM path for qualitative
    prompt = f"""Does the following CONTEXT contain enough information to answer this QUESTION
and derive the EXPECTED ANSWER?

QUESTION: {question}
EXPECTED ANSWER: {expected[:400]}
CONTEXT:
{joined[:2500]}

Return JSON: {{"score": 0.0-1.0, "detail": "one sentence"}}
Score 1.0 = context fully supports the answer, 0.0 = answer cannot be derived."""
    try:
        j = json.loads(_llm(prompt))
        return {"score": float(j.get("score", 0)), "method": "llm", "detail": j.get("detail", "")}
    except Exception as e:
        return {"score": 0.0, "method": "error", "detail": str(e)}


# ── Metric 2: Context Precision ────────────────────────────────────────────────

def context_precision(question: str, contexts: list[str]) -> dict:
    """What fraction of retrieved chunks are actually relevant to the question?"""
    scores: list[float] = []
    for ctx in contexts:
        prompt = f"""Is this PASSAGE relevant and useful for answering the QUESTION?

QUESTION: {question}
PASSAGE: {ctx[:600]}

Return JSON: {{"relevant": true/false, "score": 0.0-1.0, "reason": "one sentence"}}
Score 1.0 = highly relevant, 0.5 = partially relevant, 0.0 = not relevant."""
        try:
            j = json.loads(_llm(prompt, max_tokens=150))
            scores.append(float(j.get("score", 0)))
        except Exception:
            scores.append(0.0)

    precision = sum(scores) / len(scores) if scores else 0.0
    return {
        "score": round(precision, 3),
        "per_chunk": [round(s, 2) for s in scores],
        "n_chunks": len(contexts),
    }


# ── Metric 3: Faithfulness ─────────────────────────────────────────────────────

def faithfulness(question: str, answer: str, contexts: list[str]) -> dict:
    """Are all claims in the answer supported by the retrieved context?"""
    joined = "\n\n---\n\n".join(f"[{i+1}] {c[:400]}" for i, c in enumerate(contexts))

    # Step 1: extract atomic claims from answer
    claim_prompt = f"""Break this answer into individual factual claims (numbers, names, assertions).
List only verifiable claims, not hedges or filler.

ANSWER: {answer[:800]}

Return JSON: {{"claims": ["claim1", "claim2", ...]}}
Limit to at most 6 most important claims."""
    try:
        claims_raw = json.loads(_llm(claim_prompt, max_tokens=300))
        claims = claims_raw.get("claims", [])[:6]
    except Exception:
        return {"score": 0.0, "supported": 0, "total": 0, "detail": "claim extraction failed"}

    if not claims:
        return {"score": 1.0, "supported": 0, "total": 0, "detail": "no verifiable claims"}

    # Step 2: check each claim against context
    supported = 0
    verdicts: list[dict] = []
    for claim in claims:
        check_prompt = f"""Is this CLAIM supported by the CONTEXT below?

CLAIM: {claim}
CONTEXT:
{joined[:2000]}

Return JSON: {{"supported": true/false, "reason": "one sentence"}}"""
        try:
            j = json.loads(_llm(check_prompt, max_tokens=150))
            ok = bool(j.get("supported", False))
            verdicts.append({"claim": claim, "supported": ok, "reason": j.get("reason", "")})
            if ok:
                supported += 1
        except Exception:
            verdicts.append({"claim": claim, "supported": False, "reason": "error"})

    score = supported / len(claims) if claims else 1.0
    return {
        "score": round(score, 3),
        "supported": supported,
        "total": len(claims),
        "verdicts": verdicts,
    }


# ── Metric 4: Answer Correctness ───────────────────────────────────────────────

def answer_correctness(question: str, expected: str, response: str,
                       category: str) -> dict:
    """Is the generated answer factually correct vs. ground truth?"""
    if category in ("single_fact_prose", "single_fact_table"):
        ok = _numbers_match(expected, response)
        return {
            "score": 1.0 if ok else 0.0,
            "method": "numeric",
            "detail": "Key number(s) match" if ok else "Key number(s) do not match",
        }
    if category == "out_of_corpus":
        refused = bool(re.search(
            r"don['']t have|not in (my|the)|not loaded|cannot|not available|no information",
            response, re.IGNORECASE
        ))
        return {
            "score": 1.0 if refused else 0.0,
            "method": "refusal",
            "detail": "Correct refusal" if refused else "Failed to refuse",
        }
    # LLM semantic similarity for CCC / ADV
    prompt = f"""How factually correct is the SYSTEM RESPONSE compared to the EXPECTED ANSWER?

QUESTION: {question}
EXPECTED: {expected[:400]}
SYSTEM RESPONSE: {response[:600]}

Return JSON: {{"score": 0.0-1.0, "detail": "one sentence"}}
1.0 = fully correct, 0.5 = partially correct, 0.0 = wrong."""
    try:
        j = json.loads(_llm(prompt))
        return {"score": float(j.get("score", 0)), "method": "llm", "detail": j.get("detail", "")}
    except Exception as e:
        return {"score": 0.0, "method": "error", "detail": str(e)}


# ── Agent query ────────────────────────────────────────────────────────────────

async def query_agent(session: aiohttp.ClientSession, question: str, base_url: str,
                      retries: int = 2) -> dict:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with session.post(
                f"{base_url}/chat",
                json={"messages": [{"role": "user", "content": question}]},
                timeout=aiohttp.ClientTimeout(total=240),
            ) as r:
                data = await r.json()
                return {
                    "response": data.get("response", ""),
                    "sources":  data.get("sources", []),
                }
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                await asyncio.sleep(3 * (attempt + 1))
    raise last_exc


# ── Main ───────────────────────────────────────────────────────────────────────

async def run(suite_path: Path, base_url: str, categories: list[str],
              n: int | None, out_dir: Path, corpus_path: Path) -> None:

    _load_corpus(corpus_path)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    questions = suite["questions"]
    if categories:
        questions = [q for q in questions if q["category"] in categories]
    if n:
        questions = questions[:n]

    print(f"\n{'='*65}")
    print(f"  RAGAS-style Evaluation — {len(questions)} questions")
    print(f"  Metrics: Context Recall | Context Precision | Faithfulness | Answer Correctness")
    print(f"{'='*65}\n")

    results = []
    async with aiohttp.ClientSession() as session:
        for i, q in enumerate(questions, 1):
            print(f"  [{i:02d}/{len(questions)}] {q['id']:10s}  {q['question'][:55]}...", flush=True)
            t0 = time.perf_counter()

            # Query agent
            try:
                agent_out = await query_agent(session, q["question"], base_url)
            except Exception as e:
                print(f"             ERROR: {e}")
                continue

            response = agent_out["response"]
            sources  = agent_out["sources"]
            # SourceRef only has id/doc/section/display — look up full chunk text from corpus
            contexts: list[str] = []
            for s in sources:
                chunk_id = s.get("id", "")
                text = _CORPUS_TEXT.get(chunk_id, "")
                if text:
                    contexts.append(text)
            if not contexts:
                contexts = [response]  # fallback: evaluate against the answer itself

            expected = q.get("answer") or q.get("expected_behavior", "")
            cat = q["category"]

            # Compute metrics
            cr  = context_recall(q["question"], expected, contexts, cat)
            cp  = context_precision(q["question"], contexts) if contexts else {"score": 0}
            f   = faithfulness(q["question"], response, contexts)
            ac  = answer_correctness(q["question"], expected, response, cat)

            elapsed = time.perf_counter() - t0

            results.append({
                "id":          q["id"],
                "category":    cat,
                "difficulty":  q["difficulty"],
                "question":    q["question"],
                "expected":    expected,
                "response":    response,
                "elapsed_s":   round(elapsed, 1),
                "context_recall":    cr,
                "context_precision": cp,
                "faithfulness":      f,
                "answer_correctness":ac,
            })
            print(f"             CR={cr['score']:.2f}  CP={cp['score']:.2f}  "
                  f"F={f['score']:.2f}  AC={ac['score']:.2f}  ({elapsed:.1f}s)")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    def _avg(key: str) -> float:
        return sum(r[key]["score"] for r in results) / len(results) if results else 0

    cr_avg  = _avg("context_recall")
    cp_avg  = _avg("context_precision")
    f_avg   = _avg("faithfulness")
    ac_avg  = _avg("answer_correctness")
    overall = (cr_avg + cp_avg + f_avg + ac_avg) / 4

    cats = sorted({r["category"] for r in results})
    cat_scores: dict[str, dict] = {}
    for cat in cats:
        cat_rs = [r for r in results if r["category"] == cat]
        cat_scores[cat] = {
            "n":  len(cat_rs),
            "context_recall":    sum(r["context_recall"]["score"]    for r in cat_rs) / len(cat_rs),
            "context_precision": sum(r["context_precision"]["score"] for r in cat_rs) / len(cat_rs),
            "faithfulness":      sum(r["faithfulness"]["score"]      for r in cat_rs) / len(cat_rs),
            "answer_correctness":sum(r["answer_correctness"]["score"]for r in cat_rs) / len(cat_rs),
        }

    cat_labels = {
        "single_fact_prose":       "SFP — Prose",
        "single_fact_table":       "SFT — Table",
        "cross_company_comparison": "CCC — Cross-company",
        "out_of_corpus":           "OOC — Refusal",
        "adversarial":             "ADV — Adversarial",
    }

    print(f"\n{'='*65}")
    print(f"  RAGAS METRICS SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Category':22s} {'N':>3}  {'C-Recall':>9} {'C-Prec':>8} {'Faithful':>9} {'Correct':>8}")
    print(f"  {'-'*62}")
    for cat, s in cat_scores.items():
        print(f"  {cat_labels.get(cat, cat):22s} {s['n']:>3}  "
              f"{s['context_recall']:>9.3f} {s['context_precision']:>8.3f} "
              f"{s['faithfulness']:>9.3f} {s['answer_correctness']:>8.3f}")
    print(f"  {'-'*62}")
    print(f"  {'OVERALL':22s} {len(results):>3}  "
          f"{cr_avg:>9.3f} {cp_avg:>8.3f} {f_avg:>9.3f} {ac_avg:>8.3f}")
    print(f"\n  Composite score (mean of 4 metrics): {overall:.3f}")
    print(f"\n  Metric definitions:")
    print(f"    Context Recall    — answer info present in retrieved chunks")
    print(f"    Context Precision — retrieved chunks are relevant to the question")
    print(f"    Faithfulness      — generated answer is grounded in context (no hallucination)")
    print(f"    Answer Correctness — generated answer matches ground truth")
    print(f"{'='*65}\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = out_dir / f"ragas_eval_{ts}.json"
    out_path.write_text(json.dumps({
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": len(results),
        "overall": round(overall, 3),
        "context_recall":    round(cr_avg, 3),
        "context_precision": round(cp_avg, 3),
        "faithfulness":      round(f_avg, 3),
        "answer_correctness":round(ac_avg, 3),
        "by_category": cat_scores,
        "per_question": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Results saved → {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",      default=str(ROOT / "Experiments" / "10k_rag_eval_v2.json"))
    parser.add_argument("--url",        default="http://localhost:8000")
    parser.add_argument("--categories", nargs="*",
                        help="Filter categories, e.g. single_fact_prose single_fact_table")
    parser.add_argument("--n",          type=int, default=None, help="Limit to first N questions")
    parser.add_argument("--out",        default=str(ROOT / "Experiments" / "runs"))
    parser.add_argument("--corpus",     default=str(ROOT / "Experiments" / "corpora" / "active_corpus.json"))
    args = parser.parse_args()

    asyncio.run(run(
        Path(args.suite), args.url,
        args.categories or [],
        args.n,
        Path(args.out),
        Path(args.corpus),
    ))
