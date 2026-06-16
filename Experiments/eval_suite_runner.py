"""
End-to-end evaluation of the Document RAG Agent against a structured question suite.

Sends every question in the eval JSON to the running /chat endpoint, then uses
GPT-4o-mini as an LLM judge to score each answer against the gold answer.

Usage:
    # Server must be running: uvicorn app.backend.main:app --port 8000
    python Experiments/eval_suite_runner.py --suite path/to/eval_suite.json

Options:
    --suite   Path to the eval JSON file  (default: looks in Downloads)
    --url     Backend URL                 (default: http://localhost:8000)
    --model   Judge model                 (default: gpt-4o-mini)
    --workers Number of parallel workers  (default: 3)
    --out     Output directory            (default: Experiments/runs)
    --skip-ooc  Skip out-of-corpus questions (faster if testing only facts)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Windows: aiohttp requires SelectorEventLoop (not ProactorEventLoop default)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Windows cp1252 console can't render ✓/✗ — use ASCII fallbacks
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import aiohttp
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / "app" / "backend" / ".env")

# ── Scoring rubrics ───────────────────────────────────────────────────────────

RUBRICS = {
    "single_fact_prose": (
        "Numeric / factual answer from narrative text.\n"
        "- 2: Correct value, correct fiscal year, correct company.\n"
        "- 1: Correct order of magnitude but minor rounding error, or correct value "
        "without full context (e.g., missing company name or year).\n"
        "- 0: Wrong number, wrong year, or refusal when the answer is in scope."
    ),
    "single_fact_table": (
        "Numeric answer that lives in a financial table (income statement, segment table).\n"
        "- 2: Exact correct value from the correct row AND correct fiscal year column.\n"
        "- 1: Correct order of magnitude but off-by-one row (e.g., 2024 vs 2025 column) "
        "or slightly wrong due to rounding/version differences.\n"
        "- 0: Wrong value, fabricated, or refusal."
    ),
    "cross_company_comparison": (
        "Requires retrieving facts from multiple companies and comparing them.\n"
        "- 2: All companies correctly ranked/compared with correct values for each.\n"
        "- 1: Majority correct but one company missing, one value wrong, or comparison "
        "directionally right but figures off.\n"
        "- 0: Wrong ranking, fabricated values, or only one company retrieved."
    ),
    "out_of_corpus": (
        "The question asks about something NOT in the corpus (wrong company, wrong year, "
        "or metric not in filings). Correct behavior is to REFUSE or say 'I don't know'.\n"
        "- 2: Clearly refuses without providing any fabricated numeric answer.\n"
        "- 1: Hedges appropriately but slightly confusing (e.g., says 'I'm not certain' "
        "while still naming a vague figure).\n"
        "- 0: Provides a specific numeric answer — immediate hallucination fail."
    ),
    "adversarial": (
        "Contains a trap: wrong fiscal year end, offset FY labeling, bank metric definitions, "
        "or segment confusion. Correct behavior is to give the RIGHT answer AND flag the trap.\n"
        "- 2: Correct value AND explicitly notes the ambiguity/clarification required "
        "(e.g., 'Apple's FY ends in September, not December').\n"
        "- 1: Correct value but misses the required clarification, OR catches the trap "
        "but value is slightly off.\n"
        "- 0: Falls for the trap (gives wrong value) or refuses when answer is available."
    ),
}

JUDGE_SYSTEM = (
    "You are an expert financial research evaluator scoring RAG system responses "
    "to questions about SEC 10-K annual filings. "
    "Be strict on factual accuracy. Give the benefit of the doubt only for minor rounding "
    "within ±2% of the correct figure. Return valid JSON only."
)

JUDGE_TEMPLATE = """QUESTION: {question}

EXPECTED ANSWER: {expected_answer}

SYSTEM RESPONSE: {system_response}

CATEGORY: {category}
SCORING RUBRIC:
{rubric}

Score the response on a 0-2 scale following the rubric above.
Return ONLY this JSON (no markdown, no extra text):
{{
  "score": <0|1|2>,
  "max_score": 2,
  "correct": <true|false>,
  "key_finding": "<one sentence — what was right or wrong>",
  "reasoning": "<2-3 sentences explaining the score>"
}}"""


# ── Agent query ───────────────────────────────────────────────────────────────

async def query_agent(
    session: aiohttp.ClientSession,
    question: str,
    base_url: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list, float]:
    """Send question to the /chat endpoint. Returns (response_text, sources, elapsed)."""
    async with semaphore:
        payload = {"messages": [{"role": "user", "content": question}]}
        t0 = time.perf_counter()
        try:
            async with session.post(
                f"{base_url}/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                data = await resp.json()
                elapsed = time.perf_counter() - t0
                if resp.status != 200:
                    return f"[HTTP {resp.status}] {data}", [], elapsed
                return data.get("response", ""), data.get("sources", []), elapsed
        except asyncio.TimeoutError:
            return "[TIMEOUT after 180s]", [], time.perf_counter() - t0
        except Exception as exc:
            return f"[ERROR] {exc}", [], time.perf_counter() - t0


# ── Hybrid judge ──────────────────────────────────────────────────────────────
# For numeric fact questions (SFP, SFT) we do an exact-number pre-check:
# if the expected value appears in the response within ±2%, score 2 immediately.
# This avoids gpt-4o-mini hallucinating "incorrect" on correct answers.
# LLM judge is used for qualitative categories (CCC, ADV) and as a fallback.

_NUM_RE = re.compile(
    r"""
    \$\s*([\d,]+\.?\d*)\s*(?:billion|B)\b   # $X billion / $XB
    | \$\s*([\d,]+\.?\d*)\s*(?:million|M)\b  # $X million / $XM
    | \$\s*([\d,]+\.?\d*)                     # bare $X
    | ([\d,]+\.?\d*)\s*(?:billion|B)\b        # X billion
    | ([\d,]+\.?\d*)\s*(?:million|M)\b        # X million
    | \b([\d]+\.\d+)\b                        # decimal (EPS like 7.46)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SCALE = {0: 1000, 1: 1, 2: 1, 3: 1000, 4: 1, 5: 1}  # group → millions multiplier


def _extract_numbers(text: str) -> set[float]:
    """Return all numeric values from text, normalised to their raw scale."""
    values: set[float] = set()
    for m in _NUM_RE.finditer(text):
        for i, grp in enumerate(m.groups()):
            if grp:
                try:
                    val = float(grp.replace(",", "")) * _SCALE.get(i, 1)
                    values.add(round(val, 4))
                except ValueError:
                    pass
    return values


def _numbers_match(expected_text: str, response_text: str, tol: float = 0.025) -> bool:
    """Return True if every key number in expected appears in response within tol."""
    exp_nums = _extract_numbers(expected_text)
    resp_nums = _extract_numbers(response_text)
    if not exp_nums:
        return False
    matched = 0
    for ev in exp_nums:
        if ev < 0.01:  # skip near-zero / percentages
            matched += 1
            continue
        for rv in resp_nums:
            if abs(ev - rv) / max(abs(ev), 1e-9) <= tol:
                matched += 1
                break
    return matched >= max(1, len(exp_nums) // 2)  # at least half the key numbers match


_REFUSAL_RE = re.compile(
    r"don['']t have|not in (my|the)|not loaded|not available|cannot|"
    r"not (find|found|contain|present|disclose|reported|in my)|"
    r"no information|not provided|outside (the|my)|not in (?:the )?corpus",
    re.IGNORECASE,
)


def judge_response(
    client: OpenAI,
    question: str,
    expected: str,
    response: str,
    category: str,
    judge_model: str,
) -> dict:
    """Hybrid judge: exact-match pre-check for factual categories, LLM for the rest."""

    # ── 1. Numeric fact categories: try exact match first ─────────────────────
    if category in ("single_fact_prose", "single_fact_table"):
        if _numbers_match(expected, response):
            return {
                "score": 2, "max_score": 2, "correct": True,
                "key_finding": "Exact-match: key number(s) found in response.",
                "reasoning": "Pre-check passed — numeric value matches expected within ±2.5%.",
            }

    # ── 2. OOC: keyword-based refusal check ───────────────────────────────────
    if category == "out_of_corpus":
        refused = bool(_REFUSAL_RE.search(response))
        # If the response contains a dollar amount AND refuses, it's partial
        has_number = bool(_extract_numbers(response))
        if refused and not has_number:
            return {
                "score": 2, "max_score": 2, "correct": True,
                "key_finding": "Correctly refused without providing a fabricated number.",
                "reasoning": "Refusal detected; no specific numeric answer given.",
            }
        if not refused:
            return {
                "score": 0, "max_score": 2, "correct": False,
                "key_finding": "Failed to refuse — provided an answer for an out-of-corpus query.",
                "reasoning": "No refusal phrase detected in response.",
            }
        # Refused but included a number — fall through to LLM for partial score

    # ── 3. LLM judge for qualitative / partial cases ──────────────────────────
    rubric = RUBRICS.get(category, "Score 0-2 based on factual accuracy.")
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        expected_answer=expected,
        system_response=response[:3000],
        category=category,
        rubric=rubric,
    )
    try:
        completion = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as exc:
        return {
            "score": 0, "max_score": 2, "correct": False,
            "key_finding": f"Judge error: {exc}",
            "reasoning": str(exc),
        }


# ── Main evaluation loop ──────────────────────────────────────────────────────

async def run_eval(
    suite_path: Path,
    base_url: str,
    judge_model: str,
    workers: int,
    out_dir: Path,
    skip_ooc: bool,
) -> None:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    questions = suite["questions"]
    if skip_ooc:
        questions = [q for q in questions if q["category"] != "out_of_corpus"]

    print(f"\n{'='*65}")
    print(f"  RAG Agent Eval Suite — {suite['metadata']['suite_name']}")
    print(f"  {len(questions)} questions | judge: {judge_model} | workers: {workers}")
    print(f"  Endpoint: {base_url}")
    print(f"{'='*65}\n")

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    semaphore = asyncio.Semaphore(workers)
    results = []

    connector = aiohttp.TCPConnector(limit=workers)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Check server health first
        try:
            async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as r:
                health = await r.json()
                status = health.get("status", "?")
                if status != "ok":
                    print(f"  ⚠  Server status: {status} — agent may still be initialising")
                else:
                    print(f"  ✓  Server ready — model: {health.get('model', '?')}\n")
        except Exception as exc:
            print(f"  ✗  Cannot reach server at {base_url}: {exc}")
            print("     Start the server:  uvicorn app.backend.main:app --port 8000\n")
            sys.exit(1)

        # Create all tasks
        tasks = []
        for q in questions:
            tasks.append(
                query_agent(session, q["question"], base_url, semaphore)
            )

        # Run all in parallel (preserving order via gather)
        print(f"  Running {len(tasks)} queries (up to {workers} in parallel)...\n")
        t_start = time.perf_counter()

        # Use gather to run in parallel but keep results ordered
        ordered_responses: list[tuple[str, list, float]] = await asyncio.gather(*tasks)

        # Print progress summary
        for i, (q, (resp_text, _, elapsed)) in enumerate(zip(questions, ordered_responses), 1):
            resp_preview = resp_text[:90].replace("\n", " ") if resp_text else "—"
            print(f"  [{i:02d}/{len(tasks)}] {q['id']:10s}  {elapsed:.1f}s  {resp_preview}")

    total_elapsed = time.perf_counter() - t_start
    print(f"\n  All queries done in {total_elapsed:.0f}s. Scoring with LLM judge...\n")

    # Judge each response
    for i, (q, (resp_text, sources, q_elapsed)) in enumerate(zip(questions, ordered_responses), 1):
        expected_answer = q.get("answer") or q.get("expected_behavior", "")
        judgment = judge_response(
            openai_client,
            q["question"],
            expected_answer,
            resp_text,
            q["category"],
            judge_model,
        )
        result = {
            "id":         q["id"],
            "category":   q["category"],
            "difficulty": q["difficulty"],
            "question":   q["question"],
            "expected":   expected_answer,
            "response":   resp_text,
            "sources":    sources,
            "elapsed_s":  round(q_elapsed, 2),
            "score":      judgment.get("score", 0),
            "max_score":  judgment.get("max_score", 2),
            "correct":    judgment.get("correct", False),
            "key_finding":judgment.get("key_finding", ""),
            "reasoning":  judgment.get("reasoning", ""),
        }
        results.append(result)
        grade = "✓" if result["correct"] else "✗"
        print(f"  {grade} {q['id']:10s}  [{result['score']}/{result['max_score']}]  {result['key_finding'][:80]}")

    # ── Summary ──────────────────────────────────────────────────────────────

    categories = ["single_fact_prose", "single_fact_table", "cross_company_comparison",
                  "out_of_corpus", "adversarial"]

    cat_stats: dict[str, dict] = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        total_score  = sum(r["score"] for r in cat_results)
        max_possible = sum(r["max_score"] for r in cat_results)
        n_correct    = sum(1 for r in cat_results if r["correct"])
        cat_stats[cat] = {
            "n":            len(cat_results),
            "score":        total_score,
            "max":          max_possible,
            "pct":          total_score / max_possible if max_possible else 0,
            "accuracy":     n_correct / len(cat_results),
        }

    overall_score = sum(r["score"] for r in results)
    overall_max   = sum(r["max_score"] for r in results)
    overall_acc   = sum(1 for r in results if r["correct"]) / len(results)

    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Category':30s} {'N':>4} {'Score':>8} {'%':>7} {'Accuracy':>10}")
    print(f"  {'-'*60}")
    cat_labels = {
        "single_fact_prose":       "Single Fact — Prose",
        "single_fact_table":       "Single Fact — Table",
        "cross_company_comparison": "Cross-Company Comparison",
        "out_of_corpus":           "Out-of-Corpus (Refusal)",
        "adversarial":             "Adversarial",
    }
    for cat, stats in cat_stats.items():
        label = cat_labels.get(cat, cat)
        print(f"  {label:30s} {stats['n']:>4} "
              f"  {stats['score']:>2}/{stats['max']:>2}   "
              f"{stats['pct']*100:>5.1f}%  "
              f"{stats['accuracy']*100:>8.1f}%")
    print(f"  {'-'*60}")
    print(f"  {'OVERALL':30s} {len(results):>4} "
          f"  {overall_score:>2}/{overall_max:>2}   "
          f"{overall_score/overall_max*100:>5.1f}%  "
          f"{overall_acc*100:>8.1f}%")
    print(f"\n  Total time: {total_elapsed:.0f}s")
    print(f"{'='*65}\n")

    # ── Failures detail ───────────────────────────────────────────────────────
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"  FAILURES ({len(failures)}):")
        for r in failures:
            print(f"\n  [{r['id']}] {r['category']} / {r['difficulty']}")
            print(f"  Q: {r['question'][:100]}")
            print(f"  Expected: {r['expected'][:120]}")
            print(f"  Got:      {r['response'][:200].replace(chr(10),' ')}")
            print(f"  Finding:  {r['key_finding']}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_suite_{ts}.json"
    output = {
        "suite":       suite["metadata"]["suite_name"],
        "run_at":      datetime.now(timezone.utc).isoformat(),
        "base_url":    base_url,
        "judge_model": judge_model,
        "total_q":     len(results),
        "overall_score": overall_score,
        "overall_max":   overall_max,
        "overall_pct":   round(overall_score / overall_max * 100, 1),
        "overall_accuracy": round(overall_acc * 100, 1),
        "category_stats": cat_stats,
        "per_question": results,
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Results saved → {out_path}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG Agent eval suite.")
    parser.add_argument(
        "--suite",
        default=str(ROOT / "Experiments" / "10k_rag_eval_v2.json"),
        help="Path to the eval suite JSON",
    )
    parser.add_argument("--url",     default="http://localhost:8000")
    parser.add_argument("--model",   default="gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers. Keep at 1 for CCC questions to avoid 429 TPM errors")
    parser.add_argument("--out",     default=str(ROOT / "Experiments" / "runs"))
    parser.add_argument("--skip-ooc", action="store_true",
                        help="Skip out-of-corpus questions")
    args = parser.parse_args()

    suite_path = Path(args.suite)
    if not suite_path.exists():
        # Try a few fallback locations
        fallbacks = [
            Path.home() / "Downloads" / "10k_rag_eval_suite.json",
            ROOT / "10k_rag_eval_suite.json",
        ]
        for fb in fallbacks:
            if fb.exists():
                suite_path = fb
                break
        else:
            print(f"Eval suite not found at {args.suite}")
            sys.exit(1)

    asyncio.run(run_eval(
        suite_path  = suite_path,
        base_url    = args.url.rstrip("/"),
        judge_model = args.model,
        workers     = args.workers,
        out_dir     = Path(args.out),
        skip_ooc    = args.skip_ooc,
    ))
