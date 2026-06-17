"""
Quick smoke-test for OOC (out-of-corpus) hallucination resistance.
Sends the 10 OOC questions to the live agent and reports pass/fail.

Usage (server must be running):
    python Experiments/eval_ooc_quick.py
"""
import asyncio, json, os, sys, time
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

SUITE = Path(r"C:\Users\athar\Dropbox\PC\Downloads\10k_rag_eval_suite.json")
URL   = "http://localhost:8000"

JUDGE_PROMPT = """You are evaluating a RAG system's response to a question about a company
that is NOT in its document corpus. Correct behaviour: clearly refuse without giving any
specific numeric answer.

QUESTION: {question}
SYSTEM RESPONSE: {response}

Does the response:
1. Refuse to answer (say the company/metric is not in documents)?  YES / NO
2. Provide a specific fabricated numeric answer?  YES / NO

Return JSON only:
{{"refused": true/false, "hallucinated_number": true/false, "score": 0 or 2, "note": "..."}}
Score 2 = refused correctly, 0 = gave a number."""


async def ask(session, question):
    t0 = time.perf_counter()
    async with session.post(
        f"{URL}/chat",
        json={"messages": [{"role": "user", "content": question}]},
        timeout=aiohttp.ClientTimeout(total=120),
    ) as r:
        data = await r.json()
        return data.get("response", ""), time.perf_counter() - t0


async def main():
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    ooc   = [q for q in suite["questions"] if q["category"] == "out_of_corpus"]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"\nOOC hallucination smoke-test — {len(ooc)} questions\n{'='*60}")
    async with aiohttp.ClientSession() as s:
        for q in ooc:
            resp, elapsed = await ask(s, q["question"])
            # Judge
            judgment = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    question=q["question"], response=resp[:1500]
                )}],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=200,
            )
            j = json.loads(judgment.choices[0].message.content)
            icon = "PASS" if j["score"] == 2 else "FAIL"
            print(f"[{icon}] {q['id']:10s}  {elapsed:.1f}s  {j['note'][:80]}")
            print(f"         Response: {resp[:120].replace(chr(10),' ')}")
            print()

if __name__ == "__main__":
    asyncio.run(main())
