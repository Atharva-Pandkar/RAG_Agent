"""
LLM-based reranker — uses a cheap chat model to select the most relevant chunks.

Design
------
- Retrieve N candidates from the base retriever (e.g. 10)
- Send a numbered list of ~200-char snippet previews to gpt-4o-mini
- Model returns ONLY the indices of the top-k relevant chunks (e.g. "3, 1, 7")
- Return those chunks in ranked order

Cost: ~500 tok input + ~10 tok output per rerank call — far cheaper than
passing all 10 full chunks into the main agent's context window.
"""
from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


def llm_rerank(
    query: str,
    candidates: list[dict[str, Any]],
    return_k: int = 3,
    model: str = "gpt-4o-mini",
    preview_chars: int = 200,
) -> list[dict[str, Any]]:
    """Return the top ``return_k`` most relevant chunks from ``candidates``.

    If reranking fails for any reason, falls back to the first ``return_k``
    candidates (preserving original retrieval order).

    Args:
        query:         The original search query.
        candidates:    Chunks returned by the base retriever (dicts with at
                       least 'text', 'doc', 'section' keys).
        return_k:      How many chunks to return after reranking.
        model:         OpenAI model for reranking (default: gpt-4o-mini).
        preview_chars: Characters from each chunk shown to the reranker.

    Returns:
        ``return_k`` chunks in relevance order.
    """
    if len(candidates) <= return_k:
        return candidates

    # Build numbered snippet list
    lines: list[str] = []
    for i, c in enumerate(candidates, 1):
        text    = c.get("text", "").replace("\n", " ").strip()[:preview_chars]
        doc     = c.get("doc", "")
        section = c.get("section") or "—"
        lines.append(f"[{i}] ({doc} / {section}): {text}")
    snippets = "\n".join(lines)

    prompt = (
        f"You are a relevance filter. Given a question and {len(candidates)} passages, "
        f"select the {return_k} most relevant passages.\n\n"
        f"Question: {query}\n\n"
        f"Passages:\n{snippets}\n\n"
        f"Return ONLY {return_k} comma-separated passage numbers in order of relevance "
        f"(e.g. \"3, 1, 7\"). No other text."
    )

    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        # Parse: accept "3, 1, 7" or "3 1 7" or "3,1,7"
        numbers = re.findall(r"\d+", raw)
        indices = [int(n) - 1 for n in numbers]       # 1-based → 0-based
        valid   = [i for i in indices if 0 <= i < len(candidates)]
        unique  = list(dict.fromkeys(valid))           # deduplicate, preserve order

        if unique:
            log.info("LLM reranker selected indices (0-based): %s from %d candidates",
                     unique[:return_k], len(candidates))
            return [candidates[i] for i in unique[:return_k]]

        log.warning("LLM reranker returned no valid indices (raw: %r) — falling back", raw)

    except Exception as exc:
        log.warning("LLM reranker failed (%s) — falling back to top-%d unranked", exc, return_k)

    return candidates[:return_k]
