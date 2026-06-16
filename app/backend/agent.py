"""Simple LangChain ReAct-style agent for the 10-K RAG chatbot."""
from __future__ import annotations
import os

from langchain.agents import create_agent

from .rag_tool import search_10k_filings

SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer questions about the "
    "10-K filings of AAPL, CAT, JPM, KO, and WMT using the search_10k_filings "
    "tool to find supporting passages. Always ground your answers in retrieved "
    "passages and cite the source document. If the filings don't contain the "
    "answer, say so clearly rather than guessing."
)


def build_agent(model: str | None = None):
    model_name = model or os.environ.get("CHAT_MODEL", "openai:gpt-4o-mini")
    return create_agent(
        model=model_name,
        tools=[search_10k_filings],
        system_prompt=SYSTEM_PROMPT,
    )
