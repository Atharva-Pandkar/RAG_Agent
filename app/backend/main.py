"""
FastAPI backend for the 10-K RAG chatbot.

Run from the project root:
    uvicorn app.backend.main:app --reload --port 8000

Environment variables (set in app/backend/.env or shell):
    OPENAI_API_KEY   — required
    CHAT_MODEL       — optional, default gpt-4o-mini
"""
from __future__ import annotations
import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from .agent import build_agent  # noqa: E402 (after dotenv load)

app = FastAPI(title="10-K RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Lazy-initialised — first request triggers FAISS index build (~20s)
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []     # chunk IDs cited in the final answer


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model": os.environ.get("CHAT_MODEL", "gpt-4o-mini")}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages list is empty")

    # Convert to LangChain message objects
    lc_messages = []
    for m in req.messages:
        if m.role == "user":
            lc_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=m.content))

    agent = get_agent()

    import re

    try:
        result = agent.invoke({"messages": lc_messages})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    all_messages = result.get("messages", [])

    # The deep agent may emit planning/subagent messages before the final AI
    # response. Walk backwards to find the last AIMessage with real content.
    final_content = ""
    from langchain_core.messages import AIMessage as LCAIMessage
    for msg in reversed(all_messages):
        if isinstance(msg, LCAIMessage) and msg.content:
            final_content = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Collect chunk IDs from every search_10k_filings ToolMessage in this turn
    sources: list[str] = []
    for msg in all_messages:
        if getattr(msg, "name", None) == "search_10k_filings":
            content = getattr(msg, "content", "") or ""
            sources.extend(re.findall(r"ID:\s*([\w\-_]+)", content))

    return ChatResponse(
        response=final_content,
        sources=list(dict.fromkeys(sources)),
    )
