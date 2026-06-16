"""
FastAPI backend for the Document RAG Chatbot.

Run from the project root:
    uvicorn app.backend.main:app --reload --port 8000

Environment variables (app/backend/.env):
    OPENAI_API_KEY   — required
    CHAT_MODEL       — optional, default gpt-4o-mini
"""
from __future__ import annotations
import os
import re
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import asyncio
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from .logger import get_logger   # noqa: E402
from .agent import build_agent   # noqa: E402
from .rag_tool import get_pipeline, REGISTRY_PATH  # noqa: E402

log = get_logger(__name__)

_agent = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent

    log.info("=" * 60)
    log.info("SERVER STARTUP")
    log.info("=" * 60)

    # ── 0. Seed active_corpus.json + registry (idempotent) ──────────────────
    from src.ingestion.ingest_document import (
        ensure_active_corpus, build_registry_from_corpus, ACTIVE_CORPUS
    )
    log.info("Step 0 — Ensuring active corpus and registry...")
    ensure_active_corpus()
    if not REGISTRY_PATH.exists():
        build_registry_from_corpus(ACTIVE_CORPUS)

    # ── 1. Warm the RAG pipeline ────────────────────────────────────────────
    log.info("Step 1/2 — Loading RAG pipeline...")
    t0 = time.perf_counter()
    try:
        pipeline = get_pipeline()
        log.info("  Warming FAISS via dummy query...")
        pipeline.retrieve("warm-up", k=1)
        log.info("RAG pipeline fully warm (%.1fs)", time.perf_counter() - t0)
    except Exception:
        log.exception("FATAL — RAG pipeline failed to initialise")
        raise

    # ── 2. Build the deep agent ──────────────────────────────────────────────
    log.info("Step 2/2 — Compiling deep agent...")
    t1 = time.perf_counter()
    try:
        _agent = build_agent()
        log.info("Deep agent ready (%.1fs)", time.perf_counter() - t1)
    except Exception:
        log.exception("FATAL — Deep agent failed to build")
        raise

    log.info("=" * 60)
    log.info("SERVER READY")
    log.info("=" * 60)

    yield

    log.info("SERVER SHUTDOWN")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Document RAG Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class SourceRef(BaseModel):
    id: str
    doc: str
    section: str
    display: str      # "CompanyName / Section" shown in the UI


class ChatResponse(BaseModel):
    response: str
    sources: List[SourceRef] = []


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    log.info("→ %s %s", request.method, request.url.path)
    response = await call_next(request)
    log.info("← %s %s  status=%d  %.2fs",
             request.method, request.url.path, response.status_code,
             time.perf_counter() - t0)
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg_text(content) -> str:
    """Extract plain text from a LangChain message content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            blk.get("text", "") if isinstance(blk, dict) else str(blk)
            for blk in content
            if not isinstance(blk, dict) or blk.get("type") == "text"
        )
    return str(content)


def _doc_display(doc: str, section: str) -> str:
    """Build a human-readable 'Company / Section' label for a citation."""
    _NAMES = {
        "aapl": "Apple", "cat": "Caterpillar", "jpm": "JPMorgan Chase",
        "ko": "Coca-Cola", "wmt": "Walmart",
    }
    prefix  = doc.split("-")[0].lower()
    company = _NAMES.get(prefix, doc.upper())
    sec     = section if section and section not in ("—", "None", "") else "Annual Report"
    return f"{company} / {sec}"


# ── Citation resolution ─────────────────────────────────────────────────────

# Matches [SOURCE:some_chunk_id] in the agent response
_SOURCE_MARKER_RE = re.compile(r"\[SOURCE:([\w\-_]+)\]")

# Matches the tool-output header for each chunk
_TOOL_CHUNK_RE = re.compile(
    r"\[\d+\]\s*Source:\s*([^|]+)\|\s*Section:\s*([^|]+)\|\s*ID:\s*([\w\-_]+)"
)


def _build_sources(all_messages, final_text: str) -> tuple[str, List[SourceRef]]:
    """
    1. Collect all (doc, section, chunk_id) tuples from tool outputs.
    2. Find which chunk IDs the agent actually cited via [SOURCE:id].
    3. Replace [SOURCE:id] in the response with numbered footnotes [1], [2].
    4. Return the cleaned response + ordered list of only cited SourceRefs.
    """
    # Step 1: build a lookup from chunk_id → SourceRef
    chunk_meta: dict[str, SourceRef] = {}
    tool_order: list[str] = []          # preserves retrieval order
    for msg in all_messages:
        if getattr(msg, "name", None) == "search_documents":
            tool_text = _msg_text(getattr(msg, "content", "") or "")
            for m in _TOOL_CHUNK_RE.finditer(tool_text):
                doc     = m.group(1).strip()
                section = m.group(2).strip()
                cid     = m.group(3).strip()
                if cid not in chunk_meta:
                    chunk_meta[cid] = SourceRef(
                        id=cid, doc=doc, section=section,
                        display=_doc_display(doc, section),
                    )
                    tool_order.append(cid)

    # Step 2: find which chunks were cited in the response
    cited_ids = _SOURCE_MARKER_RE.findall(final_text)

    if cited_ids:
        # Keep only cited chunks, deduplicated, in citation order
        seen: dict[str, int] = {}
        numbered: list[SourceRef] = []
        for cid in cited_ids:
            if cid not in seen and cid in chunk_meta:
                seen[cid] = len(numbered) + 1
                numbered.append(chunk_meta[cid])

        # Replace [SOURCE:id] with footnote markers [N]
        def _replace(m: re.Match) -> str:
            cid = m.group(1)
            return f"[{seen[cid]}]" if cid in seen else ""

        cleaned = _SOURCE_MARKER_RE.sub(_replace, final_text)
        return cleaned, numbered

    # Fallback: agent did not use markers — return top-3 by retrieval order
    top3 = [chunk_meta[cid] for cid in tool_order[:3] if cid in chunk_meta]
    return final_text, top3


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    model = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
    ready = _agent is not None
    return {"status": "ok" if ready else "initialising", "model": model}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages list is empty")
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent is still initialising")

    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    log.info("CHAT REQUEST — %d message(s) | last user: %r", len(req.messages), user_msg[:120])

    lc_messages = []
    for m in req.messages:
        if m.role == "user":
            lc_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=m.content))

    log.info("Invoking deep agent...")
    t0 = time.perf_counter()
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _agent.invoke({"messages": lc_messages})
        )
    except Exception as exc:
        log.error("Agent invocation failed: %s", exc)
        log.debug("Full traceback:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))

    log.info("Agent finished in %.2fs", time.perf_counter() - t0)

    all_messages = result.get("messages", [])
    log.debug("Agent produced %d messages", len(all_messages))

    # Extract final AI text
    final_content = ""
    for msg in reversed(all_messages):
        if isinstance(msg, AIMessage) and msg.content:
            text = _msg_text(msg.content)
            if text.strip():
                final_content = text
                break

    if not final_content:
        log.warning("No AIMessage with content in agent output")

    log.info("FINAL ANSWER (%d chars): %s", len(final_content), final_content[:200])

    # Build accurate citations
    cleaned_response, sources = _build_sources(all_messages, final_content)
    log.info("Citations: %s", [s.id for s in sources])

    return ChatResponse(response=cleaned_response, sources=sources)


# ---------------------------------------------------------------------------
# Document ingestion endpoint
# ---------------------------------------------------------------------------

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Accept a document upload, parse it, add to the RAG corpus live."""
    import tempfile, shutil

    log.info("INGEST REQUEST — file: %r  size hint: %s",
             file.filename, file.size)

    # Write upload to a temp file (FastAPI streams it)
    suffix = Path(file.filename or "upload").suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        from src.ingestion.ingest_document import ingest_file
        pipeline = get_pipeline()
        result   = await asyncio.get_event_loop().run_in_executor(
            None, lambda: ingest_file(tmp_path, display_name=file.filename,
                                      pipeline=pipeline)
        )
    except Exception as exc:
        log.error("Ingestion failed: %s", exc)
        log.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    log.info("INGEST COMPLETE — %s: %d chunks", result["doc_id"], result["chunk_count"])
    return result


# ---------------------------------------------------------------------------
# Document list endpoint
# ---------------------------------------------------------------------------

@app.get("/documents")
async def list_documents():
    """Return the document registry — used by the frontend document sidebar."""
    if not REGISTRY_PATH.exists():
        return {"documents": []}
    try:
        import json
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Could not read registry: %s", exc)
        raise HTTPException(status_code=500, detail="Could not read document registry")
