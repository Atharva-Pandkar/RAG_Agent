"""
Centralised logging setup for the 10-K RAG backend.

Every module imports `get_logger(__name__)` — all loggers share the same
handlers so output is consistent whether you read the console or the file.

Log file: app/backend/logs/rag_backend.log  (rotates at 5 MB, keeps 3 files)
Console : coloured output via a simple formatter

Log levels used across the app
-------------------------------
DEBUG   — internal step detail (chunk scores, token counts, message bodies)
INFO    — normal lifecycle events (request received, tool called, answer ready)
WARNING — recoverable issues (empty retrieval, missing env var)
ERROR   — exceptions caught and returned to caller
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_INITIALISED = False

LOG_DIR  = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "rag_backend.log"

# Compact format for the console; full format (with module) for the file
_CONSOLE_FMT = "%(asctime)s [%(levelname)-8s] %(message)s"
_FILE_FMT    = "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d — %(message)s"
_DATE_FMT    = "%H:%M:%S"


def _setup() -> None:
    global _INITIALISED
    if _INITIALISED:
        return
    _INITIALISED = True

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("rag")
    root.setLevel(logging.DEBUG)

    # ── Console handler ────────────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    root.addHandler(ch)

    # ── Rotating file handler ──────────────────────────────────────────────
    fh = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    root.addHandler(fh)

    root.info("Logging initialised — file: %s", LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'rag' root hierarchy."""
    _setup()
    # Strip the package prefix so names are short: 'rag.main', 'rag.agent', etc.
    short = name.replace("app.backend.", "")
    return logging.getLogger(f"rag.{short}")
