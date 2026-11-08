"""Content-addressed response cache: ``request_hash -> successful response`` in SQLite (WAL).

Why SQLite and not "the logs are the cache" (G1's approach): G1 re-read every JSONL ledger into memory at
start-up, which costs RAM proportional to all calls ever made (the laptop has ~1 GB free) and raced with
concurrent writers. SQLite gives O(1) lookups, cross-process safety and constant memory. The JSONL ledger
stays the audit record; ``ResponseCache.rebuild_from_logs`` can regenerate the cache from it.

Only successful responses are cached. A cache hit never touches spend accounting (see ``costs``).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from .apilog import iter_jsonl, utc_now

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    request_hash TEXT PRIMARY KEY,
    provider     TEXT NOT NULL,
    model        TEXT NOT NULL,
    created_utc  TEXT NOT NULL,
    response     TEXT NOT NULL
);
"""


def payload_from_log(r: dict[str, Any]) -> dict[str, Any]:
    """The cached form of a response, rebuilt from a ledger record (same keys ``common.llm`` caches)."""
    return {"text": r.get("response_text"), "finish_reason": r.get("finish_reason"),
            "usage": r.get("usage") or {}, "model_returned": r.get("model_returned")}


def connect(path: str | os.PathLike) -> sqlite3.Connection:
    p = Path(path)
    if str(path) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=60.0, isolation_level=None, check_same_thread=False)
    con.execute("PRAGMA busy_timeout=60000")
    if str(path) != ":memory:":
        con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


class ResponseCache:
    def __init__(self, path: str | os.PathLike):
        self.path = str(path)
        self._con = connect(path)
        self._con.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def get(self, request_hash: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._con.execute("SELECT response FROM responses WHERE request_hash=?",
                                    (request_hash,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, request_hash: str, provider: str, model: str, response: dict[str, Any]) -> None:
        blob = json.dumps(response, ensure_ascii=False, default=str)
        with self._lock:
            # first writer wins: a cached response is never silently replaced
            self._con.execute("INSERT OR IGNORE INTO responses VALUES (?,?,?,?,?)",
                              (request_hash, provider, model, utc_now(), blob))

    def __contains__(self, request_hash: str) -> bool:
        with self._lock:
            return self._con.execute("SELECT 1 FROM responses WHERE request_hash=?",
                                     (request_hash,)).fetchone() is not None

    def __len__(self) -> int:
        with self._lock:
            return self._con.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def rebuild_from_logs(self, paths: Iterable[str | os.PathLike]) -> int:
        """Re-insert successful, non-cache-hit ledger records (format written by ``common.llm``)."""
        n = 0
        for p in paths:
            for r in iter_jsonl(p):
                if r.get("ok") and not r.get("cache_hit") and r.get("request_hash"):
                    self.put(r["request_hash"], r.get("provider", ""), r.get("model_requested", ""),
                             payload_from_log(r))
                    n += 1
        return n

    def close(self) -> None:
        with self._lock:
            self._con.close()
