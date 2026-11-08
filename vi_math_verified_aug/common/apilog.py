"""Append-only JSONL API ledger (06_packaging §5.2).

One JSON object per line, one file per provider: ``logs/api/<WS>/<provider>.jsonl``.

Concurrency: a line is written with a single ``write`` on a file opened in append mode, while holding
(1) a process-local ``threading.Lock`` and (2) an OS-level exclusive lock on a sidecar ``<file>.lock``
(``msvcrt.locking`` on Windows, ``fcntl.flock`` elsewhere). So many threads, asyncio tasks and processes can
log to the same file without interleaving. Transient ``PermissionError`` on Windows (a reader such as an
editor or AV scanner holding the file) is retried with backoff.

Record fields (06 §5.2): ts_utc, ws, run_id, git_sha, provider, base_url, model_requested, model_returned,
params, prompt_id, prompt_version, prompt_sha256, request (full messages), response_text, raw_response,
finish_reason, usage, latency_ms, attempt, error_class, error, status, cost_usd, cache_hit, request_hash,
item_id, client_lib_version. Keys are never logged: the client never puts them in a record, and every
record is additionally scrubbed of known secret values before it is written.
"""
from __future__ import annotations

import datetime as _dt
import functools
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from .paths import REPO_ROOT, api_log_dir

REDACTED = "***REDACTED***"
_proc_locks: dict[str, threading.Lock] = {}
_proc_locks_guard = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    key = os.path.normcase(str(path.resolve()))
    with _proc_locks_guard:
        lk = _proc_locks.get(key)
        if lk is None:
            lk = _proc_locks[key] = threading.Lock()
        return lk


class _FileLock:
    """Exclusive inter-process lock on ``<target>.lock`` (blocking, with polling)."""

    def __init__(self, target: Path, timeout: float = 60.0):
        self.path = target.with_name(target.name + ".lock")
        self.timeout = timeout
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        delay = 0.002
        while True:
            try:
                fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            except PermissionError:
                fd = None
            if fd is not None:
                try:
                    if os.name == "nt":
                        import msvcrt
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.fd = fd
                    return self
                except OSError:
                    os.close(fd)
            if time.monotonic() > deadline:
                raise TimeoutError(f"could not lock {self.path} within {self.timeout}s")
            time.sleep(delay)
            delay = min(delay * 2, 0.1)

    def __exit__(self, *exc):
        fd, self.fd = self.fd, None
        if fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def append_line(path: str | os.PathLike, line: str, *, fsync: bool = False, retries: int = 20) -> None:
    """Append one line (newline added if missing) atomically w.r.t. other ``append_line`` callers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (line if line.endswith("\n") else line + "\n").encode("utf-8")
    with _thread_lock(path), _FileLock(path):
        for k in range(retries):
            try:
                with open(path, "ab") as f:
                    f.write(data)
                    f.flush()
                    if fsync:
                        os.fsync(f.fileno())
                return
            except PermissionError:
                if k == retries - 1:
                    raise
                time.sleep(0.05 * (k + 1))


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    """Yield records; skips a torn/garbled line (e.g. a crash mid-write) instead of failing."""
    p = Path(path)
    if not p.is_file():
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


@functools.lru_cache(maxsize=1)
def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def scrub(text: str, secrets: Iterable[str]) -> str:
    for s in secrets:
        if s and s in text:
            text = text.replace(s, REDACTED)
    return text


class ApiLogger:
    """Writes ledger records for one workstream (``ws``) and optional ``run_id``."""

    def __init__(self, ws: str, *, run_id: str | None = None, log_dir: str | os.PathLike | None = None,
                 secrets: Iterable[str] = (), fsync: bool = False):
        if not ws or any(c in ws for c in "/\\:"):
            raise ValueError(f"bad workstream name {ws!r}")
        self.ws = ws
        self.run_id = run_id
        self.dir = Path(log_dir) if log_dir else api_log_dir(ws)
        self.secrets = sorted({s for s in secrets if s}, key=len, reverse=True)
        self.fsync = fsync

    def path_for(self, provider: str) -> Path:
        return self.dir / f"{provider or 'misc'}.jsonl"

    def log(self, record: dict[str, Any]) -> dict[str, Any]:
        rec = {"ts_utc": utc_now(), "ws": self.ws, "run_id": self.run_id, "git_sha": git_sha()}
        rec.update(record)
        line = json.dumps(rec, ensure_ascii=False, default=str)
        line = scrub(line, self.secrets)
        append_line(self.path_for(str(rec.get("provider") or "misc")), line, fsync=self.fsync)
        return rec

    def records(self, provider: str) -> Iterator[dict]:
        return iter_jsonl(self.path_for(provider))
