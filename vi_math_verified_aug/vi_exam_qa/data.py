"""Paths and streaming loaders for Q1 (no grader / LLM imports here, so both venvs can use it)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..common.paths import REPO_ROOT

VI_EXAM = REPO_ROOT / "data" / "vi_exam"
ITEMS = VI_EXAM / "items.jsonl"
PRIVATE = VI_EXAM / "items_private.jsonl"
CLAUDE_KEYS = VI_EXAM / "claude_keys.jsonl"
QUEUE = VI_EXAM / "qa_queue.jsonl"
QUEUE_FREEZE = VI_EXAM / "qa_queue.freeze.json"
FLAGS = REPO_ROOT / "data" / "contam" / "vi_exam_flags.jsonl"

OUT = REPO_ROOT / "data" / "vi_exam_qa"
KEYS_TABLE = OUT / "keys_table.jsonl"          # per Core item: re-typed keys + grader agreements
SELECTION = OUT / "selection.json"             # adjudication set + audit sample (seeded)
STAGE_OUT = {                                  # raw LLM results per stage (one row per item)
    "solve2": OUT / "solve2.jsonl",
    "adjudicate": OUT / "adjudications.jsonl",
    "audit": OUT / "audit_solves.jsonl",
    "audit_judge": OUT / "audit_judgements.jsonl",
    "audit2": OUT / "audit2_solves.jsonl",             # second, fresh audit of core_v1.1 (new seed)
    "audit2_judge": OUT / "audit2_judgements.jsonl",
}
CORE_V1 = OUT / "core_v1.jsonl"
DATACARD = OUT / "core_v1.datacard.md"
SHA_FILE = OUT / "core_v1.sha256"
STATS = OUT / "q1_stats.json"
REPORT = REPO_ROOT / "docs" / "research_2026-11" / "exec" / "Q1_benchmark_qa.md"

WS = "Q1"
MODEL = "claude-opus-5-5"
CAP_USD = 15.5                                 # lead approved +$5.5 on 2026-11-07 (was $10)
AUDIT_N = 150
AUDIT_SEED = 20261107
AUDIT2_SEED = 20261108


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_file(path: Path, *, normalize_eol: bool = False) -> str:
    b = path.read_bytes()
    if normalize_eol:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def core_items() -> list[dict]:
    """Core candidates of X2's items.jsonl (1,585), in file order. ~13 MB file, streamed."""
    return [it for it in iter_jsonl(ITEMS) if it.get("core_candidate")]


def private_by_id(ids: set[str] | None = None) -> dict[str, dict]:
    out = {}
    for r in iter_jsonl(PRIVATE):
        if ids is None or r["id"] in ids:
            out[r["id"]] = r.get("qa_private") or {}
    return out


def claude_keys_by_id(ids: set[str] | None = None) -> dict[str, dict]:
    """X2's blind Claude key rows (last row per id wins, as X2's build does)."""
    out = {}
    for r in iter_jsonl(CLAUDE_KEYS):
        if ids is None or r["id"] in ids:
            out[r["id"]] = r
    return out


def stage_rows(stage: str) -> dict[str, dict]:
    """Latest successful row per item for a stage."""
    out: dict[str, dict] = {}
    for r in iter_jsonl(STAGE_OUT[stage]):
        if r.get("ok") or r["item_id"] not in out:
            out[r["item_id"]] = r
    return out


def load_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
