"""G4 plumbing: the shared LLM client under ws G4 ($5.9 hard cap below the $6 Claude cap), prompt rendering and
resumable JSONL stores under data/pilot/symbolic_v2/. Reuses G3's request builder and lane settings.

Every item is a Vi-GSM8K *train* dev seed or an item generated from one (``split="dev"``), so Gemini may receive it.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from ..common.llm import LLMClient, LLMRequest, parse_json
from ..common.prompts import PromptRegistry
from ..pilot_symbolic import common as C3

ROOT = C3.ROOT
G3 = C3.OUT                                  # data/pilot/symbolic (shared seeds/slots/formalize)
G3RUN = G3 / "run_b"                         # G3 final run (edit-only renderer, X4 + G3 fixes)
OUT = ROOT / "data" / "pilot" / "symbolic_v2"
OUT.mkdir(parents=True, exist_ok=True)
WS = "G4"
BUDGET = 5.9
REG = PromptRegistry()
PROMPTS = {**C3.PROMPTS, "story": ("story_check", 1), "fluency": ("fluency", 1), "fluency2": ("fluency", 2), "irr": ("ir_render", 1)}
MODELS = C3.MODELS

_client: LLMClient | None = None
_sems: dict = {}


def client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(WS, budget_usd=BUDGET, lanes=C3.LANES, max_concurrency=200)
    return _client


async def call(rq: LLMRequest):
    prov = "claude" if rq.model.startswith("claude") else "gemini"
    s = _sems.get(rq.model)
    if s is None:
        s = _sems[rq.model] = asyncio.Semaphore(C3.PENDING[prov])
    async with s:
        return await client().call(rq)


def render(key: str, **kw):
    pid, v = PROMPTS[key]
    return REG.get(pid, v).render(**kw)


req = C3.req
resp_meta = C3.resp_meta


class Store:
    """One JSONL per stage under OUT; rows keyed by 'key'; last row wins; resumable."""

    def __init__(self, stage: str, base: Path | None = None):
        self.path = (base or OUT) / f"{stage}.jsonl"
        self.lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            for line in open(self.path, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    self.rows[r["key"]] = r

    def get(self, key: str) -> dict | None:
        return self.rows.get(key)

    def put(self, key: str, row: dict) -> dict:
        row = {"key": key, **row}
        with self.lock:
            self.rows[key] = row
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def rewrite(self) -> None:
        with self.lock, open(self.path, "w", encoding="utf-8") as f:
            for r in self.rows.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def g3(stage: str) -> dict[str, dict]:
    """Read-only G3 run_b store."""
    return Store(stage, G3RUN).rows


__all__ = ["ROOT", "G3", "G3RUN", "OUT", "WS", "MODELS", "client", "call", "render", "req", "resp_meta", "Store",
           "g3", "parse_json", "LLMRequest"]
