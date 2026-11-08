"""G3 plumbing: the shared LLM client (logs/api/G3, hard $ cap), prompt rendering and a resumable JSONL store.

Every call goes through ``common.llm.LLMClient`` (cache, budget, routing guard). Every item is a Vi-GSM8K *train*
dev seed or an item generated from one (``split="dev"``), so Gemini may receive it.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path

from ..common.llm import LLMClient, LLMRequest, parse_json
from ..common.prompts import PromptRegistry
from ..common.ratelimit import LaneConfig

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "pilot" / "symbolic"
OUT.mkdir(parents=True, exist_ok=True)
WS = "G3"
BUDGET = 5.9                       # hard cap below the $6 Claude cap given to G3
REG = PromptRegistry()

MODELS = {"opus": "claude-opus-5-5", "haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5",
          "g31l": "gemini-3.1-flash-lite"}
LANES = {"gemini": LaneConfig(rpm=10, burst=1, max_inflight=1, min_tpm=2_000, base_cooldown_s=30),
         "claude": LaneConfig(rpm=50, burst=5, max_inflight=16, base_cooldown_s=15)}
PENDING = {"claude": 16, "gemini": 6}   # admission per model (the client reserves budget before waiting)

PROMPTS = {"formalize": ("formalizer", 2), "back": ("back_formalizer", 1), "editor": ("editor", 3),
           "judge": ("validity_judge", 2), "judge3": ("validity_judge", 3), "solve": ("solution_writer", 2)}

_client: LLMClient | None = None
_sems: dict = {}


def client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(WS, budget_usd=BUDGET, lanes=LANES, max_concurrency=200)
    return _client


async def call(rq: LLMRequest):
    prov = "claude" if rq.model.startswith("claude") else "gemini"
    s = _sems.get(rq.model)
    if s is None:
        s = _sems[rq.model] = asyncio.Semaphore(PENDING[prov])
    async with s:
        return await client().call(rq)


def render(key: str, **kw):
    pid, v = PROMPTS[key]
    return REG.get(pid, v).render(**kw)


def req(model_key: str, rp, *, item: str, effort: str | None = None, max_tokens: int = 4000,
        temperature: float | None = None, cache_system: bool = False, tag: str = "", sample_idx: int = 0,
        json_mode: bool = False, extra_user: str = "") -> LLMRequest:
    model = MODELS[model_key]
    extra: dict = {}
    system = rp.system or None
    if model.startswith("claude"):
        extra["output_config"] = {"effort": effort or "low"}
        if cache_system and system:
            extra["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            system = None
        temperature = None
    user = rp.user + (("\n\n" + extra_user) if extra_user else "")
    return LLMRequest(model=model, system=system, messages=[{"role": "user", "content": user}],
                      max_tokens=max_tokens, temperature=temperature, extra=extra, item_id=item, split="dev",
                      prompt_id=rp.prompt_id, prompt_version=rp.version, prompt_sha256=rp.sha256, tag=tag,
                      sample_idx=sample_idx, json_mode=json_mode)


def resp_meta(r) -> dict:
    return {"api_ok": bool(r.ok), "cost": r.cost_usd, "usage": r.usage, "latency_ms": r.latency_ms,
            "cache_hit": r.cache_hit, "finish": r.finish_reason, "err": r.error, "model_returned": r.model_returned}


SHARED_STAGES = {"formalize"}       # shared by every run; per-run stages live in OUT/<run>/


def run_dir() -> Path:
    """Per-run directory: run_b (default; X4 with the G3 fixes, editor.v3) or run_a_x4_as_delivered."""
    d = OUT / os.environ.get("G3_RUN", "run_b")
    d.mkdir(parents=True, exist_ok=True)
    return d


class Store:
    """One JSONL per stage; rows keyed by 'key'; last row wins; resumable."""

    def __init__(self, stage: str):
        self.path = (OUT if stage in SHARED_STAGES else run_dir()) / f"{stage}.jsonl"
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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


__all__ = ["ROOT", "OUT", "run_dir", "WS", "MODELS", "client", "call", "render", "req", "resp_meta", "Store", "load_jsonl",
           "parse_json", "LLMRequest"]
