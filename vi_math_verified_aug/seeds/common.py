"""S3 shared plumbing: paths, ids, resumable JSONL stores, the LLM client (logs/api/S3, hard $ cap)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "seeds"
WORK = OUT / "_work"
REPORT = ROOT / "docs" / "research_2026-11" / "exec" / "S3_seed_pool.md"
ANN = ROOT / "docs" / "research_2026-11" / "content_review" / "annotations"

VI_GSM8K = DATA / "translation" / "gsm8k" / "gsm8k_train_vietnamese.jsonl"
EN_GSM8K = DATA / "generate" / "gsm8k" / "train.jsonl"
VI_MATH = DATA / "translation" / "MATH" / "train" / "{subject}_vi.jsonl"
EN_MATH = DATA / "generate" / "MATH" / "train" / "{subject}.json"
SUBJECTS = ("algebra", "prealgebra")
SOURCES = ("vi_gsm8k_train", "vi_math_train_algebra", "vi_math_train_prealgebra")

# Inputs never written to.
READ_ONLY = [DATA / "translation", DATA / "generate", DATA / "test", DATA / "eval", DATA / "vi_exam",
             DATA / "contam", DATA / "finetune"]

WS = "S3"
CLAUDE_CAP_USD = 4.0
MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5",
          "flash_lite": "gemini-3.1-flash-lite", "flash_lite35": "gemini-3.5-flash-lite"}
RNG_SEED = 20261107


# ------------------------------------------------------------------------------------------------ ids
def seed_id(source: str, index: int) -> str:
    """T1 §2.0: sha1(source || index)[:12]; '||' separator = the G1/G2/G3 pilot definition for GSM8K."""
    return hashlib.sha1(f"{source}||{index}".encode()).hexdigest()[:12]


def is_dev(sid: str) -> bool:
    """DEV = 5 % of seeds by hash (same rule as pilot_gen/pilot_symbolic)."""
    return int(sid, 16) % 20 == 0


# ------------------------------------------------------------------------------------------------ IO
def _guard(p: Path) -> Path:
    p = Path(p).resolve()
    for d in READ_ONLY:
        if d.resolve() == p or d.resolve() in p.parents:
            raise PermissionError(f"refusing to write inside a read-only input dir: {p}")
    return p


def read_jsonl(p: Path | str) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p: Path | str, rows: Iterable[dict]) -> int:
    """Atomic (``<p>.tmp`` then rename): an interruption never leaves a half-written output."""
    p = _guard(Path(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            s = json.dumps(r, ensure_ascii=False)
            if json.loads(s) != r:
                raise ValueError("JSON round-trip failed")
            f.write(s + "\n")
            n += 1
    os.replace(tmp, p)
    return n


def write_json(p: Path | str, obj: Any) -> None:
    p = _guard(Path(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, p)


def sha256_file(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path | str) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


class Store:
    """Append-only JSONL keyed by ``key``; last row wins; resumable across crashes."""

    def __init__(self, name: str):
        self.path = WORK / f"{name}.jsonl"
        self.lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            for line in open(self.path, encoding="utf-8"):
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:      # a torn last line after a crash
                        continue
                    self.rows[r["key"]] = r

    def get(self, key: str) -> dict | None:
        return self.rows.get(key)

    def put(self, key: str, row: dict) -> dict:
        row = {"key": key, **row}
        with self.lock:
            _guard(self.path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.rows[key] = row
        return row


# ------------------------------------------------------------------------------------------------ stats
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % CI, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


# ------------------------------------------------------------------------------------------------ LLM
_client = None
_sems: dict[str, asyncio.Semaphore] = {}
PENDING = {"claude": int(os.environ.get("S3_CLAUDE_INFLIGHT", 16)), "gemini": 14}


def client():
    from ..common.llm import LLMClient
    from ..common.ratelimit import LaneConfig
    global _client
    if _client is None:
        lanes = {"gemini": LaneConfig(rpm=12, burst=1, max_inflight=2, min_tpm=2_000, base_cooldown_s=30),
                 "claude": LaneConfig(rpm=float(os.environ.get("S3_CLAUDE_RPM", 50)), burst=5,
                                      max_inflight=int(os.environ.get("S3_CLAUDE_INFLIGHT", 16)), base_cooldown_s=15)}
        _client = LLMClient(WS, budget_usd=CLAUDE_CAP_USD, lanes=lanes, max_concurrency=64)
    return _client


async def call(rq):
    prov = "claude" if rq.model.startswith("claude") else "gemini"
    s = _sems.get(rq.model)
    if s is None:
        s = _sems[rq.model] = asyncio.Semaphore(PENDING[prov])
    async with s:
        return await client().call(rq)


_REG = None


def render(prompt_id: str, version: int = 1, **kw):
    global _REG
    if _REG is None:
        from ..common.prompts import PromptRegistry
        _REG = PromptRegistry()
    return _REG.get(prompt_id, version).render(**kw)


def req(model_key: str, rp, *, item: str, max_tokens: int = 3000, temperature: float | None = 0.0,
        effort: str | None = None, tag: str = "", sample_idx: int = 0, json_mode: bool = False,
        split: str = "train"):
    """Build an LLMRequest. Claude: ``output_config.effort`` (low by default), no temperature (the Claude 5.5
    models reject sampling params with effort set — same convention as G3). Gemini: temperature + seed 42,
    default thinking (as G3)."""
    from ..common.llm import LLMRequest
    model = MODELS[model_key]
    extra: dict = {}
    seed = None
    if model.startswith("claude"):
        extra["output_config"] = {"effort": effort or "low"}
        temperature = None
    else:
        seed = 42
    return LLMRequest(model=model, system=rp.system or None, messages=[{"role": "user", "content": rp.user}],
                      max_tokens=max_tokens, temperature=temperature, seed=seed, extra=extra, item_id=item,
                      split=split, prompt_id=rp.prompt_id, prompt_version=rp.version, prompt_sha256=rp.sha256,
                      tag=tag, sample_idx=sample_idx, json_mode=json_mode)


def resp_meta(r) -> dict:
    return {"api_ok": bool(r.ok), "model": r.model, "cost": r.cost_usd, "usage": r.usage,
            "cache_hit": r.cache_hit, "finish": r.finish_reason, "err": r.error}


def spend() -> dict:
    c = client()
    return c.costs.summary()
