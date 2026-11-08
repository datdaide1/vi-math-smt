"""G5 arm-Y plumbing: ws G5 client ($11.9 hard cap under the $12 G5 cap), prompt rendering, resumable stores, and
``call_many`` (direct calls, or Message Batches chunked under the remaining $ cap for >= 100 Claude calls)."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

from ..common.costs import worst_case_cost
from ..common.llm import LLMClient, LLMRequest, parse_json
from ..common.prompts import PromptRegistry
from ..pilot_symbolic import common as C3

ROOT = C3.ROOT
WS = "G5"
BUDGET = 20.8                     # G4+G5 cap $25 (lead 03:20): G4 spent $4.15
REG = PromptRegistry()
PROMPTS = {"formalize": ("formalizer", 2), "back": ("back_formalizer", 1), "back_en": ("back_formalizer_en", 1),
           "editor": ("editor", 3), "fluency": ("fluency", 2), "irr": ("ir_render", 1), "irr_en": ("ir_render_en", 1),
           "story": ("story_check", 1), "judge": ("validity_judge", 2), "judge3": ("validity_judge", 3),
           "solve": ("solution_writer", 2)}
MODELS = C3.MODELS
MODELS.setdefault("g35l", "gemini-3.5-flash-lite")           # second free lane family (coordinator, G5)
# NVIDIA NIM cross-family back-formalizer candidates (lead 02:10: Gemini quota exhausted; training items only)
MODELS.setdefault("nim_gemma", "google/gemma-4-31b-it")
MODELS.setdefault("nim_glm", "z-ai/glm-5.3-flash")
MODELS.setdefault("nim_bolt", "nvidia/nemotron-3.5-bolt-e4b")
_INFL = int(os.environ.get("ARMY_CLAUDE_INFLIGHT", "14"))     # org-wide concurrency is shared with D1 (lead: <= 14)
PENDING_N = {"claude": _INFL, "gemini": 6, "nim": 35}
# real-time fallback for Batch (lead): G3's 50 RPM Claude lane would make 5k solver calls take ~1.7 h
from ..common.ratelimit import LaneConfig as _LC  # noqa: E402
LANES_G5 = {**C3.LANES, "claude": _LC(rpm=400, burst=10, max_inflight=_INFL, min_tpm=1_000_000,
                                           base_cooldown_s=10)}
req = C3.req
resp_meta = C3.resp_meta

_client: LLMClient | None = None
_sems: dict = {}


def client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(WS, budget_usd=BUDGET, lanes=LANES_G5, max_concurrency=200, enable_nim=True)
    return _client


def spent() -> float:
    return client().costs.spent()


async def call(rq: LLMRequest):
    prov = "claude" if rq.model.startswith("claude") else ("nim" if "/" in rq.model else "gemini")
    s = _sems.get(rq.model)
    if s is None:
        s = _sems[rq.model] = asyncio.Semaphore(PENDING_N[prov])
    async with s:
        return await client().call(rq)


def render(key: str, **kw):
    pid, v = PROMPTS[key]
    return REG.get(pid, v).render(**kw)


PENDING = ROOT / "data" / "train" / "Y" / "_work" / "pending_batches.jsonl"


def _pending() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if PENDING.exists():
        for l in open(PENDING, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out[r["batch_id"]] = {**out.get(r["batch_id"], {}), **r}
    return out


def _mark(batch_id: str, **kw):
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    with open(PENDING, "a", encoding="utf-8") as f:
        f.write(json.dumps({"batch_id": batch_id, **kw}, ensure_ascii=False) + "\n")


async def _wait_collect(batch_id: str, poll_s: float = 20) -> dict:
    c = client()
    while True:
        st = await c.batches.status(batch_id)
        if st.get("processing_status") == "ended":
            break
        await asyncio.sleep(poll_s)
    got = await c.batches.collect(batch_id)        # caches every successful response
    _mark(batch_id, status="collected")
    return got


async def recover_pending(hashes: set[str] | None = None):
    """Collect every recorded, uncollected batch (resume after an interruption: never re-submit, never pay twice).
    Batches whose requests overlap ``hashes`` are waited for; others are collected only if already ended."""
    c = client()
    for bid, r in _pending().items():
        if r.get("status") == "collected":
            continue
        overl = hashes is None or bool(set(r.get("hashes", [])) & hashes)
        st = await c.batches.status(bid)
        if st.get("processing_status") == "ended" or overl:
            print(f"[resume] collecting pending batch {bid} ({r.get('label')})", flush=True)
            await _wait_collect(bid)


async def call_many(reqs: list[LLMRequest], batch: bool = False, label: str = "") -> list:
    """Responses in input order. batch=True and >= 100 requests -> Message Batches, chunked so each chunk's
    worst-case reservation fits under the remaining cap; otherwise concurrent direct calls. Every submitted batch is
    recorded in data/train/Y/_work/pending_batches.jsonl before waiting, and recover_pending() collects it after an
    interruption, so a restarted stage hits the cache instead of paying again."""
    if not reqs:
        return []
    if os.environ.get("ARMY_DIRECT") == "1":           # lead: real-time API when Batch turnaround > 20 min
        batch = False
    if not batch or len(reqs) < 100:
        return list(await asyncio.gather(*(call(r) for r in reqs)))
    c = client()
    await recover_pending({r.request_hash("claude") for r in reqs})
    out: list = []
    i = 0
    while i < len(reqs):
        allow = BUDGET - c.costs.spent() - 0.1
        chunk, est = [], 0.0
        for r in reqs[i:]:
            e = worst_case_cost(r.est_input_tokens(), r.max_tokens, c._price("claude", r.model), batch=True)
            if chunk and (est + e > allow or len(chunk) >= 4000):
                break
            chunk.append(r)
            est += e
        if est > allow:
            raise RuntimeError(f"budget: cannot reserve {est:.2f} (allow {allow:.2f})")
        print(f"[{label}] batch {len(chunk)} reqs, reserve ${est:.2f}, spent ${c.costs.spent():.3f}", flush=True)
        man = await c.batches.submit(chunk)
        got: dict = {}
        if man.get("batch_id"):
            _mark(man["batch_id"], status="submitted", label=label, n=len(man["items"]),
                  hashes=list(man["items"]), ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
            got = await _wait_collect(man["batch_id"])
        for r in chunk:
            h = r.request_hash("claude")
            out.append(got[h] if h in got else await c.call(r))     # cached (or collected) -> no new charge
        i += len(chunk)
        print(f"[{label}] batch done, spent ${c.costs.spent():.3f}", flush=True)
    return out


async def call_many_chunks(reqs: list, batch: bool = False, label: str = "", chunk: int = 200):
    """Yield (offset, responses) per chunk so callers append results to their store as they arrive: a killed or
    crashed stage keeps every finished chunk (and every call is cached), so a restart never pays twice."""
    for i in range(0, len(reqs), chunk):
        yield i, await call_many(reqs[i:i + chunk], batch and len(reqs) >= 100, f"{label}[{i}]")


class Store:
    """One JSONL per stage; rows keyed by 'key'; last row wins; resumable."""

    def __init__(self, base: Path, stage: str, keep_rows: bool = True):
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"{stage}.jsonl"
        self.lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        self.keep = keep_rows
        if self.path.exists() and keep_rows:
            for line in open(self.path, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    self.rows[r["key"]] = r

    def get(self, key):
        return self.rows.get(key)

    def put(self, key, row):
        row = {"key": key, **row}
        with self.lock:
            if self.keep:
                self.rows[key] = row
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def rewrite(self):
        with self.lock, open(self.path, "w", encoding="utf-8") as f:
            for r in self.rows.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


__all__ = ["ROOT", "WS", "client", "spent", "call", "call_many", "render", "req", "resp_meta", "Store", "parse_json"]
