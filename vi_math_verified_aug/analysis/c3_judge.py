"""D5 judging: Opus 5.5 + validity_judge.v2 (the G1-G4 judge of record, effort medium, max_tokens 3000) through the
Message Batches API (-50%), logged to logs/api/D5/, hard cap $10 for the whole workstream (CostTracker scope D5).

Used by C3 (R1b localization test) and by the D5 joint gate audit (c3_d5_audit). Every response is cached by the
shared LLM cache, so re-runs are free; results are appended to resumable JSONL stores (last row per key wins).

The request is built exactly as G4 built it (``pilot_symbolic.common.req`` + ``validity_judge`` v2 render), so the
labels are comparable with G1-G4. ``lang="en"`` uses ``validity_judge_en`` v1 (same rubric, English target
language; see prompts/validity_judge_en.v1.md) for the English mirror pools.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from .c3_data import ROOT, last_by_key, load_jsonl, lab_v2

WS = "D5"
BUDGET_USD = 4.8          # lead: D5 cap $10 -> $6 -> $4 -> $3.4 -> $4.8 (11-08 ~07:50, EN gate audit)
RESERVE_USD = 0.15
EFFORT, MAX_TOKENS = "medium", 1500   # D5 audit: 1500 (judge outputs ~500-700 tokens; halves the Batch worst-case reservation under the $3.4 cap). R1b used 3000.

_client = None


def client():
    global _client
    if _client is None:
        from ..common.llm import LLMClient
        from ..pilot_symbolic.common import LANES
        _client = LLMClient(WS, budget_usd=BUDGET_USD, lanes=LANES, max_concurrency=64)
    return _client


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.rows = last_by_key(load_jsonl(self.path))

    def get(self, k):
        return self.rows.get(k)

    def put(self, k, row):
        row = {"key": k, **row}
        with self.lock:
            self.rows[k] = row
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def op_instruction(op: str, lang: str = "vi") -> str:
    from ..pilot_gen.util import OP_INSTR
    if lang == "vi":
        return OP_INSTR.get(op, "")
    return OP_INSTR_EN.get(op, "")


OP_INSTR_EN = {
    "M1": "Change the numbers, keep the solution structure.",
    "M2": "Invert the problem: give the original problem's result and ask for one of the initial givens.",
    "M3": "Insert exactly one sensible calculation step in the middle of the story (not after the answer).",
    "M4": "Change one relation (add<->subtract, k times as many, percent...).",
    "M5": "Ask for a different quantity that needs at least two calculation steps.",
    "M6": "Combine with the following problem into one multi-step problem: ",
}


def build_request(it: dict):
    """it: key, seed, op, problem, lang ('vi'|'en')."""
    from ..common.prompts import PromptRegistry
    from ..pilot_symbolic.common import req
    lang = it.get("lang", "vi")
    pid, ver = ("validity_judge", 2) if lang == "vi" else ("validity_judge_en", 1)
    rp = PromptRegistry().get(pid, ver).render(seed=it["seed"], op=it["op"], op_instruction=op_instruction(it["op"], lang) + it.get("op_extra", ""),
                                               problem=it["problem"])
    return req("opus", rp, item=f"gen:D5:{it['key']}", effort=EFFORT, max_tokens=MAX_TOKENS, cache_system=True,
               tag=f"judge_opus_{pid}_v{ver}")


def _post(r) -> dict:
    from ..common.llm import parse_json
    v = parse_json(r.text) if r.ok else None
    v = v if isinstance(v, dict) else None
    return {"verdict": v, "parsed": v is not None, "labels": lab_v2(v) if v else None,
            "api_ok": bool(r.ok), "cost": r.cost_usd, "usage": r.usage, "finish": r.finish_reason, "err": r.error,
            "model_returned": r.model_returned, "cache_hit": r.cache_hit}


def estimate(items: list[dict]) -> float:
    from ..common.costs import worst_case_cost
    c = client()
    tot = 0.0
    for it in items:
        rq = build_request(it)
        tot += worst_case_cost(rq.est_input_tokens(), rq.max_tokens, c._price("claude", rq.model), batch=True)
    return tot


def _pending_path(store: Store) -> Path:
    return store.path.parent / "pending_batches.jsonl"


def _pending(store: Store) -> dict[str, dict]:
    """batch_id -> {"keys": {request_hash: item key}, "collected": bool}; append-only file, last state wins."""
    out: dict[str, dict] = {}
    for r in load_jsonl(_pending_path(store)):
        b = out.setdefault(r["batch_id"], {"keys": {}, "collected": False, "meta": {}})
        if r.get("keys"):
            b["keys"].update(r["keys"])
            b["meta"] = {k: r[k] for k in ("op", "lang") if k in r}
        if r.get("collected"):
            b["collected"] = True
    return out


def _append_pending(store: Store, row: dict) -> None:
    with open(_pending_path(store), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def _wait_collect(c, store: Store, batch_id: str, keys: dict[str, str], items: dict[str, dict], poll_s: float):
    while True:
        st = await c.batches.status(batch_id)
        if st.get("processing_status") == "ended":
            break
        await c.sleep(poll_s)
    got = await c.batches.collect(batch_id)
    for h, k in keys.items():
        r = got.get(h)
        if r is None:
            continue
        it = items.get(k, {})
        store.put(k, {**{f: it[f] for f in ("op", "lang") if f in it}, **_post(r), "batch_id": batch_id})
    _append_pending(store, {"batch_id": batch_id, "collected": True})


async def judge(items: list[dict], store: Store, *, max_usd: float | None = None, poll_s: float = 30.0) -> dict:
    """Judge every item not yet api_ok in ``store``.

    Crash-safe: every submitted batch is recorded (batch id + request-hash -> item key) in
    ``<store dir>/pending_batches.jsonl`` before polling; on restart, uncollected batches are collected first (never
    re-submitted), and results are appended to the store item by item. Chunks are sized so each batch's
    worst-case reservation fits under the D5 cap (and ``max_usd`` for this call)."""
    from ..common.costs import worst_case_cost
    c = client()
    start = c.costs.spent()
    by_key = {it["key"]: it for it in items}
    # 1) finish batches left by an interrupted run
    for bid, b in _pending(store).items():
        if not b["collected"]:
            print(f"[D5] collecting pending batch {bid} ({len(b['keys'])} items)", flush=True)
            await _wait_collect(c, store, bid, b["keys"], by_key, poll_s)
    todo = [it for it in items if not (store.get(it["key"]) or {}).get("api_ok")]
    n_done = 0
    while todo:
        allow = BUDGET_USD - c.costs.spent() - RESERVE_USD
        if max_usd is not None:
            allow = min(allow, max_usd - (c.costs.spent() - start))
        chunk, est = [], 0.0
        for it in todo:
            rq = build_request(it)
            e = worst_case_cost(rq.est_input_tokens(), rq.max_tokens, c._price("claude", rq.model), batch=True)
            if est + e > allow:
                break
            chunk.append((it, rq))
            est += e
        if not chunk:
            print(f"[D5] budget: allow ${allow:.2f} too small for the next item; {len(todo)} items left", flush=True)
            break
        print(f"[D5] batch of {len(chunk)}; reserve ${est:.2f}; spent so far ${c.costs.spent():.3f}", flush=True)
        man = await c.batches.submit([rq for _, rq in chunk])
        keys = {rq.request_hash("claude"): it["key"] for it, rq in chunk}
        if man["batch_id"]:
            sent = {h: k for h, k in keys.items() if h in man["items"]}
            _append_pending(store, {"batch_id": man["batch_id"], "keys": sent, "n": len(sent)})
            await _wait_collect(c, store, man["batch_id"], sent, by_key, poll_s)
        for it, rq in chunk:                       # requests answered from the cache at submit time: free
            if not (store.get(it["key"]) or {}).get("api_ok") and rq.request_hash("claude") not in man["items"]:
                r = await c.call(rq)
                store.put(it["key"], {**{f: it[f] for f in ("op", "lang") if f in it}, **_post(r)})
        n_done += len(chunk)
        todo = [it for it in todo[len(chunk):]]
        print(f"[D5] chunk done; spent ${c.costs.spent():.3f}", flush=True)
    return {"judged": n_done, "left": len(todo), "spent_this_call": c.costs.spent() - start, "spent_total": c.costs.spent()}


def run(coro):
    return client().run(coro)


# --------------------------------------------------------------------------------------- C3: R1b localization test
R1B_STORE = ROOT / "data" / "audit" / "c3" / "judge_opus_R1b.jsonl"


def r1b_items() -> list[dict]:
    d = ROOT / "data" / "pilot" / "symbolic_v2"
    seeds = {r["seed_id"]: r for r in load_jsonl(ROOT / "data" / "pilot" / "symbolic" / "seeds.jsonl")}
    out = []
    for r in last_by_key(load_jsonl(d / "render_R1b.jsonl")).values():
        if r.get("tag_ok") and r.get("text"):
            out.append({"key": f"R1b:{r['key']}", "slot": r["key"], "seed": seeds[r["seed_id"]]["text"], "op": r["op"],
                        "problem": r["text"], "lang": "vi"})
    return sorted(out, key=lambda x: x["key"])


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["r1b"]:
        its = r1b_items()
        print(f"R1b tag-passing items: {len(its)}; worst-case est ${estimate(its):.2f}")
        if sys.argv[2:3] == ["go"]:
            print(run(judge(its, Store(R1B_STORE), max_usd=3.0)))
