"""Solve-check remainder with Claude Haiku via the Message Batches API (50 % price), for seeds the free Gemini
pool cannot reach tonight (per-key 500 requests/day caps). Same verdict logic as ``solvecheck``:

    stage 1: VI solve (sample 0)                       -> PASS if correct
    stage 2 (stage-1 misses): VI sample 1 + EN control -> PASS if VI correct, FAIL if only EN correct, else UNINF

Results go to ``_work/solve_batch.jsonl`` (``assemble`` merges it with ``solve.jsonl``; a Gemini verdict, if any,
wins). The checker model is recorded per item. CLI:

    python -m vi_math_verified_aug.seeds.solvebatch submit1 --start N   # seeds after position N of the todo list
    python -m vi_math_verified_aug.seeds.solvebatch collect1
    python -m vi_math_verified_aug.seeds.solvebatch submit2 / collect2
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter

from .assemble import current_rows
from .common import WORK, Store, client, read_jsonl, render, req, write_json
from .mta import typed_gold

STATE = WORK / "solve_batch_state.json"
MODEL_KEY = "haiku"


def _req(problem: str, lang: str, sid: str, sample: int):
    rp = render("seed_solve" if lang == "vi" else "seed_solve_en", problem=problem)
    return req(MODEL_KEY, rp, item=f"seed:{sid}", max_tokens=1200, effort="low", sample_idx=sample,
               tag=f"S3-solvebatch-{lang}-s{sample}")


def _state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def _gemini_done() -> set[str]:
    p = WORK / "solve.jsonl"
    if not p.exists():
        return set()
    return {r["key"] for r in read_jsonl(p) if r.get("verdict") in ("PASS", "FAIL", "UNINF")}


async def _submit(reqs):
    c = client()
    try:
        return await c.batches.submit(reqs)
    finally:
        await c.aclose()


async def _collect(batch_id):
    c = client()
    try:
        st = await c.batches.status(batch_id)
        if st.get("processing_status") != "ended":
            return st, None
        return st, await c.batches.collect(batch_id)
    finally:
        await c.aclose()


def submit1(start: int, stage: str = "stage1") -> None:
    done = _gemini_done()
    st0 = _state()
    taken = set(st0.get("stage1", {}).get("seed_ids", [])) if stage != "stage1" else set()
    rows = [r for r in current_rows() if r["seed_id"] not in done][start:]
    rows = [r for r in rows if r["seed_id"] not in taken]
    reqs = [_req(r["problem_vi"], "vi", r["seed_id"], 0) for r in rows]
    man = asyncio.run(_submit(reqs))
    st = _state()
    st[stage] = {"batch_id": man["batch_id"], "seed_ids": [r["seed_id"] for r in rows],
                    "n": len(rows), "n_cached": man.get("n_cached")}
    write_json(STATE, st)
    print(json.dumps({"batch_id": man["batch_id"], "n": len(rows), "cached": man.get("n_cached")}))


def _grade(text, gold):
    from ..eval.math_verify import grade
    g = grade(text, gold)
    return bool(g["correct"]), g["span"]


def collect(stage: str) -> None:
    st = _state()
    s = st[stage]
    status, res = asyncio.run(_collect(s["batch_id"])) if s.get("batch_id") else ({}, {})
    if res is None:
        print(json.dumps({"status": status.get("processing_status"), "counts": status.get("request_counts")}))
        return
    R = {r["seed_id"]: r for r in current_rows()}
    store = Store("solve_batch")

    async def resolve(reqs):                       # cached requests are not in the batch result: read the cache
        c = client()
        try:
            return [await c.call(q) for q in reqs]
        finally:
            await c.aclose()
    if stage.startswith("stage1"):
        ids = s["seed_ids"]
        reqs = [_req(R[i]["problem_vi"], "vi", i, 0) for i in ids]
        resps = [res.get(q.request_hash("claude")) for q in reqs]
        miss_idx = [k for k, r in enumerate(resps) if r is None]
        if miss_idx:
            got = asyncio.run(resolve([reqs[k] for k in miss_idx]))
            for k, r in zip(miss_idx, got):
                resps[k] = r
        need2 = []
        for i, r in zip(ids, resps):
            if not (r and r.ok):
                need2.append(i)
                continue
            ok, span = _grade(r.text, typed_gold(R[i]))
            att = [{"lang": "vi", "solver": "haiku-batch", "model": r.model, "ok": ok, "span": span}]
            if ok:
                store.put(i, {"seed_id": i, "verdict": "PASS", "solver_models": [r.model], "attempts": att})
            else:
                store.put(i, {"seed_id": i, "verdict": "PENDING2", "solver_models": [r.model], "attempts": att})
                need2.append(i)
        st[stage]["collected"] = True
        st["need2"] = sorted(set(st.get("need2", [])) | set(need2))
        write_json(STATE, st)
        print(json.dumps({"stage1": dict(Counter(v["verdict"] for v in store.rows.values())), "need2": len(need2)}))
        return
    ids = st["need2"]
    pairs = [(i, _req(R[i]["problem_vi"], "vi", i, 1), _req(R[i]["problem_en"], "en", i, 0)) for i in ids]
    flat = [q for _, a, b in pairs for q in (a, b)]
    resps = {q.request_hash("claude"): res.get(q.request_hash("claude")) for q in flat}
    missing = [q for q in flat if resps[q.request_hash("claude")] is None]
    if missing:
        for q, r in zip(missing, asyncio.run(resolve(missing))):
            resps[q.request_hash("claude")] = r
    for i, qv, qe in pairs:
        g = typed_gold(R[i])
        old = store.get(i) or {"attempts": []}
        att = list(old.get("attempts", []))
        rv, re_ = resps[qv.request_hash("claude")], resps[qe.request_hash("claude")]
        okv, spv = _grade(rv.text, g) if rv and rv.ok else (None, None)
        oke, spe = _grade(re_.text, g) if re_ and re_.ok else (None, None)
        att += [{"lang": "vi", "solver": "haiku-batch@s1", "model": "claude-haiku-5-5", "ok": okv, "span": spv},
                {"lang": "en", "solver": "haiku-batch", "model": "claude-haiku-5-5", "ok": oke, "span": spe}]
        if okv:
            v = "PASS"
        elif okv is None or oke is None:
            v = "ERROR"
        else:
            v = "FAIL" if oke else "UNINF"
        store.put(i, {"seed_id": i, "verdict": v, "solver_models": ["claude-haiku-5-5"], "attempts": att})
    print(json.dumps({"final": dict(Counter(v["verdict"] for v in store.rows.values()))}))


def submit2() -> None:
    st = _state()
    R = {r["seed_id"]: r for r in current_rows()}
    reqs = []
    for i in st["need2"]:
        reqs += [_req(R[i]["problem_vi"], "vi", i, 1), _req(R[i]["problem_en"], "en", i, 0)]
    man = asyncio.run(_submit(reqs))
    st["stage2"] = {"batch_id": man["batch_id"], "n": len(reqs), "n_cached": man.get("n_cached")}
    write_json(STATE, st)
    print(json.dumps({"batch_id": man["batch_id"], "n": len(reqs)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit1", "collect1", "submit1b", "collect1b", "submit2", "collect2"])
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "submit1":
        submit1(a.start)
    elif a.cmd == "collect1":
        collect("stage1")
    elif a.cmd == "submit1b":
        submit1(a.start, "stage1b")
    elif a.cmd == "collect1b":
        collect("stage1b")
    elif a.cmd == "submit2":
        submit2()
    else:
        collect("stage2")


if __name__ == "__main__":
    main()
