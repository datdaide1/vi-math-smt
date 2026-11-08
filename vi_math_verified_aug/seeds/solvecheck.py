"""Stage ``solvecheck`` (T4 §3.8 step 2, R1 §6 "should fix"): blind-solve every Vietnamese seed with the free
Gemini flash-lite pool; an English control decides whether a miss is the translation's fault.

Solvers: 12 lanes = 6 keys x {gemini-3.5-flash-lite, gemini-3.1-flash-lite}. The primary model alternates by
seed_id parity; if a call fails (429 / quota / ``LaneStall``) the other Gemini model, then Haiku, is used, so an
exhausted key or model never stalls the run. The model of every attempt is recorded.

Per seed: VI solve (temperature 0); on a miss a second VI sample (0.6); still missed -> EN control (0, then 0.6).
verdict = PASS (a VI sample reproduces the gold) | FAIL (VI missed twice, EN solved: the Vietnamese wording likely
changes the math) | UNINF (neither solved) | ERROR (every solver failed at the API level; retried on resume).
With ``fix=True`` a FAIL is sent straight to MTA-vi (``mta.translate_row``; its own G2 must PASS to accept).

Safety (S3 hang of 2026-11-07, see report §2.4): ``LanePool`` now raises ``LaneStall`` instead of waiting out a
multi-hour server cool-down, and a ``Watchdog`` aborts the run if nothing completes for 5 minutes (resume restarts
where it stopped). ``deadline`` (epoch seconds) stops starting new seeds so the freeze can happen on time.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import Counter

from ..common.ratelimit import Watchdog
from .assemble import current_rows
from .common import Store, client
from .mta import solve_once, translate_row, typed_gold

GEMINI = ("flash_lite35", "flash_lite")
ORDER_OVERRIDE: tuple[str, ...] | None = None      # e.g. ("haiku",) for the live-Haiku remainder


async def _solve_any(problem: str, lang: str, suffix: str, order: tuple[str, ...], item: str, gold: dict) -> dict:
    a: dict = {}
    for m in order:
        a = await solve_once(problem, lang, m + suffix, item, gold)
        if a.get("ok") is not None:
            return a
    return a


async def check(r: dict, store: Store) -> dict:
    key = r["seed_id"]
    old = store.get(key)
    if old and old.get("verdict") in ("PASS", "FAIL", "UNINF"):
        return old
    g = typed_gold(r)
    item = f"seed:{key}"
    prim = GEMINI[int(key, 16) % 2]
    order = ORDER_OVERRIDE or (prim, GEMINI[1 - GEMINI.index(prim)], "haiku")
    att, verdict = [], None
    for suffix in ("", "@0.6"):
        a = await _solve_any(r["problem_vi"], "vi", suffix, order, item, g)
        att.append({"lang": "vi", **{k: a.get(k) for k in ("solver", "model", "ok", "span", "err")}})
        if a.get("ok"):
            verdict = "PASS"
            break
    if verdict is None:
        for suffix in ("", "@0.6"):
            a = await _solve_any(r["problem_en"], "en", suffix, order, item, g)
            att.append({"lang": "en", **{k: a.get(k) for k in ("solver", "model", "ok", "span", "err")}})
            if a.get("ok"):
                verdict = "FAIL"
                break
        if verdict is None:
            verdict = "UNINF"
    if verdict != "PASS" and any(x.get("ok") is None for x in att):
        verdict = "ERROR"
    return store.put(key, {"seed_id": key, "source": r["source"], "index": r["index"], "verdict": verdict,
                           "solver_models": sorted({x.get("model") for x in att if x.get("model")}),
                           "attempts": att})


def run(limit: int | None = None, concurrency: int = 24, fix: bool = True, deadline: float | None = None,
        watchdog_s: float = 300.0, solvers: tuple[str, ...] | None = None, skip: set[str] | None = None) -> dict:
    global ORDER_OVERRIDE
    ORDER_OVERRIDE = solvers
    rows = current_rows()
    if skip:
        rows = [r for r in rows if r["seed_id"] not in skip]
    if limit:
        rows = rows[:limit]
    store, mstore = Store("solve"), Store("mta")
    c = client()
    todo = [r for r in rows if not (store.get(r["seed_id"]) and
                                    store.get(r["seed_id"])["verdict"] in ("PASS", "FAIL", "UNINF"))]
    fails_todo = [r for r in rows if store.get(r["seed_id"]) and store.get(r["seed_id"])["verdict"] == "FAIL"
                  and not (mstore.get(r["seed_id"]) and mstore.get(r["seed_id"]).get("status") in ("ACCEPT", "FAIL"))]
    print(f"[solve] {len(todo)} to check, {len(fails_todo)} earlier FAILs to fix", flush=True)

    async def main():
        sem, msem = asyncio.Semaphore(concurrency), asyncio.Semaphore(8)
        wd = Watchdog(watchdog_s)
        n = 0

        async def fix_one(r):
            async with msem:
                rec = await translate_row(r, mstore)
                wd.tick()
                print(f"[mta] {r['source']}#{r['index']}: {rec.get('status')} {rec.get('accepted_by') or ''}",
                      flush=True)

        async def one(r):
            nonlocal n
            if deadline and time.time() > deadline:
                return
            async with sem:
                if deadline and time.time() > deadline:
                    return
                try:
                    rec = await check(r, store)
                except Exception as e:
                    rec = {"verdict": "ERROR", "err": f"{type(e).__name__}: {e}"}
                wd.tick()
                n += 1
                if n % 100 == 0:
                    print(f"[solve] +{n} {dict(Counter(v.get('verdict') for v in store.rows.values()))}", flush=True)
            if fix and rec.get("verdict") == "FAIL":
                await fix_one(r)

        work = asyncio.ensure_future(asyncio.gather(*(one(r) for r in todo), *(fix_one(r) for r in fails_todo)))
        dog = asyncio.ensure_future(wd.run(poll_s=10))
        try:
            done, _ = await asyncio.wait({work, dog}, return_when=asyncio.FIRST_COMPLETED)
            if dog in done:
                work.cancel()
                dog.result()                       # raises LaneStall
            work.result()
        finally:
            dog.cancel()
            await c.aclose()
    asyncio.run(main())
    st = Counter(store.get(r["seed_id"])["verdict"] for r in rows if store.get(r["seed_id"]))
    out = {"n": len(rows), "checked": sum(st.values()), "verdicts": dict(st),
           "by_source": {s: dict(Counter(store.get(r["seed_id"])["verdict"] for r in rows
                                         if r["source"] == s and store.get(r["seed_id"])))
                         for s in sorted({r["source"] for r in rows})},
           "models": dict(Counter(m for v in store.rows.values() for m in v.get("solver_models") or [])),
           "mta_fail_fix": dict(Counter(mstore.get(r["seed_id"])["status"] for r in rows
                                        if store.get(r["seed_id"]) and store.get(r["seed_id"])["verdict"] == "FAIL"
                                        and mstore.get(r["seed_id"]))),
           "spend": c.costs.summary()}
    print(json.dumps(out, indent=1))
    return out
