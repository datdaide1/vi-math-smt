"""D1 driver: generation -> parse/format -> blind solve -> flash-lite screen (-> Opus audit), resumable.

    python -m vi_math_verified_aug.scale_gen.run slots                         # data/train/slots_v1.jsonl
    python -m vi_math_verified_aug.scale_gen.run pilot --n 100 --cfgs A,B      # 2 x 100 VI items, provisional seeds
    python -m vi_math_verified_aug.scale_gen.run gen --langs vi,en --n 12000 --cfg B
    python -m vi_math_verified_aug.scale_gen.run audit --lang vi --cfg B --n-plus 60 --n-rej 30
    python -m vi_math_verified_aug.scale_gen.run status

Store keys are ``<lang>|<cfg>|<slot key>``; rows live in ``data/train/N/_work/<lang>/{gen,solve,lite,audit}.jsonl``.
Re-running any command resumes: finished rows are skipped, un-collected batches are re-attached, cached requests
are free.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter

from ..common.llm import parse_json
from . import checks as C
from . import lite as L
from .batchrun import BatchManager, BudgetStop
from .common import (GEN_CFG, OP_INSTR, POOL, SLOTS, WORK, Store, audit_request, client, gen_request, read_jsonl,
                     resp_meta, seed_text, solve_request)
from .slots import build_slots, create_shared, load_slots
from . import screen as SCR


def prio(slot_key: str) -> str:
    """Selection priority shared by both languages and by finalize: sha1 of 'D1sel|<slot key>'."""
    import hashlib
    return hashlib.sha1(f"D1sel|{slot_key}".encode()).hexdigest()

EST = {"gen:A": 0.00068, "gen:B": 0.0003, "solve": 0.00012, "audit": 0.010, "hv3": 0.0004}  # $/item: pilot means (A gen $0.00061, B $0.00023, solve $0.00009-0.00012) + margin
CHUNK = 1000
REALTIME: set[str] = set()      # stages run on the real-time API instead of Batch (--realtime solve,hv3)
_BM: BatchManager | None = None
_STORES: dict[tuple[str, str], Store] = {}


def bm() -> BatchManager:
    global _BM
    if _BM is None:
        _BM = BatchManager()
    return _BM


def store(lang: str, stage: str) -> Store:
    k = (lang, stage)
    if k not in _STORES:
        _STORES[k] = Store(WORK / lang / f"{stage}.jsonl")
    return _STORES[k]


# ------------------------------------------------------------------------------------------------ seeds
def pool_seeds() -> dict[str, dict]:
    return {s["seed_id"]: s for s in read_jsonl(POOL)}


def provisional_seeds() -> dict[str, dict]:
    """Pilot only (before S3 freezes pool_v1): S3's current rows, minus dev / must-fix / decontam-dropped seeds."""
    W = POOL.parent / "_work"
    det = {d["seed_id"]: d for d in read_jsonl(W / "detect.jsonl")}
    dec = {d["seed_id"]: d for d in read_jsonl(W / "decontam.jsonl")}
    out = {}
    for a in read_jsonl(W / "aligned.jsonl"):
        d = det[a["seed_id"]]
        if a["dev"] or d["must"] or dec.get(a["seed_id"], {}).get("drop"):
            continue
        out[a["seed_id"]] = {"seed_id": a["seed_id"], "source": a["source"], "dataset": a["dataset"],
                             "index": a["index"], "problem": d["problem_vi"], "en": {"problem": a["problem_en"]}}
    return out


# ------------------------------------------------------------------------------------------------ handlers
def _slot_of(key: str, slots_by_key: dict) -> dict:
    return slots_by_key[key.split("|", 2)[2]]


class Driver:
    def __init__(self, lang: str, cfg: str, slots: list[dict], seeds: dict[str, dict], *, lite: bool = True,
                 target_plus: int | None = None, target_common: int | None = None):
        self.lang, self.cfg, self.slots, self.seeds, self.do_lite = lang, cfg, slots, seeds, lite
        self.target_plus = target_plus          # critical path: stop the lite screen once this many N+ candidates
        self.stop_lite = False
        self.hold = None                        # callable -> True: do not submit generation yet (VI first, then EN)
        self.by_key = {s["key"]: s for s in slots}
        self.gen, self.solve, self.lite = store(lang, "gen"), store(lang, "solve"), store(lang, "lite")
        self.lite_task: asyncio.Task | None = None
        self.hv3 = store(lang, "hv3")
        self.target_common = target_common      # stop the Haiku-v3 screen once this many items pass the common screen
        # lead 23:20: process strictly in slots_v1 rank order (breadth over seeds first: all j=1 slots, then j=2 ...)
        self.prio_slots = sorted(slots, key=lambda s: s["rank"])
        self.hv3_only: set[str] | None = None   # slot keys eligible for the Haiku-v3 screen (e.g. D5's S+ seeds only)

    def k(self, slot: dict) -> str:
        return f"{self.lang}|{self.cfg}|{slot['key']}"

    # ---- gen
    def gen_req(self, slot: dict, attempt: int):
        seed = self.seeds[slot["seed_id"]]
        partner = self.seeds.get(slot["partner"]) if slot.get("partner") else None
        r = gen_request(slot, seed, partner, self.lang, self.cfg)
        if attempt > 1:                       # one retry after a cut-off / error: 2x max_tokens (G2 policy)
            r.max_tokens *= 2
        return r

    def exp_hashes(self, slot: dict) -> tuple[str, str]:
        """Request hashes of attempt 1 / 2 for this slot under the CURRENT seeds (text, partner, op, card, prompt)."""
        h = self._exp.get(slot["key"]) if hasattr(self, "_exp") else None
        if h is None:
            if not hasattr(self, "_exp"):
                self._exp = {}
            h = (self.gen_req(slot, 1).request_hash("claude"), self.gen_req(slot, 2).request_hash("claude"))
            self._exp[slot["key"]] = h
        return h

    def needs_gen(self, slot: dict) -> int:
        """0 = done, else the attempt number to submit. A stored row whose request differs from the current one
        (seed text re-translated by S3, M6 partner changed, ...) is regenerated; unchanged ones are reused."""
        row = self.gen.get(self.k(slot))
        if row is None:
            return 1
        h1, h2 = self.exp_hashes(slot)
        rh = row.get("request_hash")
        if rh not in (h1, h2):
            return 1
        if row.get("api_ok") and (row.get("problem") or row.get("finish") != "max_tokens"):
            return 0
        return 2 if rh == h1 else 0

    async def on_gen(self, key: str, resp, attempt: int | None = None) -> None:
        slot = _slot_of(key, self.by_key)
        att = attempt or (2 if resp.request_hash == self.exp_hashes(slot)[1] else 1)
        prev = self.gen.get(key)
        if prev is not None and prev.get("request_hash") not in self.exp_hashes(slot):
            # regenerated for a changed seed / partner: invalidate the old item's downstream rows
            for st in (self.solve, self.lite, self.hv3):
                if st.get(key) is not None:
                    st.put(key, {"api_ok": False, "superseded": True, "attempt": 0,
                                 "old_gen_request_hash": prev.get("request_hash")})
        p = C.parse_gen(resp.text if resp.ok else None, self.lang, slot["op"]) if resp.ok else \
            {"fmt_ok": False, "fmt_flags": {"api_error": True}}
        self.gen.put(key, {"lang": self.lang, "cfg": self.cfg, "slot": slot["key"], "seed_id": slot["seed_id"],
                           "op": slot["op"], "card": slot["card"], "partner": slot.get("partner"), "attempt": att,
                           **p, **resp_meta(resp), "request_hash": resp.request_hash})

    # ---- solve
    def solve_ready(self, outstanding: set[str]) -> list[str]:
        out = []
        for s in self.slots:
            k = self.k(s)
            g = self.gen.get(k)
            if g and g.get("fmt_ok") and k not in outstanding and not (self.solve.get(k) or {}).get("api_ok"):
                if (self.solve.get(k) or {}).get("attempt", 0) >= 2:
                    continue
                out.append(k)
        return out

    async def on_solve(self, key: str, resp) -> None:
        g = self.gen.get(key)
        prev = self.solve.get(key) or {}
        txt = resp.text if resp.ok else None
        self.solve.put(key, {"boxed": C.last_boxed(txt), "agree": C.agree(txt, g.get("answer"), self.lang),
                             "trace": txt, "attempt": prev.get("attempt", 0) + 1, **resp_meta(resp)})

    # ---- lite
    def lite_todo(self) -> list[tuple[str, str]]:
        out = []
        for s in self.prio_slots:
            k = self.k(s)
            g = self.gen.get(k)
            lr = self.lite.get(k) or {}
            if g and g.get("fmt_ok") and not lr.get("api_ok") and not lr.get("err"):
                out.append((k, g["problem"]))
        return out

    async def on_result(self, stage: str, key: str, resp) -> None:
        if stage == "gen":
            await self.on_gen(key, resp)
        elif stage == "solve":
            await self.on_solve(key, resp)
        elif stage == "hv3":
            await self.on_hv3(key, resp)
        elif stage == "audit":
            put_audit(store(self.lang, "audit"), key, resp)

    # ---- Haiku-v3 part of the common "+" screen (scale_gen.screen), priority order, stop at target_common passes
    def _hv3_candidates(self, outstanding: set[str]) -> tuple[list[str], int, int]:
        cand, passed, judged = [], 0, 0
        for s in self.prio_slots:
            k = self.k(s)
            h = self.hv3.get(k)
            if h is not None and h.get("api_ok"):
                judged += 1
                passed += bool(h.get("hv3_pass") and (self.solve.get(k) or {}).get("agree"))
                continue
            if k in outstanding or (h or {}).get("attempt", 0) >= 2:
                continue
            if self.hv3_only is not None and s["key"] not in self.hv3_only:
                continue
            if (self.solve.get(k) or {}).get("agree"):     # lead 02:05: lite dropped (Gemini quota)
                cand.append(k)
        return cand, passed, judged

    def hv3_req(self, k: str):
        s = self.by_key[k.split("|", 2)[2]]
        seed = self.seeds[s["seed_id"]]
        partner = self.seeds.get(s["partner"]) if s.get("partner") else None
        return SCR.haiku_v3_request(seed_text(seed, self.lang), s["op"], self.gen.get(k)["problem"], lang=self.lang,
                                    item_key=f"N:{k}", partner_text=seed_text(partner, self.lang) if partner else None)

    async def on_hv3(self, key: str, resp) -> None:
        prev = self.hv3.get(key) or {}
        p = SCR.haiku_v3_parse(resp.text if resp.ok else None)
        self.hv3.put(key, {**p, "attempt": prev.get("attempt", 0) + 1, **resp_meta(resp),
                           "api_ok": bool(resp.ok and p["parsed"])})

    async def realtime(self, jobs, handler, concurrency: int = 24) -> None:
        """Critical-path fallback (lead 03:20): direct Messages API (2x Batch price), same requests, same cache,
        same per-item append. Budget-guarded by the LLMClient's hard cap."""
        sem = asyncio.Semaphore(concurrency)

        hb = WORK / "logs" / "heartbeat"

        async def one(k, rq):
            async with sem:
                r = await client().call(rq)
                await handler(k, r)
                hb.write_text(str(time.time()), encoding="utf-8")   # long real-time stages keep the watchdog fed
        await asyncio.gather(*(one(k, rq) for k, rq in jobs))

    def _upstream_busy(self, B) -> bool:
        pre = f"{self.lang}|{self.cfg}|"
        return (any(k.startswith(pre) for k in B.outstanding_keys("gen") | B.outstanding_keys("solve"))
                or (self.lite_task is not None and not self.lite_task.done()))

    async def tick(self) -> bool:
        """One scheduling step; returns True while work remains."""
        B = bm()
        out_gen = B.outstanding_keys("gen")
        todo = []
        held = self.hold is not None and self.hold()
        for s in ([] if held else self.slots):
            k = self.k(s)
            if k in out_gen:
                continue
            a = self.needs_gen(s)
            if a:
                todo.append((k, self.gen_req(s, a)))
        try:
            for i in range(0, len(todo), CHUNK):
                chunk = todo[i:i + CHUNK]
                if "gen" in REALTIME:
                    await self.realtime(chunk, self.on_gen)
                else:
                    await B.submit("gen", self.lang, chunk, EST[f"gen:{self.cfg}"], on_cached=self.on_gen)
            out_solve = B.outstanding_keys("solve")
            ready = self.solve_ready(out_solve)
            gen_busy = any(k.startswith(f"{self.lang}|{self.cfg}|") for k in B.outstanding_keys("gen"))
            if ready and (len(ready) >= CHUNK or not gen_busy):
                for i in range(0, len(ready), CHUNK):
                    jobs = [(k, solve_request(self.gen.get(k)["problem"], self.lang, k.split("|", 2)[2]))
                            for k in ready[i:i + CHUNK]]
                    if "solve" in REALTIME:
                        await self.realtime(jobs, self.on_solve)
                    else:
                        await B.submit("solve", self.lang, jobs, EST["solve"], on_cached=self.on_solve)
            if self.target_common:
                out_h = B.outstanding_keys("hv3")
                cand, passed, judged = self._hv3_candidates(out_h)
                inflight = sum(k.startswith(f"{self.lang}|{self.cfg}|") for k in out_h)
                rate = max(0.5, passed / judged) if judged >= 200 else 0.75
                need = self.target_common - passed - inflight * rate
                if need > 0 and cand:
                    n = min(len(cand), CHUNK, int(need / rate) + 100)
                    if n >= 300 or not self._upstream_busy(B) or "hv3" in REALTIME or self.hv3_only is not None:
                        jobs = [(k, self.hv3_req(k)) for k in cand[:n]]
                        if "hv3" in REALTIME:
                            await self.realtime(jobs, self.on_hv3)
                        else:
                            await B.submit("hv3", self.lang, jobs, EST["hv3"], on_cached=self.on_hv3)
        except BudgetStop as e:
            # estimates of un-collected batches are replaced by (lower) actual costs on collection; wait for that,
            # and stop for real only when nothing is outstanding any more
            if not B.outstanding():
                raise
            if not getattr(self, "_bs_noted", False):
                print(f"[budget] waiting for collections before submitting more: {e}", flush=True)
                self._bs_noted = True
        if self.target_plus:
            self.stop_lite = progress(self)["plus"] >= self.target_plus
        lite_left = self.lite_todo() if self.do_lite else []
        if self.do_lite and lite_left and (self.lite_task is None or self.lite_task.done()):
            if self.lite_task is not None and self.lite_task.exception():
                print(f"[lite/{self.lang}] task failed: {self.lite_task.exception()!r}", flush=True)
            if not self.stop_lite:
                self.lite_task = asyncio.create_task(L.judge_all(self.lite, self.lang, lite_left,
                                                                 stop=lambda: self.stop_lite))
        if self.target_common:
            _, passed, _ = self._hv3_candidates(B.outstanding_keys("hv3"))
            self.common_done = passed >= self.target_common
        mine = [b for b in B.outstanding() if b["lang"] == self.lang
                and any(k.startswith(f"{self.lang}|{self.cfg}|") for k in b["keys"].values())]
        lite_busy = self.lite_task is not None and not self.lite_task.done()
        # lite rows that failed twice stay failed; don't loop forever on them
        lite_pending = (bool(lite_left) and not self.stop_lite
                        and any(not (self.lite.get(k) or {}).get("err") for k, _ in lite_left))
        og = B.outstanding_keys("gen")
        hv3_left = bool(self.target_common) and not getattr(self, "common_done", False) and bool(
            self._hv3_candidates(B.outstanding_keys("hv3"))[0])
        return bool(mine or lite_busy or lite_pending or hv3_left or self.solve_ready(B.outstanding_keys("solve"))
                    or any(self.needs_gen(s) for s in self.slots if self.k(s) not in og))


async def drive(drivers: list[Driver], poll_s: float = 30.0, max_hours: float = 10.0) -> None:
    t0 = time.time()
    by_lang = {}
    for d in drivers:
        by_lang.setdefault(d.lang, []).append(d)

    async def on_result(stage, key, resp):
        lang, cfg = key.split("|")[:2]
        for d in by_lang.get(lang, []):
            if d.cfg == cfg:
                await d.on_result(stage, key, resp)

    last = 0.0
    hb = WORK / "logs" / "heartbeat"
    while True:
        hb.parent.mkdir(parents=True, exist_ok=True)
        hb.write_text(str(time.time()), encoding="utf-8")
        await bm().poll(on_result)
        busy = False
        for d in drivers:
            busy |= await d.tick()
        if time.time() - last > 300:
            last = time.time()
            print(json.dumps({"t_min": round((time.time() - t0) / 60, 1), "spent": round(bm().spent_usd(), 4),
                              "committed": round(bm().committed_usd(), 4), "batches": bm().summary(),
                              **{f"{d.lang}|{d.cfg}": progress(d) for d in drivers}}), flush=True)
        if not busy or time.time() - t0 > max_hours * 3600:
            break
        await asyncio.sleep(poll_s)
    for d in drivers:
        if d.lite_task is not None:
            await d.lite_task


def progress(d: Driver) -> dict:
    ks = [d.k(s) for s in d.slots]
    g = [d.gen.get(k) for k in ks]
    return {"n": len(ks), "gen": sum(bool(r) for r in g), "fmt_ok": sum(bool(r and r.get("fmt_ok")) for r in g),
            "solved": sum(bool((d.solve.get(k) or {}).get("api_ok")) for k in ks),
            "agree": sum(bool((d.solve.get(k) or {}).get("agree")) for k in ks),
            "lite": sum(bool((d.lite.get(k) or {}).get("api_ok")) for k in ks),
            "lite_ok": sum(bool((d.lite.get(k) or {}).get("lite_ok")) for k in ks),
            "agree_lite": sum(bool((d.solve.get(k) or {}).get("agree") and (d.lite.get(k) or {}).get("lite_ok"))
                              for k in ks),
            "hv3": sum(bool((d.hv3.get(k) or {}).get("api_ok")) for k in ks),
            "plus": sum(bool((d.solve.get(k) or {}).get("agree") and (d.hv3.get(k) or {}).get("hv3_pass"))
                        for k in ks)}


# ------------------------------------------------------------------------------------------------ audit
def put_audit(st: Store, key: str, resp) -> None:
    """Opus 5.5 + validity_judge.v2 verdict -> labels (G1/G2 lab_v2: the judge of record)."""
    v = parse_json(resp.text) if resp.ok else None
    v = v if isinstance(v, dict) else None
    lab = None
    if v:
        from ..pilot_gen.g2_common import lab_v2
        lab = lab_v2(v)
    st.put(key, {"verdict": v, "labels": lab, **resp_meta(resp)})

def audit_sample(lang: str, cfg: str, slots: list[dict], n_plus: int, n_rej: int, seed: int = 20261108) -> list[str]:
    """Random audit sample: ``n_plus`` items that passed every "+" gate (agree ∧ hv3_pass) and ``n_rej``
    items rejected by at least one "+" gate, both from the screened region (lite and, for passes, hv3 judged)."""
    g, s, l, h = store(lang, "gen"), store(lang, "solve"), store(lang, "lite"), store(lang, "hv3")
    plus, rej = [], []
    for sl in slots:
        k = f"{lang}|{cfg}|{sl['key']}"
        r = g.get(k)
        if not (r and r.get("fmt_ok")):
            continue
        sv, lv, hv = s.get(k) or {}, l.get(k) or {}, h.get(k) or {}
        if not sv.get("api_ok"):
            continue
        cand = sv.get("agree")
        if cand and not hv.get("api_ok"):
            continue                                  # beyond the early-stopped Haiku-v3 screen: unknown
        (plus if cand and hv.get("hv3_pass") else rej).append(k)
    rng = random.Random(seed)
    return sorted(rng.sample(plus, min(n_plus, len(plus)))) + sorted(rng.sample(rej, min(n_rej, len(rej))))


async def run_audit(lang: str, keys: list[str], seeds: dict[str, dict], slots_by_key: dict, poll_s: float = 30.0):
    a = store(lang, "audit")
    g = store(lang, "gen")
    jobs = []
    for k in keys:
        if (a.get(k) or {}).get("api_ok"):
            continue
        r = g.get(k)
        sl = slots_by_key[k.split("|", 2)[2]]
        seed = seeds[sl["seed_id"]]
        instr = OP_INSTR[lang][sl["op"]]
        if sl["op"] == "M6" and sl.get("partner") in seeds:
            instr += seed_text(seeds[sl["partner"]], lang)
        jobs.append((k, audit_request(seed_text(seed, lang), sl["op"], instr, r["problem"], k, lang)))

    async def on_audit(key, resp):
        put_audit(a, key, resp)

    B = bm()
    if jobs:
        await B.submit("audit", lang, jobs, EST["audit"], on_cached=on_audit)
    while B.outstanding("audit"):
        await asyncio.sleep(poll_s)
        await B.poll(lambda st, k, r: on_audit(k, r) if st == "audit" else _noop())


async def _noop():
    return None


def select_slots(base: list[dict], path: str) -> list[dict]:
    """Slots of ``base`` named in ``path`` (slot keys, or seed ids -> every base slot of that seed), rank order."""
    keys, seeds_ = set(), set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            r = json.loads(line)
            k = r.get("slot") or r.get("slot_id") or r.get("key")
            (keys.add(k) if k else seeds_.add(r["seed_id"]))
        elif ":" in line:
            keys.add(line)
        else:
            seeds_.add(line)
    return [s for s in base if s["key"] in keys or s["seed_id"] in seeds_]


def _tc(spec, lang):
    if not spec:
        return None
    if ":" not in str(spec):
        return int(spec)
    d = dict(x.split(":") for x in str(spec).split(","))
    return int(d[lang]) if lang in d else None


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["slots", "pilot", "gen", "audit", "status"])
    ap.add_argument("--langs", default="vi")
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cfg", default="B")
    ap.add_argument("--cfgs", default="A,B")
    ap.add_argument("--no-lite", action="store_true")
    ap.add_argument("--n-plus", type=int, default=60)
    ap.add_argument("--n-rej", type=int, default=30)
    ap.add_argument("--keys-file", default=None)
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--provisional", action="store_true")
    ap.add_argument("--target-plus", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None, help="items per batch (default 1000)")
    ap.add_argument("--realtime", default="", help="comma list of stages to run on the real-time API: solve,hv3")
    ap.add_argument("--target-common", default=None, help="N or per language: vi:7000,en:3300")
    ap.add_argument("--hv3-only", default=None, help="e.g. en:path; restrict the Haiku-v3 screen of that language "
                    "to the slots named in the file (slot keys or seed ids; same format as --lang-slots)")
    ap.add_argument("--lang-slots", default=None, help="e.g. en:path; slot keys (or seed ids -> all base slots of that "
                    "seed) for a later language, one per line or JSONL with slot/slot_id/key or seed_id (D5's list)")
    ap.add_argument("--n-by-lang", default=None, help="e.g. vi:13500,en:5700; a later language uses the priority-"
                    "order prefix of the first language's slots (same slots as VI, E1 R1)")
    a = ap.parse_args(argv)
    global CHUNK
    if a.chunk:
        CHUNK = a.chunk
    REALTIME.update(x for x in a.realtime.split(",") if x)
    if a.cmd == "slots":
        print(json.dumps(create_shared(), ensure_ascii=False))
        return
    if a.cmd == "status":
        print(json.dumps({"spent": bm().spent_usd(), "committed": bm().committed_usd(), "batches": bm().summary()},
                         indent=1))
        return
    if a.cmd == "pilot" or a.provisional:
        seeds = provisional_seeds()
        slots = build_slots(list(seeds.values()))
    else:
        seeds = pool_seeds()
        slots = load_slots(SLOTS)
    try:
        if a.cmd == "pilot":
            sl = slots[:a.n]
            drivers = [Driver("vi", c, sl, seeds, lite=not a.no_lite) for c in a.cfgs.split(",")]
            client().run(drive(drivers, poll_s=a.poll))
            for d in drivers:
                print(d.cfg, json.dumps(progress(d)))
        elif a.cmd == "gen":
            langs = a.langs.split(",")
            nb = dict(x.split(":") for x in a.n_by_lang.split(",")) if a.n_by_lang else {}
            ls = dict(x.split(":", 1) for x in a.lang_slots.split(",")) if a.lang_slots else {}
            base = slots[:int(nb.get(langs[0], a.n))]
            drivers = []
            for i, lang in enumerate(langs):
                sl = base if i == 0 else sorted(base, key=lambda s: prio(s["key"]))[:int(nb.get(lang, len(base)))]
                if ls.get(lang):            # also for the first language (e.g. an EN-only retry run)
                    sl = select_slots(base, ls[lang])
                drivers.append(Driver(lang, a.cfg, sl, seeds, lite=not a.no_lite, target_plus=a.target_plus,
                                      target_common=_tc(a.target_common, lang)))
                ho = dict(x.split(":", 1) for x in a.hv3_only.split(",")) if a.hv3_only else {}
                if ho.get(lang):
                    drivers[-1].hv3_only = {x["key"] for x in select_slots(sl, ho[lang])}
            for prev, d in zip(drivers, drivers[1:]):   # VI first: a later language waits until the earlier
                def _hold(p=prev):                       # language's generation is fully collected
                    og = bm().outstanding_keys("gen")
                    return any(p.k(s) in og or p.needs_gen(s) for s in p.slots)
                d.hold = _hold
            client().run(drive(drivers, poll_s=a.poll))
            for d in drivers:
                print(d.lang, json.dumps(progress(d)))
        elif a.cmd == "audit":
            sl = slots[:a.n]
            if a.keys_file:
                keys = [l.strip() for l in open(a.keys_file, encoding="utf-8") if l.strip()]
            else:
                keys = audit_sample(a.lang, a.cfg, sl, a.n_plus, a.n_rej)
            client().run(run_audit(a.lang, keys, seeds, {s["key"]: s for s in slots}, poll_s=a.poll))
            print(json.dumps({"audited": sum(bool((store(a.lang, "audit").get(k) or {}).get("labels")) for k in keys)}))
    except BudgetStop as e:
        print(f"BUDGET STOP: {e}", flush=True)
        sys.exit(3)
    print(json.dumps({"spent": round(bm().spent_usd(), 4), "committed": round(bm().committed_usd(), 4)}))


if __name__ == "__main__":
    main()
