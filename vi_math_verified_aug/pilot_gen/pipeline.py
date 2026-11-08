"""G1 pilot — pipelined, resumable orchestrator (replaces the stage-by-stage run.py flow).

python -m vi_math_verified_aug.pilot_gen.pipeline [--gens ...] [--opus-sample 30]

Every (generator, seed) chain runs as soon as a worker is free; providers never wait on each other:
  native (a)  --> gemini-lite judge (all items) + [Opus judge + Opus blind-solve if seed in the Opus subsample]
  formalize (b) --> render coordinator: once the reference IR for a M2–M5 seed is decided (first passing IR in
                    REF_ORDER; fast providers first), the shared edit list is built and every generator renders (c)
  solve_seeds (d)
  writer_common (d'): once the fast generators' Opus-subsample items are scored, every generator blind-solves the
                    common set of accepted items.
Concurrency is capped inside clients.py (NIM 38 RPM bucket + 35 in flight; Claude 24 in flight; Gemini one lane per
(key, model)). Results are appended per stage to data/pilot/generator/<stage>/<gen>.jsonl; on restart, keys with a
successful row are skipped (API calls are also cached in logs/api/G1).
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

from . import clients as C
from . import seeds as S
from .ir_exec import IRError, Program, strip_tags, tag_text, untagged_numerals, validate
from .mutate import BUILD
from .run import render_check
from .util import OP_INSTR, answer_values, card_text, last_boxed, prompt, same_answer

OUT = S.OUT
GRADE = "tiểu học và THCS (lớp 3–7)"
GEN_EFFORT = "low"          # Haiku/Sonnet in every generator-side role (coordinator: low unless quality needs more)
OPUS_EFFORT = "medium"
MAXTOK = {  # task -> provider -> max_tokens (tight; NIM/Gemini reasoning tokens count against it)
    "native": {"nim": 6144, "gemini": 6144, "claude": 3000},
    "formalize": {"nim": 8192, "gemini": 8192, "claude": 6000},
    "render": {"nim": 3072, "gemini": 3072, "claude": 2000},
    "solve": {"nim": 3072, "gemini": 3072, "claude": 2000},
}
TEMP = {"native": 0.7, "formalize": 0.0, "render": 0.2, "solve": 0.6}
REF_ORDER = ["sonnet", "haiku", "gemini", "gemini31", "deepseek", "nemotron", "glm"]
FAST = ["haiku", "sonnet", "gemini", "gemini31"]
PRIMARY = {"haiku": "gemini_judge", "sonnet": "gemini_judge", "gemini": "opus", "gemini31": "opus",
           "deepseek": "both", "nemotron": "both", "glm": "both"}


# ------------------------------------------------------------------ store
class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.data: dict = {}  # (stage, gen) -> {key: row}
        for d in OUT.iterdir():
            if d.is_dir() and not d.name.startswith("_"):
                for p in d.glob("*.jsonl"):
                    rows = {}
                    for l in open(p, encoding="utf-8"):
                        r = json.loads(l)
                        rows[r["key"]] = r
                    self.data[(d.name, p.stem)] = rows

    def get(self, stage, gen, key):
        return self.data.get((stage, gen), {}).get(key)

    def done(self, stage, gen, key):
        r = self.get(stage, gen, key)
        return r is not None and r.get("api_ok", True)

    def put(self, stage, gen, key, row):
        row = {"key": key, **row}
        with self.lock:
            self.data.setdefault((stage, gen), {})[key] = row
            (OUT / stage).mkdir(parents=True, exist_ok=True)
            with open(OUT / stage / f"{gen}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


ST: Store
POOL: ThreadPoolExecutor
SEEDS: dict
OPUS_S: set
FUTS = []
FUT_LOCK = threading.Lock()


def submit(fn, *a):
    def wrapped():
        try:
            fn(*a)
        except C.BudgetExceeded as e:
            print("BUDGET", fn.__name__, a[:2], e, flush=True)
        except Exception:
            print("ERROR", fn.__name__, a[:2], traceback.format_exc()[-600:], flush=True)
    f = POOL.submit(wrapped)
    with FUT_LOCK:
        FUTS.append(f)
    return f


def kw(gen, task):
    prov = C.MODELS[gen][0]
    mt = MAXTOK[task][prov]
    if prov == "claude":
        return {"effort": GEN_EFFORT, "max_tokens": mt}
    return {"temperature": TEMP[task], "max_tokens": mt}


def meta(r):
    return {"latency": r.get("latency"), "usage": r.get("usage"), "cost": r.get("cost"), "cached": r.get("cached"),
            "api_ok": r.get("ok"), "err": r.get("err"), "finish": r.get("finish")}


# ------------------------------------------------------------------ (a) native + its judging/solving
def op_instr(s):
    return OP_INSTR[s["op"]] + (s.get("partner_text") or "") if s["op"] == "M6" else OP_INSTR[s["op"]]


def native(gen, sid):
    s = SEEDS[sid]
    row = ST.get("native", gen, sid)
    if not ST.done("native", gen, sid):
        cards = S.cards()
        target = "khó hơn một bậc" if s["op"] in ("M2", "M3", "M6") else "giữ nguyên độ khó"
        sy, us = prompt("native_v1.txt", grade_hint=GRADE, seed_vi=s["text"], op=s["op"], op_instruction=op_instr(s),
                        card=card_text(cards[s["card"]]), target=target)
        r = C.call(gen, sy, us, task="native", item=f"seed:{sid}", **kw(gen, "native"))
        j = C.parse_json(r.get("text")) or {}
        prob = j.get("problem") if isinstance(j.get("problem"), str) else None
        ans = j.get("answer")
        row = ST.put("native", gen, sid, {
            "seed_id": sid, "gen": gen, "op": s["op"], "parsed": bool(prob and ans is not None), "problem": prob,
            "answer": None if ans is None else str(ans), "op_applied": j.get("op_applied"),
            "change_summary": j.get("change_summary"), **meta(r)})
    if row and row.get("parsed"):
        submit(judge, "gemini_judge", gen, sid)
        if sid in OPUS_S:
            submit(judge, "opus", gen, sid)
            submit(opus_solve, gen, sid)


def judge(jdg, gen, sid):
    stage = f"judge_{jdg}"
    if ST.done(stage, gen, sid):
        return
    s, a = SEEDS[sid], ST.get("native", gen, sid)
    sy, us = prompt("judge_v2.txt", seed=s["text"], op=s["op"], op_instruction=op_instr(s), problem=a["problem"])
    if jdg == "opus":
        r = C.call("opus", sy, us, task="judge", item=f"gen:{gen}:{sid}", effort=OPUS_EFFORT, max_tokens=3000,
                   cache_system=True)
    else:
        r = C.call(jdg, sy, us, task="judge", item=f"gen:{gen}:{sid}", temperature=0.0, max_tokens=3072)
    j = C.parse_json(r.get("text"))
    ST.put(stage, gen, sid, {"seed_id": sid, "gen": gen, "judge": jdg, "parsed": bool(j), "verdict": j or {},
                             **meta(r)})


def opus_solve(gen, sid):
    if ST.done("opus_solve", gen, sid):
        return
    a = ST.get("native", gen, sid)
    sy, us = prompt("solve_v1.txt", problem=a["problem"])
    r = C.call("opus", sy, us, task="opus_solve", item=f"gen:{gen}:{sid}", effort=OPUS_EFFORT, max_tokens=2500)
    box = last_boxed(r.get("text"))
    ST.put("opus_solve", gen, sid, {"seed_id": sid, "gen": gen, "boxed": box, "solution": r.get("text"),
                                    "agrees": same_answer(box, answer_values(a["answer"])) if box else False,
                                    **meta(r)})


# ------------------------------------------------------------------ (b) formalization + (c) render coordination
def formalize(gen, sid, max_attempts=3):
    s = SEEDS[sid]
    if not ST.done("formalize", gen, sid):
        gold = Fraction(s["gold"])
        sy, us0 = prompt("formalize_v1.txt", text=s["text"], en_text=s["en_text"], gold=s["gold"])
        us, attempts, P, errs, flags = us0, [], None, ["not run"], []
        for k in range(max_attempts):
            r = C.call(gen, sy, us, task="formalize", item=f"seed:{sid}", tag=f"att{k}", **kw(gen, "formalize"))
            j = C.parse_json(r.get("text"))
            P, flags = None, []
            if not r.get("ok"):
                errs = [f"api: {r.get('err')}"]
            elif j is None:
                errs = ["no JSON object in reply" + (" (truncated: max_tokens)" if r.get("finish") in
                                                       ("length", "max_tokens", "MAX_TOKENS") else "")]
            else:
                try:
                    P = Program.from_json(s["text"], j)
                    errs, flags = validate(P, gold)
                except IRError as e:
                    errs = [f"schema: {e}"]
                except Exception as e:
                    errs = [f"schema: {e!r}"]
            attempts.append({"k": k, "errors": errs[:6], **meta(r)})
            if not errs or not r.get("ok"):
                break
            us = (us0 + "\n\nYOUR PREVIOUS ANSWER:\n" + (r.get("text") or "")[-6000:] +
                  "\n\nThe validator rejected it:\n- " + "\n- ".join(errs[:8]) +
                  "\nFix these problems and output the full corrected JSON object only.")
        ok = not errs
        ST.put("formalize", gen, sid, {
            "seed_id": sid, "gen": gen, "pass1": not attempts[0]["errors"], "pass": ok, "n_attempts": len(attempts),
            "errors": errs[:6], "flags": flags, "program": P.to_json() if (P is not None and ok) else None,
            "attempts": attempts, "api_ok": all(a["api_ok"] for a in attempts)})
    try_render(sid)


RENDER_LOCK = threading.Lock()
RENDER_DONE: set = set()


def try_render(sid):
    s = SEEDS[sid]
    if s["op"] not in BUILD:
        return
    with RENDER_LOCK:
        if sid in RENDER_DONE:
            return
        inp = ST.get("render_inputs", "shared", sid)
        if inp is None:
            inp = {"seed_id": sid, "op": s["op"], "ref": None, "edit": None, "tried": []}
            for g in [x for x in REF_ORDER if x in GENS]:
                f = ST.get("formalize", g, sid)
                if f is None:
                    return  # an earlier reference candidate has not finished yet
                if not f["pass"]:
                    continue
                P = Program.from_json(s["text"], f["program"])
                if validate(P, Fraction(s["gold"]))[0]:
                    continue
                e = BUILD[s["op"]](P, random.Random(S.SEED + s["index"]))
                inp["tried"].append(g)
                if e:
                    inp.update({"ref": g, "edit": e, "tagged": tag_text(P),
                                "untagged_in": untagged_numerals(tag_text(P))})
                    break
            inp = ST.put("render_inputs", "shared", sid, inp)
        RENDER_DONE.add(sid)
    if inp.get("edit"):
        for g in GENS:
            submit(render, g, sid)


def render(gen, sid):
    if ST.done("render", gen, sid):
        return
    inp = ST.get("render_inputs", "shared", sid)
    sy, us = prompt("render_v1.txt", tagged=inp["tagged"],
                    edits=json.dumps(inp["edit"]["edits"], ensure_ascii=False, indent=1))
    r = C.call(gen, sy, us, task="render", item=f"seed:{sid}", **kw(gen, "render"))
    j = C.parse_json(r.get("text")) or {}
    t = j.get("text_tagged") if isinstance(j.get("text_tagged"), str) else None
    errs = ["no text_tagged"] if t is None else render_check(t, inp)
    ST.put("render", gen, sid, {"seed_id": sid, "gen": gen, "op": inp["op"], "pass": not errs, "errors": errs,
                                "text_tagged": t, "text": strip_tags(t) if t else None, **meta(r)})


# ------------------------------------------------------------------ (d) blind solving
def solve(gen, stage, key, problem, gold, item):
    if ST.done(stage, gen, key):
        return
    sy, us = prompt("solve_v1.txt", problem=problem)
    r = C.call(gen, sy, us, task=stage, item=item, **kw(gen, "solve"))
    box = last_boxed(r.get("text"))
    ST.put(stage, gen, key, {"id": key, "gen": gen, "boxed": box, "solution": r.get("text"),
                             "correct": same_answer(box, gold) if (box is not None and gold is not None) else False,
                             **meta(r)})


def valid(v):
    try:
        return all(v.get(b) is True for b in ("atomic", "no_conflict", "complete", "single_question", "realistic")) \
            and float(v.get("natural", 0)) >= 4
    except Exception:
        return False


def writer_watch(per_gen=10):
    """When every fast generator's Opus-subsample items are scored, build the common set (accepted items:
    primary judge valid + Opus answer agrees) and let every generator blind-solve it."""
    p = OUT / "writer_common_items.jsonl"
    if not p.exists():
        while True:
            ready = all(ST.get("native", g, sid) is not None for g in FAST for sid in OPUS_S) and all(
                ST.get("opus_solve", g, sid) is not None and ST.get(f"judge_{jd}", g, sid) is not None
                        for g in FAST for sid in OPUS_S
                        if (ST.get("native", g, sid) or {}).get("parsed")
                        for jd in (["opus"] if PRIMARY[g] == "opus" else ["gemini_judge"]))
            if ready:
                break
            time.sleep(20)
        items, rng = [], random.Random(S.SEED + 2)
        for g in FAST:
            acc = []
            for sid in sorted(OPUS_S):
                a, o = ST.get("native", g, sid), ST.get("opus_solve", g, sid)
                jd = "opus" if PRIMARY[g] == "opus" else "gemini_judge"
                v = (ST.get(f"judge_{jd}", g, sid) or {}).get("verdict") or {}
                if a and a.get("parsed") and o and o.get("agrees") and valid(v):
                    acc.append({"id": f"{g}:{sid}", "problem": a["problem"], "gold": a["answer"],
                                "item": f"gen:{g}:{sid}", "src_gen": g})
            items += rng.sample(acc, min(per_gen, len(acc)))
        with open(p, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print("writer_common items:", len(items), flush=True)
    items = [json.loads(l) for l in open(p, encoding="utf-8")]
    for g in GENS:
        for it in items:
            submit(solve, g, "writer_common", it["id"], it["problem"], it["gold"], it["item"])


GENS: list = []


def status(t0):
    while True:
        time.sleep(60)
        cnt = {}
        for (stage, g), rows in list(ST.data.items()):
            cnt.setdefault(stage, {})[g] = sum(1 for r in rows.values() if r.get("api_ok", True))
        with FUT_LOCK:
            pend = sum(1 for f in FUTS if not f.done())
        print(f"[{time.time() - t0:.0f}s] pending={pend} claude=${C.claude_spent():.2f} stats={C.STATS} "
              f"gem={C.GEM_OBS} " + json.dumps(cnt), flush=True)


def main():
    global ST, POOL, SEEDS, OPUS_S, GENS
    ap = argparse.ArgumentParser()
    # NIM removed from the main pipeline (user decision 2026-11-07, coordinator); its finished rows stay as
    # partial reference data. Run with --gens including NIM keys only if that decision is reversed.
    ap.add_argument("--gens", default=",".join(FAST))
    ap.add_argument("--opus-sample", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    GENS = [g for g in args.gens.split(",") if g]
    seeds = S.load()[:args.limit] if args.limit else S.load()
    SEEDS = {s["seed_id"]: s for s in seeds}
    order = list(S.load())
    random.Random(S.SEED + 1).shuffle(order)
    OPUS_S = {s["seed_id"] for s in order[:args.opus_sample]} & set(SEEDS)
    json.dump(sorted(OPUS_S), open(OUT / "opus_subsample.json", "w"))
    ST = Store()
    POOL = ThreadPoolExecutor(max_workers=900)
    t0 = time.time()
    threading.Thread(target=status, args=(t0,), daemon=True).start()
    # fast providers first in the submission order, so they are never queued behind NIM
    gens = [g for g in REF_ORDER if g in GENS]
    for sid in SEEDS:
        for g in gens:
            submit(native, g, sid)
            submit(formalize, g, sid)
            submit(solve, g, "solve_seeds", sid, SEEDS[sid]["text"], SEEDS[sid]["gold"], f"seed:{sid}")
    threading.Thread(target=writer_watch, daemon=True).start()
    # wait until no pending futures remain (new ones are added while running)
    while True:
        time.sleep(15)
        with FUT_LOCK:
            pend = [f for f in FUTS if not f.done()]
        if not pend:
            time.sleep(30)
            with FUT_LOCK:
                if all(f.done() for f in FUTS):
                    break
    print(f"ALL DONE in {time.time() - t0:.0f}s; claude ${C.claude_spent():.3f}; stats {C.STATS}; gem {C.GEM_OBS}",
          flush=True)


if __name__ == "__main__":
    main()
