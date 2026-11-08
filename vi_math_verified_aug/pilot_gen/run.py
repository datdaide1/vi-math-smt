"""G1 pilot task runner.

python -m vi_math_verified_aug.pilot_gen.run <task> [--gens a,b] [--limit N]
tasks: native (a) | formalize (b) | render (c) | solve_seeds (d) | judge (a-items) | opus_solve (a-items)
       | writer_common (d on a common set of accepted a-items)
Outputs: data/pilot/generator/<task>/<gen>.jsonl (rewritten each run; API calls are cached in logs/api/G1).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from . import clients as C
from . import seeds as S
from .ir_exec import IRError, Program, parse_tags, strip_tags, tag_text, untagged_numerals, validate
from .numerals import parse_literal
from .util import (OP_INSTR, answer_in_text, answer_values, card_text, chrf, last_boxed, leak_regex, prompt,
                   same_answer)

OUT = S.OUT
EFFORT = {"haiku": "medium", "sonnet": "medium", "opus": "medium"}
WORKERS = {"nim": 40, "claude": 6, "gemini": 12}
GRADE = "tiểu học và THCS (lớp 3–7)"


def kw_for(gen: str, temperature: float, max_tokens: int) -> dict:
    prov = C.MODELS[gen][0]
    if prov == "claude":
        return {"effort": EFFORT[gen], "max_tokens": max_tokens}
    return {"temperature": temperature, "max_tokens": max_tokens}


def pmap(gen: str, fn, items: list) -> list:
    with ThreadPoolExecutor(WORKERS[C.MODELS[gen][0]]) as ex:
        return list(ex.map(fn, items))


def write(task: str, gen: str, rows: list):
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{gen}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read(task: str, gen: str) -> list:
    p = OUT / task / f"{gen}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def meta(r: dict) -> dict:
    return {"latency": r.get("latency"), "usage": r.get("usage"), "cost": r.get("cost"), "cached": r.get("cached"),
            "api_ok": r.get("ok"), "err": r.get("err")}


# ------------------------------------------------------------------ (a) native generation
def native(gen: str, seeds: list) -> list:
    cards = S.cards()

    def one(s):
        instr = OP_INSTR[s["op"]] + (s.get("partner_text") or "" if s["op"] == "M6" else "")
        target = "khó hơn một bậc" if s["op"] in ("M2", "M3", "M6") else "giữ nguyên độ khó"
        sy, us = prompt("native_v1.txt", grade_hint=GRADE, seed_vi=s["text"], op=s["op"], op_instruction=instr,
                        card=card_text(cards[s["card"]]), target=target)
        r = C.call(gen, sy, us, task="native", item=f"seed:{s['seed_id']}", **kw_for(gen, 0.7, 8192))
        j = C.parse_json(r.get("text")) or {}
        prob = j.get("problem") if isinstance(j.get("problem"), str) else None
        ans = j.get("answer")
        return {"seed_id": s["seed_id"], "gen": gen, "op": s["op"], "parsed": bool(prob and ans is not None),
                "problem": prob, "answer": None if ans is None else str(ans), "op_applied": j.get("op_applied"),
                "change_summary": j.get("change_summary"), **meta(r)}
    return pmap(gen, one, seeds)


# ------------------------------------------------------------------ (b) formalization
def formalize(gen: str, seeds: list, max_attempts: int = 3) -> list:
    def one(s):
        gold = Fraction(s["gold"])
        sy, us0 = prompt("formalize_v1.txt", text=s["text"], en_text=s["en_text"], gold=s["gold"])
        us, attempts, P, errs, flags = us0, [], None, ["not run"], []
        for k in range(max_attempts):
            r = C.call(gen, sy, us, task="formalize", item=f"seed:{s['seed_id']}", tag=f"att{k}",
                       **kw_for(gen, 0.0, 12000))
            j = C.parse_json(r.get("text"))
            P, flags = None, []
            if j is None:
                errs = ["no JSON object in reply"]
            else:
                try:
                    P = Program.from_json(s["text"], j)
                    errs, flags = validate(P, gold)
                except IRError as e:
                    errs = [f"schema: {e}"]
                except Exception as e:
                    errs = [f"schema: {e!r}"]
            attempts.append({"k": k, "errors": errs[:6], **meta(r)})
            if not errs:
                break
            us = (us0 + "\n\nYOUR PREVIOUS ANSWER:\n" + (r.get("text") or "")[-6000:] +
                  "\n\nThe validator rejected it:\n- " + "\n- ".join(errs[:8]) +
                  "\nFix these problems and output the full corrected JSON object only.")
        ok = not errs
        return {"seed_id": s["seed_id"], "gen": gen, "pass1": not attempts[0]["errors"], "pass": ok,
                "n_attempts": len(attempts), "errors": errs[:6], "flags": flags,
                "program": P.to_json() if (P is not None and ok) else None, "attempts": attempts}
    return pmap(gen, one, seeds)


# ------------------------------------------------------------------ (c) minimal-edit rendering
REF_ORDER = ["sonnet", "deepseek", "nemotron", "glm", "haiku", "gemini"]


def render_inputs(seeds: list) -> list:
    """One shared edit list per M2–M5 seed, built from the first passing IR in REF_ORDER."""
    from .mutate import BUILD
    forms = {g: {r["seed_id"]: r for r in read("formalize", g)} for g in REF_ORDER}
    out = []
    for s in seeds:
        if s["op"] not in BUILD:
            continue
        row = {"seed_id": s["seed_id"], "op": s["op"], "ref": None, "edit": None}
        for g in REF_ORDER:
            f = forms[g].get(s["seed_id"])
            if not (f and f["pass"]):
                continue
            P = Program.from_json(s["text"], f["program"])
            errs, _ = validate(P, Fraction(s["gold"]))
            if errs:
                continue
            e = BUILD[s["op"]](P, random.Random(S.SEED + s["index"]))
            row["ref"] = g
            if e:
                row["edit"] = e
                row["tagged"] = tag_text(P)
                row["untagged_in"] = untagged_numerals(row["tagged"])
                break
        out.append(row)
    write("render_inputs", "shared", out)
    return out


def render_check(out_tagged: str, inp: dict) -> list:
    errs = []
    tags = parse_tags(out_tagged)
    exp = inp["edit"]["expected"]
    ids = sorted(t[0] for t in tags)
    if ids != sorted(exp):
        errs.append(f"tag ids {ids} != expected {sorted(exp)}")
    for tid, lit in tags:
        if tid in exp:
            v = Fraction(exp[tid])
            vals = parse_literal(lit.replace("%", "").strip())
            if v not in vals and v * 100 not in vals:
                errs.append(f"tag {tid}: {lit!r} != {exp[tid]}")
    extra = list(untagged_numerals(out_tagged))
    for n in inp["untagged_in"]:
        if n in extra:
            extra.remove(n)
    if extra:
        errs.append(f"extra numerals {extra}")
    return errs


def render(gen: str, inputs: list) -> list:
    def one(inp):
        sy, us = prompt("render_v1.txt", tagged=inp["tagged"],
                        edits=json.dumps(inp["edit"]["edits"], ensure_ascii=False, indent=1))
        r = C.call(gen, sy, us, task="render", item=f"seed:{inp['seed_id']}", **kw_for(gen, 0.2, 8192))
        j = C.parse_json(r.get("text")) or {}
        t = j.get("text_tagged") if isinstance(j.get("text_tagged"), str) else None
        errs = ["no text_tagged"] if t is None else render_check(t, inp)
        return {"seed_id": inp["seed_id"], "gen": gen, "op": inp["op"], "pass": not errs, "errors": errs,
                "text_tagged": t, "text": strip_tags(t) if t else None, **meta(r)}
    return pmap(gen, one, [i for i in inputs if i.get("edit")])


# ------------------------------------------------------------------ (d) blind solving
def solve_rows(gen: str, items: list, task: str) -> list:
    """items: [{id, problem, gold(str|None), item}]"""
    def one(it):
        sy, us = prompt("solve_v1.txt", problem=it["problem"])
        r = C.call(gen, sy, us, task=task, item=it["item"], **kw_for(gen, 0.6, 4096))
        box = last_boxed(r.get("text"))
        row = {"id": it["id"], "gen": gen, "boxed": box, "solution": r.get("text"), **meta(r)}
        if it.get("gold") is not None:
            row["correct"] = same_answer(box, it["gold"])
        return row
    return pmap(gen, one, items)


def solve_seeds(gen: str, seeds: list) -> list:
    return solve_rows(gen, [{"id": s["seed_id"], "problem": s["text"], "gold": s["gold"],
                             "item": f"seed:{s['seed_id']}"} for s in seeds], "solve_seed")


# ------------------------------------------------------------------ judging of (a) items
JUDGE_POLICY = {  # generator -> judges to run (first = primary)
    "haiku": ["gemini_judge", "opus"], "sonnet": ["gemini_judge", "opus"], "gemini": ["opus", "gemini_judge"],
    "deepseek": ["opus", "gemini_judge"], "nemotron": ["opus", "gemini_judge"], "glm": ["opus", "gemini_judge"],
}


def judge_items(judge: str, rows: list, seeds_by: dict) -> list:
    def one(a):
        s = seeds_by[a["seed_id"]]
        instr = OP_INSTR[s["op"]] + (s.get("partner_text") or "" if s["op"] == "M6" else "")
        sy, us = prompt("judge_v1.txt", seed=s["text"], op=s["op"], op_instruction=instr, problem=a["problem"])
        kw = {"effort": EFFORT["opus"], "max_tokens": 8000} if judge == "opus" else {"temperature": 0.0,
                                                                                  "max_tokens": 8192}
        r = C.call(judge, sy, us, task="judge", item=f"gen:{a['gen']}:{a['seed_id']}", **kw)
        j = C.parse_json(r.get("text")) or {}
        return {"seed_id": a["seed_id"], "gen": a["gen"], "judge": judge, "parsed": bool(j), "verdict": j,
                **meta(r)}
    prov = C.MODELS[judge][0]
    with ThreadPoolExecutor(WORKERS[prov]) as ex:
        return list(ex.map(one, rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--gens", default=",".join(C.GENERATORS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--judges", default="")
    ap.add_argument("--sample", type=int, default=0,
                    help="budget subsample: first N seeds of random.Random(20261108).shuffle (nested across N)")
    args = ap.parse_args()
    seeds = S.load()
    if args.limit:
        seeds = seeds[:args.limit]
    if args.sample:
        order = list(seeds)
        random.Random(S.SEED + 1).shuffle(order)
        seeds = order[:args.sample]
    seeds_by = {s["seed_id"]: s for s in S.load()}
    gens = [g for g in args.gens.split(",") if g]
    t0 = time.time()
    tasks = args.task.split(",")
    if set(tasks) <= {"native", "formalize", "solve_seeds"}:
        FN = {"native": native, "formalize": formalize, "solve_seeds": solve_seeds}
        jobs = [(t, g) for t in tasks for g in gens]

        def job(tg):
            t, g = tg
            rows = FN[t](g, seeds)
            write(t, g, rows)
            print(t, g, len(rows), f"{time.time() - t0:.0f}s", flush=True)
        with ThreadPoolExecutor(len(jobs)) as ex:
            list(ex.map(job, jobs))
    elif args.task == "judge":
        jobs = []
        ids = {s["seed_id"] for s in seeds}
        for g in gens:
            rows = [a for a in read("native", g) if a["parsed"] and a["seed_id"] in ids]
            for jdg in JUDGE_POLICY[g]:
                if args.judges and jdg not in args.judges.split(","):
                    continue
                jobs.append((g, jdg, rows))
        by_judge = {}
        for g, jdg, rows in jobs:
            by_judge.setdefault(jdg, []).append((g, rows))

        def run_judge(jdg):
            out = []
            for g, rows in by_judge[jdg]:
                out.append((g, jdg, judge_items(jdg, rows, seeds_by)))
            return out
        with ThreadPoolExecutor(len(by_judge)) as ex:
            for lst in ex.map(run_judge, list(by_judge)):
                for g, jdg, rows in lst:
                    old = {(r["seed_id"]): r for r in read(f"judge_{jdg}", g)}
                    for r in rows:
                        old[r["seed_id"]] = r
                    write(f"judge_{jdg}", g, list(old.values()))
                    print(g, jdg, len(rows), flush=True)
    elif args.task == "render":
        inputs = render_inputs(seeds)
        print("render inputs:", sum(1 for i in inputs if i.get("edit")), "of", len(inputs), flush=True)
        with ThreadPoolExecutor(len(gens)) as ex:
            res = dict(zip(gens, ex.map(lambda g: render(g, inputs), gens)))
        for g, rows in res.items():
            write("render", g, rows)
            print(g, len(rows), sum(r["pass"] for r in rows), flush=True)
    elif args.task == "writer_common":
        items = [json.loads(l) for l in open(OUT / "writer_common_items.jsonl", encoding="utf-8")]
        with ThreadPoolExecutor(len(gens)) as ex:
            res = dict(zip(gens, ex.map(lambda g: solve_rows(g, items, "solve_common"), gens)))
        for g, rows in res.items():
            write("writer_common", g, rows)
            print(g, len(rows), sum(r.get("correct", False) for r in rows), flush=True)
    elif args.task == "opus_solve":
        ids = {s["seed_id"] for s in seeds}
        items = []
        for g in gens:
            for a in read("native", g):
                if a["parsed"] and a["seed_id"] in ids:
                    items.append({"id": f"{g}:{a['seed_id']}", "problem": a["problem"], "gold": None,
                                  "item": f"gen:{g}:{a['seed_id']}"})
        rows = solve_rows("opus", items, "opus_solve")
        old = {r["id"]: r for r in read("opus_solve", "opus")}
        for r in rows:
            old[r["id"]] = r
        write("opus_solve", "opus", list(old.values()))
        print("opus_solve", len(rows))
    print(f"done in {time.time() - t0:.0f}s; stats {C.STATS}; claude ${C.claude_spent():.3f}; gemini {C.GEM_OBS}",
          flush=True)


if __name__ == "__main__":
    main()
