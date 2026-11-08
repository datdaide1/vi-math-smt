"""Opus 5.5 evaluation of an arm-Y run (Message Batches): validity_judge.v2 (judge of record) + solution_writer.v2
blind solve on every gate survivor (cycle + leak + chrF + dedup), before the validity gate, so the gate's effect is
measured on the same items. Report tables per variant: before / after the Haiku validity gate.

    python -m vi_math_verified_aug.arm_y.evalrun <run_dir> [judge|report]
"""
from __future__ import annotations

import asyncio
import json
import sys
from fractions import Fraction
from pathlib import Path

from ..pilot_gen.util import OP_INSTR, answer_values, fmt_ci, last_boxed
from ..pilot_symbolic.judge import lab_v2
from .common import ROOT, Store, call_many, parse_json, render, req, resp_meta, spent
from .pipeline import cfg, load


async def judge(run: Path):
    c = cfg(run)
    seeds = {s["seed_id"]: s for s in load(run, "seeds")}
    gates = Store(run, "gates").rows
    rend = {v: Store(run, f"render_{v}").rows for v in c["variants"]}
    jo, so = Store(run, "opus_judge"), Store(run, "opus_solve")
    its = [g for g in gates.values() if g["survivor"]]
    jt = [g for g in its if not (jo.get(g["key"]) or {}).get("api_ok")]
    stt = [g for g in its if not (so.get(g["key"]) or {}).get("api_ok")]

    def seed_text(g):
        s = seeds[g["slot"].split(":")[0]]
        return s["en_text"] if g["variant"] == "EN" else s["text"]
    reqs = [req("opus", render("judge", seed=seed_text(g), op=g["op"], op_instruction=OP_INSTR[g["op"]],
                               problem=rend[g["variant"]][g["slot"]]["text"]), item=f"gen:G5:{g['key']}",
                effort="medium", max_tokens=3000, cache_system=True, tag="judge_opus_v2") for g in jt]
    reqs += [req("opus", render("solve", problem=rend[g["variant"]][g["slot"]]["text"]), item=f"gen:G5:{g['key']}",
                 effort="medium", max_tokens=4000, tag="solve_opus") for g in stt]
    resps = await call_many(reqs, True, "opus")
    for g, r in zip(jt, resps[:len(jt)]):
        v = parse_json(r.text) if r.ok else None
        v = v if isinstance(v, dict) else None
        jo.put(g["key"], {"verdict": v, "labels": lab_v2(v) if v else None, **resp_meta(r)})
    for g, r in zip(stt, resps[len(jt):]):
        b = last_boxed(r.text) if r.ok else None
        tgt = Fraction(rend[g["variant"]][g["slot"]]["target"])
        so.put(g["key"], {"boxed": b, "agree": tgt in (answer_values(b) if b else set()), **resp_meta(r)})
    print(f"opus judged {len(jt)}, solved {len(stt)}; spent ${spent():.3f}")


def report(run: Path) -> str:
    c = cfg(run)
    gates = Store(run, "gates").rows
    jo, so, vg = Store(run, "opus_judge").rows, Store(run, "opus_solve").rows, Store(run, "vgate").rows
    out = ["| variant | set | n | teacher | new M2–M5 | leak | absurd | tail | valid | unansw | natural | Opus solve = target | accept |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for v in c["variants"]:
        tag_n = sum(1 for g in gates.values() if g["variant"] == v)
        cyc = sum(g["cycle_ok"] for g in gates.values() if g["variant"] == v)
        surv = [g for g in gates.values() if g["variant"] == v and g["survivor"] and (jo.get(g["key"]) or {}).get("labels")]
        for name, S in (("survivors (before gate)", surv),
                        ("after Haiku validity gate", [g for g in surv if (vg.get(g["key"]) or {}).get("pass")])):
            L = [jo[g["key"]]["labels"] for g in S]
            n = len(L)
            nw = [l for g, l in zip(S, L) if g["op"] != "M1"]
            ag = [so[g["key"]]["agree"] for g in S if g["key"] in so]
            acc = sum(l["teacher"] and l["valid"] and not l["absurd"] and not l["leakj"] and not l["tail"] for l in L)
            nat = round(sum(l["natural"] for l in L) / n, 2) if n else None
            out.append(f"| {v} | {name} | {n} | {fmt_ci(sum(l['teacher'] for l in L), n)} | "
                       f"{fmt_ci(sum(l['new'] for l in nw), len(nw))} | {fmt_ci(sum(l['leakj'] for l in L), n)} | "
                       f"{fmt_ci(sum(l['absurd'] for l in L), n)} | {sum(l['tail'] for l in L)} | "
                       f"{sum(l['valid'] for l in L)} | {sum(l['unansw'] for l in L)} | {nat} | "
                       f"{fmt_ci(sum(ag), len(ag))} | {fmt_ci(acc, n)} |")
            res[f"{v}:{name}"] = {"n": n, "teacher": sum(l["teacher"] for l in L), "absurd": sum(l["absurd"] for l in L),
                                  "new": sum(l["new"] for l in nw), "n_new": len(nw), "agree": sum(ag), "n_agree": len(ag),
                                  "accept": acc}
        res[f"{v}:funnel"] = {"tag": tag_n, "cycle": cyc, "survivors": len(surv)}
        out.append(f"| {v} | funnel | – | tag-passing {tag_n}, cycle ok {cyc} ({fmt_ci(cyc, tag_n)}), survivors {len(surv)} | | | | | | | | | |")
    json.dump(res, open(run / "eval_metrics.json", "w"), indent=1)
    txt = "\n".join(out)
    (run / "eval_tables.md").write_text(txt, encoding="utf-8")
    return txt


if __name__ == "__main__":
    run = Path(sys.argv[1])
    run = run if run.is_absolute() else ROOT / run
    if len(sys.argv) < 3 or sys.argv[2] == "judge":
        asyncio.run(judge(run))
    print(report(run))
