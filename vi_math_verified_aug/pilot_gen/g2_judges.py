"""G2 tasks 1–2: re-judge the G1 items Opus labelled with other judges.

python -m vi_math_verified_aug.pilot_gen.g2_judges --judges g38_v2 [--set native|render|all] [--holdout]
python -m vi_math_verified_aug.pilot_gen.g2_judges --solve haiku --samples 2      # cheap answer screen

Calibration items = G1 native items on the 30-seed Opus subsample (n=119, data/pilot/generator/judge_opus) and
G1 rendered texts Opus judged (n=70, judge_render). ``--holdout`` drops every item on the 4 seeds that supply the
validity_judge.v3 few-shot anchors (G1 idx 6204, 1569, 2120, 5630).
Output: data/pilot/g2/judge_<jkey>.jsonl (key = "<set>:<gen>:<seed_id>"), solve_<model>.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import json

from . import g2_common as G
from . import seeds as S

GENS = ["haiku", "sonnet", "gemini", "gemini31"]
ANCHOR_IDX = {6204, 1569, 2120, 5630}


def calib_items(which: str = "all", holdout: bool = False) -> list[dict]:
    seeds = {s["seed_id"]: s for s in S.load()}
    out = []
    for g in GENS:
        if which in ("all", "native"):
            nat, jo, os_ = G.g1_rows("native", g), G.g1_rows("judge_opus", g), G.g1_rows("opus_solve", g)
            for sid, j in jo.items():
                if not (j.get("parsed") and j.get("verdict")):
                    continue
                a = nat[sid]
                out.append({"key": f"native:{g}:{sid}", "set": "native", "gen": g, "seed_id": sid,
                            "seed": seeds[sid], "problem": a["problem"], "answer": a["answer"],
                            "opus": j["verdict"], "opus_agrees": (os_.get(sid) or {}).get("agrees")})
        if which in ("all", "render"):
            ren, jr = G.g1_rows("render", g), G.g1_rows("judge_render", g)
            for sid, j in jr.items():
                if not j.get("verdict"):
                    continue
                out.append({"key": f"render:{g}:{sid}", "set": "render", "gen": g, "seed_id": sid,
                            "seed": seeds[sid], "problem": ren[sid]["text"], "answer": None,
                            "opus": j["verdict"], "opus_agrees": None})
    if holdout:
        out = [i for i in out if i["seed"]["index"] not in ANCHOR_IDX]
    return out


async def judge_all(jkeys, items, items_full=None):
    """``jk@all`` runs judge jk on the full set (items_full) instead of the hold-out set."""
    async def one_judge(spec):
        jk, _, scope = spec.partition("@")
        st = G.Store(f"judge_{jk}")
        jobs = []
        for it in (items_full if scope == "all" and items_full is not None else items):
            s = it["seed"]
            item_id = f"gen:{it['set']}:{it['gen']}:{it['seed_id']}"
            jobs.append((it["key"], jk, G.judge_request(jk, s["text"], s["op"], G.op_instr(s), it["problem"],
                                                        item_id)))
        await G.run_judges(st, jobs)
        n_ok = sum(st.ok(k) for k, _, _ in jobs)
        print(jk, f"ok {n_ok}/{len(jobs)}", "spent", round(G.client().costs.spent(), 4), flush=True)
    await asyncio.gather(*(one_judge(jk) for jk in jkeys))


async def solve_all(model_key, items, samples, effort):
    st = G.Store(f"solve_{model_key}")

    async def one(it, k):
        key = f"{it['key']}#s{k}"
        if st.ok(key):
            return
        rq = G.solve_request(model_key, it["problem"], f"gen:{it['set']}:{it['gen']}:{it['seed_id']}",
                             effort=effort, sample_idx=k)
        r = await G.call(rq)
        box = G.last_boxed(r.text) if r.ok else None
        st.put(key, {"item": it["key"], "k": k, "boxed": box,
                     "agrees_gen": G.same_answer(box, G.answer_values(it["answer"])) if box else False,
                     **G.resp_meta(r)})
    await asyncio.gather(*(one(it, k) for it in items if it["answer"] is not None for k in range(samples)))
    print("solve", model_key, len(st.rows), "spent", round(G.client().costs.spent(), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="")
    ap.add_argument("--set", default="all")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--solve", default="")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = calib_items(a.set, a.holdout)
    if a.limit:
        items = items[:a.limit]
    print("items", len(items), flush=True)
    c = G.client()

    async def go():
        tasks = []
        if a.judges:
            tasks.append(judge_all([j for j in a.judges.split(",") if j], items, calib_items(a.set, False)))
        if a.solve:
            tasks.append(solve_all(a.solve, items, a.samples, a.effort))
        await asyncio.gather(*tasks)
    c.run(go())
    print(json.dumps({k: v for k, v in c.summary().items() if k != "lanes"}, default=str))


if __name__ == "__main__":
    main()
