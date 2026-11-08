"""G2 task 3: native-generation prompt iteration (arm N, generator Claude Haiku 5.5).

python -m vi_math_verified_aug.pilot_gen.g2_native seeds          # 40 NEW dev seeds -> data/pilot/g2/seeds40.jsonl
python -m vi_math_verified_aug.pilot_gen.g2_native run --arms v1,v2,v3,v4 [--judges ...]

Seeds: same dev pool as G1 (Vi-GSM8K train, seed_id hash ≡ 0 mod 20, R1 corrupt + half-English rows excluded),
minus G1's 100 seeds; 40 drawn with random.Random(20261108); operators stratified to π
(M1 8, M2 8, M3 10, M4 6, M5 4, M6 4); card + M6 partner from the same RNG (partner also outside G1's seeds).
Arms: v1 = G1 native_v1 (control, pilot_gen/prompts/native_v1.txt); v2/v3/v4 = prompts/native_generator.v2–v4;
"<arm>m" = same prompt at effort=medium. Each item: Opus judge (validity_judge.v2 = G1 judge_v2, effort medium),
Opus blind-solve (solution_writer.v2), gemini-3.8-flash judge (v2), plus cheap judges for out-of-sample cascade
checks. Rows: data/pilot/g2/n_gen.jsonl, n_judge_<jkey>.jsonl, n_opus_solve.jsonl, n_solve_haiku.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random

from . import g2_common as G
from . import seeds as S
from .util import card_text

SEED2 = 20261108
STRATA = {"M1": 8, "M2": 8, "M3": 10, "M4": 6, "M5": 4, "M6": 4}
GRADE = "tiểu học và THCS (lớp 3–7)"
SEEDS_PATH = G.OUT / "seeds40.jsonl"
GEN_MAXTOK = {"low": 4000, "medium": 8000}


def build_seeds() -> dict:
    vi = [json.loads(l) for l in open(S.VI, encoding="utf-8")]
    en = [json.loads(l) for l in open(S.EN, encoding="utf-8")]
    g1 = {s["seed_id"] for s in S.load()}
    pool = []
    for v, e in zip(vi, en):
        if v["index"] in S.R1_CORRUPT | S.R1_HALF_ENGLISH:
            continue
        sid = S.seed_id(v["index"])
        if int(sid, 16) % 20 or sid in g1:
            continue
        pool.append({"seed_id": sid, "index": v["index"], "text": v["question"],
                     "gold": str(e["ground_truth"]).replace(",", ""), "en_text": e["question"]})
    pool.sort(key=lambda p: p["index"])
    rng = random.Random(SEED2)
    picked = rng.sample(pool, 40 + 4)
    seeds, partners = picked[:40], picked[40:]
    ops = [o for o, k in STRATA.items() for _ in range(k)]
    rng.shuffle(ops)
    cards = sorted(S.cards())
    for s, o in zip(seeds, ops):
        s["op"], s["card"], s["partner"] = o, rng.choice(cards), None
    for s in [s for s in seeds if s["op"] == "M6"]:
        p = partners.pop()
        s.update(partner=p["seed_id"], partner_text=p["text"], partner_gold=p["gold"], partner_index=p["index"])
    seeds.sort(key=lambda s: s["index"])
    with open(SEEDS_PATH, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    info = {"dev_pool_excl_g1": len(pool), "n": len(seeds), "ops": {o: sum(s["op"] == o for s in seeds) for o in STRATA},
            "overlap_with_g1": len({s["seed_id"] for s in seeds} & g1)}
    print(json.dumps(info))
    return info


def load_seeds() -> list[dict]:
    return [json.loads(l) for l in open(SEEDS_PATH, encoding="utf-8")]


def gen_request(arm: str, s: dict):
    effort = "medium" if arm.endswith("m") else "low"
    base = arm.rstrip("m")
    cards = S.cards()
    target = "khó hơn một bậc" if s["op"] in ("M2", "M3", "M6") else "giữ nguyên độ khó"
    kw = dict(grade_hint=GRADE, seed_vi=s["text"], op=s["op"], op_instruction=G.op_instr(s),
              card=card_text(cards[s["card"]]), target=target)
    if base == "v1":
        sy, us = G.g1_prompt("native_v1.txt", **kw)
        pid, ver, sha = "pilot_gen/native_v1", 1, None
    else:
        rp = G.render("native_generator", int(base[1:]), **kw)
        sy, us, pid, ver, sha = rp.system, rp.user, rp.prompt_id, rp.version, rp.sha256
    return G.req("haiku", sy, us, item=f"seed:{s['seed_id']}", effort=effort, max_tokens=GEN_MAXTOK[effort],
                 cache_system=True, prompt_id=pid, version=ver, sha=sha, tag=f"native_{arm}")


async def run(arms: list[str], judges: list[str], opus_solve: bool, haiku_solve: int, calib: str = ""):
    seeds = load_seeds()
    side = None
    if calib:            # finish task-1/2 calibration judging in the same process (one process per WS)
        from . import g2_judges as J
        specs = calib.split(",")
        full = J.calib_items("native", False)
        side = asyncio.ensure_future(asyncio.gather(*(
            J.judge_all([sp.split(":")[0]], [], [i for i in full if ":" not in sp or i["gen"] in sp.split(":")[1].split("+")])
            for sp in specs)))
    gst = G.Store("n_gen")

    async def gen(arm, s):
        key = f"{arm}:{s['seed_id']}"
        prev = gst.rows.get(key)
        retry = bool(prev and prev.get("api_ok") and not prev.get("parsed") and prev.get("finish") == "max_tokens"
                     and not prev.get("retried"))
        if not gst.ok(key) or retry:
            rq = gen_request(arm, s)
            if retry:                      # policy (all arms): one retry at 2x max_tokens after a thinking cut-off
                rq.max_tokens *= 2
            r = await G.call(rq)
            j = G.parse_json(r.text) if r.ok else None
            j = j if isinstance(j, dict) else {}
            prob = j.get("problem") if isinstance(j.get("problem"), str) else None
            ans = j.get("answer")
            gst.put(key, {"arm": arm, "seed_id": s["seed_id"], "op": s["op"], "parsed": bool(prob and ans is not None),
                          "problem": prob, "answer": None if ans is None else str(ans),
                          "self_check": j.get("self_check"), "solution": j.get("solution"),
                          "op_applied": j.get("op_applied"), "change_summary": j.get("change_summary"),
                          "retried": retry, **G.resp_meta(r)})
        return gst.rows[key]

    rows = await asyncio.gather(*(gen(a, s) for a in arms for s in seeds))
    print("generated", sum(r.get("parsed", False) for r in rows), "/", len(rows),
          "spent", round(G.client().costs.spent(), 4), flush=True)
    by = {s["seed_id"]: s for s in seeds}
    todo = [r for r in rows if r.get("parsed")]

    async def judge(jk):
        st = G.Store(f"n_judge_{jk}")
        jobs = [(r["key"], jk, G.judge_request(jk, by[r["seed_id"]]["text"], r["op"], G.op_instr(by[r["seed_id"]]),
                                               r["problem"], f"gen:n:{r['key']}")) for r in todo]
        await G.run_judges(st, jobs)
        print(jk, sum(st.ok(k) for k, _, _ in jobs), "/", len(jobs), "spent", round(G.client().costs.spent(), 4),
              flush=True)

    async def solve(model, effort, k, stage):
        st = G.Store(stage)

        async def one(r, i):
            key = f"{r['key']}#s{i}"
            if st.ok(key):
                return
            rq = G.solve_request(model, r["problem"], f"gen:n:{r['key']}", effort=effort, sample_idx=i)
            try:
                res = await G.call(rq)
            except Exception as e:
                st.put(key, {"api_ok": False, "err": f"{type(e).__name__}: {e}"})
                return
            box = G.last_boxed(res.text) if res.ok else None
            st.put(key, {"item": r["key"], "k": i, "boxed": box,
                         "agrees_gen": G.same_answer(box, G.answer_values(r["answer"])) if box else False,
                         **G.resp_meta(res)})
        await asyncio.gather(*(one(r, i) for r in todo for i in range(k)))
        print(stage, len(st.rows), "spent", round(G.client().costs.spent(), 4), flush=True)

    tasks = [judge(jk) for jk in judges]
    if opus_solve:
        tasks.append(solve("opus", "medium", 1, "n_opus_solve"))
    if haiku_solve:
        tasks.append(solve("haiku", "low", haiku_solve, "n_solve_haiku"))
    await asyncio.gather(*tasks)
    if side is not None:
        await side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seeds", "run"])
    ap.add_argument("--arms", default="v1,v2,v3,v4")
    ap.add_argument("--judges", default="opus_v2,g38_v2,haiku_v3,g31l_v3")
    ap.add_argument("--no-opus-solve", action="store_true")
    ap.add_argument("--haiku-solve", type=int, default=2)
    ap.add_argument("--calib", default="", help="e.g. g38_v2@all: also finish calibration judging")
    a = ap.parse_args()
    if a.cmd == "seeds":
        build_seeds()
        return
    c = G.client()
    c.run(run([x for x in a.arms.split(",") if x], [j for j in a.judges.split(",") if j], not a.no_opus_solve,
              a.haiku_solve, a.calib))
    print(json.dumps({k: v for k, v in c.summary().items() if k != "lanes"}, default=str))


if __name__ == "__main__":
    main()
