"""G4 judging (Message Batches API, 50% price; every response cached, so re-runs are free).

    python -m vi_math_verified_aug.pilot_symbolic_v2.judge native   # arm N reference: Haiku native_generator.v3
                                                                     # on the SAME (seed, operator) slots (direct)
    python -m vi_math_verified_aug.pilot_symbolic_v2.judge all      # Opus v2 judge + Opus blind solve (+ Sonnet v3)

Item sets (key = "<variant>:<slot key>"):
  E   tag-passing edit-only renders. Slots identical to G3 reuse G3's Opus judge / solve rows (same model, prompt,
      effort; G3 called them directly). Only the re-rendered slots are judged here.
  R1, R2, N   every tag-passing / parsed item.
Judges: Opus 5.5 validity_judge.v2 (effort medium; the judge of record of G1-G3), Opus 5.5 solution_writer.v2
(blind solve; agreement with the program answer = answer faithfulness), Sonnet 5.5 validity_judge.v3 (G2's best
cheap judge; effort low) for kappa against Opus. Batches are chunked so that the worst-case reservation of each
batch fits under the remaining $ cap.
"""
from __future__ import annotations

import asyncio
import sys
from fractions import Fraction

from ..common.costs import worst_case_cost
from ..pilot_gen.util import OP_INSTR, answer_values, last_boxed
from ..pilot_symbolic import seeds as SD
from ..pilot_symbolic.judge import lab_v2, lab_v3
from .common import BUDGET, Store, call, client, g3, parse_json, render, req, resp_meta
from .localize import card_for

GRADE = "tiểu học và THCS (lớp 3–7)"


# ---------------------------------------------------------------------------------------------- arm N
async def stage_native():
    seeds = SD.load_seeds()
    st = Store("native_N")
    mut = [r for r in Store("mutate").rows.values() if r.get("ok")]

    async def one(m):
        if st.get(m["key"]) and st.get(m["key"]).get("api_ok"):
            return
        s = seeds[m["seed_id"]]
        _, card = card_for(m["seed_id"])
        target = "khó hơn một bậc" if m["op"] in ("M2", "M3") else "giữ nguyên độ khó"
        rp = render_native(s["text"], m["op"], card, target)
        rq = req("haiku", rp, item=f"gen:G4:N:{m['key']}", effort="low", max_tokens=4000, cache_system=True,
                 tag="native_v3")
        r = await call(rq)
        if r.ok and r.finish_reason == "max_tokens":
            rq.max_tokens *= 2
            r = await call(rq)
        j = parse_json(r.text) if r.ok else None
        j = j if isinstance(j, dict) else {}
        prob = j.get("problem") if isinstance(j.get("problem"), str) else None
        st.put(m["key"], {"seed_id": m["seed_id"], "op": m["op"], "variant": "N", "parsed": bool(prob),
                          "text": prob, "target": None if j.get("answer") is None else str(j.get("answer")),
                          **resp_meta(r)})
    await asyncio.gather(*(one(m) for m in mut))
    print("native parsed", sum(bool(r.get("parsed")) for r in st.rows.values()), "/", len(st.rows))


def render_native(seed_vi, op, card, target):
    from ..common.prompts import PromptRegistry
    return PromptRegistry().get("native_generator", 3).render(grade_hint=GRADE, seed_vi=seed_vi, op=op,
                                                              op_instruction=OP_INSTR[op], card=card, target=target)


# ---------------------------------------------------------------------------------------------- items
def items() -> list[dict]:
    seeds = SD.load_seeds()
    mut = Store("mutate").rows
    out = []
    for v in ("E", "R1", "R2", "R1b"):
        for r in Store(f"render_{v}").rows.values():
            if not r.get("tag_ok"):
                continue
            g3copy = v == "E" and mut[r["key"]]["same_as_g3"]
            out.append({"key": f"{v}:{r['key']}", "slot": r["key"], "variant": v, "op": r["op"],
                        "seed": seeds[r["seed_id"]]["text"], "problem": r["text"], "answer": r["target"],
                        "g3copy": g3copy})
    for r in Store("native_N").rows.values():
        if r.get("parsed"):
            out.append({"key": f"N:{r['key']}", "slot": r["key"], "variant": "N", "op": r["op"],
                        "seed": seeds[r["seed_id"]]["text"], "problem": r["text"], "answer": r["target"],
                        "g3copy": False})
    return sorted(out, key=lambda x: x["key"])


def _judge_req(model_key, ver, eff, mt, it):
    rp = render("judge" if ver == 2 else "judge3", seed=it["seed"], op=it["op"], op_instruction=OP_INSTR[it["op"]],
                problem=it["problem"])
    return req(model_key, rp, item=f"gen:G4:{it['key']}", effort=eff, max_tokens=mt, cache_system=True,
               tag=f"judge_{model_key}_v{ver}")


def _solve_req(it):
    rp = render("solve", problem=it["problem"])
    return req("opus", rp, item=f"gen:G4:{it['key']}", effort="medium", max_tokens=4000, tag="solve_opus")


def _post_judge(lab):
    def post(r, it):
        v = parse_json(r.text) if r.ok else None
        v = v if isinstance(v, dict) else None
        return {"verdict": v, "parsed": v is not None, "labels": lab(v) if v else None}
    return post


def _post_solve(r, it):
    b = last_boxed(r.text) if r.ok else None
    vals = answer_values(b) if b is not None else set()
    tgt = it["answer"]
    agree = None
    if tgt is not None:
        try:
            tv = answer_values(str(tgt)) if it["variant"] == "N" else {Fraction(tgt)}
            agree = bool(vals & tv)
        except Exception:                                  # noqa: BLE001
            agree = False
    return {"boxed": b, "agree": agree, "target": tgt}


async def run_batched(jobs: list[tuple[str, Store, object, callable, dict]]):
    """jobs: (key, store, LLMRequest, post, item). Skips keys already ok; chunks under the $ cap."""
    c = client()
    todo = [j for j in jobs if not (j[1].get(j[0]) or {}).get("api_ok")]
    while todo:
        allow = BUDGET - c.costs.spent() - 0.15
        chunk, est = [], 0.0
        for j in todo:
            e = worst_case_cost(j[2].est_input_tokens(), j[2].max_tokens, c._price("claude", j[2].model), batch=True)
            if chunk and est + e > allow:
                break
            chunk.append(j)
            est += e
        if est > allow:
            print(f"budget: cannot reserve {est:.2f} (allow {allow:.2f}); stopping with {len(todo)} left", flush=True)
            return
        print(f"batch: {len(chunk)} requests, reserve ${est:.2f}, spent ${c.costs.spent():.3f}", flush=True)
        resps = await c.batches.run([j[2] for j in chunk], poll_s=30)
        for (k, st, rq, post, it), r in zip(chunk, resps):
            st.put(k, {**post(r, it), **resp_meta(r)})
        todo = todo[len(chunk):]
        print(f"done chunk; spent ${c.costs.spent():.3f}", flush=True)


async def stage_all(which: str):
    its = items()
    g3j, g3s = g3("judge_opus"), g3("solve_opus")
    jo, so, js = Store("judge_opus"), Store("solve_opus"), Store("judge_sonnet")
    for it in its:                     # E rows identical to G3: copy G3's verdicts (same model / prompt / effort)
        if it["g3copy"]:
            if it["slot"] in g3j and not jo.get(it["key"]):
                jo.put(it["key"], {**{k: v for k, v in g3j[it["slot"]].items() if k != "key"}, "source": "g3"})
            if it["slot"] in g3s and not so.get(it["key"]):
                so.put(it["key"], {**{k: v for k, v in g3s[it["slot"]].items() if k != "key"}, "source": "g3"})
    jobs = []
    if "opus" in which:
        jobs += [(it["key"], jo, _judge_req("opus", 2, "medium", 3000, it), _post_judge(lab_v2), it)
                 for it in its if not it["g3copy"]]
    if "solve" in which:
        jobs += [(it["key"], so, _solve_req(it), _post_solve, it) for it in its if not it["g3copy"]]
    if "sonnet" in which:
        jobs += [(it["key"], js, _judge_req("sonnet", 3, "low", 4000, it), _post_judge(lab_v3), it) for it in its]
    await run_batched(jobs)
    for name, st in (("opus", jo), ("solve", so), ("sonnet", js)):
        print(name, sum(bool(r.get("api_ok")) for r in st.rows.values()), "/", len(its))


def main(argv):
    stage = argv[1] if len(argv) > 1 else ""
    if stage == "native":
        asyncio.run(stage_native())
    elif stage == "all":
        asyncio.run(stage_all(argv[2] if len(argv) > 2 else "opus,solve,sonnet"))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
