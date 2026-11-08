"""G3 stage 6: judging the rendered arm-Y items.

    python -m vi_math_verified_aug.pilot_symbolic.judge opus      # Opus 5.5, validity_judge.v2 (= G1 judge_v2, the
                                                                  # judge of record of G1 Table B / D2), effort medium
    python -m vi_math_verified_aug.pilot_symbolic.judge solve     # Opus 5.5 blind solve (solution_writer.v2)
    python -m vi_math_verified_aug.pilot_symbolic.judge seeds     # Opus on the 60 seed texts (baseline)
    python -m vi_math_verified_aug.pilot_symbolic.judge second    # gemini-3.1-flash-lite, validity_judge.v3 (G2 rubric)
Every tag-passing render is judged by Opus (survivors AND items rejected by the cycle check / gates, so the gates'
discrimination can be measured); the second judge runs on the same items (free tier, dev split only).
Outputs: data/pilot/symbolic/<run>/{judge_opus,solve_opus,judge_second}.jsonl
"""
from __future__ import annotations

import asyncio
import sys
from fractions import Fraction

from ..pilot_gen.util import OP_INSTR, answer_values, last_boxed
from . import seeds as SD
from .common import Store, call, parse_json, render, req, resp_meta

BOOLS = ["atomic", "no_conflict", "complete", "single_question", "realistic"]


def _nat(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def lab_v2(v: dict) -> dict:
    """validity_judge.v2 verdict -> labels (identical to G1 analyze.lab / G2 g2_common.lab_v2)."""
    cs = set(v.get("codes") or [])
    valid = all(v.get(b) is True for b in BOOLS) and _nat(v.get("natural")) >= 4
    return {"valid": valid, "teacher": v.get("teacher_would_write") is True, "new": v.get("new_vs_seed") is True,
            "op": v.get("op_followed") is True, "natural": _nat(v.get("natural")), "leakj": "NL-LEAK" in cs,
            "absurd": "NL-ABSURD" in cs, "tail": "NL-TACKED" in cs, "neardupj": "NL-NEARDUP" in cs,
            "fluency": "NL-FLUENCY" in cs, "unansw": "NL-UNANSW" in cs, "severity": v.get("severity")}


def lab_v3(v: dict) -> dict:
    """validity_judge.v3 verdict -> the same label set (identical to G2 g2_common.lab_v3)."""
    t = lambda k: v.get(k) is True  # noqa: E731
    absurd = t("q_noninteger_count") or t("q_implausible_quantity") or t("q_incoherent_setting")
    unansw = t("q_contradiction") or t("q_missing_or_ambiguous")
    nat = _nat(v.get("natural"))
    valid = not (unansw or t("q_multi_question") or absurd) and nat >= 4
    return {"valid": valid, "teacher": t("teacher_would_write"), "new": not (t("q_same_structure") or t("q_tail")),
            "op": t("q_op_followed"), "natural": nat, "leakj": t("q_leak"), "absurd": absurd, "tail": t("q_tail"),
            "neardupj": t("q_same_structure"), "fluency": t("q_unnatural_vi") or nat < 4, "unansw": unansw,
            "judge_answer": v.get("answer")}


def items() -> list[dict]:
    """Every tag-passing render of the current run, with what the judges need."""
    seeds = SD.load_seeds()
    m = Store("mutate").rows
    out = []
    for r in Store("render").rows.values():
        if not r.get("tag_ok"):
            continue
        mr = m[r["key"]]
        op = r["op"]
        instr = OP_INSTR[op] + (seeds[mr["partner"].split("+")[-1]]["text"] if op == "M6" and mr.get("partner") else "")
        out.append({"key": r["key"], "seed_id": r["seed_id"], "op": op, "seed": seeds[r["seed_id"]]["text"],
                    "instr": instr, "problem": r["text"], "answer": mr["answer"]})
    return sorted(out, key=lambda x: x["key"])


async def _run(store: Store, jobs):
    async def one(key, rq, post):
        r0 = store.get(key)
        if r0 and r0.get("api_ok"):
            return
        try:
            r = await call(rq)
        except Exception as e:                                   # BudgetExceeded etc.
            store.put(key, {"api_ok": False, "err": f"{type(e).__name__}: {e}"})
            return
        store.put(key, {**post(r), **resp_meta(r)})
    await asyncio.gather(*(one(*j) for j in jobs))


def _judge_post(lab):
    def post(r):
        v = parse_json(r.text) if r.ok else None
        v = v if isinstance(v, dict) else None
        return {"verdict": v, "parsed": v is not None, "labels": lab(v) if v else None}
    return post


async def stage_opus():
    st = Store("judge_opus")
    jobs = []
    for it in items():
        rp = render("judge", seed=it["seed"], op=it["op"], op_instruction=it["instr"], problem=it["problem"])
        rq = req("opus", rp, item=f"gen:G3:{it['key']}", effort="medium", max_tokens=3000, cache_system=True,
                 tag="judge_opus")
        jobs.append((it["key"], rq, _judge_post(lab_v2)))
    await _run(st, jobs)
    print("opus judged", sum(bool(r.get("parsed")) for r in st.rows.values()), "/", len(jobs))


async def stage_solve():
    st = Store("solve_opus")
    jobs = []
    for it in items():
        rp = render("solve", problem=it["problem"])
        rq = req("opus", rp, item=f"gen:G3:{it['key']}", effort="medium", max_tokens=4000, tag="solve_opus")
        target = Fraction(it["answer"])

        def post(r, target=target):
            b = last_boxed(r.text) if r.ok else None
            vals = answer_values(b) if b is not None else set()
            return {"boxed": b, "agree": target in vals, "target": str(target)}
        jobs.append((it["key"], rq, post))
    await _run(st, jobs)
    print("opus solved", sum(bool(r.get("api_ok")) for r in st.rows.values()), "/", len(jobs),
          "agree", sum(bool(r.get("agree")) for r in st.rows.values()))


async def stage_second():
    st = Store("judge_second")
    jobs = []
    for it in items():
        rp = render("judge3", seed=it["seed"], op=it["op"], op_instruction=it["instr"], problem=it["problem"])
        rq = req("g31l", rp, item=f"gen:G3:{it['key']}", max_tokens=6144, temperature=0.0, json_mode=True,
                 tag="judge_g31l_v3")
        jobs.append((it["key"], rq, _judge_post(lab_v3)))
    await _run(st, jobs)
    print("second judged", sum(bool(r.get("parsed")) for r in st.rows.values()), "/", len(jobs))


async def stage_seeds():
    """Baseline: the same Opus judge on the 60 seed texts themselves (op M0 = no transformation), to separate
    seed-text quality (machine-translated Vi-GSM8K) from what the arm-Y pipeline adds or removes."""
    st = Store("judge_opus_seeds")
    jobs = []
    for s in SD.load_seeds().values():
        rp = render("judge", seed=s["text"], op="M0", op_instruction="Không biến đổi: đây là chính bài gốc "
                    "(chỉ đánh giá chất lượng đề bài; new_vs_seed và op_followed không áp dụng).", problem=s["text"])
        rq = req("opus", rp, item=f"seed:{s['seed_id']}", effort="medium", max_tokens=3000, cache_system=True,
                 tag="judge_opus_seed")
        jobs.append((s["seed_id"], rq, _judge_post(lab_v2)))
    await _run(st, jobs)
    print("seeds judged", sum(bool(r.get("parsed")) for r in st.rows.values()), "/", len(jobs))


def main(argv):
    stage = argv[1] if len(argv) > 1 else ""
    fn = {"opus": stage_opus, "solve": stage_solve, "second": stage_second, "seeds": stage_seeds}.get(stage)
    if fn is None:
        raise SystemExit(__doc__)
    asyncio.run(fn())


if __name__ == "__main__":
    main(sys.argv)
