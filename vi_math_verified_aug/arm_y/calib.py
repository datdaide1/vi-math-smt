"""Validity-gate calibration on G4's 358 Opus-labelled items (E/R1/R2 renders + native N; Opus validity_judge.v2 is
the reference, Sonnet validity_judge.v3 is already on file). Adds Haiku 5.5 validity_judge.v3 (effort low, Batch) and
reports precision / recall of candidate gate rules vs Opus. The Gemini flash-lite leg of the G2 cascade could not be
run: the free-tier daily quota (500 RPD/project) was exhausted on 2026-11-07 (G4 §8).

    python -m vi_math_verified_aug.arm_y.calib   -> data/pilot/symbolic_v3/calib.json, calib_haiku.jsonl
"""
from __future__ import annotations

import asyncio
import json

from ..pilot_gen.util import OP_INSTR, fmt_ci, kappa
from ..pilot_symbolic.judge import lab_v3
from ..pilot_symbolic_v2 import judge as J4
from ..pilot_symbolic_v2.common import Store as Store4
from .common import ROOT, Store, call_many, parse_json, render, req, resp_meta, spent

OUT = ROOT / "data" / "pilot" / "symbolic_v3"

RULES = {
    "not_absurd": lambda l: not l["absurd"],
    "not_absurd_unansw": lambda l: not l["absurd"] and not l["unansw"],
    "valid": lambda l: l["valid"],
    "not_absurd_unansw_teacher": lambda l: not l["absurd"] and not l["unansw"] and l["teacher"],
    "valid_teacher": lambda l: l["valid"] and l["teacher"],
}


def opus_accept(l):
    return l["teacher"] and l["valid"] and not l["absurd"] and not l["leakj"] and not l["tail"]


async def run():
    its = J4.items()
    st = Store(OUT, "calib_haiku")
    todo = [it for it in its if not (st.get(it["key"]) or {}).get("api_ok")]
    reqs = [req("haiku", render("judge3", seed=it["seed"], op=it["op"], op_instruction=OP_INSTR[it["op"]],
                                problem=it["problem"]), item=f"gen:G4:{it['key']}", effort="low", max_tokens=4000,
                cache_system=True, tag="calib_haiku_v3") for it in todo]
    resps = await call_many(reqs, True, "calib")
    for it, r in zip(todo, resps):
        v = parse_json(r.text) if r.ok else None
        v = v if isinstance(v, dict) else None
        st.put(it["key"], {"labels": lab_v3(v) if v else None, **resp_meta(r)})
    jo, js = Store4("judge_opus").rows, Store4("judge_sonnet").rows
    H = Store(OUT, "calib_haiku").rows
    rows = []
    for it in its:
        o = (jo.get(it["key"]) or {}).get("labels")
        h = (H.get(it["key"]) or {}).get("labels")
        s = (js.get(it["key"]) or {}).get("labels")
        if o and h and s:
            rows.append((it, o, h, s))
    res = {"n": len(rows), "opus_accept_rate": sum(opus_accept(o) for _, o, _, _ in rows) / len(rows),
           "opus_absurd_rate": sum(o["absurd"] for _, o, _, _ in rows) / len(rows), "rules": {}, "kappa": {}}
    for f in ("absurd", "unansw", "valid", "teacher"):
        res["kappa"][f"haiku_{f}"] = kappa([(bool(o[f]), bool(h[f])) for _, o, h, _ in rows])
        res["kappa"][f"sonnet_{f}"] = kappa([(bool(o[f]), bool(s[f])) for _, o, _, s in rows])
    for judge in ("haiku", "sonnet", "haiku_and_sonnet"):
        for name, rule in RULES.items():
            def ok(h, s):
                if judge == "haiku":
                    return rule(h)
                if judge == "sonnet":
                    return rule(s)
                return rule(h) and rule(s)
            passed = [(it, o) for it, o, h, s in rows if ok(h, s)]
            acc_all = sum(opus_accept(o) for _, o, _, _ in rows)
            tp = sum(opus_accept(o) for _, o in passed)
            ab = sum(o["absurd"] for _, o in passed)
            ab_all = sum(o["absurd"] for _, o, _, _ in rows)
            ab_caught = sum(o["absurd"] and not ok(h, s) for _, o, h, s in rows)
            res["rules"][f"{judge}:{name}"] = {
                "pass": len(passed), "precision_accept": fmt_ci(tp, len(passed)), "recall_accept": fmt_ci(tp, acc_all),
                "absurd_after_gate": fmt_ci(ab, len(passed)), "absurd_recall": fmt_ci(ab_caught, ab_all),
                "teacher_after_gate": fmt_ci(sum(o["teacher"] for _, o in passed), len(passed)),
                "by_variant_absurd_after": {v: f"{sum(o['absurd'] for it, o in passed if it['variant'] == v)}/"
                                               f"{sum(1 for it, _ in passed if it['variant'] == v)}"
                                            for v in ("E", "R1", "R2", "N")}}
    json.dump(res, open(OUT / "calib.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"spent ${spent():.3f}")


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    asyncio.run(run())


# ------------------------------------------------------------------------------------------------ common screen
async def run_lite():
    """D1's exact flash-lite screen (scale_gen.lite: prompts/validity_lite.v1, packs of 10, 6 keys x {3.5, 3.1}
    flash-lite, labels -> lite_ok) on the same 358 items, logged under ws G5; then candidate common screens."""
    from ..scale_gen import lite as LT
    LT.client = __import__("vi_math_verified_aug.arm_y.common", fromlist=["client"]).client
    its = J4.items()
    st = Store(OUT, "calib_lite")
    await LT.judge_all(st, "vi", [(it["key"], it["problem"]) for it in its])
    await LT.judge_all(st, "vi", [(it["key"], it["problem"]) for it in its])     # retry missing packs once
    jo = Store4("judge_opus").rows
    H = Store(OUT, "calib_haiku").rows
    L = Store(OUT, "calib_lite").rows
    rows = []
    for it in its:
        o = (jo.get(it["key"]) or {}).get("labels")
        h = (H.get(it["key"]) or {}).get("labels")
        l = L.get(it["key"]) or {}
        if o and h and l.get("api_ok"):
            rows.append((it, o, h, l))
    hr = lambda h: not h["absurd"] and not h["unansw"] and h["teacher"]  # noqa: E731
    hr2 = lambda h: not h["absurd"] and not h["unansw"]  # noqa: E731
    cands = {"lite": lambda h, l: l["lite_ok"],
             "lite_not_absurd_unansw": lambda h, l: not l["absurd"] and not l["unansw"],
             "haiku(abs,unansw,teacher)": lambda h, l: hr(h),
             "haiku(abs,unansw,teacher) AND lite": lambda h, l: hr(h) and l["lite_ok"],
             "haiku(abs,unansw) AND lite": lambda h, l: hr2(h) and l["lite_ok"],
             "haiku valid AND lite": lambda h, l: h["valid"] and l["lite_ok"]}
    res = {"n": len(rows), "kappa_lite_absurd": kappa([(bool(o["absurd"]), bool(l["absurd"])) for _, o, _, l in rows]),
           "kappa_lite_unansw": kappa([(bool(o["unansw"]), bool(l["unansw"])) for _, o, _, l in rows]), "rules": {}}
    acc_all = sum(opus_accept(o) for _, o, _, _ in rows)
    ab_all = sum(o["absurd"] for _, o, _, _ in rows)
    for name, rule in cands.items():
        P = [(it, o) for it, o, h, l in rows if rule(h, l)]
        res["rules"][name] = {
            "pass": len(P), "precision_accept": fmt_ci(sum(opus_accept(o) for _, o in P), len(P)),
            "recall_accept": fmt_ci(sum(opus_accept(o) for _, o in P), acc_all),
            "absurd_after_gate": fmt_ci(sum(o["absurd"] for _, o in P), len(P)),
            "absurd_recall": fmt_ci(ab_all - sum(o["absurd"] for _, o in P), ab_all),
            "unansw_after_gate": fmt_ci(sum(o["unansw"] for _, o in P), len(P)),
            "teacher_after_gate": fmt_ci(sum(o["teacher"] for _, o in P), len(P)),
            "by_variant_absurd_after": {v: f"{sum(o['absurd'] for it, o in P if it['variant'] == v)}/"
                                           f"{sum(1 for it, _ in P if it['variant'] == v)}" for v in ("E", "R1", "R2", "N")}}
    json.dump(res, open(OUT / "calib_common_screen.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "lite":
    asyncio.run(run_lite())
