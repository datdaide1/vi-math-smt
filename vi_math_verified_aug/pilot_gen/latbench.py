"""Clean latency benchmark (no queueing): 12 native + 12 solve calls per candidate at 3 in flight per model.

The pipeline's per-call latency includes waiting for a provider slot (semaphores / Gemini lanes), so it measures
queueing, not the API. This bench measures the API itself (logged with task=latbench_*; usage.http_latency).
python -m vi_math_verified_aug.pilot_gen.latbench
"""
from __future__ import annotations

import json
import statistics as st
from concurrent.futures import ThreadPoolExecutor

from . import clients as C
from . import seeds as S
from .pipeline import GRADE, MAXTOK, TEMP, op_instr
from .util import card_text, prompt

CAND = ["haiku", "sonnet", "gemini31"]  # gemini-3.5-flash-lite: daily quota exhausted; use pipeline http_latency


def run(gen):
    seeds = S.load()[:12]
    cards = S.cards()
    prov = C.MODELS[gen][0]
    out = []

    def one(args):
        task, s = args
        if task == "native":
            target = "khó hơn một bậc" if s["op"] in ("M2", "M3", "M6") else "giữ nguyên độ khó"
            sy, us = prompt("native_v1.txt", grade_hint=GRADE, seed_vi=s["text"], op=s["op"],
                            op_instruction=op_instr(s), card=card_text(cards[s["card"]]), target=target)
        else:
            sy, us = prompt("solve_v1.txt", problem=s["text"])
        mt = MAXTOK["native" if task == "native" else "solve"][prov]
        kw = {"effort": "low", "max_tokens": mt} if prov == "claude" else {"temperature": TEMP[
            "native" if task == "native" else "solve"], "max_tokens": mt}
        r = C.call(gen, sy, us, task=f"latbench_{task}", item=f"seed:{s['seed_id']}", tag="latbench", **kw)
        return task, (r.get("usage") or {}).get("http_latency"), (r.get("usage") or {}).get("output_tokens")
    with ThreadPoolExecutor(3) as ex:
        out = list(ex.map(one, [(t, s) for t in ("native", "solve") for s in seeds]))
    res = {}
    for task in ("native", "solve"):
        L = sorted(x[1] for x in out if x[0] == task and x[1] is not None)
        if L:
            res[task] = {"n": len(L), "p50": st.median(L), "p90": L[int(0.9 * (len(L) - 1) + 0.5)],
                         "out_mean": st.mean(x[2] or 0 for x in out if x[0] == task)}
    return gen, res


if __name__ == "__main__":
    with ThreadPoolExecutor(len(CAND)) as ex:
        R = dict(ex.map(run, CAND))
    json.dump(R, open(S.OUT / "latbench.json", "w"), indent=1)
    print(json.dumps(R, indent=1), "claude $", round(C.claude_spent(), 3))
