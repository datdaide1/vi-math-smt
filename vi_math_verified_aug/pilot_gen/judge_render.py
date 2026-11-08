"""Quality of the editor's (c) outputs beyond the mechanical tag check: Opus 5.5 judge (judge_v2, effort medium,
cached rubric) on the rendered texts that passed the tag check, for seeds in the Opus subsample only (budget).
(gemini-3.5-flash-lite was planned but its free daily quota was exhausted by the main run.)
python -m vi_math_verified_aug.pilot_gen.judge_render -> data/pilot/generator/judge_render/<gen>.jsonl
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from . import clients as C
from . import seeds as S
from .pipeline import op_instr
from .util import prompt

CAND = ["haiku", "sonnet", "gemini", "gemini31"]
OUT = S.OUT


def main():
    seeds = {s["seed_id"]: s for s in S.load()}
    jobs = []
    for g in CAND:
        rows = {}
        for l in open(OUT / "render" / f"{g}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            rows[r["key"]] = r
        sub = set(json.load(open(OUT / "opus_subsample.json")))
        jobs += [(g, r) for r in rows.values() if r["pass"] and r["seed_id"] in sub]

    def one(job):
        g, r = job
        s = seeds[r["seed_id"]]
        sy, us = prompt("judge_v2.txt", seed=s["text"], op=s["op"], op_instruction=op_instr(s), problem=r["text"])
        res = C.call("opus", sy, us, task="judge_render", item=f"gen:render:{g}:{r['seed_id']}",
                     effort="medium", max_tokens=3000, cache_system=True)
        return g, {"key": r["seed_id"], "seed_id": r["seed_id"], "gen": g, "op": s["op"],
                   "verdict": C.parse_json(res.get("text")) or {}, "api_ok": res.get("ok")}
    with ThreadPoolExecutor(16) as ex:
        out = list(ex.map(one, jobs))
    (OUT / "judge_render").mkdir(exist_ok=True)
    for g in CAND:
        with open(OUT / "judge_render" / f"{g}.jsonl", "w", encoding="utf-8") as f:
            for gg, row in out:
                if gg == g:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(len(out))


if __name__ == "__main__":
    main()
