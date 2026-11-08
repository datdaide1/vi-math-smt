"""Stage ``translationese``: Opus rates naturalness / teacher-acceptability of 60 random pool seeds.

Same judge, prompt and settings as G3's seed baseline (``pilot_symbolic.judge.stage_seeds``): Claude Opus 5.5,
``validity_judge.v2`` (the G1 judge of record), effort medium, op ``M0`` ("no transformation"), so the number is
comparable with G3's 17 % [9-28] on 60 Vi-GSM8K dev seeds. The seed texts are NOT rewritten (S3 brief): this only
measures the ceiling the arms inherit.

Sample: ``random.Random(20261107).sample`` of 60 seeds from the frozen candidate pool (non-DEV, not dropped),
sorted by seed_id first. Wilson 95 % CIs.
"""
from __future__ import annotations

import asyncio
import json
import random
from collections import Counter

from .common import RNG_SEED, Store, WORK, call, client, read_jsonl, render, req, resp_meta, wilson, write_json

N = 60
OP_M0 = ("Không biến đổi: đây là chính bài gốc (chỉ đánh giá chất lượng đề bài; new_vs_seed và op_followed "
         "không áp dụng).")
BOOLS = ["atomic", "no_conflict", "complete", "single_question", "realistic"]


def sample() -> list[dict]:
    from .assemble import current_rows
    drop = {r["seed_id"] for r in read_jsonl(WORK / "decontam.jsonl") if r["drop"]}
    rows = sorted((r for r in current_rows() if not r["dev"] and r["seed_id"] not in drop),
                  key=lambda r: r["seed_id"])
    return random.Random(RNG_SEED).sample(rows, N)


def labels(v: dict) -> dict:
    cs = set(v.get("codes") or [])
    try:
        nat = float(v.get("natural"))
    except Exception:
        nat = 0.0
    return {"teacher": v.get("teacher_would_write") is True, "natural": nat, "natural_ge4": nat >= 4,
            "fluency": "NL-FLUENCY" in cs, "unansw": "NL-UNANSW" in cs, "absurd": "NL-ABSURD" in cs,
            "latex": "NL-LATEX" in cs, "leak": "NL-LEAK" in cs,
            "valid": all(v.get(b) is True for b in BOOLS) and nat >= 4, "severity": v.get("severity")}


def run() -> dict:
    from ..common.llm import LLMRequest, parse_json
    S = sample()
    store = Store("translationese")
    c = client()

    async def one(r):
        old = store.get(r["seed_id"])
        if old and old.get("parsed"):
            return
        rp = render("validity_judge", 2, seed=r["problem_vi"], op="M0", op_instruction=OP_M0, problem=r["problem_vi"])
        rq = LLMRequest(model="claude-opus-5-5", system=None, messages=[{"role": "user", "content": rp.user}],
                        max_tokens=3000, extra={"output_config": {"effort": "medium"},
                                                "system": [{"type": "text", "text": rp.system,
                                                            "cache_control": {"type": "ephemeral"}}]},
                        item_id=f"seed:{r['seed_id']}", split="train", prompt_id=rp.prompt_id,
                        prompt_version=rp.version, prompt_sha256=rp.sha256, tag="S3-translationese-opus")
        resp = await call(rq)
        v = parse_json(resp.text) if resp.ok else None
        v = v if isinstance(v, dict) else None
        store.put(r["seed_id"], {"seed_id": r["seed_id"], "source": r["source"], "index": r["index"],
                                 "provenance": r["provenance"], "verdict": v, "parsed": v is not None,
                                 "labels": labels(v) if v else None, **resp_meta(resp)})

    async def main():
        try:
            await asyncio.gather(*(one(r) for r in S))
        finally:
            await c.aclose()
    asyncio.run(main())
    rows = [store.get(r["seed_id"]) for r in S if store.get(r["seed_id"]) and store.get(r["seed_id"]).get("parsed")]
    out = {"n_sample": len(S), "n_parsed": len(rows), "sample_by_source": dict(Counter(r["source"] for r in S)),
           "judge": "claude-opus-5-5 validity_judge.v2 effort=medium op=M0 (= G3 seed baseline)"}

    def agg(rs, tag):
        n = len(rs)
        res = {"n": n}
        for k in ("teacher", "natural_ge4", "fluency", "valid", "unansw", "absurd", "latex"):
            kk = sum(1 for r in rs if r["labels"][k])
            res[k] = {"k": kk, "pct": round(100 * kk / max(n, 1), 1), "ci95": wilson(kk, n)}
        res["natural_mean"] = round(sum(r["labels"]["natural"] for r in rs) / max(n, 1), 2)
        res["severity"] = dict(Counter(r["labels"]["severity"] for r in rs))
        out[tag] = res
    agg(rows, "all")
    for src in sorted({r["source"] for r in rows}):
        agg([r for r in rows if r["source"] == src], f"by_source:{src}")
    agg([r for r in rows if r["source"] == "vi_gsm8k_train"], "gsm8k_only")
    out["spend"] = c.costs.summary()
    write_json(WORK / "translationese_summary.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out
