"""Cross-family cycle check on NVIDIA NIM (lead 2026-11-08 02:10: the Gemini free tier is exhausted on both
flash-lite models until ~15:00; a same-family Haiku back-formalizer is only a secondary/ablation label).

    python -m vi_math_verified_aug.arm_y.nim_cycle validate <model_key> [n]   # vs G4's Gemini cycle verdicts
    python -m vi_math_verified_aug.arm_y.nim_cycle run <run_dir> <variant> <model_key>

Blind back-formalization with prompts/back_formalizer.v1 (EN: back_formalizer_en.v1), temperature 0, JSON mode,
<= 2 attempts (the retry gets only the validator/exec errors, never the answer). Training / dev items only (the
LLMClient routing guard refuses anything else). One shared 40-RPM NIM quota: run one model at a time.
Acceptance (lead): agreement with the Gemini cycle verdict >= 90% on ~100 G4 items and >= 30 items/min.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

from ..pilot_symbolic.pipeline import _feedback, _fx, load_ir
from ..symbolic import execute
from .common import MODELS, ROOT, Store, call, render, req, resp_meta

G4 = ROOT / "data" / "pilot" / "symbolic_v2"


async def back_one(model_key: str, text: str, target: str, item: str, lang: str = "vi", tag: str = "back_nim") -> dict:
    pk = "back_en" if lang == "en" else "back"
    fb, x, ans = "", None, None
    t0 = time.time()
    for a in range(2):
        x = await call(req(model_key, render(pk, problem=text), item=item, max_tokens=6144, temperature=0.0,
                           json_mode=True, tag=tag, extra_user=fb))
        if not x.ok:
            return {"done": False, "api_fail": True, "cycle_model": MODELS[model_key], **resp_meta(x)}
        P3, errs = load_ir(f"back:{item}", text, x.text, None)
        ans = None
        if P3 is not None:
            try:
                ans = execute(P3)[P3.query]
            except Exception as e:                     # noqa: BLE001
                errs = errs + [f"exec: {e}"]
        if ans is not None:
            break
        fb = _feedback(x.text or "", [e for e in errs if not e.startswith("[gold]")] or ["could not execute"])
    return {"done": True, "executed": ans is not None, "back_answer": _fx(ans), "target": target,
            "cycle_ok": ans is not None and ans == Fraction(target), "n_attempts": a + 1,
            "cycle_model": MODELS[model_key], "secs": round(time.time() - t0, 1), **resp_meta(x)}


async def validate(model_key: str, n: int = 100):
    items = []
    for v in ("R2", "R1"):
        cyc = {json.loads(l)["key"]: json.loads(l) for l in open(G4 / f"cycle_{v}.jsonl", encoding="utf-8")}
        rnd = {json.loads(l)["key"]: json.loads(l) for l in open(G4 / f"render_{v}.jsonl", encoding="utf-8")}
        for k, c in sorted(cyc.items()):
            if c.get("done") and not c.get("api_fail") and rnd.get(k, {}).get("text"):
                items.append((f"{v}:{k}", rnd[k]["text"], rnd[k]["target"], bool(c["cycle_ok"])))
    items = items[:n]
    st = Store(ROOT / "data" / "pilot" / "symbolic_v3", f"nimval_{model_key}")
    t0 = time.time()

    async def one(it):
        k, text, tgt, _ = it
        if (st.get(k) or {}).get("done"):
            return
        st.put(k, await back_one(model_key, text, tgt, f"gen:G5:nimval:{k}"))
    await asyncio.gather(*(one(it) for it in items))
    dt = (time.time() - t0) / 60
    rows = [(st.get(k) or {}, g) for k, _, _, g in items]
    done = [(r, g) for r, g in rows if r.get("done")]
    agree = sum(bool(r["cycle_ok"]) == g for r, g in done)
    res = {"model": MODELS[model_key], "n": len(items), "done": len(done), "api_fail": len(rows) - len(done),
           "agree_with_gemini": agree, "agree_rate": round(agree / max(1, len(done)), 3),
           "nim_pass": sum(bool(r["cycle_ok"]) for r, _ in done), "gemini_pass": sum(g for _, g in done),
           "minutes": round(dt, 2), "items_per_min": round(len(done) / dt, 1) if dt else None,
           "mean_attempts": round(sum(r["n_attempts"] for r, _ in done) / max(1, len(done)), 2)}
    print(json.dumps(res))
    with open(ROOT / "data" / "pilot" / "symbolic_v3" / "nim_validation.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")


async def run(run_dir: Path, v: str, model_key: str):
    """Primary cycle verdicts into cycle_<v>.jsonl (streamed, resumable); rows already decided by a cross-family
    model (Gemini, NIM) are kept. The Haiku verdicts live in cycle_haiku_<v>.jsonl (ablation only)."""
    st = Store(run_dir, f"cycle_{v}")
    rows = [r for r in Store(run_dir, f"render_{v}").rows.values() if r.get("tag_ok")]
    todo = [r for r in rows if not (st.get(r["key"]) or {}).get("done")]
    lang = "en" if v == "EN" else "vi"
    t0, n = time.time(), [0]

    async def one(r):
        res = await back_one(model_key, r["text"], r["target"], f"gen:G5:{v}:{r['key']}", lang, f"back_{v}_nim")
        st.put(r["key"], {"seed_id": r["seed_id"], "op": r["op"], "variant": v, **res})
        n[0] += 1
        if n[0] % 250 == 0:
            print(f"[nim cycle {v}] {n[0]}/{len(todo)} in {(time.time() - t0) / 60:.1f} min", flush=True)
    await asyncio.gather(*(one(r) for r in todo))
    rr = list(Store(run_dir, f"cycle_{v}").rows.values())
    print(v, "cycle_ok", sum(bool(r.get("cycle_ok")) for r in rr), "/", len(rr), "done",
          sum(bool(r.get("done")) for r in rr), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        asyncio.run(validate(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 100))
    else:
        p = Path(sys.argv[2])
        asyncio.run(run(p if p.is_absolute() else ROOT / p, sys.argv[3], sys.argv[4]))
