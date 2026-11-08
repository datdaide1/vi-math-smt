"""Local side of the self-hosted (Modal vLLM, Qwen3-32B-AWQ) cycle check.

    python -m vi_math_verified_aug.arm_y.vllm_cycle prep <run_dir> <variant> <round>      # -> <run>/vllm_in_<round>.json
    modal run -m vi_math_verified_aug.arm_y.vllm_cycle_modal --inp ... --out <run>/vllm_out_<round>.json
    python -m vi_math_verified_aug.arm_y.vllm_cycle ingest <run_dir> <variant> <round>

Round 1 also carries the 100 G4 validation items (ids "val:<v>:<key>"), scored against G4's Gemini cycle verdicts
(lead acceptance: agreement >= 90%). Round 2 re-asks items whose round-1 IR failed to load/execute, with the
validator errors (never the answer). Verdicts go to cycle_<variant>.jsonl only for items not already decided by
another cross-family model; each row records ``cycle_model``.
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction
from pathlib import Path

from ..pilot_symbolic.pipeline import _feedback, _fx, load_ir
from ..symbolic import execute
from .common import ROOT, Store, render

MODEL = "Qwen/Qwen3-32B-AWQ"
G4 = ROOT / "data" / "pilot" / "symbolic_v2"


def _val_items():
    out = []
    for v in ("R2", "R1"):
        cyc = {json.loads(l)["key"]: json.loads(l) for l in open(G4 / f"cycle_{v}.jsonl", encoding="utf-8")}
        rnd = {json.loads(l)["key"]: json.loads(l) for l in open(G4 / f"render_{v}.jsonl", encoding="utf-8")}
        for k, c in sorted(cyc.items()):
            if c.get("done") and not c.get("api_fail") and rnd.get(k, {}).get("text"):
                out.append({"id": f"val:{v}:{k}", "text": rnd[k]["text"], "target": rnd[k]["target"],
                            "gemini_ok": bool(c["cycle_ok"]), "lang": "vi"})
    return out[:100]


def _evaluate(text, raw, target):
    P3, errs = load_ir("back:vllm", text, raw, None)
    ans = None
    if P3 is not None:
        try:
            ans = execute(P3)[P3.query]
        except Exception as e:                         # noqa: BLE001
            errs = errs + [f"exec: {e}"]
    return ans, errs


def prep(run: Path, v: str, rnd: int):
    st = Store(run, f"cycle_{v}")
    lang = "en" if v.startswith("EN") else "vi"
    pk = "back_en" if lang == "en" else "back"
    vg = Store(run, "screen_haiku").rows              # cycle LAST (lead 03:40): only items that passed the screen
    rows = [r for r in Store(run, f"render_{v}").rows.values() if r.get("tag_ok")
            and (vg.get(f"{v}:{r['key']}") or {}).get("hv3_pass")
            and not ((st.get(r["key"]) or {}).get("done") and (st.get(r["key"]) or {}).get("cycle_model") == MODEL)]
    items = []
    if rnd == 1 and os.environ.get("ARMY_QWEN_VAL", "0") == "1":
        for it in _val_items():
            rp = render("back", problem=it["text"])
            items.append({"id": it["id"], "system": rp.system, "user": rp.user})
        for r in rows:
            rp = render(pk, problem=r["text"])
            items.append({"id": f"{v}:{r['key']}", "system": rp.system, "user": rp.user})
    else:
        # every tag-passing item without a Qwen verdict: earlier-round failures get the validator feedback; items never
        # sent to Qwen (new waves, items first judged by another model) get a fresh first attempt
        prev = {}
        for f in sorted(run.glob("vllm*_out_*.json*")):        # every earlier Qwen output (single + chunk files)
            if f.name.endswith(".json") and ".part" not in f.name and not f.name.startswith("vllm_out_"):
                continue                                         # merged chunk file: its parts are read instead
            try:
                prev.update({o["id"]: o for o in json.load(open(f, encoding="utf-8"))["results"]})
            except (json.JSONDecodeError, KeyError):
                continue
        for r in rows:
            oid = f"{v}:{r['key']}"
            rp = render(pk, problem=r["text"])
            o = prev.get(oid)
            if o is None:
                items.append({"id": oid, "system": rp.system, "user": rp.user})
                continue
            ans, errs = _evaluate(r["text"], o["text"], r["target"])
            if ans is None:
                fb = _feedback(o["text"] or "", [e for e in errs if not e.startswith("[gold]")] or ["could not execute"])
                items.append({"id": oid, "system": rp.system, "user": rp.user + "\n\n" + fb})
    json.dump(items, open(run / f"vllm_in_{rnd}.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("prepared", len(items))


def ingest(run: Path, v: str, rnd: int):
    out = json.load(open(run / f"vllm_out_{rnd}.json", encoding="utf-8"))
    res = {o["id"]: o for o in out["results"]}
    if rnd == 1:
        val = _val_items()
        dec = [(it, _evaluate(it["text"], res[it["id"]]["text"], it["target"])[0]) for it in val if it["id"] in res]
        agree = sum((a is not None and a == Fraction(it["target"])) == it["gemini_ok"] for it, a in dec)
        rep = {"model": MODEL, "gpu": out["gpu"], "n": len(dec), "agree_with_gemini": agree,
               "agree_rate": round(agree / max(1, len(dec)), 3), "load_s": out["load_s"], "gen_s": out["gen_s"],
               "n_items_call": len(res), "items_per_min": round(len(res) / max(1, out["gen_s"]) * 60, 1)}
        print(json.dumps(rep))
        with open(ROOT / "data/pilot/symbolic_v3/nim_validation.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rep) + "\n")
        if rep["agree_rate"] < 0.90:
            print("validation FAILED: verdicts not ingested")
            return
    st = Store(run, f"cycle_{v}")
    rows = {r["key"]: r for r in Store(run, f"render_{v}").rows.values() if r.get("tag_ok")}
    n = 0
    for oid, o in res.items():
        if not oid.startswith(f"{v}:"):
            continue
        k = oid[len(v) + 1:]
        r = rows.get(k)
        if r is None or ((st.get(k) or {}).get("done") and (st.get(k) or {}).get("cycle_model") == MODEL):
            continue                                       # ONE cycle model for the whole arm (lead 02:45)
        ans, errs = _evaluate(r["text"], o["text"], r["target"])
        if ans is None and rnd == 1:
            continue                                       # round 2 retries it
        st.put(k, {"seed_id": r["seed_id"], "op": r["op"], "variant": v, "done": True, "executed": ans is not None,
                   "back_answer": _fx(ans), "target": r["target"],
                   "cycle_ok": ans is not None and ans == Fraction(r["target"]), "n_attempts": rnd,
                   "cycle_model": MODEL, "backend": "modal_vllm", "finish": o.get("finish")})
        n += 1
    print("ingested", n)


def ingest_parts(run: Path, v: str, stem: str = "vllm2_out_1.json"):
    """Ingest every finished chunk file <stem>.partN.json (chunked Modal job) as primary Qwen verdicts; items whose
    IR did not load/execute are left for the feedback round (prep round 2 picks them up)."""
    st = Store(run, f"cycle_{v}")
    rows = {}
    for l in open(run / f"render_{v}.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("tag_ok"):
                rows[r["key"]] = {k: r[k] for k in ("key", "seed_id", "op", "text", "target")}
    n = n_ok = 0
    for p in sorted(run.glob(f"{stem}.part*.json")):
        for o in json.load(open(p, encoding="utf-8"))["results"]:
            if not o["id"].startswith(f"{v}:"):
                continue
            k = o["id"][len(v) + 1:]
            r = rows.get(k)
            if r is None or ((st.get(k) or {}).get("cycle_model") == MODEL and (st.get(k) or {}).get("done")):
                continue
            ans, errs = _evaluate(r["text"], o["text"], r["target"])
            if ans is None:
                continue
            ok = ans == Fraction(r["target"])
            st.put(k, {"seed_id": r["seed_id"], "op": r["op"], "variant": v, "done": True, "executed": True,
                       "back_answer": _fx(ans), "target": r["target"], "cycle_ok": ok, "n_attempts": 1,
                       "cycle_model": MODEL, "backend": "modal_vllm", "finish": o.get("finish"), "src": p.name})
            n += 1
            n_ok += ok
    print(f"ingested {n} new Qwen verdicts ({n_ok} pass); primary store rows {len(st.rows)}")


def prep_multi(run: Path, variants: list[str], rnd: int):
    """One Qwen job for several variants (e.g. VI retries + EN screen passes): one model load, no idle time."""
    allit = []
    for v in variants:
        prep(run, v, rnd)
        allit += json.load(open(run / f"vllm_in_{rnd}.json", encoding="utf-8"))
    json.dump(allit, open(run / f"vllm_in_{rnd}.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("prepared (all variants)", len(allit))


if __name__ == "__main__":
    run = Path(sys.argv[2])
    run = run if run.is_absolute() else ROOT / run
    if sys.argv[1] == "prep_multi":
        prep_multi(run, sys.argv[3].split(","), int(sys.argv[4]))
    elif sys.argv[1] == "ingest_parts":
        ingest_parts(run, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "vllm2_out_1.json")
    else:
        {"prep": prep, "ingest": ingest}[sys.argv[1]](run, sys.argv[3], int(sys.argv[4]))
