"""Numbers for exec/D1_N_scale.md, computed from the frozen pool files + work stores + batch ledger.

    python -m vi_math_verified_aug.scale_gen.report gates            # per-gate yields / rejection rates per language
    python -m vi_math_verified_aug.scale_gen.report audit            # Opus audit vs the "+" cascade
    python -m vi_math_verified_aug.scale_gen.report costs            # $ per stage x language (batch ledger) + Gemini calls
    python -m vi_math_verified_aug.scale_gen.report examples --lang vi --n 20
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from .common import OUT, ROOT, WORK, Store, iter_jsonl, read_jsonl


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if n == 0:
        return "–"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * p:.1f}% [{100 * max(0, c - h):.1f}–{100 * min(1, c + h):.1f}] ({k}/{n})"


def pool(lang: str) -> list[dict]:
    return read_jsonl(OUT / lang / "pool.jsonl")


def gates(lang: str) -> dict:
    P = [r for r in pool(lang) if r.get("attempt_no", 1) == 1]      # R8 rates: first attempts only
    F = lambda r: r["flags"]  # noqa: E731
    gen = len(P)
    fmt = [r for r in P if F(r)["format_ok"]]
    tr = [r for r in fmt if F(r)["trace_ok"]]
    dchk = [r for r in tr if F(r)["decontam_pass"] is not None]
    mn = [r for r in P if r["minus_eligible"]]
    ag = [r for r in mn if F(r)["solve_agree"]]
    lj = [r for r in mn if F(r)["lite_ok"] is not None]
    lok = [r for r in lj if F(r)["lite_ok"]]
    hj = [r for r in mn if F(r)["hv3_pass"] is not None]
    hp = [r for r in hj if F(r)["hv3_pass"]]
    lite_flags = Counter(f for r in lj for f in ("absurd", "unansw", "multi", "leakj") if F(r)["lite"][f])
    lite_flags["natural<4"] = sum((F(r)["lite"]["natural"] or 0) < 4 for r in lj)
    hv_flags = Counter()
    for r in hj:
        h = F(r)["hv3"]
        hv_flags["absurd"] += bool(h["absurd"])
        hv_flags["unansw"] += bool(h["unansw"])
        hv_flags["not_teacher"] += not h["teacher"]
    fmt_flags = Counter(f for r in P if not F(r)["format_ok"] for f, v in (F(r)["format"] or {}).items() if v)
    dec_flags = Counter(f for r in dchk if not F(r)["decontam_pass"] for f in ("g4_ngram", "g4_cos", "g5_dup")
                        if (F(r)["decontam"] or {}).get(f))
    # the "+" cascade rejection rate among items that reached every gate (screened region)
    # the "+" cascade (agree ∧ hv3) rejection rate among N- items that reached every "+" gate
    screened = [r for r in mn if F(r)["hv3_pass"] is not None or F(r)["solve_agree"] is False]
    return {
        "generated": gen,
        "format_ok": wilson(len(fmt), gen), "format_reject_reasons": dict(fmt_flags),
        "trace_ok | format_ok": wilson(len(tr), len(fmt)),
        "decontam_pass | trace_ok": wilson(sum(F(r)["decontam_pass"] for r in dchk), len(dchk)),
        "decontam_reject_reasons": dict(dec_flags),
        "minus_eligible (N-)": len(mn), "seeds_minus": len({r["seed_id"] for r in mn}),
        "solve_agree | N-": wilson(len(ag), len(mn)),
        "lite_ok | N-, judged (info only; not applied)": wilson(len(lok), len(lj)),"lite_flags": {k: wilson(v, len(lj)) for k, v in lite_flags.items()},
        "lite_models": dict(Counter(F(r)["lite"]["model"] for r in lj)),
        "hv3_pass | agree, judged": wilson(len(hp), len(hj)),
        "hv3_flags": {k: wilson(v, len(hj)) for k, v in hv_flags.items()},
        "verified (N+)": sum(r["verified"] is True for r in P), "verified_unknown": sum(r["verified"] is None for r in P),
        "seeds_verified": len({r["seed_id"] for r in P if r["verified"] is True}),
        "plus_cascade_reject | screened N-": wilson(sum(r["verified"] is False for r in screened), len(screened)),
        "op_mix_verified": dict(Counter(r["op"] for r in P if r["verified"] is True)),
    }


def audit(lang: str, cfg: str = "A") -> dict:
    A = Store(WORK / lang / "audit.jsonl")
    P = {f"{lang}|{cfg}|{r['slot_id']}": r for r in pool(lang)}
    rows = []
    for k, a in A.rows.items():
        if not a.get("labels") or k not in P:
            continue
        lab, r = a["labels"], P[k]
        opus_ok = lab["valid"] and lab["teacher"] and not lab["leakj"] and not lab["tail"]
        rows.append((r["verified"] is True, opus_ok, lab, r))
    pl = [x for x in rows if x[0]]
    rj = [x for x in rows if not x[0]]
    return {"n": len(rows), "n_verified": len(pl), "n_rejected": len(rj),
            "opus_accept | verified (precision)": wilson(sum(x[1] for x in pl), len(pl)),
            "opus_absurd | verified": wilson(sum(x[2]["absurd"] for x in pl), len(pl)),
            "opus_teacher | verified": wilson(sum(x[2]["teacher"] for x in pl), len(pl)),
            "opus_valid | verified": wilson(sum(x[2]["valid"] for x in pl), len(pl)),
            "opus_leak | verified": wilson(sum(x[2]["leakj"] for x in pl), len(pl)),
            "opus_new | verified, M2-M6": wilson(sum(x[2]["new"] for x in pl if x[3]["op"] != "M1"),
                                                 sum(x[3]["op"] != "M1" for x in pl)),
            "opus_accept | rejected": wilson(sum(x[1] for x in rj), len(rj)),
            "opus_absurd | rejected": wilson(sum(x[2]["absurd"] for x in rj), len(rj))}


def costs() -> dict:
    rows: dict[str, dict] = {}
    for r in iter_jsonl(WORK / "batches.jsonl"):
        rows[r["batch_id"]] = {**rows.get(r["batch_id"], {}), **r}
    out: dict[str, dict] = defaultdict(lambda: {"batches": 0, "items": 0, "cost_usd": 0.0})
    for b in rows.values():
        k = f"{b['stage']}/{b['lang']}"
        out[k]["batches"] += 1
        out[k]["items"] += b.get("n_results") or 0
        out[k]["cost_usd"] = round(out[k]["cost_usd"] + (b.get("cost_usd") or 0.0), 4)
    direct = defaultdict(float)
    gem = Counter()
    for prov in ("claude", "gemini"):
        p = ROOT / "logs" / "api" / "D1" / f"{prov}.jsonl"
        if not p.exists():
            continue
        for r in iter_jsonl(p):
            if prov == "gemini":
                gem[(r.get("model_requested"), r.get("status"))] += 1
            elif not r.get("batch_id") and r.get("cost_usd"):
                direct[r.get("tag") or "untagged"] += r["cost_usd"]
    return {"batch": dict(out), "direct_claude_usd": {k: round(v, 4) for k, v in direct.items()},
            "total_usd": round(sum(v["cost_usd"] for v in out.values()) + sum(direct.values()), 4),
            "gemini_calls": {f"{m}|{s}": n for (m, s), n in gem.items()}}


def examples(lang: str, n: int = 20, seed: int = 20261108) -> list[dict]:
    P = [r for r in pool(lang) if r["minus_eligible"]]
    rng = random.Random(seed)
    return [{"id": r["id"], "op": r["op"], "verified": r["verified"], "question": r["question"], "answer": r["answer"],
             "generator_answer": r["generator_answer"], "agree": r["flags"]["solve_agree"],
             "lite": r["flags"]["lite"], "hv3": r["flags"]["hv3"]} for r in rng.sample(P, min(n, len(P)))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gates", "audit", "costs", "examples"])
    ap.add_argument("--lang", default=None)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args(argv)
    langs = [a.lang] if a.lang else [l for l in ("vi", "en") if (OUT / l / "pool.jsonl").exists()]
    if a.cmd == "gates":
        print(json.dumps({l: gates(l) for l in langs}, ensure_ascii=False, indent=1))
    elif a.cmd == "audit":
        print(json.dumps({l: audit(l) for l in langs}, ensure_ascii=False, indent=1))
    elif a.cmd == "costs":
        print(json.dumps(costs(), indent=1))
    elif a.cmd == "examples":
        print(json.dumps({l: examples(l, a.n) for l in langs}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
