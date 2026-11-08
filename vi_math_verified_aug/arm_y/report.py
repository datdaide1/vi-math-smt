"""G5 scale tables (streamed; laptop RAM rule) -> <run>/scale_tables.md + scale_metrics.json.

    python -m vi_math_verified_aug.arm_y.report data/train/Y/_work/vi_scale
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..pilot_gen.util import fmt_ci
from .common import ROOT

OPS = ["M1", "M2", "M3", "M4", "M5"]


def _rows(p: Path):
    if p.exists():
        for l in open(p, encoding="utf-8"):
            if l.strip():
                yield json.loads(l)


def _last(p: Path, keep=None) -> dict:
    d = {}
    for r in _rows(p):
        d[r["key"]] = {k: r.get(k) for k in keep} if keep else r
    return d


def main(run: Path):
    out, M = [], {}
    P = out.append
    slots = {s["key"]: s for s in _rows(run / "slots.jsonl")}
    mut = _last(run / "mutate.jsonl", ("ok", "op", "slot", "reason"))
    rend = {v: _last(run / f"render_{v}.jsonl", ("tag_ok", "op", "slot")) for v in ("R2b", "EN")}
    gates = _last(run / "gates.jsonl", ("leak_ok", "answer_untagged", "chrf_ok", "variant", "slot"))
    scr = _last(run / "screen_haiku.jsonl", ("parsed", "hv3_pass"))
    cyc = {v: _last(run / f"cycle_{v}.jsonl", ("done", "cycle_ok", "cycle_model")) for v in ("R2b", "EN")}
    glm = _last(run / "cycle_glm_R2b.jsonl", ("done", "cycle_ok", "cycle_model"))
    sol = {}
    for v in ("R2b", "EN"):
        for r in _rows(run / f"solve_{v}.jsonl"):
            sol[(v, r["key"])] = r.get("trace")
    from ..scale_gen.solver import agree
    from .export import ans_str
    tgt = {}
    for v in ("R2b", "EN"):
        for r in _rows(run / f"render_{v}.jsonl"):
            if r.get("tag_ok"):
                tgt[(v, r["key"])] = r["target"]

    def funnel(v, keys):
        c = Counter()
        for k in keys:
            c["slots"] += 1
            if not (mut.get(k) or {}).get("ok"):
                continue
            c["mutated"] += 1
            if not (rend[v].get(k) or {}).get("tag_ok"):
                continue
            c["tag ok"] += 1
            g = gates.get(f"{v}:{k}") or {}
            tr = sol.get((v, k))
            if not (g.get("leak_ok") and not g.get("answer_untagged") and g.get("chrf_ok")):
                continue
            c["gates"] += 1
            if not (tr and agree(tr, ans_str(tgt[(v, k)]), "en" if v == "EN" else "vi")):
                continue
            c["trace match"] += 1
            s = scr.get(f"{v}:{k}") or {}
            if not s.get("parsed"):
                c["screen pending"] += 1
                continue
            if not s.get("hv3_pass"):
                continue
            c["screen pass"] += 1
            cy = cyc[v].get(k) or {}
            if not cy.get("done"):
                c["cycle pending"] += 1
                continue
            if cy.get("cycle_ok"):
                c["Y+"] += 1
        return c

    cols = ["slots", "mutated", "tag ok", "gates", "trace match", "screen pass", "Y+", "screen pending", "cycle pending"]
    P("### VI wave 1 (J=1, 7,100 pool_v1 GSM8K seeds) funnel by operator\n")
    P("| op | " + " | ".join(cols) + " |")
    P("|---|" + "---|" * len(cols))
    w1 = [k for k, s in slots.items() if s["slot"] == 1]
    for op in OPS + ["all"]:
        ks = [k for k in w1 if op == "all" or slots[k]["op"] == op]
        c = funnel("R2b", ks)
        P(f"| {op} | " + " | ".join(f"{c[x]}" + (f" ({100 * c[x] / max(1, c['slots']):.0f}%)" if x != "slots" else "")
                                    for x in cols) + " |")
        M[f"vi_w1:{op}"] = dict(c)
    yp = {k.split(":")[0] for k in w1 if (cyc["R2b"].get(k) or {}).get("cycle_ok")
          and (scr.get(f"R2b:{k}") or {}).get("hv3_pass")}
    M["vi_w1_yplus_distinct_seeds"] = len(yp)
    P(f"\nDistinct seeds with a VI Y+ item (wave 1): **{len(yp)}**\n")
    en_keys = list(rend["EN"].keys())
    c = funnel("EN", en_keys)
    P("### EN (wave-1 slots of D5's part-1 and part-2 lists) funnel\n")
    P("| " + " | ".join(cols) + " |")
    P("|" + "---|" * len(cols))
    P("| " + " | ".join(str(c[x]) for x in cols) + " |")
    M["en"] = dict(c)
    both = [k for k in en_keys if k in yp or k.split(":")[0] in yp]
    ce = funnel("EN", [k for k in en_keys if k.split(":")[0] in yp])
    P(f"\nEN on seeds that have VI Y+: {ce['slots']} slots, EN Y+ {ce['Y+']}, cycle pending {ce['cycle pending']}\n")
    M["en_given_vi_yplus"] = dict(ce)
    # Qwen vs GLM robustness
    pairs = [(bool(cyc["R2b"][k]["cycle_ok"]), bool(glm[k]["cycle_ok"])) for k in cyc["R2b"]
             if cyc["R2b"][k].get("done") and k in glm and glm[k].get("done") and glm[k].get("cycle_model")]
    ag = sum(a == b for a, b in pairs)
    P(f"Qwen3-32B vs glm-5.3-flash cycle verdicts on the same VI items: agree {fmt_ci(ag, len(pairs))}\n")
    M["qwen_vs_glm"] = {"n": len(pairs), "agree": ag}
    # wave 2 state
    w2 = [k for k, s in slots.items() if s["slot"] == 2]
    c2 = Counter()
    for k in w2:
        c2["slots"] += 1
        m = mut.get(k)
        if m is None:
            c2["not mutated yet"] += 1
        elif m.get("ok"):
            c2["mutated"] += 1
            if (rend["R2b"].get(k) or {}).get("tag_ok"):
                c2["rendered (tag ok)"] += 1
    P(f"### VI wave 2 (stopped 05:13, lead): {dict(c2)}\n")
    M["vi_w2"] = dict(c2)
    (run / "scale_tables.md").write_text("\n".join(out), encoding="utf-8")
    json.dump(M, open(run / "scale_metrics.json", "w"), indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    p = Path(sys.argv[1])
    main(p if p.is_absolute() else ROOT / p)
