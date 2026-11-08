"""C3 attribution study: which factor causes which error class, against matched baselines.

    .venv\\Scripts\\python -m vi_math_verified_aug.analysis.c3_attrib      # -> data/audit/c3/{results.json,tables.md}

Factors (one at a time, everything else held fixed by the pilot designs):
  (i)   seed text        exposure = the untransformed seed has the same defect (Opus v2 on the 60 G3/G4 seeds);
                         RD exposed - unexposed, cluster permutation p, population attributable fraction (PAF)
  (ii)  operator         rate by operator vs M1 inside one renderer; Y - N on the SAME slot per operator (the N arm
                         gets the same operator request, so an operator effect shared by both arms is the
                         transformation itself, one only in Y is the symbolic implementation); G3 -> G4 operator
                         fixes on the same 180 slots with the same renderer (per-slot "defective item produced")
  (iii) renderer         E vs R1 vs R2 on the SAME mutated programs (paired); G1 editor models on the same edit lists
  (iv)  generator model  G1: 4 models x 30 paired seeds (native); G2: 4 prompts x 40 paired seeds (Haiku)
  approach residual      best Y renderer vs N on the same slots, same seeds/operators/generator family/judge
  judge                  the renderer and approach contrasts are re-run with Sonnet 5.5 + validity_judge.v3 labels
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import c3_data as D
from .c3_stats import (attributable_fraction, cluster_boot_rd, cluster_perm_test, cochran_q_perm, fmt_pct, fmt_rd,
                       holm, mcnemar, newcombe_paired, rate)

OUT = D.ROOT / "data" / "audit" / "c3"
OPS = ("M1", "M2", "M3", "M4", "M5", "M6")
CLS = D.CLASSES + ("not_teacher",)
NEW_OPS = ("M2", "M3", "M4", "M5", "M6")


def _y(items, c):
    return np.array([i.y[c] for i in items], int)


def _applicable(c: str, op: str) -> bool:
    """near-dup is the requested behaviour of M1 (value resample), so it is scored on M2-M6 only."""
    return not (c == "neardup" and op == "M1")


# --------------------------------------------------------------------------------------------- A. rates
def base_rates(sets: dict[str, list]) -> dict:
    out = {}
    for name, items in sets.items():
        out[name] = {c: rate([i.y[c] for i in items if _applicable(c, i.op)]) for c in CLS}
    return out


def seed_rates(seeds: dict) -> dict:
    keys = ("absurd", "tail", "leak", "unansw", "fluency", "not_teacher", "not_valid")
    return {c: rate([s["y"][c] for s in seeds.values()]) for c in keys}


# --------------------------------------------------------------------------------------------- B. seed (i)
SEED_EXPOSURE = {"absurd": "absurd", "unansw": "unansw", "fluency": "fluency", "not_teacher": "not_teacher",
                 "leak": "leak"}


def seed_factor(variants: dict[str, list], seeds: dict) -> dict:
    out = {}
    for vname, items in variants.items():
        its = [i for i in items if i.seed_id in seeds]
        res = {}
        for c, ec in SEED_EXPOSURE.items():
            y = _y(its, c)
            e = np.array([seeds[i.seed_id]["y"][ec] for i in its], int)
            cl = [i.seed_id for i in its]
            if e.sum() == 0 or e.sum() == e.size:
                res[c] = {"skip": "no variation in exposure"}
                continue
            pe, pu = rate(y[e == 1]), rate(y[e == 0])
            diff_boot = _rd_boot(y, e, cl)
            perm = cluster_perm_test(y, e, cl)
            paf = attributable_fraction(y, e, cl)
            res[c] = {"exposed": pe, "unexposed": pu, "rd": diff_boot, "p_perm": perm["p_perm"], "paf": paf,
                      "n_seeds_exposed": int(len({cl[k] for k in np.flatnonzero(e)}))}
        out[vname] = res
    return out


def _rd_boot(y, e, cl, B=4000, seed=20261108):
    from .c3_stats import cluster_boot_stat

    def rd(ix):
        yy, ee = y[ix], e[ix]
        if not (ee == 1).any() or not (ee == 0).any():
            return float("nan")
        return yy[ee == 1].mean() - yy[ee == 0].mean()
    r = cluster_boot_stat(rd, y.size, cl, B=B, seed=seed)
    return {"d": r["est"], "lo": r["lo"], "hi": r["hi"]}


# --------------------------------------------------------------------------------------------- C. operator (ii)
def by_op(items: list) -> dict:
    out = {}
    for op in OPS:
        its = [i for i in items if i.op == op]
        if its:
            out[op] = {c: rate(_y(its, c)) for c in CLS if _applicable(c, op)}
    return out


def op_vs_m1(items: list) -> dict:
    """RD(op) - RD(M1) inside one variant, cluster bootstrap over seeds (indicator regression form)."""
    out = {}
    for op in OPS[1:]:
        its = [i for i in items if i.op in ("M1", op)]
        if not any(i.op == op for i in its):
            continue
        res = {}
        for c in CLS:
            if c == "neardup":
                continue
            y = _y(its, c)
            e = np.array([int(i.op == op) for i in its])
            cl = [i.seed_id for i in its]
            res[c] = _rd_boot(y, e, cl)
        out[op] = res
    return out


def paired(a_items: list, b_items: list, classes=CLS, key=lambda i: i.slot) -> dict:
    """Paired contrast b - a on the units present in both sets."""
    A = {key(i): i for i in a_items}
    B = {key(i): i for i in b_items}
    ks = sorted(set(A) & set(B))
    out = {"n_pairs": len(ks)}
    for c in classes:
        kk = [k for k in ks if _applicable(c, A[k].op)]
        x = [A[k].y[c] for k in kk]
        y = [B[k].y[c] for k in kk]
        r = newcombe_paired(x, y)
        r["rate_a"], r["rate_b"] = (float(np.mean(x)) if kk else float("nan")), (float(np.mean(y)) if kk else float("nan"))
        r["boot"] = cluster_boot_rd(np.array(y) - np.array(x), [A[k].seed_id for k in kk]) if kk else None
        out[c] = r
    ps = [out[c]["p_mcnemar"] for c in classes if out[c].get("n")]
    adj = holm(ps) if ps else []
    j = 0
    for c in classes:
        if out[c].get("n"):
            out[c]["p_holm"] = float(adj[j])
            j += 1
    return out


def y_minus_n_by_op(y_items: list, n_items: list) -> dict:
    out = {}
    for op in OPS:
        a = [i for i in n_items if i.op == op]
        b = [i for i in y_items if i.op == op]
        if a and b:
            out[op] = paired(a, b, classes=("absurd", "unansw", "fluency", "neardup", "not_teacher"))
    return out


def operator_fix(g3: list, g4e: list, mut3: dict, mut4: dict) -> dict:
    """Same 180 slots, same renderer (editor.v3 / deterministic M1): G3 operators vs G4 fixed operators.

    Per slot: 1 if the variant produced a tag-passing item that the judge flags with the class, else 0
    (an operator that refuses to produce an item counts as 0: no defective item enters the pool). Yield is reported
    alongside, because refusing more slots trivially lowers the per-slot defect rate."""
    G3 = {i.slot: i for i in g3}
    E = {i.slot: i for i in g4e}
    slots = sorted(set(mut3) | set(mut4))
    res = {"n_slots": len(slots), "produced_g3": sum(s in G3 for s in slots), "produced_g4": sum(s in E for s in slots)}
    for scope, opset in (("all", OPS), ("changed ops M3/M4/M6", ("M3", "M4", "M6")), ("unchanged ops M1/M2/M5", ("M1", "M2", "M5"))):
        ss = [s for s in slots if (mut3.get(s) or mut4.get(s) or {}).get("op") in opset]
        r = {"n_slots": len(ss), "yield_g3": sum(s in G3 for s in ss) / len(ss), "yield_g4": sum(s in E for s in ss) / len(ss)}
        for c in CLS:
            x = [G3[s].y[c] if s in G3 else 0 for s in ss]
            y = [E[s].y[c] if s in E else 0 for s in ss]
            r[c] = newcombe_paired(x, y)
        # per produced item (unpaired sets)
        r["per_item_g3"] = {c: rate([G3[s].y[c] for s in ss if s in G3]) for c in CLS}
        r["per_item_g4"] = {c: rate([E[s].y[c] for s in ss if s in E]) for c in CLS}
        # same-program control: slots where G4 kept G3's program, labels must be identical (judge rows reused)
        same = [s for s in ss if (mut4.get(s) or {}).get("same_as_g3") and s in G3 and s in E]
        r["same_program_slots"] = len(same)
        r["same_program_label_agreement"] = float(np.mean([G3[s].y == E[s].y for s in same])) if same else None
        res[scope] = r
    return res


# --------------------------------------------------------------------------------------------- E. model / prompt (iv)
def related_k(items: list, levels: list[str], ref: str, classes=CLS) -> dict:
    by = defaultdict(dict)
    for i in items:
        by[i.slot][i.variant] = i
    blocks = [b for b in by.values() if all(l in b for l in levels)]
    out = {"n_blocks": len(blocks), "levels": levels, "rates": {}, "omnibus": {}, "vs_ref": {}}
    for l in levels:
        out["rates"][l] = {c: rate([b[l].y[c] for b in blocks if _applicable(c, b[l].op)]) for c in classes}
    for c in classes:
        bb = [b for b in blocks if _applicable(c, b[ref].op)]
        m = np.array([[b[l].y[c] for l in levels] for b in bb], int)
        out["omnibus"][c] = cochran_q_perm(m, B=5000) if m.size else None
        out["vs_ref"][c] = {}
        ps = []
        for l in levels:
            if l == ref:
                continue
            r = newcombe_paired(m[:, levels.index(ref)], m[:, levels.index(l)])
            out["vs_ref"][c][l] = r
            ps.append(r["p_mcnemar"])
        for l, p in zip([l for l in levels if l != ref], holm(ps) if ps else []):
            out["vs_ref"][c][l]["p_holm"] = float(p)
    return out


# --------------------------------------------------------------------------------------------- H. summary matrix
def attribution_matrix(R: dict) -> dict:
    """One row per error class: the effect each factor can be shown to have, holding the others fixed.

    seed      PAF in R1 (and N): share of item errors removable by a clean seed
    operator  largest op - M1 excess inside R1, and whether N shows the same op excess
    renderer  largest paired |RD| among E/R1/R2 (same programs)
    generator largest |RD| vs Haiku among the 4 G1 native generators (paired seeds)
    prompt    largest |RD| vs v1 among the 4 G2 native prompts (paired seeds)
    approach  paired RD (Y renderer - N) on the same slots, for each Y renderer
    """
    M = {}
    for c in CLS:
        row = {}
        sf = R["seed"]
        for v in ("E", "R1", "R2", "N"):
            s = sf.get(v, {}).get(c if c in SEED_EXPOSURE else "", None)
            if s and "paf" in s:
                row[f"seed_paf_{v}"] = s["paf"]
                row[f"seed_rd_{v}"] = s["rd"]
        for v in ("R1", "N"):
            ov = R["op_vs_m1"].get(v, {})
            best = None
            for op, d in ov.items():
                if c in d and np.isfinite(d[c]["d"]) and (best is None or d[c]["d"] > best[1]["d"]):
                    best = (op, d[c])
            if best:
                row[f"op_max_{v}"] = {"op": best[0], **best[1]}
        rr = None
        for pair in ("E_vs_R1", "E_vs_R2", "R1_vs_R2"):
            d = R["renderer"][pair][c]
            if rr is None or abs(d["d"]) > abs(rr[1]["d"]):
                rr = (pair, d)
        row["renderer_max"] = {"pair": rr[0], **{k: rr[1][k] for k in ("d", "lo", "hi", "p_mcnemar")}}
        for fam, key in (("generator", "g1_native"), ("prompt", "g2_prompt")):
            vs = R[key]["vs_ref"][c]
            best = max(vs.items(), key=lambda kv: abs(kv[1]["d"])) if vs else None
            if best:
                row[f"{fam}_max"] = {"level": best[0], **{k: best[1][k] for k in ("d", "lo", "hi", "p_mcnemar")}}
            row[f"{fam}_omnibus_p"] = (R[key]["omnibus"][c] or {}).get("p_perm")
        for v in ("E", "R1", "R2"):
            d = R["approach"][f"{v}_vs_N"][c]
            row[f"approach_{v}"] = {k: d[k] for k in ("d", "lo", "hi", "p_mcnemar")}
        M[c] = row
    return M


# --------------------------------------------------------------------------------------------- run
def run(base: Path = D.PILOT) -> dict:
    A = D.load_all(base)
    seeds = D.seed_labels(base)
    g4 = defaultdict(list)
    for i in A["G4"]:
        g4[i.variant].append(i)
    g4s = defaultdict(list)
    for i in A["G4s"]:
        g4s[i.variant].append(i)
    g3 = A["G3"]
    g3_no6 = [i for i in g3 if i.op != "M6"]
    g1n = [i for i in A["G1"] if i.source == "G1n"]
    g1e = [i for i in A["G1"] if i.source == "G1e"]
    R: dict = {}
    R["n"] = {k: len(v) for k, v in {**{f"G4_{k}": v for k, v in g4.items()}, "G3": g3, "G1n": g1n, "G1e": g1e,
                                       "G2": A["G2"], "seeds": list(seeds)}.items()}
    sets = {"seed texts (no transformation)": None, "G3 tag-passing (all ops)": g3, "G3 tag-passing (no M6)": g3_no6,
            **{f"G4 {v} tag-passing": g4[v] for v in ("E", "R1", "R2")}, "G4 N (same slots)": g4["N"],
            **{f"G1 native {m}": [i for i in g1n if i.variant == m] for m in D.G1_MODELS},
            **{f"G2 native {p}": [i for i in A["G2"] if i.variant == p] for p in ("v1", "v2", "v3", "v4")}}
    sets.pop("seed texts (no transformation)")
    R["rates"] = base_rates(sets)
    R["seed_rates"] = seed_rates(seeds)
    R["survivor_rates"] = base_rates({f"G4 {v} survivors": [i for i in g4[v] if i.survivor] for v in ("E", "R1", "R2")}
                                     | {"G3 survivors": [i for i in g3 if i.survivor]})
    # (i) seed
    R["seed"] = seed_factor({"G3": g3, "E": g4["E"], "R1": g4["R1"], "R2": g4["R2"], "N": g4["N"]}, seeds)
    # (ii) operator
    R["by_op"] = {v: by_op(its) for v, its in (("G3", g3), ("E", g4["E"]), ("R1", g4["R1"]), ("R2", g4["R2"]),
                                               ("N", g4["N"]))}
    R["op_vs_m1"] = {v: op_vs_m1(its) for v, its in (("G3", g3), ("E", g4["E"]), ("R1", g4["R1"]), ("R2", g4["R2"]),
                                                    ("N", g4["N"]))}
    R["y_minus_n_by_op"] = {v: y_minus_n_by_op(g4[v], g4["N"]) for v in ("E", "R1", "R2")}
    R["operator_fix"] = operator_fix(g3, g4["E"], D.g3_mutations(base), D.g4_mutations(base))
    # (iii) renderer
    R["renderer"] = {"E_vs_R1": paired(g4["E"], g4["R1"]), "E_vs_R2": paired(g4["E"], g4["R2"]),
                     "R1_vs_R2": paired(g4["R1"], g4["R2"])}
    R["renderer_sonnet"] = {"E_vs_R1": paired(g4s["E"], g4s["R1"]), "E_vs_R2": paired(g4s["E"], g4s["R2"]),
                            "R1_vs_R2": paired(g4s["R1"], g4s["R2"])}
    R["g1_editor"] = related_k(g1e, ["haiku", "sonnet", "gemini31"], "haiku")
    # (iv) generator / prompt
    R["g1_native"] = related_k(g1n, list(D.G1_MODELS), "haiku")
    R["g2_prompt"] = related_k(A["G2"], ["v1", "v2", "v3", "v4"], "v1")
    # approach residual
    R["approach"] = {f"{v}_vs_N": paired(g4["N"], g4[v]) for v in ("E", "R1", "R2")}
    R["approach_survivors"] = {f"{v}_vs_N": paired(g4["N"], [i for i in g4[v] if i.survivor]) for v in ("E", "R1", "R2")}
    R["approach_sonnet"] = {f"{v}_vs_N": paired(g4s["N"], g4s[v]) for v in ("E", "R1", "R2")}
    R["matrix"] = attribution_matrix(R)
    # absurd: cause coding by reading (c3_absurd)
    from .c3_absurd import summarize
    R["absurd_causes"] = summarize({"G3": g3, "E": g4["E"], "R1": g4["R1"], "R2": g4["R2"], "N": g4["N"]})
    # literature, operator-matched: GSM-Plus human pass rate per perturbation vs our well-posed ∧ op-applied rate
    R["lit_ops"] = lit_ops({"G3": g3, "E": g4["E"], "R1": g4["R1"], "R2": g4["R2"], "N": g4["N"]})
    R["cycle_faithfulness"] = cycle_faithfulness(g3, g4)
    # localization: absurd by localized flag inside R1/R2, and the R1b re-render (x10.000 dong, no unit relabel)
    R["localized_split"] = {v: {str(flag): {c: rate([i.y[c] for i in g4[v] if bool(i.extra.get("localized")) == flag])
                                            for c in ("absurd", "not_teacher", "fluency")} for flag in (True, False)}
                            for v in ("R1", "R2")}
    r1b = D.load_r1b(base)
    R["n"]["G4_R1b"] = len(r1b)
    if r1b:
        R["absurd_causes"]["R1b"] = summarize({"R1b": r1b})["R1b"]
        R["rates"]["G4 R1b tag-passing (D5-judged)"] = base_rates({"x": r1b})["x"]
        R["r1b"] = {"R1_vs_R1b": paired(g4["R1"], r1b), "E_vs_R1b": paired(g4["E"], r1b),
                    "R1b_minus_N": paired(g4["N"], r1b),
                    "R1_vs_R1b_localized_only": paired([i for i in g4["R1"] if i.extra.get("localized")], r1b)}
    return R


GSMPLUS_PASS = {"M1": ("Numerical Substitution", 91.51), "M2": ("Reversing Operation", 29.76),
                "M3": ("Adding Operation", 75.75)}


def lit_ops(variants: dict[str, list]) -> dict:
    out = {}
    for op, (name, lit) in GSMPLUS_PASS.items():
        row = {"gsmplus_type": name, "gsmplus_pass": lit}
        for v, items in variants.items():
            ys = [i.extra.get("wellposed") for i in items if i.op == op and i.extra.get("wellposed") is not None]
            if ys:
                r = rate(ys)
                r["lit_inside_ci"] = r["lo"] <= lit / 100 <= r["hi"]
                row[v] = r
        out[op] = row
    return out


def cycle_faithfulness(g3: list, g4: dict) -> dict:
    """Answer faithfulness (Opus blind solve == program answer) before/after the cycle check, vs MathCAMPS
    (12.2% unfaithful before, 97.7% faithful after; App. F)."""
    out = {}
    for v, items in (("G3", g3), ("E", g4["E"]), ("R1", g4["R1"]), ("R2", g4["R2"])):
        its = [i for i in items if i.solve_agree is not None]
        out[v] = {"all_tag_passing_unfaithful": rate([int(not i.solve_agree) for i in its]),
                  "cycle_passed_faithful": rate([int(i.solve_agree) for i in its if i.cycle_ok]),
                  "cycle_rejected_share": rate([int(not i.cycle_ok) for i in its if i.cycle_ok is not None])}
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def main(argv=None):
    from .c3_report import tables
    R = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(_jsonable(R), ensure_ascii=False, indent=1), encoding="utf-8")
    md = tables(R)
    (OUT / "tables.md").write_text(md, encoding="utf-8")
    print(f"wrote {OUT / 'results.json'} and tables.md ({len(md)} chars)")


if __name__ == "__main__":
    main(sys.argv[1:])
