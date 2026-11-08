"""End-to-end analysis CLI.

    python -m vi_math_verified_aug.analysis.run analyze --pred preds.jsonl --out results/analysis [--B 10000]
           [--paired-seeds] [--r-results out_A_contrasts.csv out_B_contrasts.csv] [--vb-results vb.csv]
    python -m vi_math_verified_aug.analysis.run export  --pred preds.jsonl --out glmm_export
    python -m vi_math_verified_aug.analysis.run synth   --out demo.jsonl [--d_i2 0.3 --m_eff 0.2 --h4 0.3]

``analyze`` does no model fitting (memory-light): run table, the contrast library, the two-way cluster bootstrap on
the pp and marginal-logit scales, Holm / BH, TOST for non-significant confirmatory contrasts, the decision rule
(combining GLMM results if supplied -- lme4 from the R kernel, or the Python VB stand-in run off-laptop), and the
seed-variance estimate that feeds the pilot rule.  GLMM / VB fits and power simulations run on Modal or Kaggle
(``modal_jobs.py``), never on the laptop.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import bootstrap, contrasts, glmm, multiplicity, schema, variance


def analyze(d: schema.EvalData, *, B: int = 10_000, seed: int = 20261107, paired_seeds: bool = False,
            margin: float = 0.03, glmm_results: pd.DataFrame | None = None, include_exploratory: bool = True,
            chunk: int = 250) -> dict:
    C, notes = contrasts.library(d, include_exploratory=include_exploratory)
    out: dict = {"report": {k: v for k, v in d.report.items() if not k.startswith("_")}, "notes": notes}
    out["run_table"] = schema.run_table(d).to_dict("records")
    pp = bootstrap.cluster_bootstrap(d, C, B=B, seed=seed, paired_seeds=paired_seeds, chunk=chunk,
                                     keep_draws=True)
    lg = bootstrap.cluster_bootstrap(d, C, B=B, seed=seed, scale="logit", paired_seeds=paired_seeds, chunk=chunk)
    rows = []
    for c, a, b in zip(C, pp, lg):
        rows.append({"name": c.name, "family": c.family, "model": c.model, "description": c.description,
                     "est_pp": a.estimate, "se_pp": a.se, "ci_lo_pp": a.ci_lo, "ci_hi_pp": a.ci_hi,
                     "p_boot": a.p_wald, "p_perc": a.p_perc, "ci90_lo_pp": a.ci90_lo, "ci90_hi_pp": a.ci90_hi,
                     "est_mlogit": b.estimate, "se_mlogit": b.se, "p_mlogit": b.p_wald})
    res = pd.DataFrame(rows)
    res["p_adj_boot"] = np.nan
    for fam, meth in (("confirmatory", multiplicity.holm), ("exploratory", multiplicity.bh)):
        k = res["family"] == fam
        if k.any():
            res.loc[k, "p_adj_boot"] = meth(res.loc[k, "p_boot"].to_numpy())
    # GLMM side: lme4 (preferred) or VB; otherwise the marginal-logit bootstrap stands in for the logit test
    if glmm_results is not None and len(glmm_results):
        g = glmm_results.set_index("name")
        res["est_logit_glmm"] = res["name"].map(g["estimate"])
        p_col = "p" if "p" in g else None
        res["p_glmm"] = res["name"].map(g[p_col]) if p_col else np.nan
        res["glmm_source"] = res["name"].map(g["fit"]) if "fit" in g else "vb"
    else:
        res["est_logit_glmm"] = res["est_mlogit"]
        res["p_glmm"] = res["p_mlogit"]
        res["glmm_source"] = "bootstrap_marginal_logit"
    res["p_adj_glmm"] = np.nan
    for fam, meth in (("confirmatory", multiplicity.holm), ("exploratory", multiplicity.bh)):
        k = (res["family"] == fam) & res["p_glmm"].notna()
        if k.any():
            res.loc[k, "p_adj_glmm"] = meth(res.loc[k, "p_glmm"].to_numpy())
    verdicts = []
    for j, r in res.iterrows():
        dec = multiplicity.decide({"estimate": r["est_logit_glmm"]}, {"estimate": r["est_pp"]},
                                  r["p_adj_glmm"] if np.isfinite(r["p_adj_glmm"]) else None, r["p_adj_boot"])
        if r["family"] == "confirmatory" and not dec["significant"]:
            dec["tost"] = multiplicity.tost(r["est_pp"], draws=pp[j].draws, margin=margin)
        verdicts.append(dec)
    res["verdict"] = verdicts
    out["contrasts"] = res.to_dict("records")
    out["seed_variance"] = variance.seed_variance(d)
    out["item_components"] = variance.item_components(d)
    return out


def _md(out: dict) -> str:
    lines = ["# Analysis results", "", "| contrast | family | est (pp) | 95% CI (pp) | p_adj (boot) | "
             "logit est | p_adj (logit) | verdict |", "|---|---|---|---|---|---|---|---|"]
    for r in out["contrasts"]:
        v = r["verdict"]
        verdict = "significant" if v["significant"] else ("equivalent (TOST)" if v.get("tost", {}).get("equivalent")
                                                         else "n.s.")
        lines.append(f"| {r['name']} | {r['family']} | {100 * r['est_pp']:+.2f} | "
                     f"[{100 * r['ci_lo_pp']:+.2f}, {100 * r['ci_hi_pp']:+.2f}] | {r['p_adj_boot']:.4f} | "
                     f"{r['est_logit_glmm']:+.3f} ({r['glmm_source']}) | {r['p_adj_glmm']:.4f} | {verdict} |")
    lines += ["", "## Seed variance (pilot rule input)", "", "```", json.dumps(out["seed_variance"], indent=1),
              "```"]
    return "\n".join(lines)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="analysis.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--pred", nargs="+", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--B", type=int, default=10_000)
    a.add_argument("--paired-seeds", action="store_true")
    a.add_argument("--margin", type=float, default=0.03)
    a.add_argument("--r-results", nargs="*", default=None)
    a.add_argument("--vb-results", default=None)
    e = sub.add_parser("export")
    e.add_argument("--pred", nargs="+", required=True)
    e.add_argument("--out", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    for k in ("d_i2", "m_eff", "h4"):
        s.add_argument(f"--{k}", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "synth":
        from .synth import SynthSpec, generate, write_jsonl
        # demo-sized (laptop-safe): 1 model, Y+/Y-/N+/N- in both languages, 1 seed -> 8 runs x 44 items
        spec = SynthSpec(d_i2=args.d_i2, m_eff=args.m_eff, h4=args.h4, vi_models={"qwen3-0.6b": "0.6B"},
                         en_models=("qwen3-0.6b",), vi_arms=("Y+", "Y-", "N+", "N-"), n_core=24,
                         n_par={"gsm8k_platinum": 12, "math_algprealg": 8}, seeds_vi=1, seeds_en=1)
        n = write_jsonl(generate(spec, args.seed), args.out)
        print(f"wrote {n} rows to {args.out}")
        return
    d = schema.load_predictions(args.pred)
    os.makedirs(args.out, exist_ok=True)
    if args.cmd == "export":
        C, _ = contrasts.library(d)
        print(json.dumps(glmm.export_for_r(d, C, args.out), indent=1))
        return
    g = None
    if args.r_results:
        g = glmm.read_r_results(args.r_results)
    elif args.vb_results:
        g = pd.read_csv(args.vb_results)
    out = analyze(d, B=args.B, paired_seeds=args.paired_seeds, margin=args.margin, glmm_results=g)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=_default)
    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(_md(out))
    print(_md(out))


if __name__ == "__main__":
    main()
