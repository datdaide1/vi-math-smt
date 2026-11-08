"""R1 — results on the training tranche, using Y2's pipeline (D5, lead 2026-11-08).

    set OPENBLAS_NUM_THREADS=1
    .venv\\Scripts\\python -m vi_math_verified_aug.analysis.c3_r1_results --pred <files or globs> [--B 4000]
        -> data/audit/r1/{results.json, tables.md}; exec/R1_results.md is written from these tables by D5

Inputs: eval prediction rows in the Y2 schema (``schema.REQUIRED``) for
  vi_primary_1000 (datasets gsm8k_platinum / math_algprealg; K = 4 sampled + greedy),
  en_primary_1000 (same item ids, EN), and Vi-ExamMath VERIFIED (greedy only).
Arms: base, S0, Y+, Y-, N+, N-. Seeds: VI 0-3, EN 0-1.

What this adds on top of Y2 (``analysis.run.analyze``):
* Vi-ExamMath is greedy-only: Y2's estimators use sampled decodes, so greedy rows of greedy-only datasets are
  re-coded as a single sample (K = 1) for that dataset only (recorded in the report as ``greedy_as_k1``).
* Core filter: items of Vi-ExamMath are restricted to core_v1.1 ``status == VERIFIED`` and contamination flag False;
  ``certifiable`` = (yes) vs (maybe | no) is the pre-registered H4 binary (lead); yes vs no is a sensitivity re-run.
* Per-dataset contrasts (never pooled only): H2-, H3- and VxM-type contrasts and the simple effects (Y+ - Y-,
  N+ - N-, Y+ - N+, Y- - N-, each arm - S0, S0 - base) on every dataset separately (exploratory unless they are
  Y2's confirmatory family).
* Output tokens per arm x dataset (the item x run covariate; training-length covariates are aliased with arm).
* Power: minimum detectable effect MDE80 = 2.8 x bootstrap SE for each contrast, reported next to its estimate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import bootstrap, contrasts, multiplicity, schema
from .contrasts import EXPLORATORY, Contrast, Term

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "audit" / "r1"
CORE = ROOT / "data" / "vi_exam_qa" / "core_v1.1.jsonl"
SIMPLE = {"Y+ - Y-": {"Y+": 1, "Y-": -1}, "N+ - N-": {"N+": 1, "N-": -1}, "Y+ - N+": {"Y+": 1, "N+": -1},
          "Y- - N-": {"Y-": 1, "N-": -1}, "H2-type ½(Y±) - ½(N±)": {"Y+": .5, "Y-": .5, "N+": -.5, "N-": -.5},
          "H3-type ½(+) - ½(-)": {"Y+": .5, "N+": .5, "Y-": -.5, "N-": -.5},
          "VxM (Y+ - Y-) - (N+ - N-)": {"Y+": 1, "Y-": -1, "N+": -1, "N-": 1}}


def core_meta(binary: str = "yes_vs_rest") -> tuple[dict, set]:
    """item_id -> {'certifiable': bool} for VERIFIED, contamination-clean Core items; and the allowed id set.
    binary 'yes_vs_rest' (pre-registered: yes vs maybe|no) or 'yes_vs_no' (sensitivity: maybe items dropped)."""
    meta, allowed = {}, set()
    for line in open(CORE, encoding="utf-8"):
        r = json.loads(line)
        if r.get("status") != "VERIFIED" or (r.get("contamination") or {}).get("flag"):
            continue
        c = r.get("certifiable")
        if binary == "yes_vs_no" and c == "maybe":
            continue
        allowed.add(r["id"])
        meta[r["id"]] = {"certifiable": c == "yes"}
    return meta, allowed


def is_exam(ds: str) -> bool:
    d = ds.lower()
    return "exam" in d or "core" in d


EXPECTED_ROWS = {"primary": 5000, "exam": 1367}      # primary_1000: 1,000 items x (K=4 + greedy); exam: 1,367 greedy
KEEP = set(schema.REQUIRED) | {"decode", "evaltype", "certifiable", "truncated", "judged", "family", "eval_set"}


def load_rows(paths: list[str], binary: str = "yes_vs_rest"):
    """Streaming loader (memory ~ O(runs x items) downstream, never O(rows)): returns (row_iterator, item_meta,
    stats). Heavy fields (response, ...) are dropped; Vi-ExamMath rows outside VERIFIED & contamination-clean Core
    are dropped; greedy rows of greedy-only (run, dataset) pairs are recoded as one sample (K = 1)."""
    files = sorted({f for p in paths for f in (glob.glob(p) or [p])})
    # completeness: eval files are pulled while the eval runs; only complete files enter the analysis
    complete, partial = [], []
    for f in files:
        n = sum(1 for line in open(f, encoding="utf-8") if line.strip())
        need = EXPECTED_ROWS["exam" if "exam" in Path(f).name else "primary"] if Path(f).name.startswith("ev-") else 0
        (complete if n >= need else partial).append(f if n >= need else f"{f} ({n}/{need})")
    files = complete
    meta, allowed = core_meta(binary)
    stats = {"truncation": {}, "partial_files_skipped": partial, "files": files, "rows_in": 0, "exam_dropped_not_verified_or_contaminated": 0, "greedy_as_k1_rows": 0}
    has_sample, exam_sets = set(), set()
    for f in files:                                   # pass 1: tiny state only
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if not schema._is_greedy(r):
                        has_sample.add((r["run_id"], r["dataset"]))
                    if is_exam(r["dataset"]):
                        exam_sets.add(r["dataset"])
    item_meta = {f"{ds}/{i}": m for ds in exam_sets for i, m in meta.items()}

    def gen():
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    stats["rows_in"] += 1
                    if is_exam(r["dataset"]) and str(r["item_id"]) not in allowed:
                        stats["exam_dropped_not_verified_or_contaminated"] += 1
                        continue
                    r = {k: v for k, v in r.items() if k in KEEP}
                    if r["arm"] == "base":                  # two base rows: 4-shot and 0-shot (lead 08:55)
                        r["arm"] = "base-0shot" if "0shot" in r["run_id"] else "base-4shot"
                    tk = f"{r['arm']}|{r['lang']}|{r['dataset']}"
                    t = stats["truncation"].setdefault(tk, [0, 0])
                    t[0] += 1
                    t[1] += bool(r.get("truncated"))
                    if (r["run_id"], r["dataset"]) not in has_sample and schema._is_greedy(r):
                        r["sample_k"], r["decode"] = 0, "sample"
                        stats["greedy_as_k1_rows"] += 1
                    yield r
    return gen(), item_meta, stats


def per_dataset_contrasts(d: schema.EvalData) -> list[Contrast]:
    C = []
    runs = d.runs
    for ds in sorted(d.items["dataset"].unique()):
        for lang in sorted(runs["lang"].unique()):
            for model in sorted(runs.loc[runs["lang"] == lang, "model"].unique()):
                have = set(runs.loc[(runs["lang"] == lang) & (runs["model"] == model), "arm"])
                rf = {"lang": lang, "model": model}
                itf = {"dataset": ds}
                for name, coef in SIMPLE.items():
                    if all(a in have for a in coef):
                        C.append(Contrast(f"{ds}|{lang}|{model}|{name}",
                                          [Term.make(w, {**rf, "arm": a}, itf) for a, w in coef.items()],
                                          EXPLORATORY, f"{name} on {ds} ({lang}, {model})", "A"))
                for a in ("Y+", "Y-", "N+", "N-"):
                    if a in have and "S0" in have:
                        C.append(Contrast(f"{ds}|{lang}|{model}|{a} - S0",
                                          [Term.make(1, {**rf, "arm": a}, itf), Term.make(-1, {**rf, "arm": "S0"}, itf)],
                                          EXPLORATORY, f"{a} - S0 on {ds}", "A"))
                for b in ("base-4shot", "base-0shot"):
                    if "S0" in have and b in have:
                        C.append(Contrast(f"{ds}|{lang}|{model}|S0 - {b}",
                                          [Term.make(1, {**rf, "arm": "S0"}, itf), Term.make(-1, {**rf, "arm": b}, itf)],
                                          EXPLORATORY, f"S0 - {b} on {ds} (NOT a clean training contrast)", "A"))
    return C


def arm_table(d: schema.EvalData) -> pd.DataFrame:
    rt = schema.run_table(d)
    if rt.empty:
        return rt
    g = rt.groupby(["dataset", "lang", "model", "arm"])
    out = g.agg(seeds=("seed", "nunique"), n_items=("n_items", "max"), pass1=("pass1_mean", "mean"),
                pass1_sd=("pass1_mean", "std"), greedy=("greedy_acc", "mean"), greedy_sd=("greedy_acc", "std"),
                tokens=("tokens_mean", "mean")).reset_index()
    return out


def run(paths: list[str], B: int = 4000, binary: str = "yes_vs_rest", seed: int = 20261108,
        r_results: list[str] | None = None) -> dict:
    from .run import analyze
    rows, item_meta, stats = load_rows(paths, binary)
    d = schema.build(rows, item_meta=item_meta, strict=False)
    d = d.subset_items(~d.items["dataset"].map(is_exam).to_numpy() | d.items.get("certifiable", pd.Series(True, index=d.items.index)).notna().to_numpy())         if "certifiable" in d.items else d
    res = {"inputs": stats, "schema_report": {k: v for k, v in d.report.items() if not k.startswith("_")},
           "binary": binary}
    res["arm_table"] = arm_table(d).to_dict("records")
    C0, notes0 = contrasts.library(d, include_exploratory=False)
    if C0:
        g = None
        if r_results:
            from . import glmm
            g = glmm.read_r_results(r_results)
            res["glmm_source_files"] = r_results
        y2 = analyze(d, B=B, seed=seed, include_exploratory=False, glmm_results=g)
        res["confirmatory"] = [r for r in y2["contrasts"]]
        res["notes"] = y2["notes"]
        res["seed_variance"] = y2["seed_variance"]
    else:                                   # not enough arms yet (e.g. only base + S0): descriptive tables only
        res["confirmatory"], res["notes"], res["seed_variance"] = [], notes0, None
    C = per_dataset_contrasts(d)
    pp = bootstrap.cluster_bootstrap(d, C, B=B, seed=seed) if C else []
    rows_c = []
    for c, a in zip(C, pp):
        rows_c.append({"name": c.name, "est_pp": a.estimate, "se_pp": a.se, "ci_lo_pp": a.ci_lo, "ci_hi_pp": a.ci_hi,
                       "p_boot": a.p_wald, "mde80_pp": 2.8 * a.se if a.se == a.se else None})
    if rows_c:
        p = np.array([r["p_boot"] for r in rows_c], float)
        ok = np.isfinite(p)
        adj = np.full(p.shape, np.nan)
        adj[ok] = multiplicity.bh(p[ok])
        for r, q in zip(rows_c, adj):
            r["q_bh"] = float(q)
    res["per_dataset"] = rows_c
    for r in res["confirmatory"]:
        r["mde80_pp"] = 2.8 * r["se_pp"] if r.get("se_pp") == r.get("se_pp") else None
    return res


def tables(res: dict) -> str:
    L = ["# R1 tables (`python -m vi_math_verified_aug.analysis.c3_r1_results`)", "",
         f"Inputs: {len(res['inputs']['files'])} complete files (partial skipped: {len(res['inputs'].get('partial_files_skipped', []))}), {res['inputs']['rows_in']} rows; Vi-ExamMath items dropped "
         f"(not VERIFIED or contaminated): {res['inputs']['exam_dropped_not_verified_or_contaminated']}; greedy rows "
         f"recoded as K=1 (greedy-only datasets): {res['inputs']['greedy_as_k1_rows']}. H4 binary: {res['binary']}.", "",
         "## Per arm x dataset (mean over seeds; SD across seeds)", "",
         "| dataset | lang | model | arm | seeds | items | pass@1 (K) | SD | greedy | SD | out tokens |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.1f}"  # noqa: E731
    for r in res["arm_table"]:
        L.append(f"| {r['dataset']} | {r['lang']} | {r['model']} | {r['arm']} | {r['seeds']} | {r['n_items']} | "
                 f"{f(r['pass1'])} | {f(r['pass1_sd'])} | {f(r['greedy'])} | {f(r['greedy_sd'])} | "
                 f"{r['tokens']:.0f} |" if r.get("tokens") == r.get("tokens") else "")
    L += ["", "## Truncation rate (share of decodes hitting the 1,024-token cap) per arm x lang x dataset", "",
          "| arm | lang | dataset | rows | truncated |", "|---|---|---|---|---|"]
    for k, (n, t) in sorted(res["inputs"].get("truncation", {}).items()):
        a_, l_, d_ = k.split("|")
        L.append(f"| {a_} | {l_} | {d_} | {n} | {100 * t / max(1, n):.1f}% |")
    L += ["", "## Confirmatory family (Y2: Holm over H1-H4; two-way cluster bootstrap items x seeds; marginal-logit "
          "bootstrap stands in for the GLMM until lme4 runs)", "",
          "| contrast | est (pp) | 95% CI | p_adj (Holm, boot) | logit est (source) | p_adj (Holm, logit) | MDE80 (pp) | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    ms = {l: min((r["seeds"] for r in res["arm_table"] if r["arm"] in ("Y+", "Y-", "N+", "N-") and r["lang"] == l),
                 default=0) for l in ("vi", "en")}
    min_seeds = min(ms.values())
    need = lambda name: ms["en"] if name.startswith("H1") else ms["vi"]  # noqa: E731
    if min_seeds < 2:
        L.insert(len(L) - 2, f"**PROVISIONAL where marked: seeds per arm VI {ms['vi']}, EN {ms['en']}; with 1 seed the bootstrap cannot resample seeds, so CIs, "
                             "p-values, MDEs and TOST omit run-to-run variance and are too narrow. No verdict is "
                             "drawn until >= 2 seeds per arm.**\n")
    for r in res["confirmatory"]:
        v = r["verdict"]
        verdict = "significant" if v["significant"] else ("equivalent (TOST ±3 pp)" if v.get("tost", {}).get("equivalent")
                                                         else "n.s.")
        if need(r["name"]) < 2:
            verdict = f"(provisional: {need(r['name'])} seed) {verdict}"
        L.append(f"| {r['name']} | {100 * r['est_pp']:+.2f} | [{100 * r['ci_lo_pp']:+.2f}, {100 * r['ci_hi_pp']:+.2f}] | "
                 f"{r['p_adj_boot']:.4f} | {r['est_logit_glmm']:+.3f} ({r['glmm_source']}) | {r['p_adj_glmm']:.4f} | "
                 f"{100 * r['mde80_pp']:.1f} | {verdict} |")
    skipped = (res.get("notes") or {}).get("skipped") or {}
    if skipped:
        L += ["", "Skipped: " + "; ".join(f"{k}: {v}" for k, v in skipped.items())]
    L += ["", "## Per-dataset contrasts (exploratory; BH-FDR over this table)", "",
          "| contrast | est (pp) | 95% CI | p (boot) | q (BH) | MDE80 (pp) |", "|---|---|---|---|---|---|"]
    for r in res["per_dataset"]:
        if r["est_pp"] == r["est_pp"]:
            L.append(f"| {r['name']} | {100 * r['est_pp']:+.2f} | [{100 * r['ci_lo_pp']:+.2f}, {100 * r['ci_hi_pp']:+.2f}] | "
                     f"{r['p_boot']:.4f} | {r.get('q_bh', float('nan')):.4f} | {100 * (r['mde80_pp'] or 0):.1f} |")
    return "\n".join(L)


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
    ap = argparse.ArgumentParser(prog="c3_r1_results")
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--binary", default="yes_vs_rest", choices=("yes_vs_rest", "yes_vs_no"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--r-results", nargs="*", default=None, help="lme4 out_{A,B}_contrasts.csv from the Kaggle kernel")
    a = ap.parse_args(argv)
    res = run(a.pred, B=a.B, binary=a.binary, r_results=a.r_results)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if a.binary == "yes_vs_rest" else "_yes_vs_no"
    (out / f"results{suffix}.json").write_text(json.dumps(res, indent=1, default=_default), encoding="utf-8")
    md = tables(res)
    (out / f"tables{suffix}.md").write_text(md, encoding="utf-8")
    print(md[:4000])


if __name__ == "__main__":
    main()
