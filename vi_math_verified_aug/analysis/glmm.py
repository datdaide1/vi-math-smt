"""Primary binomial GLMM (T5 §2.2, E1 §5.5): cell-means design, R export, and the Python VB stand-in.

Fixed effects use the **cell-means** parametrisation ``0 + cell`` with cell = lang x arm x model x slice, which is
the same column space as T5's full factorial ``arm * dataset * model`` (and E1's ``lang * method * verif * size``
plus a dataset term).  Every hypothesis is then a weight vector over cell coefficients, exported once and used
identically by lme4 (``glmm_lme4.R``), by the Python VB fit below and -- on the probability scale -- by the
cluster bootstrap.  ``slice`` = dataset, with Core split into ``:cert`` / ``:noncert`` when items carry the
P3 ``certifiable`` flag.

Two models:

* **A (VI grid)**: VI runs, all arms, all VI items.
  RE (T5): (1|item) + (1|item:arm) + (1|item:model) + (1|run) + (1|run:evaltype)
* **B (P2 mirror)**: Y±/N± runs of the mirror models in both languages, parallel items only.
  RE (E1): (1|item) + (1|item:lang) + (1|item:arm) + (1|item:arm:lang) + (1|run) + (1|seed_key),
  seed_key = model:arm:seed (the CRN pairing term)

Pre-registered convergence fallback (T5 §2.2, extended to B in the same spirit):
  A: full -> drop (1|item:model) -> drop (1|run:evaltype) [report its variance from per-model fits] -> Python VB
  B: full -> drop (1|item:arm:lang) -> drop (1|seed_key) -> Python VB
Each lme4 fit is nAGQ = 0, then refit at nAGQ = 1 (Laplace) from the nAGQ = 0 estimates with bobyqa.

MEASURED (Y2 report §6): on a 66-run x 220-item design the VB fit did not finish within Modal's 20-min limit, and
a local attempt reached 1.5 GB.  The Python GLMM of record is therefore the NumPyro NUTS fit on GPU
(``bayes_glmm.py``, ~30 s per model there); VB is kept only for completeness and must not run on the laptop.

The Python VB stand-in (statsmodels ``BinomialBayesMixedGLM.fit_vb``, mean-field Gaussian) needs Bernoulli rows
and a dense fixed-effect matrix, so it is fitted **per model** (A) or **per size** (B) and contrasts are summed
over independent fits.  Mean-field VB understates fixed-effect uncertainty; its logit estimates are used for the
sign-agreement rule and its variance components for the pilot rule -- inference comes from the cluster bootstrap.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contrasts import Contrast, mirror_models
from .schema import EvalData, filter_mask

RE_ORDER = {
    "A": [("full", ["item", "item:arm", "item:model", "run", "run:evaltype"]),
          ("drop_item_model", ["item", "item:arm", "run", "run:evaltype"]),
          ("drop_run_evaltype", ["item", "item:arm", "run"])],
    "B": [("full", ["item", "item:lang", "item:arm", "item:arm:lang", "run", "seed_key"]),
          ("drop_item_arm_lang", ["item", "item:lang", "item:arm", "run", "seed_key"]),
          ("drop_seed_key", ["item", "item:lang", "item:arm", "run"])],
}
R_SCRIPT = os.path.join(os.path.dirname(__file__), "glmm_lme4.R")


def item_slices(d: EvalData) -> np.ndarray:
    sl = d.items["dataset"].astype(str).to_numpy().copy()
    if "certifiable" in d.items:
        c = d.items["certifiable"].to_numpy()
        nat = d.items["evaltype"].to_numpy() == "native"
        for j in np.flatnonzero(nat):
            v = c[j]
            if v is True or v is False or isinstance(v, (bool, np.bool_)):
                sl[j] = f"{sl[j]}:{'cert' if bool(v) else 'noncert'}"
    return sl


def model_subset(d: EvalData, which: str) -> EvalData:
    if which == "A":
        return d.subset_runs(d.runs["lang"].to_numpy() == "vi")
    mm = mirror_models(d)
    rm = d.runs["arm"].isin(["Y+", "Y-", "N+", "N-"]).to_numpy() & d.runs["model"].isin(mm).to_numpy()
    dd = d.subset_runs(rm)
    return dd.subset_items(dd.items["evaltype"].to_numpy() == "parallel")


def cell_labels(d: EvalData) -> tuple[np.ndarray, list[str]]:
    """(R, I) array of cell labels for observed run-item pairs, and the sorted level list."""
    sl = item_slices(d)
    rk = (d.runs["lang"] + "|" + d.runs["arm"] + "|" + d.runs["model"]).to_numpy()
    lab = np.char.add(np.char.add(rk.astype(str)[:, None], "|"), sl.astype(str)[None, :])
    lab = np.where(d.n > 0, lab, "")
    levels = sorted(set(lab[lab != ""].tolist()))
    return lab, levels


def cell_weights(d: EvalData, c: Contrast, levels: list[str]) -> np.ndarray:
    """Contrast weights over cell-mean coefficients (item-count weighted within a term's item filter)."""
    sl = item_slices(d)
    w = np.zeros(len(levels))
    pos = {l: j for j, l in enumerate(levels)}
    for t in c.terms:
        rm = filter_mask(d.runs, t.run_filter)
        im = filter_mask(d.items, t.item_filter)
        combos = sorted(set((d.runs["lang"] + "|" + d.runs["arm"] + "|" + d.runs["model"]).to_numpy()[rm]))
        if not combos or not im.any():
            raise ValueError(f"{c.name}: empty term {t}")
        s_vals, s_cnt = np.unique(sl[im], return_counts=True)
        for combo in combos:
            for s, k in zip(s_vals, s_cnt):
                key = f"{combo}|{s}"
                if key not in pos:
                    raise ValueError(f"{c.name}: cell {key} not in data")
                w[pos[key]] += t.weight * (k / s_cnt.sum()) / len(combos)
    return w


def long_table(d: EvalData) -> pd.DataFrame:
    lab, _ = cell_labels(d)
    r_idx, i_idx = np.nonzero(d.n > 0)
    runs = d.runs.iloc[r_idx].reset_index(drop=True)
    items = d.items.iloc[i_idx].reset_index(drop=True)
    return pd.DataFrame({
        "run": runs["run_id"], "lang": runs["lang"], "arm": runs["arm"], "model": runs["model"],
        "size": runs["size"], "seed": runs["seed"],
        "seed_key": runs["model"] + ":" + runs["arm"] + ":" + runs["seed"].astype(str),
        "item": items["item_key"], "dataset": items["dataset"], "evaltype": items["evaltype"],
        "cell": lab[r_idx, i_idx], "correct": d.y[r_idx, i_idx].astype(int), "K": d.n[r_idx, i_idx].astype(int)})


def export_for_r(d: EvalData, contrasts: list[Contrast], outdir: str) -> dict:
    """Write glmm_data_{A,B}.csv, contrasts_{A,B}.csv (cell x contrast weights), contrast_meta.csv."""
    os.makedirs(outdir, exist_ok=True)
    written = {}
    meta = pd.DataFrame([{"name": c.name, "family": c.family, "model": c.model, "description": c.description}
                         for c in contrasts])
    meta.to_csv(os.path.join(outdir, "contrast_meta.csv"), index=False)
    for which in ("A", "B"):
        cs = [c for c in contrasts if c.model == which]
        if not cs:
            continue
        dd = model_subset(d, which)
        if len(dd.runs) == 0:
            continue
        _, levels = cell_labels(dd)
        W = pd.DataFrame({"cell": levels})
        for c in cs:
            W[c.name] = cell_weights(dd, c, levels)
        lt = long_table(dd)
        p_data, p_w = os.path.join(outdir, f"glmm_data_{which}.csv"), os.path.join(outdir, f"contrasts_{which}.csv")
        lt.to_csv(p_data, index=False)
        W.to_csv(p_w, index=False)
        written[which] = {"data": p_data, "weights": p_w, "rows": len(lt), "cells": len(levels)}
    with open(os.path.join(outdir, "README_R.txt"), "w", encoding="utf-8") as fh:
        fh.write("Run on a Kaggle/Colab R kernel (CPU):\n"
                 "  Rscript glmm_lme4.R glmm_data_A.csv contrasts_A.csv A out_A\n"
                 "  Rscript glmm_lme4.R glmm_data_B.csv contrasts_B.csv B out_B\n"
                 "Then: python -m vi_math_verified_aug.analysis.run combine --r-results out_A_contrasts.csv "
                 "out_B_contrasts.csv ...\n")
    return written


# ---------------------------------------------------------------------------------------------- Python VB
@dataclass
class VBResult:
    contrasts: pd.DataFrame       # name, estimate (logit), sd, z, p, est_pp, sd_pp
    variance_components: dict     # name -> posterior-mean SD (logit), averaged over sub-fits
    fits: int


def _vb_fit_one(dd: EvalData, which: str, structure: list[str], max_rows: int = 30_000):
    # 30k Bernoulli rows ~ 250 MB peak in statsmodels' VB (measured: ~38k rows -> 645 MB on the laptop).
    # Full-size fits belong on a Kaggle CPU kernel; raise max_rows there.
    import scipy.sparse as sp
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    lt = long_table(dd)
    if len(lt) * int(lt["K"].max()) > max_rows:
        raise MemoryError(f"{len(lt) * int(lt['K'].max())} Bernoulli rows > max_rows={max_rows}")
    # expand binomial (correct of K) to Bernoulli rows without materialising per-sample python objects
    reps = lt["K"].to_numpy()
    idx = np.repeat(np.arange(len(lt)), reps)
    k_in = np.concatenate([np.arange(k) for k in reps])
    yb = (k_in < np.repeat(lt["correct"].to_numpy(), reps)).astype(float)
    levels = sorted(lt["cell"].unique())
    cell_codes = pd.Categorical(lt["cell"], categories=levels).codes[idx]
    X = np.zeros((len(idx), len(levels)))
    X[np.arange(len(idx)), cell_codes] = 1.0
    cols, ident, names = [], [], []
    for vi, term in enumerate(structure):
        parts = term.split(":")
        key = lt[parts[0]].astype(str)
        for p in parts[1:]:
            key = key + "|" + lt[p].astype(str)
        codes, uniq = pd.factorize(key)
        m = sp.csr_matrix((np.ones(len(idx)), (np.arange(len(idx)), codes[idx])), shape=(len(idx), len(uniq)))
        cols.append(m)
        ident += [vi] * len(uniq)
        names.append(term)
    Z = sp.hstack(cols).tocsr()
    mod = BinomialBayesMixedGLM(yb, X, Z, np.asarray(ident), vcp_p=1.0, fe_p=3.0,
                                fep_names=levels, vcp_names=names)
    res = mod.fit_vb()
    return levels, np.asarray(res.fe_mean), np.asarray(res.fe_sd), dict(zip(names, np.exp(res.vcp_mean)))


def fit_vb(d: EvalData, contrasts: list[Contrast], which: str = "A", structure: list[str] | None = None,
           split_by: str | None = None) -> VBResult:
    """Fit the Python stand-in GLMM and evaluate the model's contrasts on the logit and (delta) pp scales."""
    from .stats_util import two_sided_p
    dd = model_subset(d, which)
    structure = structure or RE_ORDER[which][0][1]
    split_by = split_by or "model"
    cs = [c for c in contrasts if c.model == which]
    groups = sorted(dd.runs[split_by].unique())
    est = {c.name: [0.0, 0.0, 0.0, 0.0] for c in cs}           # logit est, var, pp est, pp var
    vcs: dict[str, list] = {}
    nfit = 0
    for gval in groups:
        sub = dd.subset_runs(dd.runs[split_by].to_numpy() == gval)
        st = [t for t in structure if not (t.endswith("model") and sub.runs["model"].nunique() == 1)]
        levels, mu, sd, vc = _vb_fit_one(sub, which, st)
        nfit += 1
        for k, v in vc.items():
            vcs.setdefault(k, []).append(float(v))
        _, lv_all = cell_labels(dd)
        pos = {l: j for j, l in enumerate(levels)}
        for c in cs:
            w_full = cell_weights(dd, c, lv_all)
            w = np.zeros(len(levels))
            for l, wv in zip(lv_all, w_full):
                if wv != 0 and l in pos:
                    w[pos[l]] = wv
            p = 1 / (1 + np.exp(-mu))
            g = w * p * (1 - p)
            est[c.name][0] += float(w @ mu)
            est[c.name][1] += float((w ** 2) @ (sd ** 2))
            est[c.name][2] += float(w @ p)
            est[c.name][3] += float((g ** 2) @ (sd ** 2))
    rows = []
    for c in cs:
        e, v, ep, vp = est[c.name]
        z = e / np.sqrt(v) if v > 0 else np.nan
        rows.append({"name": c.name, "family": c.family, "estimate": e, "sd": float(np.sqrt(v)), "z": z,
                     "p": float(two_sided_p(z)), "est_pp": ep, "sd_pp": float(np.sqrt(vp))})
    return VBResult(pd.DataFrame(rows), {k: float(np.mean(v)) for k, v in vcs.items()}, nfit)


def read_r_results(paths: list[str]) -> pd.DataFrame:
    """Read the ``*_contrasts.csv`` written by glmm_lme4.R (name, estimate, se, z, p, est_pp, se_pp, fit)."""
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def r_status(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
