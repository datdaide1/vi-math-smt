"""Seed-variance estimator for the pilot rule (T5 §2.4 "Pilot rule", E1 §5.4).

Method of moments on the logit scale, numpy only, no GLMM needed.  Within each cell c = (lang, arm, model) with
S >= 2 training seeds and for each eval type e (native / parallel):

    a_{r,e}  = run r's mean pass@1 on the eval-type-e items (all runs of a cell see the same items)
    D_{c,e}  = mean_i p_i(1 − p_i)  (unbiased per-item estimate pooled over the cell's runs and samples)
    z_{r,e}  = (a_{r,e} − mean_c a_e) / D_{c,e}        ≈ v_r + w_{r,e} + noise   (first-order delta method)

    Var(z_e)       = σ²_run + σ²_run×eval + β_e        β_e = Bernoulli noise / D²  (known, subtracted)
    Cov(z_N, z_P)  = σ²_run

Within-cell (co)variances are pooled across cells with weights S_c − 1.  σ_run(mirror) = √(σ²_run + σ²_re) is
estimable from either eval type alone (EN runs only have the parallel type).  The delta-method conversion is
E1's ``sigma_run_pp = sigma_run x E[p(1−p)]``.  Item-side components (σ_item, τ) get crude moment estimates
(only needed to re-run the power simulation; T5 Result 4 shows they move power by <= 0.2).
"""
from __future__ import annotations

import math

import numpy as np

from .schema import EvalData
from .stats_util import norm_ppf


def _wh_chi2_q(df: float, q: float) -> float:
    """Wilson–Hilferty chi-square quantile."""
    z = norm_ppf(q)
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def _cell_stats(d: EvalData, rows: np.ndarray, cols: np.ndarray):
    n = d.n[np.ix_(rows, cols)].astype(float)
    y = d.y[np.ix_(rows, cols)].astype(float)
    keep = (n > 0).all(0)
    n, y = n[:, keep], y[:, keep]
    if n.shape[1] == 0:
        return None
    a = (y / n).mean(1)                                       # per-run accuracy
    N, Y = n.sum(0), y.sum(0)
    pq = np.where(N > 1, Y * (N - Y) / (N * np.maximum(N - 1, 1)), 0.0)   # unbiased p(1-p) per item
    D = float(pq.mean())
    I = n.shape[1]
    K = n.mean(0)
    noise = float((pq / K).sum() / I ** 2)                    # Var of a_r from Bernoulli sampling
    return a, D, noise, I


def seed_variance(d: EvalData, cell_cols=("lang", "arm", "model"), min_seeds: int = 2) -> dict:
    keys = d.runs[list(cell_cols)].astype(str).agg("|".join, axis=1).to_numpy()
    et = d.items["evaltype"].to_numpy()
    types = [t for t in ("native", "parallel") if (et == t).any()]
    sums = {"var": {t: 0.0 for t in types}, "df": {t: 0 for t in types}, "cov": 0.0, "cov_df": 0,
            "diff": 0.0, "diff_df": 0}
    D_all, cells = [], []
    for k in np.unique(keys):
        rows = np.flatnonzero(keys == k)
        if len(rows) < min_seeds:
            continue
        z = {}
        for t in types:
            st = _cell_stats(d, rows, et == t)
            if st is None:
                continue
            a, D, noise, I = st
            if D <= 0:
                continue
            zz = (a - a.mean()) / D
            beta = noise / D ** 2
            S = len(rows)
            sums["var"][t] += zz.var(ddof=1) * (S - 1) - beta * (S - 1)
            sums["df"][t] += S - 1
            z[t] = (zz, beta)
            D_all.append(D)
        if len(z) == 2:
            (zN, bN), (zP, bP) = z["native"], z["parallel"]
            S = len(rows)
            sums["cov"] += float(np.cov(zN, zP, ddof=1)[0, 1]) * (S - 1)
            sums["cov_df"] += S - 1
            dd = zN - zP
            sums["diff"] += (dd.var(ddof=1) - bN - bP) * (S - 1)
            sums["diff_df"] += S - 1
        cells.append({"cell": k, "S": len(rows), "types": sorted(z)})
    out = {"cells_used": len(cells), "mean_D": float(np.mean(D_all)) if D_all else float("nan")}
    tot_df = sum(sums["df"].values())
    if tot_df:
        s2_tot = sum(sums["var"].values()) / tot_df              # σ²_run + σ²_re (noise removed)
        out["sigma_run_mirror"] = math.sqrt(max(s2_tot, 0.0))
        out["sigma_run_mirror_raw_var"] = s2_tot
        out["df_total"] = tot_df
        lo = s2_tot * tot_df / _wh_chi2_q(tot_df, 0.975)
        hi = s2_tot * tot_df / _wh_chi2_q(tot_df, 0.025)
        out["sigma_run_mirror_ci95"] = (math.sqrt(max(lo, 0)), math.sqrt(max(hi, 0)))
    if sums["diff_df"]:
        s2_re = sums["diff"] / sums["diff_df"] / 2
        out["sigma_run_x_eval"] = math.sqrt(max(s2_re, 0.0))
        out["sigma_run_x_eval_raw_var"] = s2_re
        df = sums["diff_df"]
        out["sigma_run_x_eval_ci95"] = (math.sqrt(max(s2_re * df / _wh_chi2_q(df, 0.975), 0)),
                                        math.sqrt(max(s2_re * df / _wh_chi2_q(df, 0.025), 0)))
        s2_run = sums["cov"] / sums["cov_df"]
        out["sigma_run"] = math.sqrt(max(s2_run, 0.0))
        out["sigma_run_raw_var"] = s2_run
        out["df_paired"] = df
    return out


def item_components(d: EvalData, lang: str = "vi", arm_pair=("Y+", "N+")) -> dict:
    """Crude moment estimates of σ_item and τ (pairwise item x arm SD) on the logit scale, per eval type pooled.

    σ_item² ≈ Var_i(empirical logit of p̂_i) − mean sampling variance, using one reference arm;
    τ² ≈ Var_i(elogit p̂_iA − elogit p̂_iB) − sampling variance, averaged over models.
    """
    out = {}
    s_items, taus = [], []
    for m in sorted(d.runs.loc[d.runs["lang"] == lang, "model"].unique()):
        el, sv = {}, {}
        for a in arm_pair:
            rows = np.flatnonzero(((d.runs["lang"] == lang) & (d.runs["model"] == m) & (d.runs["arm"] == a))
                                  .to_numpy())
            if len(rows) == 0:
                break
            n = d.n[rows].sum(0).astype(float)
            y = d.y[rows].sum(0).astype(float)
            ok = n > 0
            e = np.full(n.shape, np.nan)
            e[ok] = np.log((y[ok] + .5) / (n[ok] - y[ok] + .5))
            v = np.full(n.shape, np.nan)
            v[ok] = 1 / (y[ok] + .5) + 1 / (n[ok] - y[ok] + .5)
            el[a], sv[a] = e, v
        if len(el) < 2:
            continue
        A, B = arm_pair
        ok = np.isfinite(el[A]) & np.isfinite(el[B])
        s_items.append(np.var(el[A][ok], ddof=1) - np.mean(sv[A][ok]))
        taus.append(np.var(el[A][ok] - el[B][ok], ddof=1) - np.mean(sv[A][ok] + sv[B][ok]))
    if s_items:
        out["sigma_item"] = math.sqrt(max(float(np.mean(s_items)), 0))
        out["tau"] = math.sqrt(max(float(np.mean(taus)), 0))
        out["note"] = ("empirical-logit moments; item x lang / item x model variance is absorbed into sigma_item; "
                       "prefer GLMM estimates when available")
    return out
