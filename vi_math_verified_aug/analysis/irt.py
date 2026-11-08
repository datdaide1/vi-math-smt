"""2PL IRT with binomial responses, MAP / empirical-Bayes fit in numpy (T5 §2.5).

Why not py-irt: it needs torch + pyro (several hundred MB installed, ~0.5 GB RAM at import), which does not fit the
laptop's ~1 GB free RAM.  The model is small enough for alternating Fisher scoring:

    y_ji ~ Binomial(K_ji, σ(a_i (θ_j − b_i)))             respondent j (panel model), item i
    θ_j ~ N(0, 1)                                         fixes location and scale
    b_i ~ N(μ_b, σ_b²)        μ_b, σ_b re-estimated each sweep (empirical Bayes "hierarchical prior")
    log a_i ~ N(μ_a, σ_a²)    μ_a ~ N(0, 0.5²) (T5 prior), σ_a = 0.5 by default       [positive_a=True]
    a_i ~ N(1, 1)             unconstrained variant used only to *detect* negative discrimination

Point estimates are posterior modes; SEs come from the observed Fisher information of each block.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .stats_util import norm_ppf


@dataclass
class IRTFit:
    a: np.ndarray
    b: np.ndarray
    theta: np.ndarray
    se_a: np.ndarray
    se_b: np.ndarray
    se_theta: np.ndarray
    iters: int
    converged: bool
    hyper: dict


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_2pl(y, n, *, positive_a: bool = True, sigma_a: float = 0.5, mu_a_prior_sd: float = 0.5,
            max_iter: int = 500, tol: float = 1e-5, theta_init=None) -> IRTFit:
    """Fit a 2PL to successes ``y`` and trials ``n`` (respondents x items; n == 0 = missing)."""
    y = np.asarray(y, float)
    n = np.asarray(n, float)
    J, I = y.shape
    obs = n > 0
    pj = np.where(obs, (y + .5) / (n + 1), np.nan)
    score = np.nanmean(pj, 1)
    theta = (np.asarray(theta_init, float) if theta_init is not None else
             (score - score.mean()) / (score.std() + 1e-9))
    pi = np.nanmean(pj, 0)
    b = -np.log(pi / (1 - pi))
    a = np.ones(I)
    mu_b, sd_b, mu_a = float(b.mean()), float(max(b.std(), 0.5)), 0.0
    conv = False
    it = 0
    for it in range(1, max_iter + 1):
        old = np.r_[theta, a, b]
        # ---- items: Fisher scoring on (alpha = log a, b) or (a, b), vectorised over items
        d = theta[:, None] - b[None, :]
        p = _sig(a[None, :] * d)
        r = np.where(obs, y - n * p, 0.0)
        w = np.where(obs, n * p * (1 - p), 0.0)
        if positive_a:
            ad = a[None, :] * d                       # d eta / d alpha
            g1 = (r * ad).sum(0) - (np.log(a) - mu_a) / sigma_a ** 2
            i11 = (w * ad * ad).sum(0) + 1 / sigma_a ** 2
        else:
            g1 = (r * d).sum(0) - (a - 1.0) / 1.0
            i11 = (w * d * d).sum(0) + 1.0
            ad = d
        g2 = (r * -a[None, :]).sum(0) - (b - mu_b) / sd_b ** 2
        i22 = (w * a[None, :] ** 2).sum(0) + 1 / sd_b ** 2
        i12 = (w * ad * -a[None, :]).sum(0)
        det = i11 * i22 - i12 ** 2
        det = np.where(det <= 1e-12, 1e-12, det)
        s1 = np.clip((i22 * g1 - i12 * g2) / det, -1, 1)
        s2 = np.clip((-i12 * g1 + i11 * g2) / det, -1, 1)
        if positive_a:
            a = a * np.exp(s1)
        else:
            a = a + s1
        b = b + s2
        # ---- persons: 1-D Fisher scoring
        d = theta[:, None] - b[None, :]
        p = _sig(a[None, :] * d)
        r = np.where(obs, y - n * p, 0.0)
        w = np.where(obs, n * p * (1 - p), 0.0)
        g = (r * a[None, :]).sum(1) - theta
        h = (w * a[None, :] ** 2).sum(1) + 1
        theta = theta + np.clip(g / h, -1, 1)
        # ---- empirical-Bayes hyperparameters
        mu_b, sd_b = float(b.mean()), float(max(b.std(ddof=1), 0.3))
        if positive_a:
            la = np.log(a)
            mu_a = float(la.sum() / sigma_a ** 2 / (I / sigma_a ** 2 + 1 / mu_a_prior_sd ** 2))
        if np.max(np.abs(np.r_[theta, a, b] - old)) < tol:
            conv = True
            break
    # SEs from block information at the mode
    d = theta[:, None] - b[None, :]
    p = _sig(a[None, :] * d)
    w = np.where(obs, n * p * (1 - p), 0.0)
    se_theta = 1 / np.sqrt((w * a[None, :] ** 2).sum(1) + 1)
    if positive_a:
        jac = a[None, :] * d
        i11 = (w * jac * jac).sum(0) + 1 / sigma_a ** 2
    else:
        jac = d
        i11 = (w * d * d).sum(0) + 1
    i22 = (w * a[None, :] ** 2).sum(0) + 1 / sd_b ** 2
    i12 = (w * jac * -a[None, :]).sum(0)
    det = i11 * i22 - i12 ** 2
    se_first = np.sqrt(np.maximum(i22 / det, 0))
    se_a = se_first * a if positive_a else se_first        # delta method for a = exp(alpha)
    se_b = np.sqrt(np.maximum(i11 / det, 0))
    return IRTFit(a, b, theta, se_a, se_b, se_theta, it, conv,
                  {"mu_b": mu_b, "sd_b": sd_b, "mu_a": mu_a, "sigma_a": sigma_a, "positive_a": positive_a})


def fit_theta(y, n, a, b, prior_sd: float = 1.0, iters: int = 100) -> tuple[float, float]:
    """MAP ability of one new respondent on fixed items (used for temporal-split residuals)."""
    y, n, a, b = (np.asarray(v, float) for v in (y, n, a, b))
    ok = n > 0
    y, n, a, b = y[ok], n[ok], a[ok], b[ok]
    th = 0.0
    for _ in range(iters):
        p = _sig(a * (th - b))
        g = (a * (y - n * p)).sum() - th / prior_sd ** 2
        h = (a * a * n * p * (1 - p)).sum() + 1 / prior_sd ** 2
        step = float(np.clip(g / h, -1, 1))
        th += step
        if abs(step) < 1e-8:
            break
    p = _sig(a * (th - b))
    return th, 1 / math.sqrt((a * a * n * p * (1 - p)).sum() + 1 / prior_sd ** 2)


def item_rest_corr(y, n) -> np.ndarray:
    """Item-rest point-biserial (Pearson) correlation of per-item proportion with the rest score."""
    y, n = np.asarray(y, float), np.asarray(n, float)
    obs = n > 0
    p = np.where(obs, y / np.maximum(n, 1), np.nan)
    tot = np.nansum(p, 1)
    cnt = obs.sum(1)
    out = np.full(p.shape[1], np.nan)
    for i in range(p.shape[1]):
        m = obs[:, i] & (cnt > 1)
        if m.sum() < 3:
            continue
        x = p[m, i]
        rest = (tot[m] - x) / (cnt[m] - 1)
        if x.std() == 0 or rest.std() == 0:
            continue
        out[i] = float(np.corrcoef(x, rest)[0, 1])
    return out


def item_flags(y, n, fit_pos: IRTFit | None = None, fit_free: IRTFit | None = None, a_min: float = 0.2,
               r_min: float = 0.1) -> dict:
    """T5 §2.5 use 1: items for mandatory human re-review (never auto-drop)."""
    fit_pos = fit_pos or fit_2pl(y, n, positive_a=True)
    fit_free = fit_free or fit_2pl(y, n, positive_a=False)
    r = item_rest_corr(y, n)
    low_a = fit_pos.a < a_min
    neg_a = fit_free.a < 0
    low_r = np.nan_to_num(r, nan=1.0) < r_min
    return {"low_discrimination": low_a, "negative_discrimination": neg_a, "low_item_rest": low_r,
            "item_rest": r, "flag_any": low_a | neg_a | low_r, "a_pos": fit_pos.a, "a_free": fit_free.a,
            "b": fit_pos.b}


def test_information(theta_grid, a, b, K: int = 1) -> np.ndarray:
    th = np.asarray(theta_grid, float)[:, None]
    p = _sig(a[None, :] * (th - b[None, :]))
    return (K * a[None, :] ** 2 * p * (1 - p)).sum(1)


def marginal_reliability(fit: IRTFit) -> float:
    v = float(np.var(fit.theta, ddof=1))
    return v / (v + float(np.mean(fit.se_theta ** 2)))


def core_hard_split(fit: IRTFit, small_panel: np.ndarray, top_panel: np.ndarray, margin: float = 1.5,
                    hard_gap: float = 0.5, hard_a: float = 0.5) -> dict:
    """T5 §2.5 use 2: Core = b within ±margin of the small-model ability range; Hard = above the top + gap."""
    th_s = fit.theta[np.asarray(small_panel)]
    th_top = float(np.max(fit.theta[np.asarray(top_panel)]))
    core = (fit.b >= th_s.min() - margin) & (fit.b <= th_s.max() + margin)
    hard = (fit.b > th_top + hard_gap) & (fit.a >= hard_a)
    return {"core": core, "hard": hard, "ability_range": (float(th_s.min()), float(th_s.max())),
            "top_ability": th_top}


# ------------------------------------------------------------------------------- temporal split (D5 a–c)
def matched_difficulty(b, period, pair_id, B: int = 10_000, seed: int = 20261107) -> dict:
    """(a) slot-matched pairs: mean b_post − b_pre over pairs, with a pair bootstrap CI."""
    b, period, pair_id = np.asarray(b, float), np.asarray(period), np.asarray(pair_id)
    diffs = []
    for pid in np.unique(pair_id[pd_notna(pair_id)]):
        m = pair_id == pid
        pre, post = b[m & (period == "pre")], b[m & (period == "post")]
        if pre.size and post.size:
            diffs.append(post.mean() - pre.mean())
    diffs = np.asarray(diffs)
    if diffs.size < 2:
        return {"n_pairs": int(diffs.size)}
    rng = np.random.default_rng(seed)
    bs = rng.multinomial(diffs.size, np.full(diffs.size, 1 / diffs.size), size=B) @ diffs / diffs.size
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {"n_pairs": int(diffs.size), "mean_b_post_minus_pre": float(diffs.mean()), "ci95": (float(lo), float(hi)),
            "se": float(bs.std(ddof=1))}


def pd_notna(x) -> np.ndarray:
    return np.array([v is not None and not (isinstance(v, float) and math.isnan(v)) for v in x], bool)


def reference_did(acc_m, acc_r, period, pair_id, B: int = 10_000, seed: int = 20261107) -> dict:
    """(b) C_M = [acc_M(pre) − acc_M(post)] − [acc_R(pre) − acc_R(post)], pair bootstrap; C > 0 => contamination."""
    acc_m, acc_r = np.asarray(acc_m, float), np.asarray(acc_r, float)
    period, pair_id = np.asarray(period), np.asarray(pair_id)
    rows = []
    for pid in np.unique(pair_id[pd_notna(pair_id)]):
        m = pair_id == pid
        pre, post = m & (period == "pre"), m & (period == "post")
        if pre.any() and post.any():
            rows.append((acc_m[pre].mean() - acc_m[post].mean()) - (acc_r[pre].mean() - acc_r[post].mean()))
    rows = np.asarray(rows)
    if rows.size < 2:
        return {"n_pairs": int(rows.size)}
    rng = np.random.default_rng(seed)
    bs = rng.multinomial(rows.size, np.full(rows.size, 1 / rows.size), size=B) @ rows / rows.size
    lo, hi = np.quantile(bs, [0.025, 0.975])
    se = float(bs.std(ddof=1))
    z = rows.mean() / se if se > 0 else float("nan")
    return {"n_pairs": int(rows.size), "C": float(rows.mean()), "ci95": (float(lo), float(hi)), "se": se,
            "z": float(z)}


def irt_residuals(y, n, a, b, period, B: int = 10_000, seed: int = 20261107) -> dict:
    """(c) θ̂_M from post-cutoff items only, then mean residual y/n − P̂ on pre-cutoff items (item bootstrap).

    ``a``/``b`` must come from a calibration that uses only respondents released before each item's date.
    A positive mean residual on pre items = excess accuracy on items the model could have seen.
    """
    y, n, a, b, period = (np.asarray(v) for v in (y, n, a, b, period))
    post, pre = (period == "post") & (n > 0), (period == "pre") & (n > 0)
    th, se_th = fit_theta(y[post], n[post], a[post], b[post])
    res = y[pre] / n[pre] - _sig(a[pre].astype(float) * (th - b[pre].astype(float)))
    rng = np.random.default_rng(seed)
    k = res.size
    bs = rng.multinomial(k, np.full(k, 1 / k), size=B) @ res / k
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {"theta_post": th, "se_theta_post": se_th, "n_pre": int(k), "mean_residual_pre": float(res.mean()),
            "ci95": (float(lo), float(hi))}


def flags_summary(fl: dict) -> dict:
    return {k: int(np.sum(v)) for k, v in fl.items() if isinstance(v, np.ndarray) and v.dtype == bool}


__all__ = ["fit_2pl", "fit_theta", "item_rest_corr", "item_flags", "test_information", "marginal_reliability",
           "core_hard_split", "matched_difficulty", "reference_did", "irt_residuals", "norm_ppf"]
