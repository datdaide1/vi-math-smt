"""Multiplicity control, equivalence testing and the pre-registered decision rules (T5 §2.2–§2.3, E1 §5.5)."""
from __future__ import annotations

import numpy as np

from .stats_util import norm_cdf, norm_ppf


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values (FWER)."""
    p = np.asarray(p, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def bh(p) -> np.ndarray:
    """Benjamini–Hochberg step-up adjusted p-values (q-values, FDR)."""
    p = np.asarray(p, float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def tost(est: float, se: float | None = None, margin: float = 0.03, alpha: float = 0.05,
         draws: np.ndarray | None = None) -> dict:
    """Two one-sided tests for |effect| < margin (pp on the 0–1 scale: 0.03 = 3 pp).

    Normal version (``se``): p = max(Φ((est − margin)/se), 1 − Φ((est + margin)/se)); equivalent iff the
    (1 − 2α) CI lies inside ±margin.  Bootstrap version (``draws``): the (1 − 2α) percentile CI is used and
    the p-value is the larger tail mass beyond ±margin of the draws re-centred on ±margin.
    """
    if draws is not None:
        t = np.asarray(draws, float)
        t = t[np.isfinite(t)]
        lo, hi = np.quantile(t, [alpha, 1 - alpha])
        c = t - t.mean()                       # bootstrap null distributions shifted to the margins
        p_lo = float(np.mean(c - margin >= est))   # H0: effect <= -margin
        p_hi = float(np.mean(c + margin <= est))   # H0: effect >= +margin
        p = max(p_lo, p_hi)
    else:
        z = norm_ppf(1 - alpha)
        lo, hi = est - z * se, est + z * se
        p = max(float(1 - norm_cdf((est + margin) / se)), float(norm_cdf((est - margin) / se)))
    return {"margin": margin, "ci_lo": float(lo), "ci_hi": float(hi), "p": float(p),
            "equivalent": bool(lo > -margin and hi < margin)}


def decide(glmm: dict | None, boot_pp: dict, alpha_adj_glmm: float | None, alpha_adj_boot: float | None,
           alpha: float = 0.05) -> dict:
    """Pre-registered verdict for one confirmatory contrast.

    * significance: both the GLMM (or its Python stand-in) and the pp cluster bootstrap must reject after Holm;
      if they disagree the more conservative one governs (T5 §2.3 rule 3);
    * a directional claim additionally needs the logit-scale estimate and the pp estimate to agree in sign
      (T5 §2.2 scale rule; E1 §5.5).
    """
    sig_boot = alpha_adj_boot is not None and alpha_adj_boot < alpha
    sig_glmm = None if glmm is None or alpha_adj_glmm is None else alpha_adj_glmm < alpha
    sign_pp = np.sign(boot_pp.get("estimate", np.nan))
    sign_logit = np.sign(glmm.get("estimate", np.nan)) if glmm else np.nan
    signs_agree = bool(glmm is None or sign_pp == sign_logit)
    significant = sig_boot and (sig_glmm if sig_glmm is not None else True)
    return {"significant": bool(significant and signs_agree), "sig_bootstrap": bool(sig_boot),
            "sig_glmm": sig_glmm, "signs_agree": signs_agree,
            "governing": "bootstrap" if sig_glmm is None else ("both" if sig_boot == sig_glmm else "conservative")}
