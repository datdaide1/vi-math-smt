"""Small-slice intervals and exact paired tests (T5 §2.3 "other tests", §3.2; Bowyer 2503.01747).

* ``wilson``, ``clopper_pearson`` (two- or one-sided), ``beta_posterior`` (Beta(1,1) default; Jeffreys = 0.5).
* ``slice_ci``: per-model accuracy on a slice with K > 1 samples -- average K within item, then an item
  bootstrap (T5: "For K > 1, average K within item first, then use an item bootstrap"); K = 1 -> Beta/Wilson.
* ``mcnemar``: exact two-sided and mid-p, for deterministic single-decode comparisons only.
* ``residual_error_upper``: Clopper–Pearson one-sided upper bound for the random audit (0/150 -> 2.0 %).
"""
from __future__ import annotations

import math

import numpy as np

from .stats_util import beta_ppf, binom_cdf, binom_pmf, norm_ppf


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = norm_ppf(1 - (1 - conf) / 2)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def clopper_pearson(k: int, n: int, conf: float = 0.95, one_sided: bool = False) -> tuple[float, float]:
    a = 1 - conf
    if one_sided:
        lo = 0.0
        hi = 1.0 if k == n else beta_ppf(1 - a, k + 1, n - k)
        return (lo, hi)
    lo = 0.0 if k == 0 else beta_ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_ppf(1 - a / 2, k + 1, n - k)
    return (lo, hi)


def beta_posterior(k: float, n: float, conf: float = 0.95, a0: float = 1.0, b0: float = 1.0) -> dict:
    a, b = k + a0, n - k + b0
    q = (1 - conf) / 2
    return {"mean": a / (a + b), "lo": beta_ppf(q, a, b), "hi": beta_ppf(1 - q, a, b)}


def residual_error_upper(errors: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper–Pearson upper bound on the label-error rate of an audited benchmark."""
    return clopper_pearson(errors, n, conf, one_sided=True)[1]


def slice_ci(y: np.ndarray, n: np.ndarray, conf: float = 0.95, B: int = 10_000, seed: int = 20261107,
             method: str = "auto") -> dict:
    """Accuracy CI for one run (or a pooled set of runs) on a small slice.

    ``y``/``n`` are per-item successes / samples (1-D) or (runs, items); items with n == 0 are ignored.
    K = 1 everywhere -> Beta(1,1) posterior and Wilson; otherwise item-mean then item bootstrap.
    """
    y, n = np.atleast_2d(np.asarray(y, float)), np.atleast_2d(np.asarray(n, float))
    obs = (n > 0).all(0)
    y, n = y[:, obs], n[:, obs]
    per_item = (y / n).mean(0)                     # average K (and runs) within item
    I = per_item.size
    est = float(per_item.mean()) if I else float("nan")
    if method == "auto":
        method = "beta" if (n == 1).all() and y.shape[0] == 1 else "item_bootstrap"
    if method == "beta":
        k = int(y.sum())
        post = beta_posterior(k, I, conf)
        return {"estimate": est, "n_items": I, "method": "beta(1,1)", "lo": post["lo"], "hi": post["hi"],
                "wilson": wilson(k, I, conf)}
    rng = np.random.default_rng(seed)
    out = []
    for s in range(0, B, 1000):
        b = min(1000, B - s)
        w = rng.multinomial(I, np.full(I, 1 / I), size=b)
        out.append(w @ per_item / I)
    t = np.concatenate(out)
    a = (1 - conf) / 2
    lo, hi = np.quantile(t, [a, 1 - a])
    return {"estimate": est, "n_items": I, "method": "item_bootstrap", "lo": float(lo), "hi": float(hi),
            "se": float(t.std(ddof=1))}


def mcnemar(a, b=None) -> dict:
    """Exact McNemar on paired binary outcomes.

    ``mcnemar(x, y)`` with two 0/1 arrays, or ``mcnemar((n01, n10))`` with the discordant counts.
    Returns the exact two-sided p (doubling the smaller tail) and the mid-p.
    """
    if b is None:
        n01, n10 = (int(v) for v in a)
    else:
        x, y = np.asarray(a, int), np.asarray(b, int)
        n01 = int(((x == 0) & (y == 1)).sum())
        n10 = int(((x == 1) & (y == 0)).sum())
    m = n01 + n10
    if m == 0:
        return {"n01": n01, "n10": n10, "p_exact": 1.0, "p_mid": 1.0}
    k = min(n01, n10)
    tail = binom_cdf(k, m, 0.5)
    p = min(1.0, 2 * tail)
    pmid = min(1.0, 2 * (tail - 0.5 * binom_pmf(k, m, 0.5)))
    return {"n01": n01, "n10": n10, "p_exact": p, "p_mid": pmid}
