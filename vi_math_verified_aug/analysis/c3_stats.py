"""C3 attribution study: small-sample paired statistics (numpy only, laptop-safe).

* ``wilson``                 Wilson score interval (re-exported from analysis.intervals)
* ``newcombe_paired``        risk difference of two paired proportions, Newcombe (1998) method 10 (hybrid score,
                             with the continuity-corrected phi), the interval recommended for paired binary data
* ``newcombe_unpaired``      risk difference of two independent proportions, Newcombe (1998) method 10
* ``mcnemar``                exact McNemar (re-exported)
* ``cluster_boot_rd``        percentile CI of a mean difference by resampling clusters (seeds) with replacement
* ``cluster_perm_test``      permutation p for an exposure defined at the cluster level (seed label shuffled
                             across seeds; keeps the within-seed correlation of the 1-3 slots per seed)
* ``attributable_fraction``  population attributable fraction (p_all - p_unexposed) / p_all, cluster-bootstrap CI
* ``cochran_q_perm``         omnibus test that k related binary outcomes have equal rates (Cochran's Q statistic,
                             p by permuting the k labels within each block; exact in the limit, no chi2 needed)
* ``holm``                   Holm FWER adjustment (re-exported)
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from .intervals import mcnemar, wilson
from .multiplicity import holm

__all__ = ["wilson", "mcnemar", "holm", "newcombe_paired", "newcombe_unpaired", "cluster_boot_rd",
           "cluster_perm_test", "attributable_fraction", "cochran_q_perm", "rate", "fmt_pct", "fmt_rd"]


def rate(y: Sequence[int]) -> dict:
    y = np.asarray(y, int)
    n, k = int(y.size), int(y.sum())
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": k / n if n else float("nan"), "lo": lo, "hi": hi}


def newcombe_unpaired(k1: int, n1: int, k2: int, n2: int) -> dict:
    """D = p2 - p1 with the Newcombe hybrid-score interval (independent samples)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return {"d": d, "lo": max(-1.0, lo), "hi": min(1.0, hi)}


def newcombe_paired(x: Sequence[int], y: Sequence[int]) -> dict:
    """D = mean(y) - mean(x) for paired 0/1 vectors; Newcombe (1998) method 10.

    2x2: a = both 1, b = x1y0, c = x0y1, d = both 0. p_x = (a+b)/n, p_y = (a+c)/n.
    """
    x, y = np.asarray(x, int), np.asarray(y, int)
    n = int(x.size)
    if n == 0:
        return {"d": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "b": 0, "c": 0}
    a = int(((x == 1) & (y == 1)).sum())
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    dd = n - a - b - c
    px, py = (a + b) / n, (a + c) / n
    lx, ux = wilson(a + b, n)
    ly, uy = wilson(a + c, n)
    num = a * dd - b * c
    if num > 0:
        num = max(num - n / 2, 0.0)
    den = math.sqrt((a + b) * (c + dd) * (a + c) * (b + dd))
    phi = num / den if den > 0 else 0.0
    d = py - px
    lo = d - math.sqrt(max(0.0, (py - ly) ** 2 - 2 * phi * (py - ly) * (ux - px) + (ux - px) ** 2))
    hi = d + math.sqrt(max(0.0, (uy - py) ** 2 - 2 * phi * (uy - py) * (px - lx) + (px - lx) ** 2))
    return {"d": d, "lo": max(-1.0, lo), "hi": min(1.0, hi), "n": n, "b": b, "c": c,
            "p_mcnemar": mcnemar((b, c))["p_exact"]}


def _groups(clusters: Sequence) -> list[np.ndarray]:
    cl = np.asarray(clusters)
    _, inv = np.unique(cl, return_inverse=True)
    return [np.flatnonzero(inv == g) for g in range(inv.max() + 1)] if cl.size else []


def cluster_boot_rd(diff: Sequence[float], clusters: Sequence, B: int = 4000, seed: int = 20261108,
                    conf: float = 0.95) -> dict:
    """Mean of per-unit differences (e.g. y_i - x_i on paired slots) with a cluster-bootstrap percentile CI."""
    diff = np.asarray(diff, float)
    groups = _groups(clusters)
    if not groups:
        return {"d": float("nan"), "lo": float("nan"), "hi": float("nan")}
    sums = np.array([diff[g].sum() for g in groups])
    cnts = np.array([g.size for g in groups], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(B, len(groups)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    a = (1 - conf) / 2
    return {"d": float(diff.mean()), "lo": float(np.quantile(bs, a)), "hi": float(np.quantile(bs, 1 - a)),
            "n": int(diff.size), "clusters": len(groups)}


def cluster_boot_stat(stat: Callable[[np.ndarray], float], n_units: int, clusters: Sequence, B: int = 4000,
                      seed: int = 20261108, conf: float = 0.95) -> dict:
    """Generic cluster bootstrap: ``stat(index_array)`` on resampled clusters (index arrays may repeat units)."""
    groups = _groups(clusters)
    rng = np.random.default_rng(seed)
    full = np.arange(n_units)
    est = stat(full)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), size=len(groups))
        ix = np.concatenate([groups[g] for g in pick])
        v = stat(ix)
        if np.isfinite(v):
            vals.append(v)
    a = (1 - conf) / 2
    vals = np.asarray(vals)
    return {"est": float(est), "lo": float(np.quantile(vals, a)) if vals.size else float("nan"),
            "hi": float(np.quantile(vals, 1 - a)) if vals.size else float("nan"), "B_ok": int(vals.size)}


def cluster_perm_test(y: Sequence[int], exposure: Sequence[int], clusters: Sequence, B: int = 10000,
                      seed: int = 20261108) -> dict:
    """Two-sided permutation p for RD = P(y|exposed) - P(y|unexposed), exposure constant within a cluster.

    The cluster-level exposure vector is shuffled across clusters (each cluster keeps its items)."""
    y, e = np.asarray(y, int), np.asarray(exposure, int)
    groups = _groups(clusters)
    ce = np.array([e[g[0]] for g in groups])
    cy = np.array([y[g].sum() for g in groups], float)
    cn = np.array([g.size for g in groups], float)

    def rd(cexp):
        ne, nu = cn[cexp == 1].sum(), cn[cexp == 0].sum()
        if ne == 0 or nu == 0:
            return float("nan")
        return cy[cexp == 1].sum() / ne - cy[cexp == 0].sum() / nu

    obs = rd(ce)
    rng = np.random.default_rng(seed)
    cnt, tot = 0, 0
    for _ in range(B):
        v = rd(rng.permutation(ce))
        if np.isfinite(v):
            tot += 1
            cnt += abs(v) >= abs(obs) - 1e-12
    return {"rd": obs, "p_perm": (cnt + 1) / (tot + 1)}


def attributable_fraction(y: Sequence[int], exposure: Sequence[int], clusters: Sequence, B: int = 4000,
                          seed: int = 20261108) -> dict:
    """PAF = (P(y) - P(y | unexposed)) / P(y): share of the item errors that would vanish if no seed had the
    defect (under no confounding). Cluster-bootstrap CI over seeds."""
    y, e = np.asarray(y, float), np.asarray(exposure, int)

    def paf(ix):
        yy, ee = y[ix], e[ix]
        pt = yy.mean() if yy.size else float("nan")
        if not (ee == 0).any() or pt == 0:
            return float("nan")
        return (pt - yy[ee == 0].mean()) / pt

    r = cluster_boot_stat(paf, y.size, clusters, B=B, seed=seed)
    return {"paf": r["est"], "lo": r["lo"], "hi": r["hi"]}


def cochran_q(m: np.ndarray) -> float:
    """Cochran's Q for an (n blocks x k treatments) 0/1 matrix."""
    m = np.asarray(m, int)
    k = m.shape[1]
    col, row, N = m.sum(0), m.sum(1), m.sum()
    den = k * N - (row ** 2).sum()
    if den == 0:
        return 0.0
    return float((k - 1) * (k * (col ** 2).sum() - N ** 2) / den)


def cochran_q_perm(m: np.ndarray, B: int = 10000, seed: int = 20261108) -> dict:
    m = np.asarray(m, int)
    q = cochran_q(m)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        perm = np.argsort(rng.random(m.shape), axis=1)
        cnt += cochran_q(np.take_along_axis(m, perm, 1)) >= q - 1e-12
    return {"Q": q, "p_perm": (cnt + 1) / (B + 1), "n_blocks": int(m.shape[0]), "k": int(m.shape[1])}


def fmt_pct(r: dict) -> str:
    return f"{100 * r['p']:.0f}% [{100 * r['lo']:.0f}–{100 * r['hi']:.0f}] ({r['k']}/{r['n']})"


def fmt_rd(r: dict, key: str = "d") -> str:
    return f"{100 * r[key]:+.0f} [{100 * r['lo']:+.0f}, {100 * r['hi']:+.0f}]"
