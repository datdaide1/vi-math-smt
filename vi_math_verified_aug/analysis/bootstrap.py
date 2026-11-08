"""Two-way cluster bootstrap (T5 §2.3 item 2; E1 §5.5), vectorised numpy, chunked for low memory.

Each draw resamples
  * **items within dataset** (multinomial counts over the item keys of each dataset; a parallel item's EN and VI
    versions share one key, so the language pairing is preserved), and
  * **seeds within stratum** (default stratum = lang x arm x model, i.e. "seeds within arm x model" of T5 and
    "seeds within language x size x arm" of E1).  ``paired_seeds=True`` resamples seed *indices* within
    arm x model jointly for both languages (the CRN design of E1 §2.2).

The statistic is recomputed exactly as on the data: every cell value is the weighted mean over resampled runs of
each run's weighted item mean.  This is the "pigeonhole" bootstrap for crossed designs (Owen 2007), which is
mildly conservative; the coverage simulation in ``tests/test_analysis_coverage.py`` checks it.

Memory: one chunk holds (chunk x items) item weights + (chunk x runs) per cell group; with chunk = 500,
2,500 items and 120 runs that is < 20 MB.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contrasts import Contrast
from .schema import EvalData, filter_mask
from .stats_util import norm_ppf, two_sided_p


@dataclass
class BootResult:
    name: str
    estimate: float
    se: float
    ci_lo: float
    ci_hi: float
    p_wald: float               # two-sided, z = estimate / bootstrap SE
    p_perc: float               # two-sided percentile p: 2 min(P*(T <= 0), P*(T >= 0)) (bounded below by 1/B)
    ci90_lo: float
    ci90_hi: float
    draws: np.ndarray | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "draws"}


class _Prepared:
    """Unique cells across a set of contrasts, compiled to masks."""

    def __init__(self, d: EvalData, contrasts: list[Contrast]):
        self.d = d
        acc = d.acc
        self.obs = (d.n > 0).astype(np.float64)                 # (R, I)
        self.accz = np.where(d.n > 0, acc, 0.0).astype(np.float64)
        groups: dict[tuple, int] = {}
        cells: dict[tuple, int] = {}
        self.group_masks, self.cell_spec = [], []
        self.contrast_terms = []
        for c in contrasts:
            tl = []
            for t in c.terms:
                g = groups.get(t.items)
                if g is None:
                    g = groups[t.items] = len(self.group_masks)
                    self.group_masks.append(filter_mask(d.items, t.item_filter))
                key = (t.runs, g)
                ci = cells.get(key)
                if ci is None:
                    ci = cells[key] = len(self.cell_spec)
                    self.cell_spec.append((filter_mask(d.runs, t.run_filter), g))
                tl.append((t.weight, ci))
            self.contrast_terms.append(tl)

    def cell_values(self, item_w: np.ndarray, run_w: np.ndarray) -> np.ndarray:
        """item_w (B, I) and run_w (B, R) -> cell values (B, n_cells)."""
        B = item_w.shape[0]
        out = np.empty((B, len(self.cell_spec)))
        per_group = {}
        for g, gm in enumerate(self.group_masks):
            w = item_w[:, gm]
            num = w @ self.accz[:, gm].T                          # (B, R)
            den = w @ self.obs[:, gm].T
            with np.errstate(invalid="ignore", divide="ignore"):
                per_group[g] = num / den                          # nan where a run saw none of the drawn items
        for ci, (rm, g) in enumerate(self.cell_spec):
            a = per_group[g][:, rm]
            w = run_w[:, rm] * np.isfinite(a)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[:, ci] = np.nansum(np.where(w > 0, a, 0.0) * w, axis=1) / w.sum(1)
        return out

    def contrasts(self, cells: np.ndarray, scale: str) -> np.ndarray:
        if scale == "logit":
            p = np.clip(cells, 1e-6, 1 - 1e-6)
            cells = np.log(p / (1 - p))
        out = np.empty((cells.shape[0], len(self.contrast_terms)))
        for j, tl in enumerate(self.contrast_terms):
            out[:, j] = sum(w * cells[:, ci] for w, ci in tl)
        return out


def _item_weights(d: EvalData, rng, B: int, resample: bool) -> np.ndarray:
    I = len(d.items)
    if not resample:
        return np.ones((B, I))
    W = np.zeros((B, I))
    ds = d.items["dataset"].to_numpy()
    for name in np.unique(ds):
        idx = np.flatnonzero(ds == name)
        W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=B)
    return W


def _run_weights(d: EvalData, rng, B: int, strata: tuple, unit: str, resample: bool) -> np.ndarray:
    R = len(d.runs)
    if not resample:
        return np.ones((B, R))
    W = np.zeros((B, R))
    keys = d.runs[list(strata)].astype(str).agg("|".join, axis=1).to_numpy()
    units = d.runs[unit].astype(str).to_numpy()
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        uv, inv = np.unique(units[idx], return_inverse=True)
        cnt = rng.multinomial(len(uv), np.full(len(uv), 1 / len(uv)), size=B)   # (B, n_units)
        W[:, idx] = cnt[:, inv]
    return W


def cluster_bootstrap(d: EvalData, contrasts: list[Contrast], *, B: int = 10_000, seed: int = 20261107,
                      scale: str = "pp", resample_items: bool = True, resample_seeds: bool = True,
                      paired_seeds: bool = False, chunk: int = 500, keep_draws: bool = False,
                      conf: float = 0.95) -> list[BootResult]:
    """Bootstrap every contrast jointly (same draws), returning one :class:`BootResult` each."""
    if not contrasts:
        return []
    rng = np.random.default_rng(seed)
    prep = _Prepared(d, contrasts)
    est = prep.contrasts(prep.cell_values(np.ones((1, len(d.items))), np.ones((1, len(d.runs)))), scale)[0]
    strata, unit = (("arm", "model"), "seed") if paired_seeds else (("lang", "arm", "model"), "run_id")
    draws = []
    done = 0
    while done < B:
        b = min(chunk, B - done)
        iw = _item_weights(d, rng, b, resample_items)
        rw = _run_weights(d, rng, b, strata, unit, resample_seeds)
        draws.append(prep.contrasts(prep.cell_values(iw, rw), scale))
        done += b
    T = np.vstack(draws)                                           # (B, C)
    out = []
    a = 1 - conf
    for j, c in enumerate(contrasts):
        t = T[:, j]
        t = t[np.isfinite(t)]
        se = float(t.std(ddof=1)) if t.size > 1 else float("nan")
        lo, hi = np.quantile(t, [a / 2, 1 - a / 2]) if t.size else (np.nan, np.nan)
        lo90, hi90 = np.quantile(t, [0.05, 0.95]) if t.size else (np.nan, np.nan)
        pp = 2 * min((t <= 0).mean(), (t >= 0).mean()) if t.size else np.nan
        pp = float(max(min(pp, 1.0), 1 / max(t.size, 1)))
        z = est[j] / se if se and se > 0 else np.nan
        out.append(BootResult(c.name, float(est[j]), se, float(lo), float(hi),
                              float(two_sided_p(z)) if np.isfinite(z) else float("nan"), pp,
                              float(lo90), float(hi90), t if keep_draws else None))
    return out


def normal_ci(est: float, se: float, conf: float = 0.95) -> tuple[float, float]:
    z = norm_ppf(1 - (1 - conf) / 2)
    return est - z * se, est + z * se
