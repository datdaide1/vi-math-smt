"""Synthetic item-level predictions with *planted* effects, for tests and for dry-running the pipeline.

Generative model (logit), union of T5 §2.4 and E1 §3.1:

    logit p[r,i] = b(lang_r, size_r, slice_i, arm_r)                      fixed, contains the planted effects
                 + u_i + e_{i,lang} + g_{i,arm} + h_{i,model}            item, item x lang, item x arm, item x model
                 + v_r + w_{r,evaltype}                                   run, run x eval-type
    y[r,i] ~ Binomial(K, p[r,i])

Planted fixed effects (logit units): verification premium ``vp_en`` in English and ``vp_en + d_i2`` in
Vietnamese (=> the P2 interaction), method effect ``m_eff`` (Y − N), certifiable x method interaction ``h4``
(Y − N is ``m_eff + h4/2`` on certifiable Core items and ``m_eff − h4/2`` on the rest).

``truth()`` gives the population value of any contrast: on the pp scale it is E[estimator] (cell expectations
integrated over every random term by Gauss–Hermite), on the conditional logit scale it is the fixed-effect
combination that the GLMM estimates.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .contrasts import Contrast, cell_value
from .schema import EvalData, arm_method_verif

GH_x, GH_w = np.polynomial.hermite.hermgauss(60)


def marg(b, sd):
    """E[expit(b + sd Z)], vectorised over b."""
    b = np.asarray(b, float)[..., None]
    z = math.sqrt(2.0) * GH_x
    return (GH_w / (1 + np.exp(-(b + sd * z)))).sum(-1) / math.sqrt(math.pi)


@dataclass
class SynthSpec:
    vi_models: dict = field(default_factory=lambda: {"qwen3-0.6b": "0.6B", "qwen3-1.7b": "1.7B"})
    en_models: tuple = ("qwen3-0.6b", "qwen3-1.7b")
    vi_arms: tuple = ("S0", "T", "M", "Y+", "Y-", "N+", "N-")
    en_arms: tuple = ("Y+", "Y-", "N+", "N-")
    seeds_vi: int = 3
    seeds_en: int = 3
    n_core: int = 200
    n_par: dict = field(default_factory=lambda: {"gsm8k_platinum": 150, "math_algprealg": 100})
    cert_share: float = 0.5
    K: int = 4
    # fixed effects (logit)
    base: dict = field(default_factory=lambda: {"0.6B": -0.8, "1.7B": 0.0, "4B": 0.7})
    en_shift: float = 0.7
    native_shift: float = -0.4
    other_arms: dict = field(default_factory=lambda: {"S0": -0.25, "T": 0.0, "M": 0.05, "T-off": -0.1})
    vp_en: float = 0.10
    d_i2: float = 0.0
    m_eff: float = 0.0
    h4: float = 0.0
    # random effects (SD, logit)
    s_item: float = 1.5
    s_lang: float = 0.5
    tau: float = 0.5           # pairwise item x arm SD (per-arm SD tau / sqrt 2), as T5/E1
    s_im: float = 0.3
    s_run: float = 0.10
    s_re: float = 0.07
    paired_runs_rho: float = 0.0   # CRN correlation of v between the vi and en run with the same (model, arm, seed)

    @property
    def sd_total(self) -> float:
        return math.sqrt(self.s_item ** 2 + self.s_lang ** 2 + self.tau ** 2 / 2 + self.s_im ** 2
                         + self.s_run ** 2 + self.s_re ** 2)


def _fixed(spec: SynthSpec, lang: str, size: str, arm: str, evaltype: str, cert) -> float:
    b = spec.base[size] + (spec.en_shift if lang == "en" else 0.0) + (spec.native_shift if evaltype == "native"
                                                                       else 0.0)
    method, verif = arm_method_verif(arm)
    if method is None:
        return b + spec.other_arms.get(arm, 0.0)
    vp = spec.vp_en + (spec.d_i2 if lang == "vi" else 0.0)
    m = spec.m_eff
    if evaltype == "native" and cert is not None:
        m += spec.h4 / 2 if bool(cert) else -spec.h4 / 2
    return b + (m / 2 if method == "Y" else -m / 2) + (vp / 2 if verif == "+" else -vp / 2)


def generate(spec: SynthSpec, seed: int = 0, with_truth_tables: bool = True) -> EvalData:
    rng = np.random.default_rng(seed)
    # items
    rows = [{"dataset": "vi_exam_core", "item_id": f"c{j:04d}", "evaltype": "native",
             "certifiable": bool(j < round(spec.cert_share * spec.n_core))} for j in range(spec.n_core)]
    for ds, k in spec.n_par.items():
        rows += [{"dataset": ds, "item_id": f"{ds[:3]}{j:04d}", "evaltype": "parallel", "certifiable": None}
                 for j in range(k)]
    items = pd.DataFrame(rows)
    items.insert(0, "item_key", items["dataset"] + "/" + items["item_id"])
    I = len(items)
    # runs
    rr = []
    for m, size in spec.vi_models.items():
        for a in spec.vi_arms:
            for s in range(spec.seeds_vi):
                rr.append(("vi", m, size, a, s))
    for m in spec.en_models:
        for a in spec.en_arms:
            for s in range(spec.seeds_en):
                rr.append(("en", m, spec.vi_models.get(m, "0.6B"), a, s))
    runs = pd.DataFrame(rr, columns=["lang", "model", "size", "arm", "seed"])
    runs.insert(0, "run_id", runs["lang"] + ":" + runs["model"] + ":" + runs["arm"] + ":s" + runs["seed"].astype(str))
    runs["family"] = runs["model"].str.split("-").str[0]
    mv = [arm_method_verif(a) for a in runs["arm"]]
    runs["method"], runs["verif"] = [m for m, _ in mv], [v for _, v in mv]
    R = len(runs)
    # visibility: VI runs see every item, EN runs only parallel items
    is_par = (items["evaltype"] == "parallel").to_numpy()
    vis = np.ones((R, I), bool)
    vis[(runs["lang"] == "en").to_numpy()] = is_par
    # fixed part
    eta0 = np.zeros((R, I))
    slices = []
    for key, grp in items.groupby(["evaltype", "certifiable"], dropna=False).groups.items():
        et, cert = key
        cert = None if (cert is None or (isinstance(cert, float) and np.isnan(cert))) else cert
        slices.append((et, cert, np.asarray(grp)))
    for r in range(R):
        L, sz, a = runs.at[r, "lang"], runs.at[r, "size"], runs.at[r, "arm"]
        for et, cert, grp in slices:
            eta0[r, grp] = _fixed(spec, L, sz, a, et, cert)
    # random part
    u = rng.normal(0, spec.s_item, I)
    e = {L: rng.normal(0, spec.s_lang, I) for L in ("vi", "en")}
    arms = sorted(set(spec.vi_arms) | set(spec.en_arms))
    g = {a: rng.normal(0, spec.tau / math.sqrt(2), I) for a in arms}
    models = sorted(set(spec.vi_models) | set(spec.en_models))
    h = {m: rng.normal(0, spec.s_im, I) for m in models}
    crn = {}
    eta = eta0.copy()
    for r in range(R):
        L, m, a, s = runs.at[r, "lang"], runs.at[r, "model"], runs.at[r, "arm"], runs.at[r, "seed"]
        zc = crn.setdefault((m, a, s), rng.normal())
        v = spec.s_run * (math.sqrt(spec.paired_runs_rho) * zc + math.sqrt(1 - spec.paired_runs_rho) * rng.normal())
        w = np.where(is_par, rng.normal(0, spec.s_re), rng.normal(0, spec.s_re))
        eta[r] += u + e[L] + g[a] + h[m] + v + w
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(spec.K, p).astype(np.float32)
    n = np.full((R, I), spec.K, np.int16)
    y[~vis], n[~vis] = 0, 0
    tok = np.where(vis, 300.0, np.nan).astype(np.float32)
    greedy = np.full((R, I), np.nan, np.float32)
    d = EvalData(runs, items, y, n, tok, greedy, {"synthetic": True, "seed": seed})
    if with_truth_tables:
        d.report["_eta_fixed"] = np.where(vis, eta0, np.nan)
        d.report["_p_marg"] = np.where(vis, marg(eta0, spec.sd_total), np.nan)
    return d


def truth(d: EvalData, c: Contrast, scale: str = "pp") -> float:
    """Population value of a contrast for synthetic data.

    ``pp``: E[pp estimator]; ``logit_marginal``: Σ w logit(E[cell value]) (target of the marginal-logit
    bootstrap); ``logit_conditional``: Σ w (mean fixed logit of the cell) (target of the GLMM contrast).
    """
    tab = d.report["_eta_fixed"] if scale == "logit_conditional" else d.report["_p_marg"]
    obs = np.isfinite(tab)
    dd = EvalData(d.runs, d.items, np.where(obs, tab, 0).astype(np.float64), obs.astype(np.int16), d.tok,
                  d.greedy, {})
    vals = [cell_value(dd, t)[0] for t in c.terms]
    if scale == "logit_marginal":
        vals = [math.log(v / (1 - v)) for v in vals]
    return float(sum(t.weight * v for t, v in zip(c.terms, vals)))


def to_rows(d: EvalData, seed: int = 0):
    """Expand an EvalData back to schema rows (one per sample) -- for schema round-trip tests and demos."""
    rng = np.random.default_rng(seed)
    for r in range(len(d.runs)):
        run = d.runs.iloc[r]
        for i in np.flatnonzero(d.n[r] > 0):
            it = d.items.iloc[i]
            k, yv = int(d.n[r, i]), int(d.y[r, i])
            outcomes = np.zeros(k, int)
            outcomes[rng.choice(k, yv, replace=False)] = 1
            for s in range(k):
                row = {"run_id": run["run_id"], "arm": run["arm"], "model": run["model"], "size": run["size"],
                       "seed": int(run["seed"]), "lang": run["lang"], "dataset": it["dataset"],
                       "item_id": it["item_id"], "sample_k": s, "correct": int(outcomes[s]), "tokens": 300,
                       "evaltype": it["evaltype"]}
                if it.get("certifiable") is not None and not (isinstance(it["certifiable"], float)):
                    row["certifiable"] = bool(it["certifiable"])
                yield row


def write_jsonl(d: EvalData, path: str, seed: int = 0) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in to_rows(d, seed):
            fh.write(json.dumps(row) + "\n")
            n += 1
    return n
