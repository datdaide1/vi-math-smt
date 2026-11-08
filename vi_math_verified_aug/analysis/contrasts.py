"""Declarative contrasts over (run filter, item filter) cells, and the pre-registered contrast library.

A **cell value** is the mean over the runs matching ``Term.runs`` of each run's mean pass@1 over the items matching
``Term.items``.  A **contrast** is a weighted sum of cell values (pp scale) or of cell logits (logit scale).  The
same contrast is exported as a weight vector over GLMM cell-mean coefficients (``glmm.cell_weights``) so that
lme4/emmeans, the Python VB fit and the cluster bootstrap all test the *identical* linear combination.

Confirmatory family (Holm, m = 4), adapted from T5 §2.3 to the approved problems (01_TECHNICAL_DESIGN §1):

====  =====================================================  ==========  ==============================================
H     contrast                                               data        source
====  =====================================================  ==========  ==============================================
H1    P2 language x verification, I2 = VP_vi - VP_en on the   model B     E1 §1, §5.5 (VP = ½[(Y+−Y−)+(N+−N−)]),
      parallel items, mean over sizes run in both languages  (mirror)    averaged over the sizes in both languages
H2    C2 method on native items: ½(Y+ + Y−) − ½(N+ + N−)      model A     user brief "arm contrasts Y± vs N±";
      on Vi-ExamMath Core, mean over models                  (VI grid)   T5 H1–H3 re-targeted from T to Y/N
H3    verification main effect on Core:                      model A     T5 H4 unchanged
      ½(Y+ + N+) − ½(Y− + N−), mean over models
H4    P3 coverage transfer: (Y − N) on certifiable Core      model A     01 §1 P3; within-run DiD, so the run effect
      items − (Y − N) on non-certifiable Core items                      cancels (T5 §2.4 logic)
====  =====================================================  ==========  ==============================================

Everything else (I3, language x method, per-size/per-model versions, capability-matched premium, arm x eval-type
interactions of T/M, single-arm pairs) is exploratory (BH-FDR).  The lead ratifies this family in the
pre-registration; the library is data, so changing it does not touch the inference code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schema import EvalData, filter_mask

CONFIRMATORY, EXPLORATORY = "confirmatory", "exploratory"


@dataclass(frozen=True)
class Term:
    weight: float
    runs: tuple          # sorted tuple of (column, value) pairs; value may be a tuple of allowed values
    items: tuple

    @staticmethod
    def make(weight: float, runs: dict, items: dict) -> "Term":
        def fz(d):
            return tuple(sorted((k, tuple(v) if isinstance(v, (list, set, tuple)) else v) for k, v in d.items()))
        return Term(float(weight), fz(runs), fz(items))

    @property
    def run_filter(self) -> dict:
        return dict(self.runs)

    @property
    def item_filter(self) -> dict:
        return dict(self.items)


@dataclass
class Contrast:
    name: str
    terms: list
    family: str = EXPLORATORY
    description: str = ""
    model: str = "A"                       # which GLMM the contrast belongs to (A = VI grid, B = mirror)
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------- cell evaluation
def cell_value(d: EvalData, term: Term) -> tuple[float, int, int]:
    """(value, n_runs, n_items) of one cell; value is nan if the cell is empty."""
    rm = filter_mask(d.runs, term.run_filter)
    im = filter_mask(d.items, term.item_filter)
    if not rm.any() or not im.any():
        return float("nan"), int(rm.sum()), int(im.sum())
    n = d.n[np.ix_(rm, im)]
    y = d.y[np.ix_(rm, im)]
    obs = n > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        per_run = np.where(obs, y / np.maximum(n, 1), 0).sum(1) / obs.sum(1)
    per_run = per_run[obs.any(1)]
    return (float(per_run.mean()) if per_run.size else float("nan")), int(rm.sum()), int(im.sum())


def estimate(d: EvalData, c: Contrast, scale: str = "pp") -> float:
    tot = 0.0
    for t in c.terms:
        v, _, _ = cell_value(d, t)
        if not np.isfinite(v):
            return float("nan")
        tot += t.weight * (v if scale == "pp" else float(np.log(np.clip(v, 1e-6, 1 - 1e-6)
                                                                    / np.clip(1 - v, 1e-6, 1))))
    return tot


def missing_cells(d: EvalData, c: Contrast) -> list[str]:
    out = []
    for t in c.terms:
        v, nr, ni = cell_value(d, t)
        if not np.isfinite(v):
            out.append(f"runs={t.run_filter} (n={nr}) items={t.item_filter} (n={ni})")
    return out


# -------------------------------------------------------------------------------------------- library builders
def _vals(d: EvalData, col: str, **flt) -> list:
    m = filter_mask(d.runs, flt)
    return sorted(set(d.runs.loc[m, col].dropna().tolist()))


def _has_arms(d: EvalData, arms, **flt) -> bool:
    have = set(_vals(d, "arm", **flt))
    return all(a in have for a in arms)


def _arm_terms(coef: dict, runs: dict, items: dict, scale: float = 1.0) -> list[Term]:
    return [Term.make(w * scale, {**runs, "arm": a}, items) for a, w in coef.items() if w != 0]


# coefficient vectors over the four Y/N arms
VP = {"Y+": .5, "Y-": -.5, "N+": .5, "N-": -.5}          # verification premium
METHOD = {"Y+": .5, "Y-": .5, "N+": -.5, "N-": -.5}      # symbolic minus LLM-native
VxM = {"Y+": 1, "Y-": -1, "N+": -1, "N-": 1}            # (Y+ − Y−) − (N+ − N−)
YN4 = ("Y+", "Y-", "N+", "N-")


def native_items(d: EvalData) -> dict:
    return {"evaltype": "native"}


def parallel_items(d: EvalData) -> dict:
    return {"evaltype": "parallel"}


def floor_excluded_models(d: EvalData, floor: float = 0.10, ref_arm: str = "S0") -> list[str]:
    """T5 §4.2: drop a model from pooled confirmatory contrasts if its S0 Core accuracy is < 10 %."""
    out = []
    for m in _vals(d, "model", lang="vi", arm=ref_arm):
        v, _, _ = cell_value(d, Term.make(1, {"lang": "vi", "arm": ref_arm, "model": m}, native_items(d)))
        if np.isfinite(v) and v < floor:
            out.append(m)
    return out


def mirror_models(d: EvalData) -> list[str]:
    """Models that have all four Y/N arms in both languages (the P2 cells)."""
    return [m for m in _vals(d, "model") if _has_arms(d, YN4, lang="vi", model=m) and
            _has_arms(d, YN4, lang="en", model=m)]


def vi_grid_models(d: EvalData, exclude: list[str] = ()) -> list[str]:
    return [m for m in _vals(d, "model", lang="vi") if _has_arms(d, YN4, lang="vi", model=m) and m not in exclude]


def _size_of(d: EvalData, model: str) -> str:
    return str(d.runs.loc[d.runs["model"] == model, "size"].iloc[0])


def i2_terms(models: list[str], items: dict) -> list[Term]:
    """VP_vi − VP_en averaged over models."""
    ts = []
    for m in models:
        ts += _arm_terms(VP, {"lang": "vi", "model": m}, items, 1 / len(models))
        ts += _arm_terms(VP, {"lang": "en", "model": m}, items, -1 / len(models))
    return ts


def library(d: EvalData, *, cap_pair: tuple[str, str] = ("1.7B", "0.6B"), floor: float = 0.10,
            include_exploratory: bool = True) -> tuple[list[Contrast], dict]:
    """Build the contrast list for the data at hand. Returns (contrasts, notes)."""
    notes: dict = {"skipped": {}}
    excl = floor_excluded_models(d, floor)
    notes["floor_excluded_models"] = excl
    mm = mirror_models(d)
    vm = vi_grid_models(d, excl)
    notes["mirror_models"], notes["vi_grid_models"] = mm, vm
    NAT, PAR = native_items(d), parallel_items(d)
    has_cert = "certifiable" in d.items and d.items.loc[filter_mask(d.items, NAT), "certifiable"].notna().any()
    C: list[Contrast] = []

    # ---------------- confirmatory
    if mm:
        C.append(Contrast("H1_P2_lang_x_verif", i2_terms(mm, PAR), CONFIRMATORY,
                          "I2 = VP_vi − VP_en on parallel items, mean over mirror models", "B"))
    else:
        notes["skipped"]["H1_P2_lang_x_verif"] = "no model has Y±/N± in both languages"
    if vm:
        k = 1 / len(vm)
        C.append(Contrast("H2_method_Y_vs_N_core",
                          sum((_arm_terms(METHOD, {"lang": "vi", "model": m}, NAT, k) for m in vm), []),
                          CONFIRMATORY, "½(Y+ + Y−) − ½(N+ + N−) on Core, mean over VI models", "A"))
        C.append(Contrast("H3_verif_core",
                          sum((_arm_terms(VP, {"lang": "vi", "model": m}, NAT, k) for m in vm), []),
                          CONFIRMATORY, "½(Y+ + N+) − ½(Y− + N−) on Core, mean over VI models", "A"))
        if has_cert:
            ts = []
            for m in vm:
                ts += _arm_terms(METHOD, {"lang": "vi", "model": m}, {**NAT, "certifiable": True}, k)
                ts += _arm_terms(METHOD, {"lang": "vi", "model": m}, {**NAT, "certifiable": False}, -k)
            C.append(Contrast("H4_P3_method_x_certifiable", ts, CONFIRMATORY,
                              "(Y − N) on certifiable Core items − (Y − N) on non-certifiable Core items", "A"))
        else:
            notes["skipped"]["H4_P3_method_x_certifiable"] = "items carry no `certifiable` flag"
    else:
        for h in ("H2_method_Y_vs_N_core", "H3_verif_core", "H4_P3_method_x_certifiable"):
            notes["skipped"][h] = "no VI model with all four Y/N arms"
    if not include_exploratory:
        return C, notes

    # ---------------- exploratory (BH-FDR)
    E = []
    if mm:
        E.append(Contrast("I3_lang_x_method_x_verif",
                          sum(([*_arm_terms(VxM, {"lang": "vi", "model": m}, PAR, 1 / len(mm)),
                                *_arm_terms(VxM, {"lang": "en", "model": m}, PAR, -1 / len(mm))] for m in mm), []),
                          description="[(Y+−Y−)−(N+−N−)]_vi − [same]_en", model="B"))
        E.append(Contrast("lang_x_method",
                          sum(([*_arm_terms(METHOD, {"lang": "vi", "model": m}, PAR, 1 / len(mm)),
                                *_arm_terms(METHOD, {"lang": "en", "model": m}, PAR, -1 / len(mm))] for m in mm), []),
                          description="[Y − N]_vi − [Y − N]_en on parallel items", model="B"))
        for m in mm:
            E.append(Contrast(f"I2[{m}]", i2_terms([m], PAR), description=f"I2 for {m}", model="B"))
        by_size = {_size_of(d, m): m for m in mm}
        hi, lo = cap_pair
        if hi in by_size and lo in by_size:
            E.append(Contrast("cap_VPvi_hi_minus_VPen_lo",
                              [*_arm_terms(VP, {"lang": "vi", "model": by_size[hi]}, PAR),
                               *_arm_terms(VP, {"lang": "en", "model": by_size[lo]}, PAR, -1)],
                              description=f"capability-matched VP_vi({hi}) − VP_en({lo})", model="B"))
        for lang in ("vi", "en"):
            E.append(Contrast(f"VP_{lang}_parallel",
                              sum((_arm_terms(VP, {"lang": lang, "model": m}, PAR, 1 / len(mm)) for m in mm), []),
                              description=f"verification premium in {lang} on parallel items", model="B"))
    if vm:
        for m in vm:
            E.append(Contrast(f"H2[{m}]", _arm_terms(METHOD, {"lang": "vi", "model": m}, NAT), model="A"))
            E.append(Contrast(f"H3[{m}]", _arm_terms(VP, {"lang": "vi", "model": m}, NAT), model="A"))
        for a, b in (("Y+", "Y-"), ("N+", "N-")):
            E.append(Contrast(f"{a}_vs_{b}_core", sum((_arm_terms({a: 1 / len(vm), b: -1 / len(vm)},
                                                                  {"lang": "vi", "model": m}, NAT) for m in vm), []),
                              model="A"))
        # VP on Core vs VP on the parallel set (descriptive; different item sets)
        E.append(Contrast("VP_vi_core_minus_parallel",
                          sum(([*_arm_terms(VP, {"lang": "vi", "model": m}, NAT, 1 / len(vm)),
                                *_arm_terms(VP, {"lang": "vi", "model": m}, PAR, -1 / len(vm))] for m in vm), []),
                          description="arm-pair x eval-type: VP on Core − VP on parallel", model="A"))
    # T5-style arm x eval-type interactions for baseline arms (T, M, T-off, S0) vs Y+/N+
    par_ds = sorted(d.items.loc[filter_mask(d.items, PAR), "dataset"].unique())
    vi_models = [m for m in _vals(d, "model", lang="vi") if m not in excl]
    for a, b in (("T", "N+"), ("T", "Y+"), ("T", "M"), ("S0", "Y+"), ("S0", "N+"), ("T-off", "T")):
        ms = [m for m in vi_models if _has_arms(d, (a, b), lang="vi", model=m)]
        if not ms or not par_ds:
            continue
        ts = []
        for m in ms:
            k = 1 / len(ms)
            ts += _arm_terms({a: 1, b: -1}, {"lang": "vi", "model": m}, NAT, k)
            for ds in par_ds:
                ts += _arm_terms({a: 1, b: -1}, {"lang": "vi", "model": m}, {"dataset": ds}, -k / len(par_ds))
        E.append(Contrast(f"{a}_vs_{b}_x_evaltype", ts,
                          description=f"[{a}−{b} on Core] − mean_ds [{a}−{b} on translated ds] (T5 H-form)",
                          model="A"))
    for c in E:
        c.family = EXPLORATORY
    return C + E, notes
