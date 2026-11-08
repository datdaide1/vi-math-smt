"""Power simulations, ported from T5 Appendix A (arm x eval-type DiD within runs) and E1 Appendix A (P2
language x verification between runs).  numpy only; chunked so peak memory stays < ~150 MB.

Both simulators take an explicit ``rng`` so results are reproducible and tests can run tiny replicate counts.
CLI::

    python -m vi_math_verified_aug.analysis.power t5 --pN .45 --pT .55 --delta .05 --nN 500 --S 3 --K 4
    python -m vi_math_verified_aug.analysis.power mirror --sizes 0.6B 1.7B --S 5 --s_run 0.122
    python -m vi_math_verified_aug.analysis.power pilot --s_run 0.10 --s_re 0.07 --s_item 1.5 --tau 0.5
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from .stats_util import norm_cdf, norm_ppf

SQ2 = math.sqrt(2.0)
GH_x, GH_w = np.polynomial.hermite.hermgauss(60)
Z80 = norm_ppf(0.8)


def marg(b: float, sd: float) -> float:
    z = SQ2 * GH_x
    return float(np.sum(GH_w / (1 + np.exp(-(b + sd * z)))) / math.sqrt(math.pi))


def solve_b(p: float, sd: float) -> float:
    lo, hi = -15.0, 15.0
    for _ in range(80):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if marg(m, sd) < p else (lo, m)
    return (lo + hi) / 2


def zcrit(alpha: float) -> float:
    return norm_ppf(1 - alpha / 2)


def mde(sd: float, alpha: float = 0.0125, power: float = 0.8) -> float:
    return (zcrit(alpha) + norm_ppf(power)) * sd


def tost_power(sd: float, margin: float = 0.03, alpha: float = 0.05) -> float:
    """P(TOST declares equivalence | true effect 0) = 2 Φ(margin/sd − z_{1−α}) − 1 (E1 App. D)."""
    return max(0.0, 2 * norm_cdf(margin / sd - norm_ppf(1 - alpha)) - 1)


# --------------------------------------------------------------------------------------------------- T5 sim
def simulate_t5(pN, pT, Delta, nN, nT, S, K, *, s_item=1.5, tau=0.5, s_run=0.10, s_re=0.07, reps=2000,
                chunk=100, rng=None):
    """T5 App. A: arm A vs B, DiD = mean_native(y_A − y_B) − mean_trans(y_A − y_B), per model size.

    Returns (estimates, feasible SEs), both in proportion units.
    """
    rng = rng if rng is not None else np.random.default_rng(20261107)
    sd_tot = math.sqrt(s_item ** 2 + (tau / 2) ** 2 + s_run ** 2 + s_re ** 2)
    bN, bT = solve_b(pN, sd_tot), solve_b(pT, sd_tot)
    dN = solve_b(pN + Delta, sd_tot) - bN
    n = nN + nT
    b = np.r_[np.full(nN, bN), np.full(nT, bT)]
    isN = np.r_[np.ones(nN, bool), np.zeros(nT, bool)]
    c = np.array([0.5, -0.5]).reshape(1, 2, 1, 1)
    delta = np.zeros((1, 2, 1, n))
    delta[0, 0, 0, isN] = dN
    est, feas, done = [], [], 0
    while done < reps:
        R = min(chunk, reps - done)
        done += R
        u = rng.normal(0, s_item, (R, 1, 1, n))
        g = rng.normal(0, tau, (R, 1, 1, n))
        v = rng.normal(0, s_run, (R, 2, S, 1))
        w = np.where(isN.reshape(1, 1, 1, n), rng.normal(0, s_re, (R, 2, S, 1)), rng.normal(0, s_re, (R, 2, S, 1)))
        p = 1 / (1 + np.exp(-(b.reshape(1, 1, 1, n) + u + c * g + delta + v + w)))
        y = rng.binomial(K, p) / K
        d = y[:, 0] - y[:, 1]
        Is = d[:, :, isN].mean(axis=2) - d[:, :, ~isN].mean(axis=2)
        est.append(Is.mean(axis=1))
        Di = d.mean(axis=1)
        se2 = Di[:, isN].var(axis=1, ddof=1) / nN + Di[:, ~isN].var(axis=1, ddof=1) / nT
        if S >= 2:
            se2 = se2 + Is.var(axis=1, ddof=1) / S
        feas.append(np.sqrt(se2))
    return np.concatenate(est), np.concatenate(feas)


def power_t5(pN, pT, Delta, nN, nT, S, K, alpha=0.0125, **kw) -> dict:
    est, se = simulate_t5(pN, pT, Delta, nN, nT, S, K, **kw)
    sd = est.std(ddof=1)
    z = zcrit(alpha)
    return {"mean": float(est.mean()), "sd": float(sd), "power_oracle": float(np.mean(np.abs(est) / sd > z)),
            "power_feasible": float(np.mean(np.abs(est) / se > z)), "mde80": float(mde(sd, alpha))}


# ------------------------------------------------------------------------------------------- E1 mirror sim
ARMS = ("Y+", "Y-", "N+", "N-")
PLUS = np.array([1, 0, 1, 0], bool)
MIRROR_ACC = {"0.6B": (0.31, 0.49), "1.7B": (0.49, 0.63), "4B": (0.63, 0.74)}   # (vi, en) of unverified arms


def simulate_mirror(sizes, S_vi, S_en, *, K=4, n=1000, reps=2000, chunk=20, s_item=1.5, s_lang=0.5, tau=0.5,
                    rho_l=0.5, rho_s=0.5, s_run=0.122, rho_crn=0.0, prem_en=0.02, D2=0.0, D3=0.0,
                    S_by_size=None, acc=None, rng=None) -> dict:
    """E1 App. A ``simulate``: returns draws of I2/I3 per size, pooled, and the capability-matched contrast."""
    rng = rng if rng is not None else np.random.default_rng(20261107)
    acc = acc or MIRROR_ACC
    sizes = list(sizes)
    M = len(sizes)
    if S_by_size is None:
        S_by_size = [(S_vi, S_en)] * M
    Smax = max(max(a, b) for a, b in S_by_size)
    ta = tau / SQ2
    sd_tot = math.sqrt(s_item ** 2 + s_lang ** 2 + ta ** 2 + s_run ** 2)
    off = np.zeros((M, 2, 4))
    for mi, sz in enumerate(sizes):
        for li, p in enumerate(acc[sz]):
            b = solve_b(p, sd_tot)
            for ai, a in enumerate(ARMS):
                if not PLUS[ai]:
                    off[mi, li, ai] = b
                    continue
                prem = prem_en + (D2 if li == 0 else 0.0)
                if li == 0:
                    prem += D3 / 2 if a[0] == "Y" else -D3 / 2
                off[mi, li, ai] = solve_b(min(p + prem, 0.99), sd_tot)
    w0, wL, wm = rho_l * rho_s, rho_s * (1 - rho_l), rho_l * (1 - rho_s)
    wi = max(0.0, 1 - w0 - wL - wm)
    mask = np.zeros((M, 2, Smax), bool)
    for mi, (sv, se) in enumerate(S_by_size):
        mask[mi, 0, :sv] = True
        mask[mi, 1, :se] = True
    keys = ["I2_pool", "I3_pool"] + [f"I2_{s}" for s in sizes] + [f"I3_{s}" for s in sizes]
    if "0.6B" in sizes and "1.7B" in sizes:
        keys.append("cap")
    out = {k: [] for k in keys}
    done = 0
    while done < reps:
        R = min(chunk, reps - done)
        done += R
        u = rng.normal(0, s_item, (R, 1, 1, 1, 1, n))
        e = rng.normal(0, s_lang, (R, 1, 2, 1, 1, n))
        sl = ta * (math.sqrt(w0) * rng.normal(size=(R, 1, 1, 4, 1, n))
                   + math.sqrt(wL) * rng.normal(size=(R, 1, 2, 4, 1, n))
                   + math.sqrt(wm) * rng.normal(size=(R, M, 1, 4, 1, n))
                   + math.sqrt(wi) * rng.normal(size=(R, M, 2, 4, 1, n)))
        zc = rng.normal(size=(R, M, 1, 4, Smax, 1))
        zl = rng.normal(size=(R, M, 2, 4, Smax, 1))
        v = s_run * (math.sqrt(rho_crn) * zc + math.sqrt(1 - rho_crn) * zl)
        p = 1 / (1 + np.exp(-(off.reshape(1, M, 2, 4, 1, 1) + u + e + sl + v)))
        y = rng.binomial(K, p).mean(axis=-1) / K
        y = np.where(mask.reshape(1, M, 2, 1, Smax), y, np.nan)
        cell = np.nanmean(y, axis=-1)
        dY = cell[..., 0] - cell[..., 1]
        dN = cell[..., 2] - cell[..., 3]
        VP = (dY + dN) / 2
        I2 = VP[:, :, 0] - VP[:, :, 1]
        I3 = (dY - dN)[:, :, 0] - (dY - dN)[:, :, 1]
        for mi, sz in enumerate(sizes):
            out[f"I2_{sz}"].append(I2[:, mi])
            out[f"I3_{sz}"].append(I3[:, mi])
        out["I2_pool"].append(I2.mean(axis=1))
        out["I3_pool"].append(I3.mean(axis=1))
        if "cap" in out:
            out["cap"].append(VP[:, sizes.index("1.7B"), 0] - VP[:, sizes.index("0.6B"), 1])
    return {k: np.concatenate(v) for k, v in out.items()}


def summarize(res: dict) -> dict:
    out = {}
    for k, v in res.items():
        sd = float(v.std(ddof=1))
        out[k] = {"mean_pp": 100 * float(v.mean()), "sd_pp": 100 * sd, "mde80_05_pp": 100 * mde(sd, 0.05),
                  "mde80_0125_pp": 100 * mde(sd, 0.0125), "tost3_power": tost_power(sd)}
    return out


# --------------------------------------------------------------------------------------------- pilot rule
def pilot_rule(s_run: float, s_re: float, s_item: float = 1.5, tau: float = 0.5, *, reps: int = 400,
               rng=None) -> dict:
    """T5 §2.4 + E1 §5.4 pilot rules, re-simulated with the measured variance components.

    ``s_run`` = run main effect SD, ``s_re`` = run x eval-type SD (logit).  σ_run(mirror) = √(s_run² + s_re²).
    """
    rng = rng if rng is not None else np.random.default_rng(20261107)
    s_mirror = math.sqrt(s_run ** 2 + s_re ** 2)
    out = {"inputs": {"s_run": s_run, "s_re": s_re, "s_item": s_item, "tau": tau, "s_run_mirror": s_mirror}}
    # T5: per-model 2B DiD at the planned spec (n_N = 500, n_T = 1000, S = 3 and 5, K = 4); pooled over 3 sizes
    # approximated as SD / sqrt(3) (T5 Result 3 with rho = 1 gives ~ this; conservative vs rho = 0.5)
    t5 = {}
    for S in (3, 5):
        r = power_t5(0.45, 0.55, 0.0, 500, 1000, S, 4, s_item=s_item, tau=tau, s_run=s_run, s_re=s_re, reps=reps,
                     rng=rng)
        t5[f"S{S}"] = {"sd_pp_2B": 100 * r["sd"], "mde80_2B_pp": 100 * r["mde80"],
                       "mde80_pooled3_pp": 100 * mde(r["sd"] / math.sqrt(3))}
    out["t5"] = t5
    rec = []
    if t5["S3"]["mde80_pooled3_pp"] > 4.0:
        rec.append("T5: pooled MDE > 4 pp at S=3 -> raise S to 5 everywhere")
    if s_re >= 0.12:
        rec.append("T5: sigma_run x eval >= 0.12 logit -> S = 8-10 in the 2B primary cells")
    # E1 mirror: O3 with the measured sigma_run(mirror)
    o3 = summarize(simulate_mirror(["0.6B", "1.7B"], 5, 5, s_run=s_mirror, s_item=s_item, tau=tau, reps=reps,
                                   rng=rng))
    out["e1_O3"] = {k: o3[k] for k in ("I2_pool", "I3_pool", "cap")}
    if s_mirror <= 0.08:
        rec.append("E1: sigma_run(mirror) <= 0.08 -> O3b (S = 3), saves ~54 T4-h")
    elif s_mirror < 0.15:
        rec.append("E1: 0.08 < sigma_run(mirror) < 0.15 -> keep O3 (S = 5)")
    elif s_mirror < 0.20:
        rec.append("E1: sigma_run(mirror) >= 0.15 -> O7 (0.6B at S = 8), +30 T4-h")
    else:
        rec.append("E1: sigma_run(mirror) >= 0.20 -> O7 plus 1.7B at S = 8, +~70 T4-h")
    out["recommendations"] = rec
    return out


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="power")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("t5")
    for k, v in (("pN", .45), ("pT", .55), ("delta", .05), ("alpha", .0125), ("s_item", 1.5), ("tau", .5),
                 ("s_run", .10), ("s_re", .07)):
        a.add_argument(f"--{k}", type=float, default=v)
    for k, v in (("nN", 500), ("nT", 1000), ("S", 3), ("K", 4), ("reps", 1500)):
        a.add_argument(f"--{k}", type=int, default=v)
    b = sub.add_parser("mirror")
    b.add_argument("--sizes", nargs="+", default=["0.6B", "1.7B"])
    for k, v in (("S", 5), ("K", 4), ("n", 1000), ("reps", 2000)):
        b.add_argument(f"--{k}", type=int, default=v)
    for k, v in (("s_run", .122), ("D2", 0.0), ("rho_crn", 0.0), ("s_item", 1.5), ("tau", .5)):
        b.add_argument(f"--{k}", type=float, default=v)
    c = sub.add_parser("pilot")
    for k, v in (("s_run", .10), ("s_re", .07), ("s_item", 1.5), ("tau", .5)):
        c.add_argument(f"--{k}", type=float, default=v)
    c.add_argument("--reps", type=int, default=400)
    args = ap.parse_args(argv)
    import json
    if args.cmd == "t5":
        print(json.dumps(power_t5(args.pN, args.pT, args.delta, args.nN, args.nT, args.S, args.K, args.alpha,
                                  s_item=args.s_item, tau=args.tau, s_run=args.s_run, s_re=args.s_re,
                                  reps=args.reps), indent=1))
    elif args.cmd == "mirror":
        r = simulate_mirror(args.sizes, args.S, args.S, K=args.K, n=args.n, reps=args.reps, s_run=args.s_run,
                            D2=args.D2, rho_crn=args.rho_crn, s_item=args.s_item, tau=args.tau)
        print(json.dumps(summarize(r), indent=1))
    else:
        print(json.dumps(pilot_rule(args.s_run, args.s_re, args.s_item, args.tau, reps=args.reps), indent=1))


if __name__ == "__main__":
    _main()
