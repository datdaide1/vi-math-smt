"""GPU (JAX) versions of the heavy numerics: the two-way cluster bootstrap and the T5 / E1 power simulations.

Same algorithms as ``bootstrap.py`` and ``power.py`` (which stay the numpy reference implementations); only the
random streams differ, so GPU vs reference agreement is checked statistically (``modal_jobs.py``).  JAX is
imported lazily: the laptop has no JAX and never runs this module.  float32 throughout.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .bootstrap import BootResult, _Prepared
from .contrasts import Contrast
from .schema import EvalData
from .stats_util import two_sided_p

LAST_TIMING: dict = {}   # first chunk (includes XLA compile) vs the remaining chunks, seconds


def _run_chunks(f, key, total, chunk, name):
    """Call the jitted chunk function; record compile-inclusive first-chunk time and steady-state time."""
    jax, _ = _jax()
    out, done, t_first, t0 = [], 0, None, time.perf_counter()
    while done < total:
        b = min(chunk, total - done)
        key, sk = jax.random.split(key)
        r = f(sk, b)
        r = jax.tree_util.tree_map(np.asarray, r)        # blocks until the device finishes
        out.append(r)
        done += b
        if t_first is None:
            t_first = time.perf_counter() - t0
            t1 = time.perf_counter()
    LAST_TIMING[name] = {"first_chunk_s": t_first, "rest_s": time.perf_counter() - t1, "chunks": len(out),
                         "chunk": chunk, "total": total}
    return out


def _jax():
    import jax
    import jax.numpy as jnp
    return jax, jnp


# ------------------------------------------------------------------------------------------- bootstrap (GPU)
def cluster_bootstrap_jax(d: EvalData, contrasts: list[Contrast], *, B: int = 10_000, seed: int = 20261107,
                          scale: str = "pp", paired_seeds: bool = False, chunk: int = 2000,
                          conf: float = 0.95, keep_draws: bool = False) -> list[BootResult]:
    jax, jnp = _jax()
    prep = _Prepared(d, contrasts)
    est = prep.contrasts(prep.cell_values(np.ones((1, len(d.items))), np.ones((1, len(d.runs)))), scale)[0]
    accz = jnp.asarray(prep.accz, jnp.float32)
    obs = jnp.asarray(prep.obs, jnp.float32)
    I, R = accz.shape[1], accz.shape[0]
    ds = d.items["dataset"].to_numpy()
    item_blocks = [np.flatnonzero(ds == name) for name in np.unique(ds)]
    strata, unit = (("arm", "model"), "seed") if paired_seeds else (("lang", "arm", "model"), "run_id")
    keys = d.runs[list(strata)].astype(str).agg("|".join, axis=1).to_numpy()
    units = d.runs[unit].astype(str).to_numpy()
    run_blocks = []
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        uv, inv = np.unique(units[idx], return_inverse=True)
        run_blocks.append((idx, len(uv), inv))
    gmasks = [jnp.asarray(np.flatnonzero(m)) for m in prep.group_masks]
    cells = [(jnp.asarray(np.flatnonzero(rm)), g) for rm, g in prep.cell_spec]
    C = np.zeros((len(prep.cell_spec), len(contrasts)), np.float32)
    for j, tl in enumerate(prep.contrast_terms):
        for w, ci in tl:
            C[ci, j] += w
    Cm = jnp.asarray(C)

    def counts(key, b, n):
        draws = jax.random.randint(key, (b, n), 0, n)
        return jax.vmap(lambda r: jnp.bincount(r, length=n))(draws).astype(jnp.float32)

    def one_chunk(key, b):
        ks = jax.random.split(key, len(item_blocks) + len(run_blocks))
        iw = jnp.zeros((b, I), jnp.float32)
        for k, idx in zip(ks, item_blocks):
            iw = iw.at[:, idx].set(counts(k, b, len(idx)))
        rw = jnp.zeros((b, R), jnp.float32)
        for k, (idx, nu, inv) in zip(ks[len(item_blocks):], run_blocks):
            rw = rw.at[:, idx].set(counts(k, b, nu)[:, inv])
        per_group = []
        for gm in gmasks:
            w = iw[:, gm]
            per_group.append((w @ accz[:, gm].T) / (w @ obs[:, gm].T))
        cv = []
        for rm, g in cells:
            a = per_group[g][:, rm]
            ww = rw[:, rm] * jnp.isfinite(a)
            cv.append(jnp.sum(jnp.where(ww > 0, a, 0.0) * ww, 1) / jnp.sum(ww, 1))
        cv = jnp.stack(cv, 1)
        if scale == "logit":
            p = jnp.clip(cv, 1e-6, 1 - 1e-6)
            cv = jnp.log(p / (1 - p))
        return cv @ Cm

    f = jax.jit(one_chunk, static_argnums=1)
    out = _run_chunks(f, jax.random.PRNGKey(seed), B, chunk, "bootstrap")
    T = np.vstack(out).astype(np.float64)
    res, a = [], 1 - conf
    for j, c in enumerate(contrasts):
        t = T[:, j][np.isfinite(T[:, j])]
        se = float(t.std(ddof=1))
        lo, hi = np.quantile(t, [a / 2, 1 - a / 2])
        lo90, hi90 = np.quantile(t, [0.05, 0.95])
        pp = float(max(min(2 * min((t <= 0).mean(), (t >= 0).mean()), 1.0), 1 / t.size))
        z = est[j] / se if se > 0 else np.nan
        res.append(BootResult(c.name, float(est[j]), se, float(lo), float(hi), float(two_sided_p(z)), pp,
                              float(lo90), float(hi90), t if keep_draws else None))
    return res


# ------------------------------------------------------------------------------------------- power (GPU)
def simulate_t5_jax(pN, pT, Delta, nN, nT, S, K, *, s_item=1.5, tau=0.5, s_run=0.10, s_re=0.07, reps=2000,
                    chunk=500, seed=20261107):
    from .power import solve_b
    jax, jnp = _jax()
    sd_tot = math.sqrt(s_item ** 2 + (tau / 2) ** 2 + s_run ** 2 + s_re ** 2)
    bN, bT = solve_b(pN, sd_tot), solve_b(pT, sd_tot)
    dN = solve_b(pN + Delta, sd_tot) - bN
    n = nN + nT
    isN = jnp.asarray(np.r_[np.ones(nN, bool), np.zeros(nT, bool)])
    b = jnp.asarray(np.r_[np.full(nN, bN), np.full(nT, bT)], jnp.float32)
    c = jnp.asarray([0.5, -0.5], jnp.float32).reshape(1, 2, 1, 1)
    delta = jnp.zeros((1, 2, 1, n), jnp.float32).at[0, 0, 0, :nN].set(dN)

    def run(key, R):
        k = jax.random.split(key, 6)
        u = s_item * jax.random.normal(k[0], (R, 1, 1, n))
        g = tau * jax.random.normal(k[1], (R, 1, 1, n))
        v = s_run * jax.random.normal(k[2], (R, 2, S, 1))
        w = jnp.where(isN.reshape(1, 1, 1, n), s_re * jax.random.normal(k[3], (R, 2, S, 1)),
                      s_re * jax.random.normal(k[4], (R, 2, S, 1)))
        p = jax.nn.sigmoid(b.reshape(1, 1, 1, n) + u + c * g + delta + v + w)
        y = jax.random.binomial(k[5], K, p) / K
        dd = y[:, 0] - y[:, 1]
        Is = dd[:, :, :nN].mean(2) - dd[:, :, nN:].mean(2)
        Di = dd.mean(1)
        se2 = Di[:, :nN].var(1, ddof=1) / nN + Di[:, nN:].var(1, ddof=1) / nT
        if S >= 2:
            se2 = se2 + Is.var(1, ddof=1) / S
        return Is.mean(1), jnp.sqrt(se2)

    f = jax.jit(run, static_argnums=1)
    out = _run_chunks(f, jax.random.PRNGKey(seed), reps, chunk, "t5")
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def simulate_mirror_jax(sizes, S, *, K=4, n=1000, reps=2000, chunk=200, s_item=1.5, s_lang=0.5, tau=0.5,
                        rho_l=0.5, rho_s=0.5, s_run=0.122, rho_crn=0.0, prem_en=0.02, D2=0.0, seed=20261107):
    """E1 simulate() with equal seeds per cell (S_vi = S_en = S): pooled and per-size I2, I3."""
    from .power import ARMS, MIRROR_ACC, PLUS, solve_b
    jax, jnp = _jax()
    sizes = list(sizes)
    M = len(sizes)
    ta = tau / math.sqrt(2)
    sd_tot = math.sqrt(s_item ** 2 + s_lang ** 2 + ta ** 2 + s_run ** 2)
    off = np.zeros((M, 2, 4))
    for mi, sz in enumerate(sizes):
        for li, p in enumerate(MIRROR_ACC[sz]):
            bb = solve_b(p, sd_tot)
            for ai, a in enumerate(ARMS):
                off[mi, li, ai] = bb if not PLUS[ai] else solve_b(min(p + prem_en + (D2 if li == 0 else 0), .99),
                                                                  sd_tot)
    off = jnp.asarray(off, jnp.float32).reshape(1, M, 2, 4, 1, 1)
    w0, wL, wm = rho_l * rho_s, rho_s * (1 - rho_l), rho_l * (1 - rho_s)
    wi = max(0.0, 1 - w0 - wL - wm)

    def run(key, R):
        k = jax.random.split(key, 9)
        nrm = jax.random.normal
        u = s_item * nrm(k[0], (R, 1, 1, 1, 1, n))
        e = s_lang * nrm(k[1], (R, 1, 2, 1, 1, n))
        sl = ta * (math.sqrt(w0) * nrm(k[2], (R, 1, 1, 4, 1, n)) + math.sqrt(wL) * nrm(k[3], (R, 1, 2, 4, 1, n))
                   + math.sqrt(wm) * nrm(k[4], (R, M, 1, 4, 1, n)) + math.sqrt(wi) * nrm(k[5], (R, M, 2, 4, 1, n)))
        v = s_run * (math.sqrt(rho_crn) * nrm(k[6], (R, M, 1, 4, S, 1))
                     + math.sqrt(1 - rho_crn) * nrm(k[7], (R, M, 2, 4, S, 1)))
        p = jax.nn.sigmoid(off + u + e + sl + v)
        cell = (jax.random.binomial(k[8], K, p).mean(-1) / K).mean(-1)       # (R, M, L, A)
        dY = cell[..., 0] - cell[..., 1]
        dN = cell[..., 2] - cell[..., 3]
        I2 = ((dY + dN) / 2)[:, :, 0] - ((dY + dN) / 2)[:, :, 1]
        I3 = (dY - dN)[:, :, 0] - (dY - dN)[:, :, 1]
        return I2, I3

    f = jax.jit(run, static_argnums=1)
    out = _run_chunks(f, jax.random.PRNGKey(seed), reps, chunk, "mirror")
    I2, I3 = np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])
    out = {"I2_pool": I2.mean(1), "I3_pool": I3.mean(1)}
    for mi, sz in enumerate(sizes):
        out[f"I2_{sz}"] = I2[:, mi]
    return out


def timed(fn, *a, **kw):
    t = time.perf_counter()
    r = fn(*a, **kw)
    return r, time.perf_counter() - t
