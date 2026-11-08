"""Bayesian GLMM sensitivity fit (T5 §2.2 table: "Bayesian GLMM (brms/NumPyro) ... sensitivity analysis for H1"),
NumPyro NUTS on GPU.  Same cell-means fixed effects and random-effect structure as the lme4 model, so the posterior
of every contrast is directly comparable with the lme4 Wald result (``glmm_lme4.R``).

    beta_cell ~ N(0, 2.5²)                                     weakly informative, logit scale
    sd_k ~ HalfNormal(1),  z_k ~ N(0, 1),  re_k = sd_k z_k     non-centred, one per random-effect term
    correct ~ Binomial(K, logit = beta[cell] + Σ_k re_k[level_k])

Reported per contrast: posterior mean / SD / 95 % CrI on the logit and (conditional) pp scale and P(H > 0).
Never run on the laptop (JAX is not installed there); see ``modal_jobs.py``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .contrasts import Contrast
from .glmm import RE_ORDER, cell_labels, cell_weights, long_table, model_subset
from .schema import EvalData


def _design(dd: EvalData, structure: list[str]):
    lt = long_table(dd)
    _, levels = cell_labels(dd)
    cell = pd.Categorical(lt["cell"], categories=levels).codes
    re = []
    for term in structure:
        parts = term.split(":")
        key = lt[parts[0]].astype(str)
        for p in parts[1:]:
            key = key + "|" + lt[p].astype(str)
        codes, uniq = pd.factorize(key)
        if len(uniq) > 1:
            re.append((term, codes.astype(np.int32), len(uniq)))
    return lt, levels, cell.astype(np.int32), re


def fit_numpyro(d: EvalData, contrasts: list[Contrast], which: str = "A", structure: list[str] | None = None,
                num_warmup: int = 500, num_samples: int = 500, num_chains: int = 2, seed: int = 0,
                max_tree_depth: int = 8) -> dict:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    dd = model_subset(d, which)
    structure = structure or RE_ORDER[which][0][1]
    lt, levels, cell, re = _design(dd, structure)
    y = jnp.asarray(lt["correct"].to_numpy(), jnp.float32)
    K = jnp.asarray(lt["K"].to_numpy(), jnp.float32)
    cell_j = jnp.asarray(cell)
    re_j = [(name, jnp.asarray(codes), L) for name, codes, L in re]

    def model():
        beta = numpyro.sample("beta", dist.Normal(0.0, 2.5).expand([len(levels)]).to_event(1))
        eta = beta[cell_j]
        for name, codes, L in re_j:
            sd = numpyro.sample(f"sd[{name}]", dist.HalfNormal(1.0))
            z = numpyro.sample(f"z[{name}]", dist.Normal(0.0, 1.0).expand([L]).to_event(1))
            eta = eta + sd * z[codes]
        numpyro.sample("y", dist.Binomial(total_count=K, logits=eta), obs=y)

    t0 = time.perf_counter()
    mcmc = MCMC(NUTS(model, max_tree_depth=max_tree_depth, target_accept_prob=0.85), num_warmup=num_warmup,
                num_samples=num_samples, num_chains=num_chains, chain_method="vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    secs = time.perf_counter() - t0
    post = mcmc.get_samples(group_by_chain=True)
    beta = np.asarray(post["beta"]).reshape(-1, len(levels))
    rows = []
    for c in [c for c in contrasts if c.model == which]:
        w = cell_weights(dd, c, levels)
        lg = beta @ w
        pp = (1 / (1 + np.exp(-beta))) @ w
        rows.append({"name": c.name, "family": c.family, "estimate": float(lg.mean()), "sd": float(lg.std(ddof=1)),
                     "ci_lo": float(np.quantile(lg, .025)), "ci_hi": float(np.quantile(lg, .975)),
                     "p_gt0": float((lg > 0).mean()), "est_pp": float(pp.mean()), "sd_pp": float(pp.std(ddof=1))})
    sds = {k: float(np.asarray(v).mean()) for k, v in post.items() if k.startswith("sd[")}
    # split-R-hat on the contrast-relevant parameters and the SDs
    from numpyro.diagnostics import split_gelman_rubin
    rhat_beta = float(np.max(np.asarray(split_gelman_rubin(np.asarray(post["beta"])))))
    rhat_sd = {k: float(split_gelman_rubin(np.asarray(v))) for k, v in post.items() if k.startswith("sd[")}
    return {"model": which, "contrasts": rows, "sd_posterior_mean": sds, "rhat_beta_max": rhat_beta,
            "rhat_sd": rhat_sd, "seconds": secs, "n_obs": int(len(lt)), "n_cells": len(levels),
            "re_levels": {name: L for name, _, L in re}, "device": str(jax.devices()[0])}
