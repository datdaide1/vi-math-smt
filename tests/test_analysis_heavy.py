"""Y2 analysis -- simulation-scale checks.  NOT run on the laptop: they run on Modal via
``modal run vi_math_verified_aug/analysis/modal_jobs.py`` (which sets VIMATH_HEAVY=1).

Each test prints ``METRIC {json}`` lines that the Modal job collects into logs/api/Y2/.
"""
import json
import math
import os

import numpy as np
import pytest

from vi_math_verified_aug.analysis import bootstrap, contrasts, irt, power, synth, variance

pytestmark = pytest.mark.skipif(os.environ.get("VIMATH_HEAVY") != "1",
                                reason="simulation-scale test; runs on Modal (VIMATH_HEAVY=1)")
CONF = ["H1_P2_lang_x_verif", "H2_method_Y_vs_N_core", "H3_verif_core", "H4_P3_method_x_certifiable"]


def metric(name, **kw):
    print("METRIC " + json.dumps({"test": name, **kw}, default=float), flush=True)


def cov_spec(**kw):
    base = dict(vi_arms=("Y+", "Y-", "N+", "N-"), seeds_vi=3, seeds_en=3, n_core=120,
                n_par={"gsm8k_platinum": 80, "math_algprealg": 60}, K=4)
    base.update(kw)
    return synth.SynthSpec(**base)


def test_planted_effects_recovered_full_design():
    spec = synth.SynthSpec(d_i2=0.4, m_eff=0.3, h4=0.4, seeds_vi=5, seeds_en=5, n_core=300,
                           n_par={"gsm8k_platinum": 300, "math_algprealg": 200})
    d = synth.generate(spec, seed=11)
    C, _ = contrasts.library(d, include_exploratory=False)
    res = bootstrap.cluster_bootstrap(d, C, B=4000)
    for c, r in zip(C, res):
        t = synth.truth(d, c)
        metric("planted_recovery", contrast=c.name, truth_pp=t, est_pp=r.estimate, lo=r.ci_lo, hi=r.ci_hi,
               p=r.p_wald)
        assert r.ci_lo < t < r.ci_hi, c.name
        assert r.ci_lo > 0, f"{c.name}: planted effect not detected"


@pytest.mark.parametrize("planted", [dict(d_i2=0.3, m_eff=0.2, h4=0.3), dict(vp_en=0.0)])
def test_ci_coverage_and_type1(planted):
    reps, B = 200, 500
    hits = {h: 0 for h in CONF}
    hits_logit = {h: 0 for h in CONF}
    rej = {h: 0 for h in CONF}
    bias = {h: [] for h in CONF}
    for k in range(reps):
        d = synth.generate(cov_spec(**planted), seed=1000 + k)
        C, _ = contrasts.library(d, include_exploratory=False)
        pp = bootstrap.cluster_bootstrap(d, C, B=B, seed=k)
        lg = bootstrap.cluster_bootstrap(d, C, B=B, seed=k, scale="logit")
        for c, a, b in zip(C, pp, lg):
            t = synth.truth(d, c)
            tl = synth.truth(d, c, "logit_marginal")
            hits[c.name] += a.ci_lo <= t <= a.ci_hi
            hits_logit[c.name] += b.ci_lo <= tl <= b.ci_hi
            rej[c.name] += a.p_wald < 0.05
            bias[c.name].append(a.estimate - t)
    for h in CONF:
        cov, cov_l = hits[h] / reps, hits_logit[h] / reps
        metric("coverage", planted=planted != {"vp_en": 0.0}, contrast=h, coverage_pp=cov, coverage_mlogit=cov_l,
               reject_rate_05=rej[h] / reps, mean_bias_pp=float(np.mean(bias[h])), reps=reps, B=B)
        assert 0.90 <= cov <= 0.995, (h, cov)
        assert 0.90 <= cov_l <= 0.995, (h, cov_l)
        if planted == {"vp_en": 0.0}:   # global null: every confirmatory truth is 0
            assert rej[h] / reps <= 0.08, (h, rej[h] / reps)


def test_irt_recovery_and_negative_discrimination_flags():
    rng = np.random.default_rng(7)
    J, I, K = 60, 300, 8
    th = rng.normal(size=J)
    a = np.exp(rng.normal(0.0, 0.3, I))
    b = rng.normal(0.0, 1.0, I)
    neg = np.arange(10)
    a[neg] = -0.8
    p = 1 / (1 + np.exp(-a * (th[:, None] - b)))
    y = rng.binomial(K, p)
    n = np.full_like(y, K)
    fl = irt.item_flags(y, n)
    good = np.setdiff1d(np.arange(I), neg)
    fit = irt.fit_2pl(y, n)
    r_b = np.corrcoef(fit.b[good], b[good])[0, 1]
    r_a = np.corrcoef(fit.a[good], a[good])[0, 1]
    r_t = np.corrcoef(fit.theta, th)[0, 1]
    tp = int(fl["negative_discrimination"][neg].sum())
    fp = float(fl["flag_any"][good].mean())
    metric("irt", corr_b=r_b, corr_a=r_a, corr_theta=r_t, neg_flagged=tp, false_flag_rate=fp,
           converged=fit.converged, iters=fit.iters, reliability=irt.marginal_reliability(fit))
    assert r_b > 0.95 and r_a > 0.7 and r_t > 0.95
    assert tp >= 8 and fp < 0.10


def test_seed_variance_estimator_recovers_planted_components():
    s_run, s_re = 0.15, 0.10
    ests = []
    for k in range(20):
        spec = synth.SynthSpec(vi_models={"qwen3-1.7b": "1.7B"}, en_models=(), vi_arms=("Y+", "Y-", "N+", "N-"),
                               en_arms=(), seeds_vi=12, n_core=300, n_par={"gsm8k_platinum": 300},
                               s_run=s_run, s_re=s_re, K=4)
        ests.append(variance.seed_variance(synth.generate(spec, seed=500 + k)))
    m = {k: float(np.mean([e[k] for e in ests])) for k in ("sigma_run", "sigma_run_x_eval", "sigma_run_mirror")}
    metric("seed_variance", planted_run=s_run, planted_re=s_re, planted_mirror=math.hypot(s_run, s_re), **m,
           sd_sigma_run_x_eval=float(np.std([e["sigma_run_x_eval"] for e in ests])))
    assert abs(m["sigma_run_x_eval"] - s_re) < 0.025
    assert abs(m["sigma_run"] - s_run) < 0.03
    assert abs(m["sigma_run_mirror"] - math.hypot(s_run, s_re)) < 0.025


def test_power_sims_reproduce_t5_and_e1_tables():
    r = power.power_t5(0.45, 0.55, 0.0, 400, 1000, 3, 4, reps=1500, rng=np.random.default_rng(20261107))
    o3 = power.summarize(power.simulate_mirror(["0.6B", "1.7B"], 5, 5, reps=1000,
                                               rng=np.random.default_rng(20261107)))
    metric("power_numpy", t5_2B_n400_S3_K4_sd_pp=100 * r["sd"], t5_table=1.79,
           e1_O3_I2_sd_pp=o3["I2_pool"]["sd_pp"], e1_table=0.95, e1_O3_I3_sd_pp=o3["I3_pool"]["sd_pp"])
    assert 1.6 < 100 * r["sd"] < 2.0
    assert 0.85 < o3["I2_pool"]["sd_pp"] < 1.06


def test_pilot_rule_branches():
    lo = power.pilot_rule(0.05, 0.04, reps=150)
    hi = power.pilot_rule(0.15, 0.13, reps=150)
    metric("pilot_rule", low=lo["recommendations"], high=hi["recommendations"],
           low_O3_I2_mde=lo["e1_O3"]["I2_pool"]["mde80_0125_pp"], high_O3_I2_mde=hi["e1_O3"]["I2_pool"]["mde80_0125_pp"])
    assert any("O3b" in s for s in lo["recommendations"])
    assert any("O7" in s for s in hi["recommendations"]) and any("8-10" in s for s in hi["recommendations"])
