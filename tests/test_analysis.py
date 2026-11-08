"""Y2 analysis pipeline -- fast local unit tests on tiny inputs (<= 50 items, a handful of runs, < 200 MB).

Simulation-scale checks (planted-effect recovery, CI coverage, lme4/VB fits, IRT and seed-variance recovery,
power reproduction) live in ``test_analysis_heavy.py`` and run on Modal CPU (``analysis/modal_jobs.py``).
"""
import json
import math

import numpy as np
import pytest

from vi_math_verified_aug.analysis import (bootstrap, contrasts, glmm, intervals, irt, multiplicity, power, schema,
                                           stats_util, synth, variance)


def tiny_spec(**kw):
    base = dict(vi_models={"qwen3-0.6b": "0.6B"}, en_models=("qwen3-0.6b",), vi_arms=("Y+", "Y-", "N+", "N-"),
                en_arms=("Y+", "Y-", "N+", "N-"), seeds_vi=1, seeds_en=1, n_core=24,
                n_par={"gsm8k_platinum": 10, "math_algprealg": 6}, K=4)
    base.update(kw)
    return synth.SynthSpec(**base)


# ------------------------------------------------------------------------------------------- special functions
def test_normal_functions():
    assert stats_util.norm_ppf(0.975) == pytest.approx(1.959963985, abs=1e-8)
    assert stats_util.norm_ppf(1e-6) == pytest.approx(-4.753424309, abs=1e-6)
    assert stats_util.norm_cdf(1.0) == pytest.approx(0.841344746, abs=1e-9)
    assert stats_util.two_sided_p(1.959963985) == pytest.approx(0.05, abs=1e-8)


def test_beta_and_binomial():
    assert stats_util.betainc(3.0, 3.0, 0.5) == pytest.approx(0.5, abs=1e-12)
    assert stats_util.betainc(1.0, 1.0, 0.3) == pytest.approx(0.3, abs=1e-12)
    assert stats_util.beta_ppf(0.975, 1, 11) == pytest.approx(1 - 0.025 ** (1 / 11), abs=1e-9)
    direct = sum(math.comb(20, k) * 0.3 ** k * 0.7 ** (20 - k) for k in range(8))
    assert stats_util.binom_cdf(7, 20, 0.3) == pytest.approx(direct, abs=1e-12)


# ------------------------------------------------------------------------------------------------ intervals
def test_wilson_cp_beta():
    lo, hi = intervals.wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    # T5 §3.2 residual-error bounds: 0/150 -> ~2.0 %, 1/150 -> ~3.1 %, 2/150 -> ~4.1 % (Poisson 2.0/3.2/4.2)
    assert intervals.residual_error_upper(0, 150) == pytest.approx(1 - 0.05 ** (1 / 150), abs=1e-9)
    assert intervals.residual_error_upper(0, 150) < 0.0201
    assert 0.029 < intervals.residual_error_upper(1, 150) < 0.032
    assert 0.039 < intervals.residual_error_upper(2, 150) < 0.042
    lo, hi = intervals.clopper_pearson(3, 10)
    assert (lo, hi) == pytest.approx((0.06674, 0.65245), abs=1e-4)
    post = intervals.beta_posterior(0, 10)
    assert post["hi"] == pytest.approx(1 - 0.025 ** (1 / 11), abs=1e-9)


def test_mcnemar_exact_and_midp():
    r = intervals.mcnemar((1, 9))
    assert r["p_exact"] == pytest.approx(2 * 11 / 1024, abs=1e-12)
    assert r["p_mid"] == pytest.approx(2 * 6 / 1024, abs=1e-12)
    x = np.array([1, 1, 0, 0, 1, 0])
    y = np.array([1, 0, 1, 1, 1, 1])
    assert intervals.mcnemar(x, y)["n01"] == 3 and intervals.mcnemar(x, y)["n10"] == 1


def test_slice_ci_methods():
    y1 = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    r = intervals.slice_ci(y1, np.ones(10))
    assert r["method"] == "beta(1,1)" and r["lo"] < 0.7 < r["hi"]
    y4 = np.array([4, 3, 0, 2, 4, 1, 3, 4, 0, 2])
    r = intervals.slice_ci(y4, np.full(10, 4), B=2000)
    assert r["method"] == "item_bootstrap" and r["lo"] < r["estimate"] < r["hi"]
    assert r["estimate"] == pytest.approx(y4.mean() / 4)


# --------------------------------------------------------------------------------------------- multiplicity
def test_holm_bh_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert multiplicity.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert multiplicity.bh(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert multiplicity.holm([0.5]) == pytest.approx([0.5])


def test_tost():
    assert multiplicity.tost(0.0, 0.01)["equivalent"]
    assert not multiplicity.tost(0.025, 0.01)["equivalent"]
    rng = np.random.default_rng(0)
    draws = rng.normal(0.002, 0.008, 4000)
    t = multiplicity.tost(0.002, draws=draws)
    assert t["equivalent"] and t["p"] < 0.05


def test_decision_rule_needs_sign_agreement_and_both_tests():
    d = multiplicity.decide({"estimate": 0.2}, {"estimate": 0.03}, 0.01, 0.02)
    assert d["significant"]
    d = multiplicity.decide({"estimate": -0.2}, {"estimate": 0.03}, 0.01, 0.02)
    assert not d["significant"] and not d["signs_agree"]
    d = multiplicity.decide({"estimate": 0.2}, {"estimate": 0.03}, 0.20, 0.02)
    assert not d["significant"] and d["governing"] == "conservative"


# --------------------------------------------------------------------------------------------------- schema
def test_schema_roundtrip_and_validation(tmp_path):
    d = synth.generate(tiny_spec(), seed=3)
    p = tmp_path / "pred.jsonl"
    synth.write_jsonl(d, str(p))
    with open(p, "a", encoding="utf-8") as fh:      # one greedy row
        r = dict(json.loads(open(p, encoding="utf-8").readline()))
        r.update(sample_k=-1, correct=1)
        fh.write(json.dumps(r) + "\n")
    e = schema.load_predictions(str(p))
    assert e.report["K_values"] == [4] and e.report["greedy_rows"] == 1
    # align and compare counts
    key = lambda dd: {(dd.runs.run_id[r], dd.items.item_key[i]): (dd.y[r, i], dd.n[r, i])
                      for r in range(len(dd.runs)) for i in range(len(dd.items)) if dd.n[r, i] > 0}
    assert key(d) == key(e)
    assert set(e.runs["method"].dropna()) == {"Y", "N"}
    assert e.items.loc[e.items["dataset"] == "vi_exam_core", "evaltype"].eq("native").all()
    assert len(schema.run_table(e)) > 0
    assert len(schema.to_long(e)) == int((e.n > 0).sum())


def test_schema_rejects_bad_rows():
    good = dict(run_id="r1", arm="Y+", model="m", size="0.6B", seed=0, lang="vi", dataset="core", item_id="1",
                sample_k=0, correct=1, tokens=10)
    with pytest.raises(schema.SchemaError):
        schema.build([good, dict(good)])                                  # duplicate sample
    with pytest.raises(schema.SchemaError):
        schema.build([good, dict(good, sample_k=1, arm="N+")])            # inconsistent run metadata
    with pytest.raises(schema.SchemaError):
        schema.build([dict(good, lang="fr")])
    with pytest.raises(schema.SchemaError):
        schema.build([{k: v for k, v in good.items() if k != "tokens"}])
    assert schema.arm_method_verif("Y−") == ("Y", "-")
    assert schema.arm_method_verif("T-off") == (None, None)


# ------------------------------------------------------------------------------------------------ contrasts
def test_library_recovers_planted_logit_effects_exactly():
    spec = tiny_spec(d_i2=0.4, m_eff=0.3, h4=0.5, vp_en=0.1)
    d = synth.generate(spec, seed=1)
    C, notes = contrasts.library(d)
    by = {c.name: c for c in C}
    assert [c.name for c in C if c.family == "confirmatory"] == [
        "H1_P2_lang_x_verif", "H2_method_Y_vs_N_core", "H3_verif_core", "H4_P3_method_x_certifiable"]
    assert synth.truth(d, by["H1_P2_lang_x_verif"], "logit_conditional") == pytest.approx(0.4)
    assert synth.truth(d, by["H2_method_Y_vs_N_core"], "logit_conditional") == pytest.approx(0.3)
    assert synth.truth(d, by["H3_verif_core"], "logit_conditional") == pytest.approx(0.5)     # vp_en + d_i2
    assert synth.truth(d, by["H4_P3_method_x_certifiable"], "logit_conditional") == pytest.approx(0.5)
    assert synth.truth(d, by["I3_lang_x_method_x_verif"], "logit_conditional") == pytest.approx(0.0)
    # every arm contrast sums its weights to zero within each (lang, item set) block -> intercepts cancel
    for c in C:
        assert abs(sum(t.weight for t in c.terms)) < 1e-12, c.name


def test_glmm_cell_weights_match_terms(tmp_path):
    spec = tiny_spec(d_i2=0.4, m_eff=0.3, h4=0.5)
    d = synth.generate(spec, seed=1)
    C, _ = contrasts.library(d, include_exploratory=False)
    for which in ("A", "B"):
        dd = glmm.model_subset(d, which)
        lab, levels = glmm.cell_labels(dd)
        eta = dd.report.get("_eta_fixed")
        if eta is None:   # subsets drop the report tables; recompute from the parent
            pass
        for c in [c for c in C if c.model == which]:
            w = glmm.cell_weights(dd, c, levels)
            assert abs(w.sum()) < 1e-12
            # fixed logit of each cell from the generator
            full_eta = d.report["_eta_fixed"]
            rmap = {r: j for j, r in enumerate(d.runs.run_id)}
            imap = {k: j for j, k in enumerate(d.items.item_key)}
            cell_eta = {}
            for r in range(len(dd.runs)):
                for i in range(len(dd.items)):
                    if lab[r, i]:
                        cell_eta[lab[r, i]] = full_eta[rmap[dd.runs.run_id[r]], imap[dd.items.item_key[i]]]
            got = float(sum(wv * cell_eta[l] for l, wv in zip(levels, w)))
            assert got == pytest.approx(synth.truth(d, c, "logit_conditional"), abs=1e-9), c.name
    out = glmm.export_for_r(d, C, str(tmp_path))
    assert set(out) == {"A", "B"}
    import pandas as pd
    W = pd.read_csv(out["A"]["weights"])
    assert "H3_verif_core" in W and abs(W["H3_verif_core"].sum()) < 1e-12
    L = pd.read_csv(out["B"]["data"])
    assert set(L["lang"]) == {"vi", "en"} and (L["evaltype"] == "parallel").all()


# ------------------------------------------------------------------------------------------------ bootstrap
def test_bootstrap_vectorised_matches_explicit_replication():
    d = synth.generate(tiny_spec(seeds_vi=2, seeds_en=1, vi_arms=("Y+", "N+"), en_arms=("Y+",)), seed=2)
    c = contrasts.Contrast("x", [contrasts.Term.make(1, {"lang": "vi", "arm": "Y+"}, {"evaltype": "native"}),
                                 contrasts.Term.make(-1, {"lang": "vi", "arm": "N+"}, {"evaltype": "native"})])
    prep = bootstrap._Prepared(d, [c])
    rng = np.random.default_rng(5)
    iw = bootstrap._item_weights(d, rng, 1, True)
    rw = bootstrap._run_weights(d, rng, 1, ("lang", "arm", "model"), "run_id", True)
    fast = prep.contrasts(prep.cell_values(iw, rw), "pp")[0, 0]
    # explicit: replicate items and runs according to the counts, then recompute the plain estimate
    acc = d.acc
    def cell(arm):
        rows = np.flatnonzero((d.runs["arm"] == arm).to_numpy() & (d.runs["lang"] == "vi").to_numpy())
        cols = np.flatnonzero((d.items["evaltype"] == "native").to_numpy())
        vals = []
        for r in rows:
            per_item = np.repeat(acc[r, cols], iw[0, cols].astype(int))
            vals += [per_item.mean()] * int(rw[0, r])
        return np.mean(vals)
    assert fast == pytest.approx(cell("Y+") - cell("N+"), abs=1e-6)   # float32 acc
    res = bootstrap.cluster_bootstrap(d, [c], B=300, chunk=100)[0]
    assert res.estimate == pytest.approx(contrasts.estimate(d, c), abs=1e-12)
    assert res.se > 0 and res.ci_lo < res.estimate < res.ci_hi
    nores = bootstrap.cluster_bootstrap(d, [c], B=50, resample_items=False, resample_seeds=False)[0]
    assert nores.se == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_paired_seeds_and_logit_scale():
    d = synth.generate(tiny_spec(), seed=4)
    C, _ = contrasts.library(d, include_exploratory=False)
    a = bootstrap.cluster_bootstrap(d, C, B=200, paired_seeds=True)
    b = bootstrap.cluster_bootstrap(d, C, B=200, scale="logit")
    assert len(a) == len(b) == len(C)
    assert all(np.isfinite(r.estimate) for r in a + b)


# ------------------------------------------------------------------------------------- variance / IRT / power
def test_seed_variance_runs_on_tiny_data():
    d = synth.generate(tiny_spec(seeds_vi=3, seeds_en=1, vi_arms=("Y+", "N+"), en_arms=("Y+",)), seed=6)
    sv = variance.seed_variance(d)
    assert {"sigma_run_mirror", "sigma_run_x_eval", "sigma_run"} <= set(sv)
    assert sv["cells_used"] == 2


def test_irt_tiny_and_theta():
    rng = np.random.default_rng(0)
    th = rng.normal(size=6)
    a, b = np.exp(rng.normal(0, .3, 40)), rng.normal(0, 1, 40)
    p = 1 / (1 + np.exp(-a * (th[:, None] - b)))
    y = rng.binomial(8, p)
    n = np.full_like(y, 8)
    fit = irt.fit_2pl(y, n)
    assert fit.converged and np.corrcoef(fit.theta, th)[0, 1] > 0.8
    t_hi, _ = irt.fit_theta(np.full(40, 7), np.full(40, 8), a, b)
    t_lo, _ = irt.fit_theta(np.full(40, 1), np.full(40, 8), a, b)
    assert t_hi > t_lo
    assert irt.test_information(np.array([0.0]), a, b).shape == (1,)


def test_power_helpers():
    assert power.mde(1.0, 0.0125) == pytest.approx(3.339, abs=1e-3)       # E1 App. B constant
    assert power.tost_power(0.0095) == pytest.approx(0.87, abs=0.01)        # E1 §3.2, O3
    assert power.tost_power(0.0119) == pytest.approx(0.62, abs=0.01)        # O3b
    est, se = power.simulate_t5(0.45, 0.55, 0.0, 20, 20, 2, 2, reps=20, chunk=10,
                                rng=np.random.default_rng(1))
    assert est.shape == se.shape == (20,)
    r = power.simulate_mirror(["0.6B"], 2, 2, n=20, reps=10, chunk=5, rng=np.random.default_rng(1))
    assert r["I2_pool"].shape == (10,)


# ------------------------------------------------------------------------------------------- end-to-end CLI
def test_run_analyze_end_to_end(tmp_path):
    from vi_math_verified_aug.analysis import run
    d = synth.generate(tiny_spec(d_i2=0.4), seed=8)
    out = run.analyze(d, B=200)
    names = [r["name"] for r in out["contrasts"]]
    assert names[:4] == ["H1_P2_lang_x_verif", "H2_method_Y_vs_N_core", "H3_verif_core",
                         "H4_P3_method_x_certifiable"]
    conf = [r for r in out["contrasts"] if r["family"] == "confirmatory"]
    assert all(np.isfinite(r["p_adj_boot"]) for r in conf)
    assert all(r["glmm_source"] == "bootstrap_marginal_logit" for r in conf)
    assert all("tost" in r["verdict"] for r in conf if not r["verdict"]["significant"])
    p = tmp_path / "demo.jsonl"
    run.main(["synth", "--out", str(p), "--d_i2", "0.3"])
    run.main(["export", "--pred", str(p), "--out", str(tmp_path / "exp")])
    run.main(["analyze", "--pred", str(p), "--out", str(tmp_path / "res"), "--B", "100"])
    res = json.loads((tmp_path / "res" / "results.json").read_text(encoding="utf-8"))
    assert res["contrasts"][0]["name"] == "H1_P2_lang_x_verif"
    assert (tmp_path / "exp" / "contrasts_A.csv").exists() and (tmp_path / "exp" / "glmm_data_B.csv").exists()
