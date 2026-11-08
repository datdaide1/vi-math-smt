"""Off-laptop jobs for the Y2 analysis pipeline, on a Modal L4 container (GPU for JAX/NumPyro, its CPUs for R).

    E:\\venv_tools\\Scripts\\modal.exe run vi_math_verified_aug/analysis/modal_jobs.py --parts pytest,gpu,glmm

Parts (each a separate container call, timeout 20 min):
  pytest  simulation-scale tests (tests/test_analysis_heavy.py): planted recovery, CI coverage, type-I error,
          IRT recovery, seed-variance recovery, numpy power-table reproduction, pilot rule
  gpu     JAX bootstrap + power sims on the GPU vs the numpy reference implementations (agreement + timing)
  lme4    glmm_lme4.R (reference, container CPUs) on a planted synthetic design, models A and B
  numpyro NumPyro NUTS Bayesian GLMM (GPU) on the same design
  vb      statsmodels mean-field VB (CPU) on the same design -- timed out at 20 min in run 3; not in the default

Results are written to logs/api/Y2/modal_<part>.json.  Budget: <= $2 total for Y2 (L4 $0.80/h).
"""
from __future__ import annotations

import json
import os
import pathlib

import modal

_HERE = pathlib.Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent   # in the container the file sits at /root
_LOCAL = (ROOT / "tests").is_dir()
app = modal.App("y2-analysis")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("r-base-core", "r-cran-lme4")
    .pip_install("numpy==2.2.4", "pandas==2.2.3", "scipy==1.15.2", "statsmodels==0.14.4", "pytest==8.3.5",
                 "jax[cuda12]==0.5.3", "numpyro==0.18.0")
    .env({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    .add_local_dir(str(ROOT / "vi_math_verified_aug" / "analysis"), "/root/vi_math_verified_aug/analysis")
    .add_local_file(str(ROOT / "vi_math_verified_aug" / "__init__.py"), "/root/vi_math_verified_aug/__init__.py")
    .add_local_file(str(ROOT / "tests" / "test_analysis_heavy.py"), "/root/tests/test_analysis_heavy.py")
    .add_local_file(str(ROOT / "tests" / "test_analysis.py"), "/root/tests/test_analysis.py")
)


def _env():
    import sys
    sys.path.insert(0, "/root")
    os.chdir("/root")


@app.function(image=image, gpu=["L4", "A10G", "L40S"], cpu=4.0, memory=16384, timeout=1200, retries=0)
# retries=0: a startup bug must fail once, not loop (run 1 crash-looped on IndexError(2) from
# Path(__file__).parents[2] evaluated inside the container, where the file sits at /root/modal_jobs.py).
def run_part(part: str) -> dict:
    _env()
    import time
    t0 = time.time()
    if part == "pytest":
        import subprocess
        env = dict(os.environ, VIMATH_HEAVY="1", PYTHONPATH="/root")
        r = subprocess.run(["python", "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", "tests/test_analysis.py",
                            "tests/test_analysis_heavy.py"], capture_output=True, text=True, env=env)
        metrics = [json.loads(l.split("METRIC ", 1)[1]) for l in r.stdout.splitlines() if "METRIC " in l]
        tail = [l for l in r.stdout.splitlines() if "METRIC " not in l][-40:]
        return {"part": part, "returncode": r.returncode, "metrics": metrics, "stdout_tail": tail,
                "stderr_tail": r.stderr[-3000:], "seconds": time.time() - t0}
    if part == "gpu":
        return {"part": part, **_gpu_part(), "seconds": time.time() - t0}
    if part in ("lme4", "numpyro", "vb"):
        return {"part": part, **_glmm_part(part), "seconds": time.time() - t0}
    raise ValueError(part)


def _gpu_part() -> dict:
    import jax
    import numpy as np
    from vi_math_verified_aug.analysis import bootstrap, contrasts, gpu, power, synth
    out = {"device": str(jax.devices()[0])}
    # realistic-size design: 2 models; VI 7 arms x 5 seeds, EN 4 arms x 5 seeds; Core 500 + parallel 1,000
    spec = synth.SynthSpec(d_i2=0.3, m_eff=0.2, h4=0.3, seeds_vi=5, seeds_en=5, n_core=500,
                           n_par={"gsm8k_platinum": 600, "math_algprealg": 400})
    d = synth.generate(spec, seed=3)
    C, _ = contrasts.library(d)
    out["design"] = {"runs": len(d.runs), "items": len(d.items), "contrasts": len(C)}
    (_, t_compile) = gpu.timed(gpu.cluster_bootstrap_jax, d, C, B=2000)     # compile at the chunk shape
    (g, tg) = gpu.timed(gpu.cluster_bootstrap_jax, d, C, B=10_000)
    t_boot = dict(gpu.LAST_TIMING["bootstrap"])
    (n, tn) = gpu.timed(bootstrap.cluster_bootstrap, d, C, B=10_000, chunk=500)
    rows = []
    for a, b in zip(n, g):
        rows.append({"name": a.name, "est_ref": a.estimate, "est_gpu": b.estimate, "se_ref": a.se, "se_gpu": b.se,
                     "se_ratio": b.se / a.se, "ci_ref": [a.ci_lo, a.ci_hi], "ci_gpu": [b.ci_lo, b.ci_hi]})
    ratios = np.array([r["se_ratio"] for r in rows])
    out["bootstrap"] = {"B": 10_000, "seconds_gpu": tg, "seconds_gpu_first_call_B2000": t_compile, "seconds_numpy": tn, "max_abs_est_diff":
                        max(abs(r["est_ref"] - r["est_gpu"]) for r in rows), "se_ratio_min": float(ratios.min()),
                        "se_ratio_max": float(ratios.max()), "se_ratio_mean": float(ratios.mean()), "gpu_timing": t_boot,
                        "rows": rows}
    # power: T5 2B cell and E1 O3, GPU vs numpy reference
    gpu.simulate_t5_jax(0.45, 0.55, 0.0, 400, 1000, 3, 4, reps=500)       # warm-up at chunk shape
    (tj, t_tj) = gpu.timed(gpu.simulate_t5_jax, 0.45, 0.55, 0.0, 400, 1000, 3, 4, reps=4000)
    t_t5 = dict(gpu.LAST_TIMING["t5"])
    (tn_, t_tn) = gpu.timed(power.simulate_t5, 0.45, 0.55, 0.0, 400, 1000, 3, 4, reps=1500,
                            rng=np.random.default_rng(1))
    gpu.simulate_mirror_jax(["0.6B", "1.7B"], 5, reps=200)
    (mj, t_mj) = gpu.timed(gpu.simulate_mirror_jax, ["0.6B", "1.7B"], 5, reps=4000)
    t_mi = dict(gpu.LAST_TIMING["mirror"])
    (mn, t_mn) = gpu.timed(power.simulate_mirror, ["0.6B", "1.7B"], 5, 5, reps=1000, rng=np.random.default_rng(1))
    out["power"] = {
        "t5_2B_n400_S3_K4": {"sd_pp_gpu": 100 * float(tj[0].std(ddof=1)), "sd_pp_numpy": 100 * float(tn_[0].std(ddof=1)),
                             "t5_table_sd_pp": 1.79, "reps_gpu": 4000, "reps_numpy": 1500,
                             "seconds_gpu": t_tj, "seconds_numpy": t_tn, "gpu_timing": t_t5},
        "e1_O3": {"I2_sd_pp_gpu": 100 * float(mj["I2_pool"].std(ddof=1)),
                  "I2_sd_pp_numpy": 100 * float(mn["I2_pool"].std(ddof=1)), "e1_table_sd_pp": 0.95,
                  "I3_sd_pp_gpu": 100 * float(mj["I3_pool"].std(ddof=1)),
                  "I3_sd_pp_numpy": 100 * float(mn["I3_pool"].std(ddof=1)), "e1_table_I3_sd_pp": 1.94,
                  "reps_gpu": 4000, "reps_numpy": 1000, "seconds_gpu": t_mj, "seconds_numpy": t_mn,
                  "gpu_timing": t_mi}}
    return out


def _glmm_part(method: str) -> dict:
    """One estimation method on the shared planted design (same seed -> identical data in every part)."""
    import subprocess
    import tempfile
    import time
    import pandas as pd
    from vi_math_verified_aug.analysis import bayes_glmm, contrasts, glmm, synth
    spec = synth.SynthSpec(d_i2=0.4, m_eff=0.3, h4=0.4, seeds_vi=3, seeds_en=3, n_core=120,
                           n_par={"gsm8k_platinum": 60, "math_algprealg": 40})
    d = synth.generate(spec, seed=21)
    C, _ = contrasts.library(d, include_exploratory=False)
    out = {"truth_logit": {c.name: synth.truth(d, c, "logit_conditional") for c in C},
           "truth_pp_marginal": {c.name: synth.truth(d, c) for c in C},
           "planted_sd": {"item": spec.s_item, "item_lang": spec.s_lang, "item_arm_per_arm": spec.tau / 2 ** .5,
                          "item_model": spec.s_im, "run": spec.s_run, "run_evaltype": spec.s_re},
           "design": {"runs": len(d.runs), "items": len(d.items)}}
    if method == "lme4":
        tmp = tempfile.mkdtemp()
        out["export"] = glmm.export_for_r(d, C, tmp)
        rscript = os.path.join("/root/vi_math_verified_aug/analysis", "glmm_lme4.R")
        # three R processes in parallel on the container's 4 CPUs: A and B with the Laplace refit, and A at
        # nAGQ = 0 only (the pre-registered time fallback), so a slow Laplace fit still leaves a result
        jobs = {}
        for tag, which, env_extra in (("A", "A", {}), ("B", "B", {}), ("A_nagq0", "A", {"GLMM_NAGQ0_ONLY": "1"})):
            cmd = ["Rscript", rscript, os.path.join(tmp, f"glmm_data_{which}.csv"),
                   os.path.join(tmp, f"contrasts_{which}.csv"), which, os.path.join(tmp, f"out_{tag}")]
            jobs[tag] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                          env=dict(os.environ, **env_extra)), time.time())
        for tag, (proc, t) in jobs.items():
            try:
                so, se = proc.communicate(timeout=max(60, 1080 - (time.time() - t)))
                res = {"returncode": proc.returncode, "stdout_tail": so[-2500:], "stderr_tail": se[-2500:]}
            except subprocess.TimeoutExpired:
                proc.kill()
                res = {"returncode": "timeout"}
            res["seconds"] = time.time() - t
            for suffix in ("contrasts", "varcomp"):
                p = os.path.join(tmp, f"out_{tag}_{suffix}.csv")
                if os.path.exists(p):
                    res[suffix] = pd.read_csv(p).to_dict("records")
            p = os.path.join(tmp, f"out_{tag}_status.json")
            if os.path.exists(p):
                res["status"] = open(p).read()
            out[tag] = res
    elif method == "numpyro":
        for which in ("A", "B"):
            try:
                out[which] = bayes_glmm.fit_numpyro(d, C, which, num_warmup=300, num_samples=300, max_tree_depth=7)
            except Exception as e:  # report, do not hide
                out[which] = {"error": repr(e)}
    else:
        for which in ("A", "B"):
            t = time.time()
            try:
                v = glmm.fit_vb(d, C, which)
                out[which] = {"contrasts": v.contrasts.to_dict("records"), "vc": v.variance_components,
                              "fits": v.fits, "seconds": time.time() - t}
            except Exception as e:
                out[which] = {"error": repr(e), "seconds": time.time() - t}
    return out


@app.local_entrypoint()
def main(parts: str = "pytest,gpu,lme4,numpyro"):
    outdir = ROOT / "logs" / "api" / "Y2"
    outdir.mkdir(parents=True, exist_ok=True)
    plist = [p.strip() for p in parts.split(",") if p.strip()]
    for res in run_part.map(plist, return_exceptions=True, order_outputs=False):
        if isinstance(res, Exception):
            print("ERROR", repr(res))
            continue
        path = outdir / f"modal_{res['part']}.json"
        path.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(f"{res['part']}: {res.get('seconds', 0):.0f}s -> {path}", flush=True)
