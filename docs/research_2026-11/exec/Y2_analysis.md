# Y2 — Statistical analysis pipeline (2026-11-07)

**Scope.** This is the inferential pipeline for the study. It follows `tech/T5_evaluation.md` §2–§3 and
`exec/E1_english_mirror.md` §5.5 (the language axis).
- Code: `vi_math_verified_aug/analysis/` (all new; no legacy code reused).
- Local tests: `tests/test_analysis.py` (tiny inputs).
- Simulation-scale tests: `tests/test_analysis_heavy.py`, run on Modal.
- Raw evidence: `logs/api/Y2/` (`modal_*.json`, `modal_run*.log`, `billing_*`).

**Compute rules.**
- Laptop: unit tests on tiny inputs only (≤ 50 items, ≤ 8 runs, < 200 MB).
- Everything else ran on Modal GPU containers (L4, fallback A10G or L40S):
  - JAX and NumPyro on the GPU;
  - lme4 (R) on the same container's CPUs, as the reference.
- History: two early local statsmodels-VB attempts were killed by the coordinator (645 MB, then 1.5 GB). Since then
  no fit or simulation has run on the laptop.

---

## 0. Summary

| Item | Status | Evidence |
|---|---|---|
| Input schema + validation + aggregation | **built**, tested | §1; `test_schema_*` |
| Confirmatory family H1–H4 adapted to P1–P3; exploratory family | **built**; the lead must ratify it | §2 |
| GLMM (a): lme4 R script, cell-means design, pre-registered fallback order | **built, run on Modal**. Both models converge on the full structure (Laplace), and their estimates recover the planted effects | §3, §6.4 |
| GLMM (b): Python. NumPyro NUTS on GPU (primary Python GLMM), marginal-logit cluster bootstrap, statsmodels VB | NumPyro **agrees with lme4** (§6.4). VB **too slow**: did not finish in 20 min, so it is not used | §3, §6.4 |
| Two-way cluster bootstrap, 10k draws: numpy reference + JAX/GPU | **built**. CI coverage 0.95–0.995; type-I error 0.01–0.04; GPU vs numpy SE ratio 0.997–1.028 | §6.1, §6.2 |
| TOST, Holm, BH, decision rule | **built**, tested against hand-computed values | §4 |
| Wilson / Clopper–Pearson / Beta intervals, item-bootstrap slice CI, exact and mid-p McNemar | **built**, tested against closed forms | §4 |
| IRT 2PL MAP in numpy, item flags, Core/Hard split, temporal-split tools | **built**. Recovery: r_b = 0.98, r_a = 0.88; 10/10 negative-discrimination items flagged, 0 % false flags | §5, §6.3 |
| Power simulation (T5 App. A, E1 App. A ported; numpy + JAX); seed-variance estimator; pilot rule | **built**. Reproduces T5 (1.79 pp) and E1 (0.95 / 1.94 pp) | §5, §6.2, §6.3 |
| Total Modal spend for Y2 | **$1.35** of the $2 budget | §7, `logs/api/Y2/billing_report_today_final.json` |

---

## 1. Input schema (`schema.py`)

**One JSONL row = one decoded sample:**
`run_id, arm, model, size, seed, lang, dataset, item_id, sample_k, correct, tokens`.

Optional fields:
- `decode` (`sample`/`greedy`; `sample_k = -1` also means greedy);
- `family`;
- `evaltype` (`native`/`parallel`);
- `certifiable` (P3);
- `period`, `pair_id` (temporal split);
- `truncated`, `judged`.

**Conventions the producers must follow.**
- `dataset` is **language-neutral**, e.g. `gsm8k_platinum`, `math_algprealg`, `vi_exam_core`.
- `item_id` is **shared by an EN original and its VI translation**. The cluster key is `dataset/item_id`, so
  resampling and the item random effects keep the parallel pairing. This is E1 R5.
- Each run has exactly one `lang`, which is both its training and its evaluation language.
- `evaltype` defaults from the dataset name: names containing `exam`, `core` or `hard` are `native`; everything else
  is `parallel`.
- Arms `Y±` and `N±` are parsed into `method` and `verif`. A Unicode minus is accepted.

**Validation.** The loader rejects:
- missing fields;
- `lang ∉ {vi, en}`;
- `correct ∉ {0, 1}`;
- duplicate (run, item, sample) triples;
- a run whose `arm`, `model`, `size`, `seed` or `lang` changes between rows.

It reports:
- the K values used, and any runs with mixed K;
- greedy, truncated and judged row counts.

**Memory.** Rows are streamed into per-cell accumulators. Memory is O(runs × items), never O(rows): about 110 runs ×
1,500 items is 0.7 MB per array.

**Outputs.**
- `EvalData`: dense `y`, `n`, `tok` and `greedy` arrays, each (runs × items).
- `run_table()`: the T5 §2.6 per-run × dataset table (mean pass@1, greedy accuracy, tokens, n_items).
- `to_long()`: the binomial table the GLMMs use.

---

## 2. Hypotheses: adapting T5's H1–H4 to the approved problems (`contrasts.py`)

T5's H1–H3 were "translated arm T vs X × eval type". The approved design (`01_TECHNICAL_DESIGN.md` §1) makes
translation a baseline only. The arms that matter are **Y±** (symbolic, ± verification) and **N±** (LLM-native,
± verification), and **P2 is language × verification**.

**Proposed confirmatory family (Holm, m = 4; FWER 0.05):**

| H | Contrast (equal weights over models/sizes) | Data | Why |
|---|---|---|---|
| **H1 (P2)** | I2 = VP_vi − VP_en on the 1,000 parallel items, where VP_L = ½[(Y+ − Y−) + (N+ − N−)] | Model B (mirror runs, both languages) | E1 §1, §5.5 |
| **H2 (C2)** | ½(Y+ + Y−) − ½(N+ + N−) on Vi-ExamMath Core | Model A (VI grid) | The brief's "arm contrasts Y± vs N±" |
| **H3** | ½(Y+ + N+) − ½(Y− + N−) on Core | Model A | T5 H4, unchanged |
| **H4 (P3)** | (Y − N) on certifiable Core items − (Y − N) on non-certifiable Core items | Model A | 01 §1 P3 "does certifiable training transfer?". A within-run DiD, so the run effect cancels |

**Exploratory family (BH-FDR, q = 0.05).** It contains:
- I3 (language × method × verification) and language × method;
- I2 per size, and the capability-matched VP_vi(1.7B) − VP_en(0.6B);
- VP_vi and VP_en separately;
- H2 and H3 per model;
- Y+ vs Y− and N+ vs N− on Core;
- VP on Core − VP on the parallel set;
- T5-form arm × eval-type DiDs for the baseline arms: T vs N+, T vs Y+, T vs M, S0 vs Y+, S0 vs N+, T-off vs T.

**How the library is built.**
- The family is **data**: `contrasts.library()` builds it from the runs present.
- A contrast with missing cells is skipped, with a reason in `notes`.
- T5 §4.2's floor rule (a model with S0 Core accuracy < 10 %) is applied automatically.
- **⚑ The lead must ratify H2 and H4 in the pre-registration.** In particular H4 needs the `certifiable` flag on Core
  items (X2/X3). Without that flag H4 is skipped and Holm runs with m = 3.

**Decision rule** (`multiplicity.decide`; T5 §2.2–§2.3, E1 §5.5). A confirmatory contrast is "significant" only if
all three hold:
1. it rejects in the GLMM after Holm;
2. it rejects in the pp-scale cluster bootstrap after Holm. If (1) and (2) disagree, the more conservative one
   governs;
3. the logit estimate and the pp estimate agree in sign.

If the contrast is not significant, run TOST with a ±3 pp margin on the bootstrap draws. "No meaningful effect" may be
claimed only if the 90 % CI lies inside ±3 pp.

---

## 3. GLMM (`glmm.py`, `glmm_lme4.R`, `bayes_glmm.py`)

**Fixed effects** use the cell-means parametrisation `0 + cell`, with cell = lang × arm × model × slice:
- `slice` is the dataset, with Core split into `:cert` and `:noncert`;
- this spans the same space as T5's `arm * dataset * model` and E1's `lang * method * verif * size`;
- every hypothesis is one weight vector over the cell coefficients. `glmm.cell_weights` exports it once.

The **same weights** are used by lme4, by NumPyro and, on the pp scale, by the bootstrap. The unit test
`test_glmm_cell_weights_match_terms` checks that w·η_fixed equals the planted logit effect exactly.

| Model | Data | Random effects (full) | Fallback order (pre-registered) |
|---|---|---|---|
| A (VI grid; H2–H4) | VI runs, all arms, all VI items | (1\|item) + (1\|item:arm) + (1\|item:model) + (1\|run) + (1\|run:evaltype) | 1. drop item:model → 2. drop run:evaltype (report its SD from per-model fits) → 3. Python (NumPyro) |
| B (mirror; H1) | Y±/N± runs of the models run in both languages; parallel items | (1\|item) + (1\|item:lang) + (1\|item:arm) + (1\|item:arm:lang) + (1\|run) + (1\|seed_key) | 1. drop item:arm:lang → 2. drop seed_key → 3. Python (NumPyro) |

Here `seed_key` = model:arm:seed, i.e. E1's CRN pairing term `(1|size:arm:seed_idx)`.

**Fit procedure.**
- Each candidate is fitted with nAGQ = 0, then refitted with nAGQ = 1 (Laplace) from the nAGQ = 0 estimates, using
  bobyqa.
- Contrasts:
  - logit scale: Wald z = w′β / √(w′Vw);
  - pp scale: delta method on plogis(β), which is exactly emmeans `regrid = "response"` without the emmeans
    dependency.
- The script applies Holm and BH itself.

**Python GLMM of record: NumPyro NUTS on GPU** (`bayes_glmm.fit_numpyro`).
- Same cells and random-effect structure as lme4.
- Priors: β ~ N(0, 2.5²); SD ~ HalfNormal(1); non-centred random effects.
- 2 chains run vectorised on the GPU.
- Reports the posterior mean, SD, 95 % CrI and P(H > 0) per contrast, the SD components, and split-R̂.
- This also serves as T5's planned "Bayesian GLMM sensitivity analysis".

**statsmodels VB** (`glmm.fit_vb`, the brief's suggestion) is kept in the code but is **not used**:
- it needs Bernoulli-expanded rows plus a dense fixed-effect matrix;
- it reached 1.5 GB locally;
- it did not finish within 20 minutes on Modal (run 3, `FunctionTimeoutError`);
- mean-field VB also understates fixed-effect uncertainty.

---

## 4. Robustness, multiplicity, small slices

**Two-way cluster bootstrap** (`bootstrap.py`, numpy reference; `gpu.py`, JAX).
- Each draw resamples:
  - items within dataset, as multinomial counts over item keys. This keeps the EN/VI pairing;
  - seeds within lang × arm × model, as multinomial counts over runs.
- `paired_seeds=True` instead resamples seed indices within arm × model jointly for both languages (the CRN design).
- The statistic is recomputed exactly: the cell value is the weighted mean, over resampled runs, of each run's
  weighted item mean. A unit test checks this against explicit replication of the drawn items and runs.
- Scales: pp, and the marginal logit Σ w·logit(cell). The marginal logit is the GLMM-free stand-in for the logit-scale
  test.
- Outputs: percentile 95 % and 90 % CIs, bootstrap SE, Wald p from the bootstrap SE, and percentile p.
- Memory is chunked: about 20 MB per 500-draw chunk at 110 runs × 1,500 items.

**Multiple testing and equivalence** (`multiplicity.py`):
- `holm`, `bh`;
- `tost` (normal, or bootstrap draws shifted to the ±margin);
- `decide` (§2).

**Small slices** (`intervals.py`):
- Wilson, Clopper–Pearson (two- or one-sided), and Beta(1,1)/Jeffreys posteriors;
- `slice_ci`: with K > 1, average K within item, then an item bootstrap (T5 §2.3); with K = 1, Beta;
- exact McNemar with mid-p, for deterministic single-decode comparisons only;
- `residual_error_upper` (T5 §3.2): 0/150 → 1.98 %, 1/150 → 3.1 %, 2/150 → 4.1 %.

## 5. IRT, power and the pilot rule

**IRT** (`irt.py`).
- **py-irt is not used.** It needs torch + pyro (hundreds of MB, about 0.5 GB of RAM at import).
- Model: 2PL with binomial responses (K samples per item), fitted by MAP / empirical Bayes with alternating Fisher
  scoring:
  - θ ~ N(0, 1);
  - b ~ N(μ_b, σ_b) with the hyperparameters re-estimated each sweep;
  - log a ~ N(μ_a, 0.5²) (T5 prior).
- An **unconstrained-a** variant, a ~ N(1, 1), is used only to detect **negative discrimination**.
- `item_flags` implements T5 §2.5 use 1 (flag for mandatory human re-review, never auto-drop):
  - a < 0.2;
  - item-rest point-biserial < 0.1;
  - free a < 0.
- Also provided:
  - `test_information` and `marginal_reliability`;
  - `core_hard_split`, T5's ±1.5-logit Core rule and Hard = b > top ability + 0.5 with a ≥ 0.5;
  - the temporal-split tools of T5 D5: (a) `matched_difficulty`, slot-matched pairs, b_post − b_pre with a pair
    bootstrap; (b) `reference_did`, C_M with a pair bootstrap; (c) `irt_residuals`, θ̂ from post-cutoff items only,
    then the mean residual on pre-cutoff items.

**Power** (`power.py` numpy, `gpu.py` JAX).
- `simulate_t5` / `power_t5` port T5 App. A.
- `simulate_mirror` / `summarize` port E1 App. A (I2, I3, per-size, cap).
- Also: `mde`, `tost_power`, and a CLI `python -m vi_math_verified_aug.analysis.power {t5|mirror|pilot}`.
- Run it on Modal or Kaggle, not on the laptop.

**Seed-variance estimator** (`variance.py`): a method of moments on the delta-method logit scale.
- Within each lang × arm × model cell with S ≥ 2 seeds, the run deviations are z_{r,e} = (a_{r,e} − ā_e) / E[p(1−p)].
- The known Bernoulli noise is subtracted.
- Estimates:
  - Cov(z_native, z_parallel) gives σ²_run;
  - ½Var(z_N − z_P) gives σ²_run×eval;
  - Var(z_e) gives σ²_run(mirror) = σ²_run + σ²_run×eval;
  - all are pooled with weights S − 1, and Wilson–Hilferty CIs are reported.
- `item_components` gives crude σ_item and τ.

**Pilot rule** (`power.pilot_rule`). It re-simulates with the measured components and applies:
- T5: pooled MDE > 4 pp → S = 5; σ_run×eval ≥ 0.12 → S = 8–10 in the 2B cells;
- E1 §5.4: σ_run(mirror) ≤ 0.08 → O3b; 0.08–0.15 → O3; ≥ 0.15 → O7; ≥ 0.20 → O7 + 1.7B at S = 8.

---

## 6. Validation on synthetic data with planted effects

**Generator** (`synth.py`). It is the union of the T5 and E1 models: fixed logit cells plus item, item × lang,
item × arm, item × model, run and run × evaltype effects, with K = 4.

Planted effects:

| Planted parameter | Meaning |
|---|---|
| `d_i2` | P2 (the language × verification interaction) |
| `m_eff` | H2 (method main effect) |
| `vp_en` + `d_i2` | H3 (the Vietnamese verification premium) |
| `h4` | P3 (method × certifiability interaction) |

`truth()` gives the population value of any contrast on three scales:
- pp: E[estimator], via Gauss–Hermite integration over every random term;
- marginal logit;
- conditional logit: what the GLMM estimates.

### 6.1 Planted-effect recovery and CI coverage

Source: numpy bootstrap on Modal, `modal_pytest.json`; 24/24 tests passed.

**Full design.**
- 2 models; 7 VI arms; 4 EN arms; 5 seeds; Core 300 + parallel 500 items; B = 4,000.
- Planted (logit): d_i2 = 0.4, m_eff = 0.3, h4 = 0.4.
- Every 95 % CI covers the truth and excludes 0.

| H | truth (pp) | estimate (pp) | 95 % CI (pp) |
|---|---|---|---|
| H1 P2 | 6.50 | 6.49 | [4.38, 8.55] |
| H2 Y vs N | 4.67 | 5.00 | [3.25, 6.78] |
| H3 verif | 7.79 | 7.68 | [5.87, 9.47] |
| H4 P3 | 6.22 | 6.99 | [4.26, 9.69] |

**Coverage simulation.**
- 200 independent synthetic datasets. Each has 2 models × Y±/N± × 3 seeds in both languages and 260 items.
- B = 500 per dataset.
- Truth = E[estimator]. The marginal-logit CI is checked against Σ w·logit(E cell).

| H | planted: coverage pp / marginal logit | planted: rejection @ .05 | global null: coverage pp / marginal logit | null: rejection @ .05 (type I) | mean bias (pp) |
|---|---|---|---|---|---|
| H1 | 0.955 / 0.950 | 0.725 | 0.955 / 0.950 | **0.040** | −0.21 |
| H2 | 0.960 / 0.960 | 0.555 | 0.970 / 0.975 | 0.025 | −0.07 |
| H3 | 0.990 / 0.990 | 1.000 | 0.975 / 0.985 | 0.015 | +0.06 |
| H4 | 0.990 / 0.995 | 0.515 | 0.990 / 0.990 | 0.010 | +0.12 |

**What this shows.**
- Coverage is at or above nominal.
- The two-way ("pigeonhole") bootstrap is conservative, as expected. It is most conservative for the Core-only
  contrasts H3 and H4: coverage ≈ 0.99 and a type-I rate of 0.01–0.015.
- The GLMM is therefore the more powerful of the two required tests. Under T5's rule ("the more conservative one
  governs"), the bootstrap will usually decide.
- The Monte Carlo error of each coverage figure, with 200 replicates, is ±1.5 pp.
- A first version of this test was wrong. Its "null" scenario still had the default premium vp_en = 0.1 planted,
  which inflated H3's rejection rate. The fix sets the null to vp_en = 0, so every truth is exactly 0.

### 6.2 GPU (JAX) vs numpy reference

Source: Modal L4, `modal_gpu.json`.

**Bootstrap.**
- Design: 110 runs × 1,500 items, 23 contrasts, B = 10,000.
- Point estimates are identical (maximum difference 0).
- The GPU/numpy SE ratio is 0.997–1.028 (mean 1.009), which is within Monte Carlo error.

**Power simulations.**

| Simulation | GPU SD (pp) | numpy SD (pp) | Table SD (pp) |
|---|---|---|---|
| T5, 2B, n = 400, S = 3, K = 4 | 1.80 (4,000 reps) | 1.81 (1,500 reps); 1.79 in the pytest run | 1.79 (T5) |
| E1 O3, I2 | 0.94 | 0.94 | 0.95 (E1) |
| E1 O3, I3 | 1.89 | 1.94 | 1.94 (E1) |

**Speed.**
- XLA compilation dominates a single call: 14.5 s for the first bootstrap chunk, and 2.7–3.5 s for the power
  simulations.

After compilation:

| Task | GPU | numpy CPU | Speed-up |
|---|---|---|---|
| Bootstrap | 8,000 draws in 0.005 s | 10,000 draws in 1.2 s | ≈ 200× |
| T5 power sim | 3,500 reps in 0.06 s | 1,500 reps in 0.9 s | ≈ 35× |
| E1 power sim | 3,800 reps in 0.58 s | 1,000 reps in 5.5 s | ≈ 35× |

- For **one** analysis of the real data, the numpy bootstrap (≈ 1 s) is enough.
- The GPU pays off for coverage studies, power grids and the pilot rule.

### 6.3 IRT, seed variance, pilot rule

Source: `modal_pytest.json`.

**IRT.** Setup: 60 respondents × 300 items, K = 8, with 10 items planted at a = −0.8.

| Quantity | Result |
|---|---|
| corr(b̂, b) | 0.984 |
| corr(â, a) | 0.877 |
| corr(θ̂, θ) | 0.999 |
| Negative-discrimination items flagged | **10 of 10** |
| False-flag rate on good items | **0 %** |
| Marginal reliability | 0.997 |

**Caveat:** the stopping rule max-|Δ| < 1e-5 was not met within 500 sweeps (`converged = False`). The estimates are
nevertheless clearly recovered. Use a relative tolerance before the real panel fit.

**Seed variance.** Setup: 20 replicates of 1 model × 4 arms × 12 seeds × (300 native + 300 parallel) items.

| Component | Planted | Mean estimate over 20 replicates |
|---|---|---|
| σ_run | 0.15 | 0.145 |
| σ_run×eval | 0.10 | 0.091 (SD 0.017) |
| σ_run(mirror) | 0.180 | 0.172 |

- The estimates are 4–9 % low; the first-order delta method attenuates them slightly.
- At S = 3 the estimate is noisy, so always report its CI.

**Pilot rule.**

| Inputs | Recommendation | E1 O3 I2 MDE (α = .0125) |
|---|---|---|
| σ_run = 0.05, σ_re = 0.04 | O3b | 2.0 pp |
| σ_run = 0.15, σ_re = 0.13 | S = 5 everywhere; S = 8–10 in the 2B cells; O7 | 5.0 pp |

### 6.4 GLMM: lme4 (reference) vs NumPyro (GPU) vs truth

**Setup.**
- A planted design with 66 runs × 220 items, generated with the same seed in every part.
- Logit scale. Truth = the conditional fixed effect.
- Sources: `modal_lme4.json` (run 4) and `modal_numpyro.json` (run 3).

| H | truth | lme4 Laplace est (SE) | lme4 nAGQ = 0 est (SE) | NumPyro mean (SD) | NumPyro 95 % CrI |
|---|---|---|---|---|---|
| H1 (model B) | 0.400 | 0.538 (0.074) | — | 0.536 (0.095) | [0.36, 0.74] |
| H2 | 0.300 | 0.271 (0.063) | 0.264 (0.063) | 0.269 (0.076) | [0.12, 0.42] |
| H3 | 0.500 | 0.532 (0.063) | 0.520 (0.063) | 0.534 (0.073) | [0.39, 0.68] |
| H4 | 0.400 | 0.580 (0.113) | 0.566 (0.113) | 0.588 (0.118) | [0.37, 0.83] |

**GPU vs reference agreement.**
- Point estimates from NumPyro and lme4 agree within **0.008 logit**.
- NumPyro posterior SDs are 4–28 % wider. This is expected: the posterior propagates the uncertainty in the variance
  components, which the Wald SE ignores.
- Every truth lies within 2 SE of both methods.
- H1 and H4 sit 1.6–1.9 SE high in this single dataset. Both methods agree on this, so it is sampling, not method;
  calibration across many datasets is §6.1.
- NumPyro split-R̂: the maximum over β is 1.08 (model A) and 1.04 (model B); all SD components are ≤ 1.04. Use ≥ 1,000
  draws for the real fit.

**Variance components (SD, logit).**

| Component | Planted | lme4 | NumPyro |
|---|---|---|---|
| item | 1.5 | 1.51 | 1.53 |
| item:arm | 0.354 | 0.33 | 0.35 |
| item:model | 0.3 | 0.28 | 0.30 |
| run | 0.10 | 0.064 | 0.067 |
| run:evaltype | 0.07 | 0.024 | 0.084 |
| B: item:lang | 0.5 | 0.44 | 0.46 |
| B: item:arm:lang | 0 | 0 (singular) | 0.08 |

- The run-level components are poorly determined with 3 seeds × 14 cells.
- This is exactly why the pilot rule must report a CI, not a point value.

**Convergence and timing on Modal.**

| Fit | Time | Result |
|---|---|---|
| Model A, nAGQ = 0 | 27 s | converged on the full structure |
| Model A, Laplace refit | 668 s | converged on the full structure |
| Model B, full structure | 108 s | converged; singular because item:arm:lang = 0, which is correct (none was planted) |

- The Laplace refit moves the contrasts by ≤ 0.014 logit (≤ 0.2 SE) but costs 24× the time.
- **Recommendation for the pre-registration:**
  - At full scale (≈ 134k rows, about 15× this test) the Laplace refit may take hours on a Kaggle CPU.
  - Add the rule: "if the nAGQ = 1 refit exceeds 3 h, report nAGQ = 0" (set `GLMM_NAGQ0_ONLY=1`).
  - Show the nAGQ = 0 vs nAGQ = 1 difference on a per-model subset.

**statsmodels VB:** not usable. It did not finish within 20 min on Modal (`FunctionTimeoutError`, run 3).

---

## 7. Bugs found and fixed; cost

| # | Bug | Effect | Fix |
|---|---|---|---|
| 1 | `modal_jobs.py` computed `Path(__file__).parents[2]` at import time. Inside the container the file is `/root/modal_jobs.py`, so this raised `IndexError(2)` | Run 1 crash-looped at startup because Modal retried. App `ap-rFfLlUqzM6mfpPlCHjo8UO` was stopped; cost $0.009 | Path computation guarded; `retries=0`; GPU fallback list `["L4", "A10G", "L40S"]` |
| 2 | `glmm_lme4.R` counted lme4's "boundary (singular) fit" warning as non-convergence | Model B dropped to fallback step 1 although the full fit was valid (run 3) | Singular fits are now recorded, not treated as failures, as the script header always stated. Run 4 keeps the full structure |
| 3 | Model A's Laplace refit used `calc.derivs = TRUE`, on a container with the default fractional CPU | Timed out at 9 min (run 3) | `calc.derivs = FALSE`, `cpu = 4`, and the three R fits run in parallel; model A's Laplace fit then took 668 s. **Not isolated:** which of the two changes removed most of the cost is unknown |
| 4 | Local `timeout 1500` on `modal run` | Run 2's GLMM part was lost when the client was killed | Parts are split, and results are written unordered as soon as each one arrives |
| 5 | The heavy test's "null" scenario still had vp_en = 0.1 planted | H3's "type-I" rate of 0.19 was really power | The null now sets vp_en = 0 |
| 6 | GPU timing included XLA compilation, and the jit is rebuilt on every call | 19.7 s reported against 1.2 s for numpy | First-chunk and steady-state times are recorded separately (§6.2) |

**Modal spend for Y2: $1.35** (budget ≤ $2).
- Source: `modal billing report --for today --json`, filtered to the `y2-analysis` apps
  (`logs/api/Y2/billing_report_today_final.json`).

| Run | Cost |
|---|---|
| Run 1 (crash loop) | $0.01 |
| Run 2 | $0.30 |
| Run 3 (includes the 20-min VB timeout) | $0.77 |
| Run 4 (lme4) | $0.26 |

- The scheduler placed most jobs on **A10G** ($0.95 of the total) rather than L4 ($0.28).
- The before/after workspace summaries (`billing_before.txt`, `billing_after.txt`) include other workstreams.
- The last hour may not be billed yet.

## 8. How to run

```bash
# local, tiny (laptop-safe): 18 tests, < 10 s
.venv/Scripts/python -m pytest tests/test_analysis.py
# real data
python -m vi_math_verified_aug.analysis.run analyze --pred preds/*.jsonl --out results/analysis --B 10000 [--paired-seeds]
python -m vi_math_verified_aug.analysis.run export  --pred preds/*.jsonl --out glmm_export   # -> R inputs
Rscript glmm_lme4.R glmm_data_A.csv contrasts_A.csv A out_A        # Kaggle/Modal CPU; GLMM_NAGQ0_ONLY=1 = fallback
python -m vi_math_verified_aug.analysis.run analyze ... --r-results out_A_contrasts.csv out_B_contrasts.csv
# off-laptop heavy checks (Modal GPU; retries=0)
E:/venv_tools/Scripts/modal.exe run vi_math_verified_aug/analysis/modal_jobs.py --parts pytest,gpu,lme4,numpyro
python -m vi_math_verified_aug.analysis.power pilot --s_run <sigma_run> --s_re <sigma_run_x_eval>   # on Modal/Kaggle
```

## 9. Open items

1. **⚑ Ratify H2 and H4** as the confirmatory family. H4 needs `certifiable` tags on Core items from X2/X3. Without
   them, H4 is skipped and Holm uses m = 3.
2. **Producers must emit the §1 schema**, with a language-neutral `dataset` and a parallel `item_id`. This affects the
   eval harness (S2) and the eval sets (X3).
3. **Laplace fit time at full scale is unmeasured.** Measure it on the first real pilot (6 runs), then fix the nAGQ
   rule.
4. **IRT stopping rule:** switch to a relative tolerance before the panel fit.
5. **Seed-variance estimator:** it reads 4–9 % low. Report its CI, and prefer the GLMM variance components when the
   fit converges.
6. **Not implemented:**
   - the Fantastic-Bugs tetrachoric / scalability Z_j flags (only the a and item-rest flags exist);
   - the Rogan–Gladen judge correction (T5 §3.3);
   - the agreement statistics κ, AC1 and Krippendorff α (T5 §3.2).

   These belong to the benchmark-QA workstream unless they are reassigned.
7. **emmeans is not required.** The pp-scale contrasts are computed directly by the delta method, which is the same
   calculation as `regrid = "response"`. They have not been cross-checked against emmeans itself.
