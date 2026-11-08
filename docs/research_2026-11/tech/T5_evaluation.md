# T5 — Evaluation & benchmark science (2026-11-07)

Scope: the statistics of the arm × eval-type comparison (RQ-1, `00_SOLUTION.md` §2.3), IRT for Vi-ExamMath, the benchmark-QA
and contamination protocol, and the decoding protocol for 0.6B–4B students. Starting points: `04_data_eval.md` §2 and §5,
`01_literature_novelty.md` §1E. Where this report disagrees with them, the reason is given.

Evidence tags: **[O]** = I opened the arXiv abstract, HTML, or the official page in this session. **[S]** = seen only in a search
snippet or summary, so **[UNVERIFIED]**. **[P:x]** = verified in an earlier project report x. **[R]** = a command I ran
(simulation code in Appendix A).

---

## 0. Decisions at a glance

| # | Decision | Choice | Main alternative rejected | Why (one line) |
|---|---|---|---|---|
| D1 | Primary inferential model | **Binomial GLMM** with crossed item and run random effects, and one pre-registered contrast per hypothesis | Per-set accuracy + paired bootstrap | Handles item heterogeneity and seed variance together. NIST AI 800-3 (2026) recommends GLMMs for "generalized accuracy" [O] |
| D2 | Multiple comparisons | **Holm** over the 4 confirmatory contrasts (FWER 0.05). **BH-FDR q=0.05** for exploratory ones | Bonferroni over all pairs (28+) | Keeps power for the 4 RQ hypotheses and is honest about the rest |
| D3 | Sample size | **Native Core ≥ 500 scored items; 3 seeds per cell, 5 seeds in the 2B primary cells; K = 4 samples per item** | 400 items, 3 seeds, greedy | Simulated per-model MDE drops from 8.4 to 4.6–4.9 pp. Pooled over sizes, about 3.4 pp (2.6 pp with 5 seeds) [R] |
| D4 | IRT model | **Bayesian 2PL with hierarchical priors**, fitted on a frozen reference panel. Rasch as a robustness check | 3PL (ATLAS default); CAT item subsetting | Free response means no guessing parameter. Fluid uses 2PL plus MCMC [O]. Item subsetting does not reduce seed variance (Madaan) [O] |
| D5 | Temporal-split difficulty control | **Slot-matched pairs + reference-model difference-in-differences + IRT residuals** | Raw pre/post accuracy gap | Post-cutoff decay alone is "unreliable" (Test of Time, ACL 2026) [O] |
| D6 | Item QA | **Three independent keys (official, SymPy program, blind Claude solve) + Claude critique + 2 humans on every flagged item + a 150-item random audit + an IRT anomaly screen** | Platinum-style (inspect only items some model fails) | Platinum's count is "only a lower bound" [O]. The random audit gives a residual-error upper bound |
| D7 | Contamination evidence | **Temporal DiD + training-corpus overlap (13-gram, MinHash, cross-lingual embedding) + numeric perturbation + partial-prompt completion.** Min-K%++ in an appendix only | MIA-based "not contaminated" claims | MIAs score near chance (Duan 2402.07841 [P:04]; Fu, NAACL-F 2025 [S]) |
| D8 | Decoding | **T = 0.6, top-p = 0.95, K = 4 samples, mean pass@1 as the primary metric**, plus a greedy secondary, maj@4 reported, max_new_tokens = 2048 (4096 if truncation > 1 %) | Greedy only | Halves the noise term (Miller [O]). The Qwen3 card warns against greedy [O]. Greedy is kept as a cross-check (Song 2407.10457 [O]) |
| D9 | LLM judge | **Adjudicator only** (rule scorer unparseable or disagreeing). Calibrated on 300 human-labelled outputs. Rogan–Gladen correction if invoked on > 5 % of outputs | Judge as primary scorer | Judge bias is fixable only with sensitivity/specificity estimates (Lee et al., ICML 2026) [O] |

---

## 1. Evidence table (2024–2026, newest first where possible)

| # | Method / topic | Paper (id, date, venue) | What it showed (numbers) | Tag |
|---|---|---|---|---|
| E1 | GLMM for benchmarks | Keller et al., *Expanding the AI Evaluation Toolbox with Statistical Models*, NIST AI 800-3, **2026-02-17** | Separates *benchmark accuracy* (this fixed item set) from *generalized accuracy* (similar unseen items). GLMMs give valid uncertainty more efficiently than regression-free methods, plus a variance decomposition and item difficulties. 22 frontier LLMs × 3 benchmarks | [O] |
| E2 | Fluid Benchmarking (IRT + adaptive) | Hofmann et al., 2509.11106, Sep 2025, COLM 2025 | **2PL**, MCMC with hierarchical priors, calibrated on **102 LMs**. At 500 items: rank distance 8.3 (Fluid) vs 9.1 (accuracy); total variation 4.9 vs 23.8. Mislabeled items selected: 0.01 vs 0.75 (random). The IRT component improves validity; dynamic selection reduces variance | [O] |
| E3 | ATLAS adaptive testing | 2511.04689, v1 Oct 2025, v3 Jul 2026, ICML 2026 | **3PL** default (2PL compared). Up to 90 % fewer items; HellaSwag ability from 41 items, MAE 0.157. Drops items with point-biserial < 0.1. Exposure < 12 %. SE stopping τ ∈ {0.1, 0.2, 0.3}, 30–500 items. Calibration used 3.3k–4.7k models. A "3–6 % negative-discrimination items" figure appears only in a search snippet | [O] / [S] |
| E4 | tinyBenchmarks | Maia Polo et al., 2402.14992, ICML 2024 | About 100 curated items estimate MMLU-scale performance | [P:01 E7] |
| E5 | Variance in benchmarks | Madaan et al., 2406.10229, Jun 2024 | Quantifies seed variance across training runs. **IRT and item analysis "struggle to meaningfully reduce variance"** | [O] |
| E6 | Signal & Noise | Heineman et al., 2508.13144, Aug 2025, NeurIPS 2025 (listing) | 30 benchmarks, 375 models, 900K results. A high signal-to-noise ratio predicts reliable small-scale decisions. Interventions: better metric, filtering noisy subtasks, checkpoint averaging | [O] |
| E7 | Benchmark bug finding | Truong et al., *Fantastic Bugs*, 2511.16842, Nov 2025, NeurIPS 2025 D&B | Flags items by response-pattern statistics (mean tetrachoric correlation, scalability Z_j < 0, item-total correlation). **Precision@50 up to 84 %** across 9 benchmarks. LLM first pass (o1) has 98 % precision on the first 100 GSM8K items. Error classes: ambiguous, wrong key, grading issue | [O] |
| E8 | Platinum benchmarks | Vendrow et al., 2502.03461, Feb 2025 | Manually inspect every item that ≥ 1 LLM fails. GSM8K sample: 26/300 removed and 1 relabeled. SVAMP: 27/300 and 3. The authors call it a lower bound | [O] |
| E9 | GSM8K-Platinum (full test) | madrylab | 110 removed + 10 revised out of 1,319 | [P:04 §2.2] |
| E10 | Error bars for evals | Miller, 2411.00640, Nov 2024 | Var(μ̂) = (Var(x) + E[σ²_i]/K)/n. Paired SE uses the difference scores. MDE = (z_{α/2} + z_β)·√((ω² + σ²_A/K_A + σ²_B/K_B)/n). At n = 198, K from 1 to 10 cuts the MDE from 13.2 to 7.5 pp. **Don't lower temperature to reduce variance.** Clustered SEs can be about 3× larger | [O] |
| E11 | Small-n uncertainty | Bowyer et al., 2503.01747, ICML 2025 position paper | CLT intervals are too narrow below a few hundred items. Beta/Bayesian and other frequentist intervals are better (`bayes_evals`) | [O] |
| E12 | Reasoning-eval reproducibility | Hochlehnert et al., 2504.07086, COLM 2025 | 1.5B-model pass@1 SD across seeds is AIME'24 ±4.6–6.6 and MATH500 ±0.3–1.0. One AIME question = 2.5–3.3 pp. Temperature 0→1 moves results up to 15 %, top-p up to 8 %, hardware up to 8 %. Recommends many seeds, fixed T = 0.8 / top-p 0.9, no truncation, math-verify-style extraction, paired tests | [O] |
| E13 | Seed effects in fine-tuning | Bui et al., 2503.07329, IJCNLP-AACL 2025 | Significant macro (metric) and micro (per-item) variance across fine-tuning seeds. ">50 % of 85 ACL papers misuse seeds" comes from a snippet | [O] / [S] |
| E14 | Greedy vs sampling | Song et al., 2407.10457, Jul 2024 | Greedy is generally ≥ sampling for most tasks. Alignment reduces sampling variance | [O] |
| E15 | Qwen3 decoding guidance | HF model card Qwen/Qwen3-1.7B | Thinking mode: T 0.6, top-p 0.95, top-k 20, with "**DO NOT use greedy decoding** … endless repetitions". Non-thinking: T 0.7, top-p 0.8 | [O] |
| E16 | Prompt-format sensitivity | Sclar et al., FormatSpread, 2310.11324, ICLR 2024 | Up to **76 points** spread from formatting alone (LLaMA-2-13B) | [O] |
| E17 | Fresh-item rerun | Zhang et al., GSM1k, 2405.00332, NeurIPS 2024 D&B | Up to 8 % drop GSM8K → GSM1k. r² = 0.36 between P(generate GSM8K item) and the gap | [O] |
| E18 | Perturbation | Mirzadeh et al., GSM-Symbolic, 2410.05229, ICLR 2025 | Every model drops when only the numbers change. An irrelevant clause causes drops of up to 65 % | [O] |
| E19 | Contamination of Qwen2.5 | Wu et al., 2507.10532, AAAI 2026 | Qwen2.5 is contaminated on MATH-500/AMC/AIME. A clean generator (RandomCalculation) removes spurious-reward gains. Partial-prompt test: Qwen2.5-Math-7B rebuilds 54.6 % of the hidden 40 % of MATH-500 items (the number is from a snippet) | [O] / [S] |
| E20 | Temporal-signal critique | *Test of Time*, 2509.00072, ACL 2026 | Post-cutoff decay is "highly sensitive to how benchmark questions are constructed". An LLM rewrite removes LiveCodeBench's decay. Conclusion: "sensitive but unreliable" signal | [O] |
| E21 | Variant contamination | DVD, 2601.04895, Jan 2026 | PPL, Min-k%++, CDD (edit distance) and embedding detectors fail on paraphrased contamination. DVD (variance of the generation distribution) is better. Tested by fine-tuning Qwen2.5 / Llama3.1 | [O] |
| E22 | Cross-lingual contamination | Yao et al., 2406.13236, EMNLP 2024 | Training on translated benchmarks inflates scores and evades n-gram detectors | [S] |
| E23 | MIA unreliability | Duan et al., 2402.07841 (COLM 2024); Fu et al., 2410.18966 (NAACL-F 2025) | MIAs are near random on IID member/non-member splits | [P:04] / [S] |
| E24 | Judge bias correction | Lee et al., 2511.21140, ICML 2026 | Plug-in correction using judge sensitivity/specificity estimated on a human calibration set. CIs include calibration uncertainty. Adaptive calibration allocation | [O] |
| E25 | PPI for noisy judges | Chen et al., 2601.05420, ICML 2026 | PPI-type estimators can have smaller asymptotic variance than measurement-error correction | [S] |
| E26 | Answer matching | Chandak et al., 2507.02856 | Free-form + reference matching ≈ inter-human agreement. Reference-free judges do poorly | [P:01 E4] |
| E27 | Zero-leakage temporal eval | Pyeon et al., 2511.18649 | 2026 Korean CSAT digitized within 2 h of release. 20B model ≈ 95.7 | [P:01 E1] |
| E28 | IRT software | Lalor & Rodriguez, py-irt, 2203.01282, INFORMS JoC 2023 | Bayesian 1PL/2PL/3PL on Pyro, GPU, SVI. Accuracy comparable to `mirt` | [S] |

**What the evidence implies for this project.**
1. With 3 seeds and a few hundred items, seed variance and item sampling are the same order of magnitude (E5, E12, E13). The
   analysis must model both, which a GLMM does (E1).
2. IRT helps **validity and item diagnosis** (E2, E3, E7) but not variance in a full-item evaluation (E5). Use it for item
   analysis and difficulty control, not for subsetting.
3. No single contamination signal is credible on its own in 2026 (E20, E21, E23). Report a **convergent set** in which every
   signal has a falsifiable prediction.

---

## 2. Analysis plan

### 2.1 Design facts the plan relies on

- **Run** r = (family, size, arm, training seed). Each run is evaluated on **every** item of both eval types, so items are
  crossed with arms, models and seeds. Seeds are not shared across arms. Each seed draws its own training subsample from
  the arm pool, so run variance includes data-sampling variance. The generalization target is "data from source X", not
  "this particular 5k sample".
- **Eval types** for the primary contrast:
  - **native** = Vi-ExamMath-Core, all years.
  - **translated** = a fixed, stratified 1,000-item primary subsample: 600 Vi-GSM8K-Platinum + 400 Vi-MATH alg/prealg.
  - The rest of the translated sets and Hard are secondary. Decoding is identical across all primary items (§4), so eval
    type is not confounded with decoding.
- **Outcome** y_{r,i} = number correct out of K = 4 samples, scored by `math_verify`, with judge adjudication per §3.3.

### 2.2 Primary statistical model (D1)

lme4 syntax, binomial with logit link, one row per (run, item):

```r
glmer(cbind(correct, K - correct) ~ arm * dataset * model          # fixed: model = family×size (Q0.8, Q2, Q4, G-E2B)
        + (1 | item) + (1 | item:arm) + (1 | item:model)           # item difficulty, item-specific arm effect, item×model
        + (1 | run) + (1 | run:evaltype),                          # seed main effect (cancels in H), seed × eval-type
      family = binomial, data = d, nAGQ = 0)  # then refit nAGQ = 1 from the nAGQ = 0 estimates; bobyqa
```

- **Hypotheses are linear contrasts, not raw coefficients.** For arms A vs B, the interaction is
  `H_AB = [A−B on Core] − ½([A−B on GSM8K-Plat] + [A−B on MATH-sub])`, averaged with equal weights over the four models.
- **Scale.** The contrast is tested on the logit scale (Wald z). It is reported as an average marginal effect in pp with
  `emmeans(..., regrid = "response")`.
- **Logit vs pp.** Interactions can differ in sign between the logit and probability scales when baselines differ (the
  native and translated accuracies will). Pre-register this rule: claim an interaction only if the logit-scale test and the
  pp-scale cluster bootstrap (§2.3) agree in sign.
- **Difficulty-adjusted secondary model (H1b).** Add `arm × b_i`, where b_i is the IRT difficulty from the frozen reference
  panel (§2.5). The arm × dataset contrast then means "native vs translated **at equal difficulty**". Without this, a
  "translated data helps less on native items" finding could just be "helps less on harder items".
- **Size.** About 134k rows (96 runs × 1,400 items). The random-effect levels are about 1.4k items, 11k item:arm, 5.6k
  item:model, about 100 runs and about 200 run:evaltype. glmer with nAGQ = 0 handles this in minutes on CPU. R is not on
  the laptop (`Rscript: command not found` [R]); use a Kaggle or Colab R kernel (free).
- **Convergence fallback, pre-registered in this order:**
  1. Drop (1|item:model).
  2. Drop (1|run:evaltype). If dropped, report its estimated variance from a separate per-model fit.
  3. Python `statsmodels BinomialBayesMixedGLM`, which supports variance components only.

| Alternative | Handles item heterogeneity | Handles seed variance | Interaction test | Cost | Verdict |
|---|---|---|---|---|---|
| Per-set accuracy, mean ± SD over seeds, unpaired t | no | yes (df = 2) | no | trivial | Rejected: CLT/df problems (E11), ignores pairing (E10) |
| Paired item-level bootstrap (items only) | yes | **no** | DiD on bootstrap | minutes | Under-covers: run×eval variance dominates at n ≥ 400 (§2.4). Kept as a **robustness check, after adding seed resampling** |
| **Binomial GLMM, crossed random effects (chosen)** | yes | yes | logit contrast, Wald | minutes (R) | Matches the data-generating structure (E1). One model gives all contrasts and the variance decomposition |
| Bayesian GLMM (brms/NumPyro) | yes | yes | posterior P(H > 0) | hours of CPU | **Sensitivity analysis** for H1 only (weakly-informative priors; Bowyer E11 recommends Bayesian for small n) |
| IRT ability θ per run, compare arms on θ | via IRT | via θ per run | θ_native − θ_trans | moderate | Secondary (E2 validity argument). Needs two calibrated scales, so it cannot be primary |

### 2.3 Tests and multiple-comparison control (D2)

**Confirmatory family** (Holm, FWER = 0.05; m = 4; the smallest threshold is 0.0125):

| H | Contrast | Directional prediction (for interpretation only; tests are two-sided) |
|---|---|---|
| H1 | T vs N-checked, arm × eval-type interaction | RQ-1: the translated-data advantage shrinks on native items |
| H2 | T vs Y-checked (symbolic) interaction | — |
| H3 | T vs M (= T+N) interaction | Mixing recovers native performance |
| H4 | Verification main effect on Core: mean(N-chk, Y-chk) − mean(N-unchk, Y-unchk) | C2 / RQ-3 |

**Per confirmatory contrast, report:**
1. GLMM Wald estimate in pp, its 95 % CI and the Holm-adjusted p.
2. A **two-way cluster bootstrap**: 10,000 draws, resampling items within dataset and seeds within arm×model; the
   statistic is the same DiD in pp. Report it alongside.
3. If 1 and 2 disagree in significance, the more conservative one governs (pre-registered).
4. If H1 is non-significant, a TOST equivalence test with margin ±3 pp. "No meaningful interaction" may only be claimed if
   the 90 % CI lies inside ±3 pp. At the planned design the pooled SE is about 1.0 pp (§2.4), so this is achievable.

**Exploratory** (BH-FDR, q = 0.05, labelled as exploratory):
- the same contrasts per model;
- all other arm pairs;
- per-dataset contrasts;
- budget-curve slopes (2k/5k/10k/20k);
- arm × eval × size;
- robustness slices (§3.5);
- temporal pre/post (§3.4).

**Other tests and where they apply:**
- **McNemar (exact, mid-p)** only for deterministic single-decode comparisons: zero-shot baselines and reference-panel
  models on one dataset.
- **Per-model accuracy CIs on small slices** (< 300 items: the 2026 post-cutoff slice, Hard): Beta(1,1)-posterior or Wilson
  intervals (E11). For K > 1, average K within item first, then use an item bootstrap.
- **English control:** the same GLMM with dataset ∈ {GSM8K-EN, MATH-500-EN}. It enters H-tests only as a descriptive
  contrast.

### 2.4 Power analysis (D3) [R]

**Generative model.** Logit scale, per model size. Code in Appendix A; numpy only; 1,500 Monte-Carlo replicates per cell;
seed 20261107.

```
y[r,i] ~ Binomial(K, p[r,i])/K
logit p[r,i] = b_e(i) + u_i + c_a·g_i + δ_{a,e(i)} + v_r + w_{r,e(i)}

u_i  ~ N(0, 1.5²)    item difficulty
g_i  ~ N(0, 0.5²)    item-specific arm effect (c_A = +½, c_B = −½)
v_r  ~ N(0, 0.10²)   run main effect. Cancels in the interaction because every run sees both eval types
w_re ~ N(0, 0.07²)   run × eval-type ("seed luck" that differs between native and translated)
```

- Arm B accuracy: native p_N, translated p_N + 10 pp. Arm A = B on translated, B + Δ on native, with Δ ∈ {3, 5, 8} pp.
- n_translated = 1,000; α = 0.0125 (first Holm step); two-sided.
- **Estimator** Î = mean_native(y_A − y_B) − mean_trans(y_A − y_B).
- **Power** is reported two ways:
  - **oracle:** Monte-Carlo SD, i.e. a correctly calibrated Wald test;
  - **feasible:** SE² = var_items(D_N)/n_N + var_items(D_T)/n_T + var_seeds(Î_s)/S, a conservative stand-in for the
    cluster bootstrap.
- **Calibration check, under Δ = 0 at nominal 5 %:** the feasible test rejects 2.0–3.4 %, so it is slightly conservative.
  The simulated single-seed between-arm discordance is 28–36 % of items [R].

**Closed-form check (Miller E10 extended to a DiD), at 2B, n_N = 400, S = 3, K = 1:**

```
Var(Î) ≈ Σ_e (ω_e² + 2·E[p(1−p)]/(S·K)) / n_e + 4·σ²_{w,pp} / S

ω² ≈ τ²·E[(p(1−p))²] ≈ 0.25 × 0.03 = 0.0075
E[p(1−p)] ≈ 0.16
σ_{w,pp} ≈ 0.07 × 0.2475 = 0.0173

native:     (0.0075 + 0.107) / 400  = 2.9e-4
translated: (0.0075 + 0.107) / 1000 = 1.1e-4
run×eval:   4 × 0.0173² / 3         = 4.0e-4
→ SD ≈ 2.8 pp. The simulation gives 2.5 pp; the closed form overstates p(1−p) at the mean.
```

The **run × eval term is the largest single component** once n_N ≥ 400. More seeds buy more than more items; more samples
K shrink the Bernoulli term.

**Result 1: power per model** (oracle / feasible), α = 0.0125:

| size (native acc of B) | Δ | n=200,S=3,K=1 | n=400,S=1,K=1 | n=400,S=3,K=1 | n=400,S=3,K=4 | n=600,S=3,K=4 | n=400,S=5,K=4 | n=600,S=5,K=4 | n=800,S=5,K=4 |
|---|---|---|---|---|---|---|---|---|---|
| 0.6-0.8B (0.25) | 3 pp | 0.08 / 0.04 | 0.04 / 0.09 | 0.12 / 0.07 | 0.24 / 0.24 | 0.31 / 0.30 | 0.42 / 0.35 | 0.45 / 0.39 | 0.50 / 0.43 |
| 0.6-0.8B (0.25) | 5 pp | 0.23 / 0.13 | 0.11 / 0.22 | 0.37 / 0.23 | 0.73 / 0.62 | 0.73 / 0.66 | 0.92 / 0.82 | 0.93 / 0.86 | 0.97 / 0.89 |
| 0.6-0.8B (0.25) | 8 pp | 0.63 / 0.40 | 0.34 / 0.49 | 0.82 / 0.62 | 0.99 / 0.94 | 0.99 / 0.95 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| 1.7-2B (0.45) | 3 pp | 0.06 / 0.03 | 0.04 / 0.08 | 0.10 / 0.06 | 0.22 / 0.21 | 0.20 / 0.24 | 0.34 / 0.29 | 0.38 / 0.32 | 0.41 / 0.35 |
| 1.7-2B (0.45) | 5 pp | 0.21 / 0.11 | 0.10 / 0.20 | 0.30 / 0.19 | 0.62 / 0.56 | 0.68 / 0.60 | 0.80 / 0.72 | 0.88 / 0.78 | 0.90 / 0.80 |
| 1.7-2B (0.45) | 8 pp | 0.51 / 0.33 | 0.23 / 0.43 | 0.75 / 0.56 | 0.97 / 0.91 | 0.99 / 0.91 | 1.00 / 0.98 | 1.00 / 1.00 | 1.00 / 1.00 |
| 4B (0.65) | 3 pp | 0.08 / 0.04 | 0.05 / 0.09 | 0.10 / 0.07 | 0.25 / 0.25 | 0.28 / 0.32 | 0.39 / 0.33 | 0.43 / 0.39 | 0.52 / 0.46 |
| 4B (0.65) | 5 pp | 0.22 / 0.12 | 0.11 / 0.20 | 0.38 / 0.23 | 0.75 / 0.62 | 0.77 / 0.69 | 0.91 / 0.81 | 0.96 / 0.87 | 0.96 / 0.91 |
| 4B (0.65) | 8 pp | 0.64 / 0.41 | 0.33 / 0.49 | 0.89 / 0.65 | 0.99 / 0.95 | 1.00 / 0.97 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

(In the S = 1 column the feasible power exceeds the oracle because one seed cannot estimate run × eval variance, so that
test is anti-conservative. **Never run single-seed cells for the primary contrasts.**)

**Result 2: SD of Î (pp) → MDE at 80 % power,** where MDE = (2.498 + 0.842)·SD:

| size | n=200,S=3,K=1 | n=400,S=1,K=1 | n=400,S=3,K=1 | n=400,S=3,K=4 | n=600,S=3,K=4 | n=400,S=5,K=4 | n=600,S=5,K=4 | n=800,S=5,K=4 |
|---|---|---|---|---|---|---|---|---|
| 0.6-0.8B | 2.82 → 9.4 | 3.92 → 13.1 | 2.31 → 7.7 | 1.64 → 5.5 | 1.57 → 5.2 | 1.33 → 4.4 | 1.25 → 4.2 | 1.21 → 4.0 |
| 1.7-2B | 3.09 → 10.3 | 4.32 → 14.4 | 2.51 → 8.4 | 1.79 → 6.0 | 1.74 → 5.8 | 1.48 → 4.9 | 1.37 → 4.6 | 1.32 → 4.4 |
| 4B | 2.86 → 9.5 | 3.90 → 13.0 | 2.28 → 7.6 | 1.60 → 5.3 | 1.55 → 5.2 | 1.29 → 4.3 | 1.22 → 4.1 | 1.15 → 3.8 |

**Result 3: pooled over 3 sizes** (same items; item × arm slopes correlated ρ across sizes; Appendix A, `power_pooled.py`):

| ρ | n_N, S, K | SD (pp) | MDE80 (α = .0125) |
|---|---|---|---|
| 0.5 | 400, 3, 1 | 1.42 | 4.7 |
| 0.5 | 400, 3, 4 | 1.01 | **3.4** |
| 0.5 | 600, 3, 4 | 0.95 | 3.2 |
| 0.5 | 400, 5, 4 | 0.79 | **2.6** |
| 1.0 | 400, 3, 4 | 1.07 | 3.6 |
| 1.0 | 400, 5, 4 | 0.86 | 2.9 |

**Result 4: sensitivity at 2B** (oracle power at α = .0125; the unknown variance components varied one at a time):

| scenario | n=400,S=3,K=1: SD / P(3) / P(5) / P(8) | n=400,S=3,K=4: SD / P(3) / P(5) / P(8) | n=600,S=3,K=4: SD / P(3) / P(5) / P(8) |
|---|---|---|---|
| base (σ_item 1.5, τ 0.5, σ_run×eval 0.07) | 2.50 / 0.11 / 0.35 / 0.75 | 1.84 / 0.19 / 0.60 / 0.97 | 1.73 / 0.22 / 0.68 / 0.99 |
| σ_item = 2.5 (more deterministic items) | 2.07 / 0.14 / 0.49 / 0.90 | 1.44 / 0.31 / 0.82 / 1.00 | 1.33 / 0.39 / 0.90 / 1.00 |
| τ = 0.8 (more item-specific arm effects) | 2.58 / 0.08 / 0.30 / 0.73 | 1.94 / 0.15 / 0.53 / 0.95 | 1.81 / 0.20 / 0.62 / 0.97 |
| τ = 0.3 | 2.51 / 0.10 / 0.30 / 0.76 | 1.78 / 0.22 / 0.62 / 0.97 | 1.69 / 0.23 / 0.65 / 0.99 |
| σ_run×eval = 0.03 | 2.12 / 0.16 / 0.41 / 0.91 | 1.32 / 0.40 / 0.92 / 1.00 | 1.16 / 0.53 / 0.97 / 1.00 |
| **σ_run×eval = 0.15** | 3.71 / 0.04 / 0.12 / 0.37 | 3.25 / 0.06 / 0.16 / 0.49 | 3.18 / 0.06 / 0.17 / 0.49 |
| σ_item 2.5 + σ_run×eval 0.03 | 1.83 / 0.18 / 0.57 / 0.98 | 1.11 / 0.61 / 0.97 / 1.00 | 0.97 / 0.74 / 1.00 / 1.00 |

How to read this table. The decisive unknown is **σ_run×eval**, i.e. how much a training seed's luck differs between native
and translated items. Item-side parameters (σ_item, τ) move power by at most about 0.2. If σ_run×eval is 0.15 logit, items
and K stop mattering: SD ≈ 3.2 pp at n = 600.

Worked example for σ_run×eval = 0.15, using the two simulated SDs:

```
non-run part (n = 600, K = 4, taken from the 0.03 row)   ≈ 1.16 pp
run part at S = 3: √(3.18² − 1.16²)                      ≈ 2.96 pp
per-seed run part: 2.96 × √3                             ≈ 5.1 pp
at S = 10: 5.1/√10                                       = 1.62 pp
total SD at S = 10: √(1.62² + 1.16²)                     ≈ 2.0 pp
per-model MDE: 3.34 × 2.0                                ≈ 6.7 pp
pooled over 3 sizes: ≈ 3.34 × 2.0/√3                     ≈ 3.9 pp
```

So the remedy for large seed × eval variance is **seeds, not items**. This is why the pilot (below) measures σ_run×eval
before the grid is fixed.

**Answers to "how many items and seeds?"**

| Target interaction | Per model (one size) | Pooled over 3 sizes (homogeneous effect) |
|---|---|---|
| **8 pp** | n_N = 400, S = 3, K = 4 → power ≥ 0.97 at every size | trivially |
| **5 pp** | n_N ≥ 600, S = 5, K = 4 → 0.88–0.96. With S = 3, power is only 0.62–0.77 | n_N = 400, S = 3, K = 4 → MDE 3.4 pp, so yes |
| **3 pp** | **Not reachable** with free compute (n = 800, S = 5 gives ≤ 0.52) | n_N = 400, **S = 5**, K = 4 → MDE 2.6–2.9 pp; marginal |

**Decision D3:**
1. **Core ≥ 500 scored items** after QA. Scrape about 650 candidates, expecting about 20 % QA loss.
2. **K = 4 samples on all primary items.**
3. **S = 3 seeds everywhere, S = 5 for the arms in H1–H3 at the 2B size** (T, N-chk, Y-chk, M, S0).
4. H1–H3 are tested **pooled over models**, because the confirmatory contrasts average over models (§2.2). Per-model
   contrasts are exploratory.

**Expected resolution:** about 3–3.5 pp pooled, about 5 pp at 2B.

**Extra compute:** 2 extra seeds × 5 arms × 1 size = 10 extra SFT runs at 2B. At about 1 L4-h each (the training cost per
`03`), that is about 10 L4-h. The K = 4 eval cost is in §4.3.

**Pilot rule (do before the full grid):**
1. Train S0 and T at 0.8B with 3 seeds each.
2. Evaluate on the available 2025 native items and the translated subsample with K = 4.
3. Fit the §2.2 GLMM and read off σ_item, τ, σ_run×eval.
4. Rerun `power_sim.py` with them.
5. Raise seeds before adding sizes or budgets:
   - if the pooled MDE is above 4 pp, raise S to 5 everywhere;
   - if σ_run×eval ≥ 0.12 logit, use S = 8–10 in the 2B primary cells (see the worked example under Result 4).

Cost: 6 runs at 0.8B on the laptop or Kaggle, plus about 1 GPU-h of eval.

### 2.5 IRT for Vi-ExamMath (D4, D5)

**Model choice**

| Model | Parameters / item | Pros | Cons | Verdict |
|---|---|---|---|---|
| Rasch (1PL) | b | Stable with few respondents; sufficient statistic = sum score | Assumes equal discrimination, so it cannot flag bad items | **Robustness fit** and equating check |
| **2PL, Bayesian hierarchical priors (chosen)** | a, b | Gives discrimination for QA flags (E3, E7) and difficulty for stratification. Fluid uses the same model (E2) | Needs regularization with about 40–60 respondents | **Primary** |
| 3PL | a, b, c | ATLAS default (E3) | c (guessing) is about 0 for free response with numeric answers. Unidentifiable with few respondents | Rejected |
| Multidimensional / PSN-IRT | vectors | Captures topic structure | Overfits at this respondent count | Rejected. Revisit if the 2PL residual analysis shows strong topic clustering |

**Priors and fit**
- Priors: θ ~ N(0,1); b_i ~ N(μ_b, σ_b) with μ_b ~ N(0,1) and σ_b ~ HalfNormal(1); log a_i ~ N(μ_a, σ_a) with μ_a ~ N(0, 0.5).
  The log-normal prior forces a > 0. Negative discrimination is detected through the item-rest correlation instead
  (below).
- Fit with NumPyro NUTS, or py-irt SVI (E28) for speed. This is minutes on CPU at 40–60 respondents × about 800 items.
- **Binomial responses:** each respondent answers each item K = 8 times (T = 0.6), so the likelihood is
  Binomial(8, σ(a(θ − b))). This gives 8× the information per respondent-item and is a standard way to fight the small
  respondent count.

**Calibration panel**
- The panel is frozen before any arm is trained: about 40–60 open models, 0.5B–14B, base and instruct, at least 4 families,
  run with K = 8 on Kaggle T4 × 2 under vLLM. Add one high-ability anchor: Claude, via the API, which `00_SOLUTION` §4
  assigns to eval items.
- Panel members **must not be sent through NIM** (contamination control; `00_SOLUTION` §4).
- The experimental arms are **not** in the calibration. Using them would make the `arm × b_i` covariate in H1b circular.
- Responses from the arms are used only for the post-hoc anomaly screen (§3.1, stage 6).

**Uses, each with a decision rule**

1. **Item QA flags.** Send an item to mandatory human re-review (never auto-drop) if any of these hold:
   - posterior mean a_i < 0.2;
   - item-rest point-biserial < 0.1 (the ATLAS cutoff, E3);
   - Fantastic-Bugs mean tetrachoric < 0 or Z_j < 0 (E7).
2. **Core / Hard split.** Core = items whose b_i lies within ±1.5 logits of the ability range of the 0.6B–4B panel members.
   Hard = b_i above the 4B panel ability + 0.5 with a_i ≥ 0.5. Hard is leaderboard-only; Core feeds H1–H4.
3. **Test information.** Report the test information function I(θ) = Σ a_i² P_i(1 − P_i) and the marginal reliability.
   Requirement: I(θ) ≥ 10 (SE(θ) ≤ 0.32) across the 0.6B–4B ability band. Otherwise add Core items in the deficient
   difficulty band.
4. **Difficulty covariate b_i** for H1b (§2.2). Translated items get b_i from the same panel, run on them with K = 8.
5. **Temporal-split difficulty control (D5).** Raw pre/post accuracy gaps confound contamination with year-to-year
   difficulty, and E20 shows decay depends on item construction. The design has four parts:
   - **(a) Slot matching.** Grade-10 entrance exams keep a fixed per-province structure across years (the same "Bài k" slot
     type) [UNVERIFIED as a general rule; check per province during scraping]. Pair every post-cutoff item with a
     pre-cutoff item from the same province, slot and topic tag. All items go through the identical scrape → QA pipeline,
     which satisfies E20's "same construction" condition.
   - **(b) Reference-model DiD.** For target model M (cutoff c_M) and a reference model R whose cutoff precedes both item
     years, or whose release lies inside a window where both sets are unseen:
     `C_M = [acc_M(pre) − acc_M(post)] − [acc_R(pre) − acc_R(post)]`.
     Concretely: Qwen3 (2025-04-28) and Qwen3.5 (2026-02-28) on June-2025 vs June-2026 items. Qwen3 is clean on both, so
     it absorbs difficulty differences; Qwen3.5 is "pre" on the 2025 items. C > 0 indicates contamination.
     Test it with the GLMM `correct ~ model × period + (1|item) + (1|pair)`.
   - **(c) IRT residuals.** Fit item parameters using only respondents released before each item's date. Then compute each
     target model's residual r_i = y_i − P̂_i(θ̂_M), with θ̂_M estimated on post-cutoff items only. A positive mean residual
     on pre-cutoff items = excess accuracy on items it could have seen.
   - **(d) Report a, b and c side by side.** Claim contamination only if at least 2 of them agree in sign with p < 0.05.

**Why no adaptive / tinyBenchmarks subsetting.**
- Every run evaluates all items anyway; the cost is in §4.3.
- Subsetting breaks the item pairing that H1 relies on.
- Madaan (E5) found IRT does not reduce seed variance, and the power analysis shows that seed variance is the bottleneck.
- Fluid's gains are mainly validity and cost (E2). Cost is not our constraint at eval time.

### 2.6 What every results table must contain

Per run × dataset:
- mean pass@1 over K = 4, plus greedy accuracy and maj@4;
- truncation rate;
- judge-invocation rate;
- n_items.

Per contrast:
- estimate in pp with 95 % CI (GLMM and cluster bootstrap);
- Holm or BH p, labelled as confirmatory or exploratory.

Variance decomposition: σ²_item, σ²_item:arm, σ²_run, σ²_run:eval, from the GLMM.

Release item-level prediction JSONL for every run (`04` §2.3 already requires this).

---

## 3. Benchmark QA + contamination protocol for Vi-ExamMath

### 3.1 Item validation pipeline (D6)

**Unit of evaluation:** a **sub-question with an auto-gradable final answer** (number, expression, set, interval, tuple),
with its parent context prepended.
- Proof-only parts and figure-dependent parts are excluded from Core, unless the figure can be described in text losslessly.
  A text-described figure is tagged `figure_text`.
- Dependent sub-questions (b uses a's result) keep a's statement but not a's answer.

| Stage | Who | What | Output / rule |
|---|---|---|---|
| 0 Ingest | scraper | Official text, official key (Sở GD-ĐT PDF when available), `source_url`, `exam_date`, `first_seen` (Wayback CDX), province, slot, topic | Dedup inside the benchmark: MinHash (5-gram shingles, 128 permutations) Jaccard ≥ 0.8 → merge |
| 1 Normalise | script + human spot-check (5 %) | Vietnamese decimal comma, units, LaTeX; structured gold `{type, value, tolerance, form}` | Gold parses in `math_verify` round-trip |
| 2 Three independent keys | (i) official key; (ii) **Claude writes a SymPy/Z3 program, executed locally**; (iii) **Claude solves blind** in a separate call, without the key or the program. Optional (iv): a self-hosted open model ≥ 14B on Kaggle | All agree → status `consensus` (GSM8K-Platinum style status, E9). Any disagreement → stage 4 |
| 3 LLM critique | Claude, structured JSON | Checks ambiguity, missing conditions, multiple valid answers, unstated rounding/format, figure dependence, out-of-scope (proof), translation-free wording issues. Each check gives a label, a category and a ≤ 2-sentence justification (Fantastic-Bugs first-pass design, E7) | Any flag → stage 4 |
| 4 Human adjudication | **2 independent annotators** (math-trained, Vietnamese native), blind to each other, shown the item, the keys and the LLM justification | Label ∈ {valid, fix-key, fix-text, drop} × category {ambiguous, wrong-key, grading-format, figure, out-of-scope}. Disagreement → **3rd adjudicator** decides |
| 5 Random audit | the same 2 annotators | **150 items sampled uniformly from items with no flag** in stages 2–3 | Estimates the pipeline miss rate (Platinum's "lower bound" problem, E8) |
| 6 Psychometric screen | script (after the reference-panel run, §2.5) | IRT/statistical flags (a_i < 0.2, point-biserial < 0.1, tetrachoric/Z_j < 0) | Flagged → back to stage 4. Report precision of the flags (E7 reports ≤ 84 % precision@50) |
| 7 Freeze | — | Version, hash, canary GUID, password-zip release (`04` §2.1 DeepSEA practice) | Item-level status field: `consensus` / `verified` / `revised` |

**Why not Platinum-only (flag only items some model fails).** At grade-10 level, frontier models solve almost everything
(E27). Items with a wrong key that every model "agrees" with get past a failure-triggered review. Stages 2 and 5 close that
gap.

**Claude usage.** About 650 candidate items × 3 calls (program, blind solve, critique) × about 4k tokens ≈ **8M tokens**.
Check the per-model price before the run to stay inside the $100 eval-side budget [UNVERIFIED pricing]. Test items never go
to NIM or Gemini (`00_SOLUTION` §4).

### 3.2 Agreement statistics and sample sizes

| Quantity | Statistic | Target / decision rule | Sample |
|---|---|---|---|
| Annotator agreement, valid vs invalid | **Cohen's κ + Gwet's AC1 + raw % agreement**, each with a bootstrap CI | κ ≥ 0.6 **or** AC1 ≥ 0.8. Report both: with about 85–90 % valid items, κ is depressed by the prevalence paradox, which AC1 is robust to (standard psychometrics, not re-verified here) | All stage-4 items + the 150 audit items (≈ 300–350 double-labelled) |
| Agreement on the error category | **Krippendorff's α (nominal)** | α ≥ 0.67 for tentative conclusions | Items labelled invalid by ≥ 1 annotator |
| LLM-pipeline flagger vs final human label | **Sensitivity, specificity**, with Wilson CIs | Report. If sensitivity < 0.8, add a second critique pass with a different prompt | Stage-4 items (positives) + audit (negatives) |
| **Residual error rate of the released benchmark** | Poisson / Clopper–Pearson upper bound from the audit | 0 errors in 150 → ≤ 2.0 % (rule of three = 3/150). 1 error → ≤ 3.2 % (4.74/150). 2 errors → ≤ 4.2 % (6.30/150). Claim "estimated label-error rate ≤ X % (95 % upper bound)" | 150 audit items |
| Scorer validity (`math_verify` + Vietnamese normaliser) | FP and FN rate vs 2 humans | FP ≤ 1 %, FN ≤ 2 % on model outputs | 300 outputs, stratified by model size × dataset × scorer verdict |

**Human load:** about 350 double-labelled items + 300 scorer checks ≈ 650 reviews per annotator × about 3 min ≈
**33 h per annotator**, plus the adjudicator on about 15 % of items. This is the key open resource question from
`00_SOLUTION` open question 4 ("Người chấm": who annotates).

### 3.3 LLM-as-judge, adjudicator only (D9)

- **Invoke only when** (a) `math_verify` cannot parse the prediction or the gold, or (b) the parse succeeds but the result
  type mismatches (e.g. a set vs a tuple). The judge sees the reference answer: reference-guided binary judgement (E26).
- **Judge:** Claude (eval side). It must differ from the generator family (DeepSeek), which addresses self-preference.
- **Calibration:** the 300 human-labelled outputs of §3.2 → judge sensitivity Se and specificity Sp.
- **Correction:** if the judge decides > 5 % of a run's outputs, report the corrected accuracy among judged outputs with the
  Rogan–Gladen estimator θ̂ = (p̂ + Sp − 1)/(Se + Sp − 1). The CI must include calibration uncertainty (Lee et al., E24).
  PPI is the variance-efficient alternative (E25 [S]).
- Below 5 %, the maximum bias is ≤ 5 % × (1 − min(Se, Sp)), i.e. < 0.5 pp if Se and Sp ≥ 0.9. Report it, without
  correction.
- **Balance check:** report the judge-invocation rate per arm. An arm with systematically more unparseable outputs (format
  drift) has its accuracy partly judge-determined. If the invocation rates differ by > 3 pp between arms in a confirmatory
  contrast, rerun that contrast on the judge-free subset (items parsed for all runs) as a robustness check.

### 3.4 Contamination protocol: what to report (D7)

| Evidence | Applies to | Procedure | Report | Credibility 2026 |
|---|---|---|---|---|
| **C1 Temporal DiD + IRT residual** | pretraining of every evaluated model | §2.5 (5a–d); dates = exam date and `first_seen`; per-model cutoff = HF repo creation date (`00_SOLUTION` §4) | Table per model: pre/post accuracy, DiD with CI, mean IRT residual pre/post | High *when construction is identical* (E20 caveat). **Headline** |
| **C2 Training-corpus overlap** (corpora we control) | every training arm, the seed sets, 5CD-AI, LuminarAI, hllj, toanmath-alpha labels | (i) exact **13-gram** on normalised text (NFC, lowercase, numbers → `<num>`), plus a diacritic-stripped pass; (ii) **MinHash LSH** on 5-gram shingles, Jaccard ≥ 0.5; (iii) **cross-lingual embedding** (multilingual encoder, e.g. BGE-M3 [UNVERIFIED choice]), Vi benchmark vs EN seed (GSM8K/MATH train) and vs Vi corpora. Threshold calibrated so that FPR = 1 % on random pairs and recall is measured on known translation pairs (BenchMAX-vi ↔ Vi-GSM8K, 250 pairs) | Hit counts per arm × eval set, before and after removal. Remove all hits from training **before** training (target 0 after) | High for what we can see. Catches the cross-lingual route (E22) |
| **C3 Numeric perturbation** (GSM-Symbolic style) | all models | ≥ 150 Core items with a verified SymPy program (stage 2) → resample the numbers within the original domain, recompute the gold, human-check 20 % | Paired drop (original − perturbed) per model, paired GLMM/McNemar. Pre- vs post-cutoff items × perturbation interaction | High (E17, E18). A memorised item drops more |
| **C4 Partial-prompt completion** | **base** models only | Give the first 60 % of the item text; measure ROUGE-L / exact completion of the rest and answer accuracy without the full question (Wu et al., E19) | Rate on pre vs post items; flag items with exact completion | Medium-high when positive; a negative is weak |
| C5 Min-K%++ / other MIA | base models | Standard | **Appendix only**, never "not contaminated" | Low (E21, E23) |
| C6 Distractor-clause slice | all | Add one irrelevant Vietnamese clause to 100 Core items (E18) | Drop per model | Robustness, not contamination |
| Prevention | release | Canary GUID, password zip, no test text to NIM/Gemini, Claude only on the eval side | Statement in the datasheet | — |

**Interpretation rule:**
- Say "**no evidence of contamination**" for model M only if C1 DiD ≤ 2 pp with CI upper bound ≤ 5 pp, C3 shows no
  pre > post perturbation interaction, and C4 has no exact completions.
- Never say "uncontaminated".
- For the arms (our SFT data), the C2 post-removal count of 0 is the claim.

### 3.5 Robustness slices (exploratory, BH-FDR)

- (i) Numeric perturbation (C3).
- (ii) Distractor clause (C6).
- (iii) **Translation-variance probe:** BenchMAX-vi vs our Vi-GSM8K on the same 250 items (`04` §2.3). Paired per model;
  separates translation artifacts from data-source effects.
- (iv) **Prompt paraphrase:** 2 alternative Vietnamese instruction wordings on Core. Report the max−min spread per arm
  (FormatSpread-lite, E16).
- (v) **Answer-form slice:** items whose gold is a fraction, decimal or interval, to catch format-convention effects of
  translated data.

---

## 4. Decoding and evaluation protocol for small models (D8)

### 4.1 Decisions

| Setting | Choice | Alternatives | Evidence / reason |
|---|---|---|---|
| Primary metric | **mean pass@1 over K = 4 samples** per item | greedy pass@1; maj@k; pass@k | Halves the per-item Bernoulli term at fixed n. The power gain is visible in §2.4 (SD 2.51 → 1.79 pp at 2B, n = 400, S = 3) [R]. Miller: resample rather than lower the temperature (E10) |
| Temperature / top-p / top-k | **T = 0.6, top-p = 0.95, top-k = 20, min-p = 0**, identical for every arm, model and dataset | T = 0.7 / top-p 0.8 (Qwen3 non-thinking); T = 0.8 / top-p 0.9 (Sober Look); tune per model | Qwen3 card values (E15). Sober Look shows T alone moves results by up to 15 % (E12), so **fix one setting**. Per-arm tuning would leak into the arm effect. Sweep T ∈ {0.3, 0.6, 1.0} once, on S0 and T at 2B only, as a sensitivity row |
| Greedy | **Secondary**, reported for every run | Primary greedy | Literature comparability. Song: greedy ≥ sampling on many tasks (E14). Qwen3 warns of repetition loops (E15). Report greedy repetition and truncation rates |
| maj@k / pass@k | **maj@4 reported** (free given K = 4). **pass@k only** for the C4 (RLVR) analysis as a ceiling, with the unbiased estimator | pass@k as headline | pass@k rises with k from the tail and does not measure deployable accuracy (Lewkowycz et al., via a search summary [S]) |
| max_new_tokens | **2048** for short-CoT SFT students. **4096** if the S0 pilot shows > 1 % truncation; then use it for all arms | 32k (Qwen3 card, for thinking models) | Students are trained on short CoT (`00_SOLUTION` §6.1). Truncated = wrong, and the truncation rate is reported per run. Sober Look: near-limit outputs are almost always wrong (E12) |
| Prompt | **Exactly the SFT training template**, zero-shot, Vietnamese instruction, `\boxed{}` final answer | few-shot for SFT models | Fine-tuned arms share a template, so prompt sensitivity is held constant. Base/zero-shot baselines: **4-shot fixed Vietnamese exemplars** plus a 3-template spread (E16) |
| Chat template | Same tokenizer template as training. Base models: plain text | — | Sober Look: omitting templates causes "consistent performance drops" (E12) |
| Answer extraction | Last `\boxed{}` → `math_verify` 0.9.0 + Vietnamese normaliser (decimal comma, units, "x = ") → judge only per §3.3 | Regex last number | One scoring function across arms (`04` §4.4) |
| Determinism | vLLM pinned version, fixed `seed` per (run, item, sample), same GPU type per model size, log the hardware | — | Hardware/stack differences of up to 8 % (E12). Re-score, never re-generate, when the scorer changes |
| Reference panel (IRT) | K = 8, same T / top-p | — | §2.5 |

### 4.2 Small-model specifics

- **Repetition loops at 0.6–0.8B.** Count outputs that hit the max-token limit with a repeated n-gram (n = 20, ≥ 3
  repeats). If a confirmatory arm has > 5 % such outputs, report it. Do **not** add a repetition penalty to one arm only;
  any penalty is global and pre-registered.
- **Floor effects.** If a model's Core accuracy is < 10 % for the S0 arm, exclude that model from the pooled H1–H3 contrasts
  (pre-registered). Logit-scale interactions are unstable near the floor (`00_SOLUTION` §5.2 already drops Gemma-3-1B for
  this reason).
- **Seeds.** Training seed = data order + LoRA init + the arm's subsample. Sampling seed = a hash of (run, item, k). Report
  both.

### 4.3 Compute cost of evaluation

**Per run:**

| Part | Items | Samples each | Generations |
|---|---|---|---|
| Primary items (Core 500 + translated subsample 1,000) | 1,500 | K = 4 | 6,000 |
| Secondary items, greedy (remaining Vi-GSM8K / Vi-MATH, PolyMath, BenchMAX, Hard, slices) | ≈ 3,600 | 1 | ≈ 3,600 |
| Greedy pass on primary items | 1,500 | 1 | 1,500 |
| **Total** | | | **≈ 11k generations** |

At about 350 output tokens each, that is ≈ 3.9M tokens per run.

**Throughput:**
- Assumed ≈ 4k tok/s for a 2B model on an L4, and ≈ 2.5k tok/s for 4B [UNVERIFIED; measure in the pilot].
- That gives ≈ 16 min per run for 2B and ≈ 26 min for 4B.

**Grid totals:**
- About 110 runs (8 arms × 4 models × 3 seeds + 10 extra 2B seeds) ≈ **30–35 GPU-h**.
- Reference panel: 50 models × 1,500 items × 8 samples ≈ **15 GPU-h** on Kaggle T4 × 2.

**Where it runs:**
- Kaggle (30 h/week, free) carries the panel and most eval.
- Modal handles 4B and Gemma eval (`00_SOLUTION` §6.2 budget).
- GLMM, bootstrap and IRT fits run on CPU (Kaggle/Colab R kernel and the laptop), at $0.

---

## 5. Open risks and unverified items

1. **Variance components are assumed, not measured** (σ_item 1.5, τ 0.5, σ_run×eval 0.07 logit). The sensitivity table
   shows how the conclusions move. The **pilot rule in §2.4** replaces them with measured values before the grid is
   committed.
2. Core ≥ 500 auto-gradable sub-questions depends on scraping yield. If the yield is below 400, the per-model 5 pp target is
   lost; the pooled about-3.5 pp target survives with S = 5.
3. Slot-matching stability across years is assumed [UNVERIFIED].
4. Claude token pricing, vLLM throughput, and the BGE-M3 choice are [UNVERIFIED].
5. E19's numbers, E22, E23 (Fu), E25 and E28 were seen in snippets only.

---

## Appendix A — power simulation code [R]

Run: `python -I power_sim.py {typeI|main|sens}`. The pooled variant (`power_pooled.py`) imports `simulate`'s helpers and
averages the per-size DiD over 3 sizes with shared items and item-slope correlation ρ. Runtime: about 25 min on the laptop
CPU for `main` (225 cells × 1,500 replicates). Only numpy is needed.

```python
import math, sys, itertools
import numpy as np
rng = np.random.default_rng(20261107)
SQ2 = math.sqrt(2.0)
def Phi(x): return 0.5 * (1 + math.erf(x / SQ2))
def zq(p):                                   # inverse normal by bisection
    lo, hi = -10, 10
    for _ in range(100):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if Phi(m) < p else (lo, m)
    return (lo + hi) / 2
GH_x, GH_w = np.polynomial.hermite.hermgauss(60)
def marg(b, sd):                             # E[expit(b + sd*Z)]
    z = SQ2 * GH_x
    return float(np.sum(GH_w / (1 + np.exp(-(b + sd * z)))) / math.sqrt(math.pi))
def solve_b(p, sd):                          # intercept giving marginal accuracy p
    lo, hi = -15, 15
    for _ in range(80):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if marg(m, sd) < p else (lo, m)
    return (lo + hi) / 2

def simulate(pN, pT, Delta, nN, nT, S, K, s_item=1.5, tau=0.5, s_run=0.10, s_re=0.07, reps=2000, chunk=250):
    sd_tot = math.sqrt(s_item**2 + (tau / 2)**2 + s_run**2 + s_re**2)
    bN, bT = solve_b(pN, sd_tot), solve_b(pT, sd_tot)
    dN = solve_b(pN + Delta, sd_tot) - bN            # logit shift of arm A on native items
    n = nN + nT
    b = np.r_[np.full(nN, bN), np.full(nT, bT)]
    isN = np.r_[np.ones(nN, bool), np.zeros(nT, bool)]
    est, feas_se, done = [], [], 0
    c = np.array([0.5, -0.5]).reshape(1, 2, 1, 1)    # arm A, arm B
    while done < reps:
        R = min(chunk, reps - done); done += R
        u = rng.normal(0, s_item, (R, 1, 1, n)); g = rng.normal(0, tau, (R, 1, 1, n))
        delta = np.zeros((1, 2, 1, n)); delta[0, 0, 0, isN] = dN
        v = rng.normal(0, s_run, (R, 2, S, 1))
        w = np.where(isN.reshape(1, 1, 1, n), rng.normal(0, s_re, (R, 2, S, 1)), rng.normal(0, s_re, (R, 2, S, 1)))
        p = 1 / (1 + np.exp(-(b.reshape(1, 1, 1, n) + u + c * g + delta + v + w)))
        y = rng.binomial(K, p) / K                      # (R, arm, seed, item)
        d = y[:, 0] - y[:, 1]
        Is = d[:, :, isN].mean(axis=2) - d[:, :, ~isN].mean(axis=2)   # per-seed DiD
        est.append(Is.mean(axis=1))
        Di = d.mean(axis=1)
        se2 = Di[:, isN].var(axis=1, ddof=1) / nN + Di[:, ~isN].var(axis=1, ddof=1) / nT
        if S >= 2: se2 = se2 + Is.var(axis=1, ddof=1) / S
        feas_se.append(np.sqrt(se2))
    return np.concatenate(est), np.concatenate(feas_se)

def power(pN, pT, Delta, nN, nT, S, K, alpha=0.0125, **kw):
    z = zq(1 - alpha / 2)
    est, se = simulate(pN, pT, Delta, nN, nT, S, K, **kw)
    sd = est.std(ddof=1)
    return est.mean(), sd, float(np.mean(np.abs(est) / sd > z)), float(np.mean(np.abs(est) / se > z))
# sizes: {"0.6-0.8B": (0.25, 0.35), "1.7-2B": (0.45, 0.55), "4B": (0.65, 0.75)}; nT = 1000
```
