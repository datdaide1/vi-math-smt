# E1 — Optimal design of the English mirror for P2 (2026-11-07)

**Question (P2, `01_TECHNICAL_DESIGN.md` §1, §3.7):** is the symbolic/verification premium larger in Vietnamese than in
English? This report decides how big the English mirror must be, which arms, sizes, seeds and data budgets it needs,
what it costs in T4-hours and generation calls, and which English evaluation sets to use.

**Evidence tags.**
- **[O]** = paper or page opened in this session (arXiv HTML/abstract).
- **[P:x]** = taken from project report x.
- **[M]** = measured in the project (file path given).
- **[R]** = my own computation. All code and raw outputs are in the appendices.
- **[EST]** = an assumption or arithmetic estimate. Treat compute figures as ±2× (same convention as T3/T6).
- **[UNVERIFIED]** = not checked against a primary source.

---

## 0. Decision

**Choose option O3 ("interaction core at two sizes, 5 seeds"), plus two small anchors.**

| Block | Arms | Sizes (Qwen3-Base) | Seeds | Data budget | Runs |
|---|---|---|---|---|---|
| **EN mirror (confirmatory)** | Y+, Y−, N+, N− | 0.6B, 1.7B | **5** | 10k items, 2 epochs, frozen VI recipe | 40 |
| EN anchor (descriptive) | S0-EN (seed problems only) | 0.6B, 1.7B | 3 | the same seed problems in English | 6 |
| EN recipe-transfer check | S0-EN at LR ×½ and ×2 | 1.7B | 1 | as above | 2 |
| **VI top-up (balances P2 cells)** | Y+, Y−, N+, N− seeds 4–5 | 0.6B | +2 | 10k (existing VI pools) | 8 |
| VI top-up | Y−, N− seeds 4–5 (Y+/N+ already have 5, T5 D3) | 1.7B | +2 | 10k | 4 |
| **Total** | | | | | **48 EN + 12 VI = 60 runs** |

- **Cost [R, ±2×]:** 76.7 T4-h training + 37.1 T4-h evaluation = **113.8 T4-h**.
  - At **30 T4-h/week** that is **3.8 weeks**.
  - If Kaggle bills one quota-hour for both GPUs of a T4×2 session (T3 gate G8, still unverified), two lanes give
    60 T4-h/week and it takes **1.9 weeks**.
  - This adds about 17–20 % to T3's 560–680 T4-h plan.
  - Everything stays on Kaggle T4 fp16, the same hardware and precision as the VI cells it is compared with.
- **Generation [R]:** about **269k NIM calls** for English pools of 20k per generated arm, the same pool size as the VI
  side. That is **4.7 days at 40 RPM**, $0, and runs in parallel with the GPU work.
- **Power [R]:** for the P2 contrast (language × verification, pooled over the two sizes):
  - MDE80 = **3.2 pp** at α = .0125 (the first Holm step in T5) and **2.7 pp** at α = .05;
  - power is 0.73 / 0.88 for a true 3 pp effect, and 0.95 / 0.99 for 4 pp;
  - a TOST with margin ±3 pp has **0.87** power when the true interaction is 0, so a null result is also publishable.
- **English evaluation sets:** the **exact English originals of the Vietnamese parallel primary subsample**:
  - 600 GSM8K-Platinum items;
  - 400 MATH-test algebra/prealgebra items, which must include all 206 MATH-500 algebra/prealgebra items.
  - Secondary sets: full GSM8K-Platinum (1,209) and MATH-500 algebra/prealgebra (206), for comparison with the
    literature.
  - Not used: MGSM, the original GSM8K test, the rest of MATH-500, or an English translation of Core (reasons in §5.6).
- **3-way interaction** (language × method × verification): MDE80 = 6.2 pp pooled. It is **exploratory** (BH-FDR).
  No affordable design brings it below about 5 pp (§3.3).
- **Pilot-adaptive rule:** after the T5 pilot measures the run-to-run variance (§5.4):
  - if σ_run ≥ 0.15 logit, add seeds at 0.6B (S = 8, +30 T4-h; this is option O7);
  - if σ_run ≤ 0.08 logit, cut to S = 3 (option O3b, −54 T4-h).

---

## 1. What P2 needs to estimate

Notation:
- VP_L = the verification premium in language L, averaged over the two generation methods:
  ½[(Y+ − Y−) + (N+ − N−)].
- Every accuracy below is on the **parallel** item set: the same problems in English and in Vietnamese.

| Contrast | Formula | Role |
|---|---|---|
| **I2: language × verification (P2)** | VP_vi − VP_en, averaged over sizes | **Confirmatory** |
| I3: language × method × verification | [(Y+−Y−) − (N+−N−)]_vi − [same]_en | Exploratory: "is the *symbolic* certificate's premium what grows?" |
| Language × method | [mean(Y) − mean(N)]_vi − [same]_en | Exploratory. It has the same variance class as I2 |
| **cap: capability-matched premium** | VP_vi(1.7B) − VP_en(0.6B) | Separates "it is Vietnamese" from "the student is weaker in this language" (§5.2) |
| Manipulation checks | Verification rejection rate per language; audited error rate of Y−/N− data per language | Without them a null I2 cannot be interpreted |

**Why this differs from T5's design.** T5's arm × eval-type contrast is computed *within* runs: every run sees both native
and translated items, so the run main effect cancels and only σ_run×eval (0.07) remains (T5 §2.4). In the mirror, each run
is trained in **one** language and evaluated in **that** language. I2 is therefore a contrast **between runs**. The
whole run variance enters, and it supplies **≈ 91 % of Var(I2)** at the planned design [R, `decomp.py`, App. D]. This is
the single most important design fact: **seeds buy power; extra samples K and extra items buy almost nothing.**

---

## 2. How prior cross-lingual data-method studies built their language controls

### 2.1 Evidence table

| Study | How the language conditions were built | English control | Seeds / statistics | Evaluation |
|---|---|---|---|---|
| **2508.14828** Barua et al., *Long CoT across languages* (ICLR 2026) [O] | s1K (1,000 English traces) machine-translated into 9 languages (*Translated-s1k*) vs traces distilled in the target language (*Distilled-s1k*). Qwen3-8B-Base, one model per dataset × language, **trained separately** | §6's main English baseline is **OpenThoughts3-20k**, 20× more data. A **compute-matched English-s1k** control appears only in App. B.4, and a replication on Olmo3-7B in B.5 | Training seeds not reported; no CIs or tests. AIME uses 5 samples per item (evaluation variance only) | MATH-500 and AIME machine-translated (Gemini-2.0-Flash), MMLU-ProX |
| **2505.19848** Ogundepo et al., *African languages* [O] | 30k translated OpenMathInstruct / BigMath vs 30k native synthetic (AfriPersona), Llama-3.1-8B | **None.** No English-trained condition. "Eng" is only an evaluation column of AfriMGSM | Single run per cell; no variance, CIs or tests. Some reported differences are < 1 pt (e.g. 39.4 vs 38.7) | AfriMGSM (250 human-translated GSM8K items), GPT-4o judge |
| **2604.13286** *English is Not All You Need* [O] | 22 mixtures × 2 tasks × 5 models (Qwen3 0.6/1.7/8B, Gemma-3 1/4B) = 220 runs. **Parallel** mCoT-MATH, 10k per language, equal examples per language, **joint** multilingual training | "English-only post-training" is the main control | No seeds reported (one run per configuration). Bootstrap and Wilson CIs on *aggregates across configurations*. Adding one non-English language raises English math by a median **+3.4 %** | MGSM, human-translated |
| **2509.23657** (multilingual SFT vs RL, Qwen2.5-3B) [O] | GSM8K + LUFFY translated by Qwen3-30B-A3B into En/Zh/De(+4). **Same source problems**, separate models per training language | Training on the English originals | Training seeds not stated; evaluation averaged over 6 runs. SFT: En 45.2 vs Zh 46.9, De 46.3 (Table 7). RL: En 62.7 vs De 71.5 | MGSM (10 languages incl. English), MMath500 |
| **2602.19187** Yeo et al., symbolic vs NL modification [O] | Same 1,500 GSM8K seeds, same generator (GPT-5-mini), 4 prompts × 1 datapoint per seed per arm; RLVR on Qwen2.5-1.5B/3B | English only, so no language contrast | No training seeds or CIs. Sym − NL ranges from −0.6 to +2.1 pts | GSM8K, GSM-Sym, MATH-500, AIME24, greedy |
| **MGSM** (2210.03057) [O abstract] | 250 GSM8K test problems **manually translated** into 10 languages | The same 250 items in English | — | The template for a parallel evaluation set |
| **Multilingual GSM-Symbolic** (2610.03367) [O abstract] | 30k QA in 15 languages, **item-matched** from symbolic templates | The same items in English | Regression on language/model factors; 92 % of between-language variance explained | Item-matched parallel evaluation |

The 2026 search for a study that runs **the same synthesis and verification pipeline in a low-resource language and in
English** and compares the verification premium found nothing [O: search 2026-11-07; nearest hits were 2605.26132,
English-only, and 2505.19848, no verification]. P2 is therefore open. It is also first-of-kind, so the control has to be
defensible without a precedent to point to.

### 2.2 What a reviewer will accept: requirements derived from the table

| # | Requirement | Where the evidence comes from | How E1 meets it |
|---|---|---|---|
| R1 | **The same source problems in both languages** ("same seeds" = the same seed items) | 2509.23657 and 2604.13286 use the same or parallel sources. MGSM and M-GSM-Symbolic make items parallel so that language is not confounded with content | English arms are seeded from the **English originals of the same GSM8K/MATH-alg/prealg train rows** whose translations seed the VI arms. Same operator schedule π, same slot IDs, same RNG seed per slot |
| R2 | **The same data budget, recipe and compute in both languages** | 2508.14828 had to add a compute-matched English-s1k control (App. B.4) because its main English baseline used 20× more data | 10k items, 2 epochs, the same 32 seq/step, and LR/epoch **frozen from VI S0**. One recipe-transfer check (S0-EN at LR ×½, ×2) answers "was the recipe tuned for Vietnamese?" |
| R3 | **The same generator and the same gate models** | 2602.19187 holds the generator fixed across arms. Otherwise "language" is confounded with "model" | `deepseek-v4.1-flash` generates in both languages. The formalizer-B, judges and solution writer are the same models. Only the prompt language changes |
| R4 | **Translated prompts with structural parity** | No study reports a parity protocol, which makes it an easy reviewer objection | VI prompts are translated to English; structure, constraints, schema and few-shot items are kept 1:1 (the exemplars are the English originals of the VI exemplars); length within ±15 %. Both versions are versioned and published. Language-specific gate parts (numeral parser, the `vi_ratio` residue check) are replaced by their English equivalents, and nothing else changes |
| R5 | **Parallel evaluation items, scored by the same grader logic** | MGSM, M-GSM-Symbolic, 2509.23657 | The **same 1,000 item IDs** in English and Vietnamese (§5.6). One `math_verify` with a locale switch. A grader-parity audit scores the 1,000 gold answers plus 200 outputs per language, and the target is 0 disagreements |
| R6 | **Multiple training seeds and run-aware inference** | None of the five studies reports training seeds. 2505.19848 interprets sub-1-pt differences from single runs. A 2027 reviewer of a *variance-sensitive interaction claim* will not accept that | 5 seeds per cell, a GLMM with a run random effect, a cluster bootstrap over seeds and items, and TOST (§5.5) |
| R7 | **Separate (monolingual) training, not a joint mixture** | 2604.13286: adding one language changes English math by a median +3.4 %, so a joint model's premium is contaminated by transfer | Separate models per language (option O8 rejected, §6) |
| R8 | **Manipulation check: verification must actually remove more defects in VI** | P2's mechanism is that machine-made VI math has 2–19 % systematic semantic errors [P: R1, R2, R6] | Rejection rate per gate per language, plus the T1 quality gate (200 items per arm) on the English arms too |

**Same vs different training seeds.** "Same seeds" also has a second meaning: common random numbers. The VI run and the
EN run with the same (size, arm, seed index) share the LoRA-init seed, the data-order seed and the subsample slot IDs.
- This costs nothing and cannot bias the result.
- If it induces a run correlation ρ_crn = 0.3, the pooled MDE drops from 3.2 to 2.8 pp [R, `sens`].
- Adopt it. Model it with a `(1 | size:arm:seed_idx)` term, and report the estimated correlation.

**Different seed problems** (for example, English-native problems not drawn from GSM8K/MATH) would add a content
confound that no analysis can remove. Rejected.

---

## 3. Power analysis [R]

### 3.1 Model (adapted from T5 §2.4 / App. A; code in App. A)

```
logit p[m,L,a,s,i] = b_{m,L} + d_{m,L,a} + u_i + e_{i,L} + slope_{m,L,a,i} + v_{m,L,a,s}
u_i      ~ N(0, 1.5²)     item difficulty, shared by the EN original and its VI translation (T5 σ_item)
e_{i,L}  ~ N(0, 0.5²)     language-specific difficulty (the translation shifts difficulty)  [EST; 1.0 in sensitivity]
slope    ~ per-arm SD 0.5/√2 (pairwise SD 0.5 = T5 τ); corr ρ_l = 0.5 across languages, ρ_s = 0.5 across sizes
v        ~ N(0, 0.122²)   run effect = √(0.10² + 0.07²): T5's run main effect plus run×eval-type,
                           because in the mirror nothing cancels
K = 4 samples per item, n = 1,000 parallel items, 2,000 Monte-Carlo replicates, seed 20261107
```

**Accuracies of the unverified arms on the parallel set** [EST], as (VI, EN):
- 0.6B: (0.31, 0.49); 1.7B: (0.49, 0.63); 4B: (0.63, 0.74).

They are anchored on these numbers:
- the measured 4-shot Vi-GSM8K accuracies, 38.3 % (0.6B) and 58.7 % (1.7B) [M: `jobs/outputs/w2-pilot-v1/results/*/eval.json`];
- the Qwen3 technical-report English GSM8K figures, 59.6 / 75.4 / 87.8 [P: 00_SOLUTION §5.2];
- scaled down for the 40 % MATH share.

**The premium.** VP_en = 2 pp (the order of 2602.19187's Sym−NL gaps [O]). The alternative adds D to VP_vi.

**Robustness to the assumptions.**
- SD(I2) moves little with the accuracy levels (§3.3); the item parameters move it by ≤ 0.1 pp.
- What matters is σ_run, which the T5 pilot measures.
- **Normal-approximation check:** for D = 2/3/4 pp the simulated power and the approximation Φ(D/SD − z) agree within
  0.01 [R, `power`]. The MDEs below are (z_{1−α/2} + 0.842) × SD.

### 3.2 MDE of each option (pooled over the sizes in the option) [R, `main`]

| Option | Design (EN arms; sizes; seeds VI/EN) | SD(I2) pp | MDE80 I2, α=.05 / .0125 | MDE80 I3 (3-way), α=.05 / .0125 | Capability-matched contrast, α=.05 | TOST ±3 pp power (true 0) |
|---|---|---|---|---|---|---|
| O1 Full mirror | 7 EN arms; 0.6/1.7/4B; S3 (S5 for 5 arms at 1.7B) | 0.90 | 2.5 / 3.0 | 5.0 / 6.0 | 4.4 | 0.91 |
| O2 Interaction, one size | Y±N±; 1.7B; 5/5 | 1.34 | 3.8 / 4.5 | 7.6 / 9.0 | — | 0.45 |
| **O3 Interaction, two sizes (chosen)** | Y±N±; 0.6+1.7B; 5/5 | **0.95** | **2.7 / 3.2** | 5.4 / 6.5 | **3.9** | **0.87** |
| O3b as O3, 3 seeds | Y±N±; 0.6+1.7B; 3/3 | 1.19 | 3.3 / 4.0 | 6.8 / 8.1 | 5.0 | 0.62 |
| O4 Interaction, three sizes | Y±N±; 0.6/1.7/4B; 3/3 | 1.00 | 2.8 / 3.3 | 5.4 / 6.4 | 5.2 | 0.82 |
| O5a Fewer EN seeds, K=4 | Y±N±; 1.7B; 5/2 | 1.78 | 5.0 / 5.9 | 9.7 / 11.6 | — | 0.03 |
| O5b Fewer EN seeds, K=8 | Y±N±; 1.7B; 5/2 | 1.68 | 4.7 / 5.6 | 9.8 / 11.7 | — | 0.11 |
| O5c Fewer EN seeds, K=8, two sizes | Y±N±; 0.6+1.7B; 5/3 | 1.09 | 3.1 / 3.6 | 6.0 / 7.1 | 4.4 | 0.73 |
| O6 Small size only, many seeds | Y±N±; 0.6B; 8/8 | 1.07 | 3.0 / 3.6 | 5.9 / 7.1 | — | 0.75 |
| O7 = O3 + 0.6B at S8 | Y±N±; 0.6B S8 + 1.7B S5 | 0.88 | 2.5 / 2.9 | 4.8 / 5.7 | 3.5 | 0.92 |
| O8 Joint vi+en training | per method: VI± × EN± × S3; 1.7B | 1.24 | 3.5 / 4.2 | — | — | 0.56 |

How each column is defined:
- **O1's** I2 uses only its Y±/N± cells. The extra arms (S0, T, M) add **no** power to P2.
- **O8** assumes zero cross-lingual transfer between the two halves of the mixture. That is the design's most favourable
  case; 2604.13286 shows the transfer is real.
- **TOST power** = 2Φ(3/SD − 1.645) − 1 [R, `tost.py`].

### 3.3 Sensitivity of O3 [R, `sens`]

| Scenario | SD(I2) pooled | MDE80 .0125 | MDE80 I3 .0125 | MDE80 I2 1.7B alone .0125 |
|---|---|---|---|---|
| Base | 0.96 | 3.2 | 6.2 | 4.6 |
| σ_run = 0.06 | 0.56 | 1.9 | 3.7 | 2.6 |
| **σ_run = 0.20** | **1.48** | **5.0** | 10.1 | 7.0 |
| ρ_l = 0 (item×arm effects language-specific) | 0.99 | 3.3 | 6.5 | 4.6 |
| ρ_l = 1 | 0.91 | 3.0 | 6.2 | 4.4 |
| τ = 0.8 | 0.95 | 3.2 | 6.5 | 4.5 |
| ρ_crn = 0.3 (paired training seeds) | 0.83 | 2.8 | 5.5 | 4.0 |
| n = 600 (GSM8K-Platinum only) | 0.97 | 3.2 | 6.6 | 4.6 |
| K = 1 (greedy-like) | 1.07 | 3.6 | 6.9 | 5.1 |
| σ_lang-item = 1.0 | 0.91 | 3.0 | 5.8 | 4.2 |

**Reading.**
1. Only σ_run matters. Every item-side parameter moves the MDE by ≤ 0.3 pp.
2. Cutting items from 1,000 to 600 costs nothing, and K = 4 vs 1 is worth only 0.4 pp.
3. The closed form agrees with the simulation (App. D): at 1.7B, S = 5, SD = 1.37 (closed form) vs 1.34 pp
   (simulation), with run variance 91 % of the total. So "fewer English seeds, compensated by K" (O5a/b) is the wrong
   trade: going from K = 4 to K = 8 buys 0.3 pp of MDE, while dropping from 5 seeds to 2 costs 1.4 pp.
4. **The 3-way contrast stays ≥ 4.8 pp** even in the most expensive design (O7). It is therefore exploratory. No option
   with a credible cost gets it near 3 pp.

---

## 4. Compute and generation per option [R, `compute_cost.py`]

### 4.1 Per-run cost (T4 fp16)

**Training:**
- T3 §1.2 FLOP-based hours (padding-free; 9M tokens per run): 0.42 h (0.6B), 1.83 h (1.7B, gradient checkpointing on),
  4.27 h (4B).
- **×1.3 for padded, length-grouped batching.** Padding-free packing leaked across sequences on T4. Measured:
  `packcheck.json` shows max |Δlogit| up to 23 and packed NLL 3.7–4.2 vs 0.66–0.88 separate, while padded rows match
  (≤ 0.18) [M: `jobs/outputs/w2-pilot-v1/results/Qwen3-0.6B-Base/packcheck.json`].
- **×0.873 for English.** The Qwen3 tokenizer gives VI/EN = 1.146 token ratio on the parallel train data (problem +
  solution): GSM8K 1.170, MATH-algebra 1.081, MATH-prealgebra 1.105 [R, `fertility.py`, App. C].

**Evaluation:**
- Measured anchor: 300 four-shot items take 87 s (0.6B) and 92 s (1.7B) of generation on one T4, plus 118–150 s of
  model load [M: `eval.json`].
- Linear extrapolation is an **upper bound**, because vLLM batches better with thousands of prompts.
- English: 1,000 items × (K = 4 + greedy) + ≈ 900 greedy secondary items = 5.9k generations → **0.53 T4-h**.
- Vietnamese (T5 §4.3): 11k generations → **0.96 T4-h**.
- 4B: ×2.

| Run type | Train T4-h | Eval T4-h | Total |
|---|---|---|---|
| VI 0.6B / 1.7B / 4B | 0.55 / 2.38 / 5.55 | 0.96 / 0.96 / 1.92 | 1.50 / 3.34 / 7.47 |
| EN 0.6B / 1.7B / 4B | 0.48 / 2.08 / 4.84 | 0.53 / 0.53 / 1.07 | 1.01 / 2.61 / 5.91 |

### 4.2 Option table

| Option | EN runs | VI top-up runs | T4-h | Weeks at 30 / 60 T4-h per week | MDE80 I2 (.0125) | Precision per T4-h = 1/(SD²·h) | Answers P2 credibly? |
|---|---|---|---|---|---|---|---|
| O1 Full mirror | 73 | 4 | 239.5 (+≈ 83 with the 2nd family) | 8.0 / 4.0 | 3.0 | 0.0052 | Yes, but at 2.4× O3's cost for ≈ 0.2 pp. T, T-off and M have no English meaning (T-EN is just MetaMathQA in its original language), so a "full" mirror is not well defined |
| O2 | 20 | 4 | 65.5 | 2.2 / 1.1 | 4.5 | 0.0085 | No: one size, TOST power 0.45 |
| **O3** | **40** | **12** | **97.8** | **3.3 / 1.6** | **3.2** | **0.0113** | **Yes** |
| O3b | 24 | 0 | 43.4 | 1.4 / 0.7 | 4.0 | **0.0163** | Borderline: best per hour, but a 4 pp MDE misses the plausible 2–3 pp effect, and a null cannot be certified (TOST 0.62) |
| O4 | 36 | 0 | 114.3 | 3.8 / 1.9 | 3.3 | 0.0087 | Yes, but costlier than O3 per unit precision, and it has no capability-matched pair |
| O5a | 8 | 4 | 34.2 | 1.1 / 0.6 | 5.9 | 0.0092 | No |
| O5b | 8 | 4 | 36.9 | 1.2 / 0.6 | 5.6 | 0.0096 | No |
| O5c | 24 | 12 | 76.8 | 2.6 / 1.3 | 3.6 | 0.0110 | Weak (TOST 0.73) |
| O6 | 32 | 20 | 62.4 | 2.1 / 1.0 | 3.6 | 0.0140 | No: one size, and the weakest students cannot separate language from competence |
| O7 | 52 | 24 | 127.9 | 4.3 / 2.1 | 2.9 | 0.0101 | Yes: the escalation path |
| O8 Joint | (24 bilingual runs) | — (no reuse of VI cells) | 142.7 | 4.8 / 2.4 | 4.2 | 0.0046 | No (§6) |

**The chosen package** = O3 + S0-EN anchor (6 runs) + recipe check (2 runs) = **113.8 T4-h** (76.7 train + 37.1 eval).
That is **3.8 weeks at 30 T4-h/week**, or 1.9 weeks if two lanes count as one quota-hour.

### 4.3 Generation API calls (NIM, $0) [R]

T6 §4.2 placeholder rates:
- N arm: 5 calls per accepted checked item.
- Y arm: 6 calls per accepted item, plus 1 formalization per seed (11k seeds per 20k items).
- Plus 10 % retries, plus ≈ 15k calls to regenerate English S0 solutions in the shared teacher style.

| EN pool per generated arm | NIM calls | Days at 40 RPM | Note |
|---|---|---|---|
| **20k (= the VI pool, chosen)** | **≈ 269k** | **4.7** | Same pool-to-budget ratio (10k drawn from 20k) as the VI cells, so run variance includes the same data-sampling variance |
| 12.5k (lean) | ≈ 174k | 3.0 | Use only if NIM becomes binding. EN run variance would then omit part of the data-sampling variance; report it |

Y−/N− pools come from the same streams (the pre-verification outputs), so they need no extra calls.

Claude spend: **$0**. The English evaluation sets are originals, so no translation is needed.

Human ratings: the T1 quality gate on the 4 English arms is about 4 × 200 × 1.5 min ≈ 20 person-h per rater [EST].

---

## 5. Decision and justification

### 5.1 Why O3

O3 is the best power per GPU-hour **among the designs that answer P2 credibly**. "Credible" here means all three of:
1. MDE ≤ about 3 pp at the confirmatory α, because the plausible effect is small (§5.3);
2. TOST ±3 pp power ≥ 0.8, so that a null result can be published;
3. at least two sizes, including a capability-matched pair (§5.2).

How the alternatives compare:
- O3b has higher precision per hour (0.0163 vs 0.0113) but fails 1 and 2.
- O6 is efficient (0.0140) but fails 3.
- O4 meets 1–2 at 0.0087, worse per hour, and fails 3.
- O1 and O7 meet everything, but cost 2.4× and 1.3× as much.

O3 therefore stays the base design. O7 is the pre-registered escalation if the pilot shows a large σ_run.

### 5.2 Why 0.6B + 1.7B rather than 1.7B + 4B

The estimated parallel-set accuracies are 0.49 for 0.6B in English and 0.49 for 1.7B in Vietnamese. The underlying
facts:
- English GSM8K 59.6 for Qwen3-0.6B-Base [P: Qwen3 TR via 00_SOLUTION §5.2];
- Vietnamese GSM8K 58.7 for Qwen3-1.7B-Base [M].

So **0.6B-EN and 1.7B-VI start at the same competence.**

This yields the contrast cap = VP_vi(1.7B) − VP_en(0.6B), with MDE 3.9 pp at α = .05. It answers the first question a
reviewer will ask: "Is the bigger Vietnamese premium about Vietnamese, or just about a weaker student?"

| If I2 > 0 and … | Then |
|---|---|
| cap > 0 too | The premium is language-specific: generator errors in Vietnamese |
| cap ≈ 0 | The premium tracks student competence, not language |

A 4B mirror cannot form such a pair. It also costs 5.9 T4-h per run against 1.0 for 0.6B.

**Day-1 check:** confirm the match with the same harness. Run 4-shot English GSM8K-Platinum 300 on Qwen3-0.6B-Base;
this costs about 0.05 T4-h.

### 5.3 Why ≈ 3 pp is the right resolution target

The literature's English verification and correctness premiums are small:
- Sym − NL ranges from −0.6 to +2.1 pts [O 2602.19187];
- 100 % wrong-answer traces cost only 3.2 pts at 32B [P: 01 A4];
- FLAMES finds coverage matters more than reliability [P: 01 A2].

The Vietnamese side has more defects to remove (2–19 % systematic errors [P: R1, R2, R6]), so a VP_vi − VP_en of 2–4 pp
is the plausible range [EST].

- **MDE ≥ 4 pp (O2, O3b, O5*):** likely to miss a real effect and unable to certify a null.
- **O3:** detects 3 pp with 0.73 power at α = .0125 (0.88 at α = .05) and 4 pp with 0.95. It certifies |I2| < 3 pp with
  0.87 power if there is no effect.

### 5.4 Pilot-adaptive rule (extends T5 §2.4)

The T5 pilot (S0 and T at 0.6B, 3 seeds) estimates σ_run and σ_run×eval. Compute σ_run(mirror) = √(σ_run² + σ_run×eval²)
and rerun `power_mirror.py main`.

| Measured σ_run(mirror) | Action | Pooled MDE (.0125) | Extra cost |
|---|---|---|---|
| ≤ 0.08 | O3b (S = 3) | ≈ 2.6 [R, linear in σ_run] | −54 T4-h |
| 0.08–0.15 | **O3 (S = 5)** | 3.2–3.9 | — |
| ≥ 0.15 | O7 (0.6B → S = 8) | ≈ 3.5–3.9 | +30 T4-h |
| ≥ 0.20 | O7 plus 1.7B → S = 8 | ≈ 3.9 [EST] | +≈ 70 T4-h |

**Raise seeds, never K or items** (§3.3).

### 5.5 Analysis plan for P2

**Model** (lme4; the same family and fallbacks as T5 §2.2):

```r
glmer(cbind(correct, K - correct) ~ lang * method * verif * size
        + (1 | item) + (1 | item:lang) + (1 | item:arm) + (1 | item:arm:lang)
        + (1 | run) + (1 | size:arm:seed_idx),           # seed_idx term exploits the paired (CRN) seeds
      family = binomial, data = d_parallel, nAGQ = 0)    # item = parallel item ID shared by EN and VI
```

**P2 (confirmatory).**
- The lang × verif contrast, averaged over method and size, tested by a logit Wald test.
- Reported in pp via `emmeans(regrid = "response")`.
- Add P2 to T5's Holm family. The design is sized for the worst case α = .0125.
- **Scale rule:** VI and EN baselines differ by about 14–18 pp, so pre-register T5's rule. Claim an interaction only if
  the logit test and the pp cluster bootstrap (seeds within language × size × arm, items within dataset; 10,000 draws)
  agree in sign.
- The capability-matched contrast is the scale-free cross-check, because its baselines are equal.

**Null result.** If P2 is non-significant, run TOST with margin ±3 pp. Claim "no meaningful language difference" only if
the 90 % CI lies inside ±3 pp.

**Exploratory (BH-FDR):** I3; language × method; per-size I2; cap; I2 restricted to paired slots (slots where both
languages' pipelines succeeded); and VP_vi on Vi-ExamMath-Core vs VP_vi on the parallel set. The last one is
descriptive only, because its item set differs.

**Manipulation checks**, reported per language × arm:
- the rejection rate of each gate (cycle, validity, answer);
- the T1 quality gate (200 items): teacher-acceptable, absurd and leak rates for Y−/N−;
- mean and p99 tokens per item;
- formalization coverage, i.e. the share of seeds with a valid IR.

### 5.6 English evaluation sets

| Set | Role | Why |
|---|---|---|
| **GSM8K-Platinum, the 600 IDs of the VI subsample** | **Primary (P2)** | Exact originals of the VI items, so items are parallel (R5). Platinum removed 110 flawed items and revised 10 golds [M: `data/clean/vi_gsm8k_test/vi_gsm8k_platinum_v1.0.datacard.json`]; the original GSM8K test would add label noise that differs between languages |
| **MATH-test algebra/prealgebra, the 400 IDs of the VI subsample, ⊇ all 206 MATH-500 alg/prealg items** | **Primary (P2)** | Exact originals. MATH-500 has exactly 124 algebra + 82 prealgebra items, aligned 500/500 [M: `vi_math500_algprealg_v1.1.datacard.json`], so including all 206 lets the subset be reported as "MATH-500 alg/prealg" for comparison with the literature |
| Full GSM8K-Platinum (1,209), greedy | Secondary | Comparability with published GSM8K numbers |
| MATH-500 alg/prealg (206), reported separately | Secondary | Comparability with the literature, e.g. 2602.19187 and 2508.14828 report MATH-500 |
| **Exclusion rule** | — | Drop from **both** languages any item still `pending` in the VI review queues (126 + 123 rows [M]) or flagged as a translation defect. A broken VI item would otherwise depress only the VI side |
| MGSM-en | **No** | Its 250 items are a subset of the GSM8K test [O abstract], so Platinum already covers them; and it adds no parallel Vietnamese split of our own |
| Rest of MATH-500 (number theory, counting & probability, geometry, intermediate algebra, precalculus) | **No** for P2 | Outside the seed scope, and C&P is 0 % certifiable [P: R4]. That is P3's question, not P2's |
| Vi-ExamMath-Core translated to English | **No** | It would reverse the translationese confound, costs Claude Opus spend, and sends benchmark items through translation. VP_vi on Core vs on the parallel set already answers "does it hold on native items?" for the VI side |

**Note to the T5 owner (no edit made):** the VI primary subsample should be drawn so that its 400 MATH IDs contain the 206
MATH-500 algebra/prealgebra items. Freeze the 1,000 IDs once, and use them for both languages.

### 5.7 What this adds to the Vietnamese grid

1. **12 VI runs:**
   - 0.6B Y+, Y−, N+, N− seeds 4–5 (8 runs);
   - 1.7B Y−, N− seeds 4–5 (4 runs). Y+ and N+ at 1.7B already have 5 seeds under T5 D3.

   Cost: 25.4 T4-h including evaluation [R].
2. **No new VI data.** The P2 cells are the existing 10k-budget points drawn from the 20k VI pools.
3. **One protocol requirement on the VI side:** paired training seeds (CRN). The VI run with seed index s must use the same
   LoRA-init seed, data-order seed and subsample slot IDs as the EN run with seed index s. This is a free choice that
   must be fixed **before** the VI grid runs.
4. **This supersedes the old "English control S0–S3 × 3 seeds × 1.7B" (12 runs)** in `03` §6.6 and `T3` §5.2.

---

## 6. Rejected options

| Option | Reason (evidence) |
|---|---|
| O1 Full mirror (every arm × every size × every seed) | 239.5 T4-h (+≈ 83 for a 2nd family) for an MDE of 3.0 vs 3.2 pp. T, T-off and M have no English counterpart: translating English into English is the identity, and 5CD-AI has no English analogue. Only Y±, N± and S0 mirror cleanly. Precision per hour 0.0052, half of O3's |
| O2 One size (1.7B) | MDE 4.5 pp and TOST power 0.45. It cannot separate language from competence. A single size is also the setting where 2602.19187's Sym advantage flipped between 1.5B and 3B, so one size invites a "size-specific" objection |
| O3b 3 seeds | MDE 4.0 pp; TOST 0.62. Kept only as the low-variance branch of §5.4 |
| O4 Three sizes, 3 seeds | Similar MDE to O3 at 117 % of the hours. The 4B runs (5.9 T4-h each) buy no capability-matched pair |
| O5 Fewer English seeds, compensated by K = 4 or 8 | 91 % of Var(I2) is run variance. K = 8 improves the MDE by only 0.3 pp, while losing seeds costs 1.4 pp (O5a/b: 5.6–5.9 pp). This is the clearest negative result of the simulation |
| O6 0.6B only | Efficient (0.0140 per h) but one size, at the weakest end. No competence-matched pair. The base accuracy (VI 0.31) is close to the region where T5's floor rule starts to bite on Core |
| O8 Joint multilingual training (one model on VI+EN, verification crossed per language) | (i) Estimand drift: P2 asks about the premium of the *Vietnamese grid's* data. In a mixture, the VI premium is measured while English data is present, and 2604.13286 measured a median +3.4 % math transfer from adding one language [O]. (ii) No reuse of VI runs, and each run costs VI + EN tokens: 142.7 T4-h for MDE 4.2 even under the favourable no-transfer assumption [R, `joint`]. (iii) Precision per hour 0.0046 |
| Different seed problems in English | Adds a content confound (§2.2 R1) |
| Translate-test as the "English side" | Answers a different question: the language of *inference*, not of the training data |

---

## 7. Risks and unverified items

1. **σ_run is assumed** (0.122 logit, from T5's assumed 0.10 and 0.07). The whole decision scales with it, which is why
   the §5.4 rule exists. No primary source for GSM8K-scale SFT seed variance at ≤ 2B was found. The search returned only
   GLUE-type studies (2503.07329) and 2504.07086's "≥ 10 seeds for small benchmarks" advice [O: search 2026-11-07,
   snippets only → UNVERIFIED].
2. **The parallel-set accuracies are [EST].** The 0.6B-EN ≈ 1.7B-VI match mixes Qwen3 TR numbers with our harness; check
   it on day 1 (§5.2).
3. **The ×1.3 padding factor** is T3's assumption [EST]. The T4 per-run hours are FLOP-based ±2× (T3). Evaluation hours
   are a linear upper bound from the 300-item measurement.
4. **Kaggle T4×2 quota accounting** (1 or 2 quota-hours per session-hour) is unverified (T3 G8). Hence two calendar
   figures.
5. **NIM pass rates** are T6 placeholders. English pass rates are probably higher than Vietnamese ones, so 269k calls is
   likely an overestimate.
6. **A generator-language asymmetry** is part of the treatment, not a confound. P2 asks exactly whether the same pipeline
   is worth more in Vietnamese. The manipulation checks (§5.5) make the mechanism visible.
7. The MGSM language list was not verified (only the abstract was opened). That does not affect the decision.

---

## Appendix A — `power_mirror.py` [R]

Run: `python -I power_mirror.py {main|sens|power|joint}`. numpy only. Runtime is about 2 min for `main` on the laptop.
Outputs are in App. E.

```python
"""E1 power simulation: language x method x verification (English mirror).
Adapts T5 Appendix A (same seed, same generative structure). numpy only.
Run: python -I power_mirror.py {main|sens|power|joint|alloc}

Model (logit), per size m, language L in {vi,en}, arm a in {Y+,Y-,N+,N-}, training seed s, parallel item i:
  logit p = b_{m,L} + d_{m,L,a} + u_i + e_{i,L} + slope_{m,L,a,i} + v_{m,L,a,s}
  u_i ~ N(0,1.5^2)     item difficulty, shared by the EN original and its VI translation
  e_{i,L} ~ N(0,0.5^2) language-specific item difficulty (translation shifts difficulty)
  slope ~ item x arm effect; per-arm SD tau/sqrt2 (tau = 0.5 gives T5's pairwise SD 0.5);
          correlation rho_l across languages, rho_s across sizes
  v ~ N(0, s_run^2), s_run = sqrt(0.10^2 + 0.07^2) = 0.122 (T5 run + run x eval-type). In the mirror every
          language contrast is BETWEEN runs, so the run term never cancels (unlike T5's arm x eval-type DiD).
  rho_crn: optional correlation between the vi run and the en run that share (size, arm, seed index)
Estimators (pp):
  VP_L = mean over methods of [acc(+) - acc(-)]          verification premium in language L
  I2   = VP_vi - VP_en                                    P2 primary (language x verification)
  I3   = [(Y+ - Y-) - (N+ - N-)]_vi - [same]_en           3-way (language x method x verification)
  cap  = VP_vi(1.7B) - VP_en(0.6B)                         capability-matched (both ~49 % baseline)
"""
import math, sys
import numpy as np
rng = np.random.default_rng(20261107)
SQ2 = math.sqrt(2.0)
GH_x, GH_w = np.polynomial.hermite.hermgauss(60)
def marg(b, sd):
    z = SQ2 * GH_x
    return float(np.sum(GH_w / (1 + np.exp(-(b + sd * z)))) / math.sqrt(math.pi))
def solve_b(p, sd):
    lo, hi = -15, 15
    for _ in range(80):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if marg(m, sd) < p else (lo, m)
    return (lo + hi) / 2
Z = {0.05: 1.959964, 0.0125: 2.497705}
Z80 = 0.841621
ARMS = ['Y+', 'Y-', 'N+', 'N-']
PLUS = np.array([1, 0, 1, 0], bool)
# accuracy of the unverified arms on the 1,000-item parallel set (600 GSM8K-Plat + 400 MATH alg/prealg),
# (vi, en) [ESTIMATE, E1 §3.1]
SIZES = {'0.6B': (0.31, 0.49), '1.7B': (0.49, 0.63), '4B': (0.63, 0.74)}


def simulate(sizes, S_vi, S_en, K=4, n=1000, reps=2000, chunk=50, s_item=1.5, s_lang=0.5, tau=0.5,
             rho_l=0.5, rho_s=0.5, s_run=0.122, rho_crn=0.0, prem_en=0.02, D2=0.0, D3=0.0, S_by_size=None):
    M = len(sizes)
    if S_by_size is None: S_by_size = [(S_vi, S_en)] * M
    Smax = max(max(a, b) for a, b in S_by_size)
    ta = tau / SQ2
    sd_tot = math.sqrt(s_item**2 + s_lang**2 + ta**2 + s_run**2)
    off = np.zeros((M, 2, 4))
    for mi, sz in enumerate(sizes):
        for li, p in enumerate(SIZES[sz]):              # li 0 = vi, 1 = en
            b = solve_b(p, sd_tot)
            for ai, a in enumerate(ARMS):
                if not PLUS[ai]:
                    off[mi, li, ai] = b; continue
                prem = prem_en + (D2 if li == 0 else 0.0)
                if li == 0: prem += (D3 / 2 if a[0] == 'Y' else -D3 / 2)
                off[mi, li, ai] = solve_b(min(p + prem, 0.99), sd_tot)
    w0, wL, wm = rho_l * rho_s, rho_s * (1 - rho_l), rho_l * (1 - rho_s)
    wi = max(0.0, 1 - w0 - wL - wm)
    mask = np.zeros((M, 2, Smax), bool)
    for mi, (sv, se) in enumerate(S_by_size): mask[mi, 0, :sv] = True; mask[mi, 1, :se] = True
    keys = ['I2_pool', 'I3_pool'] + [f'I2_{s}' for s in sizes] + [f'I3_{s}' for s in sizes]
    if '0.6B' in sizes and '1.7B' in sizes: keys.append('cap')
    out = {k: [] for k in keys}
    done = 0
    while done < reps:
        R = min(chunk, reps - done); done += R
        u = rng.normal(0, s_item, (R, 1, 1, 1, 1, n))
        e = rng.normal(0, s_lang, (R, 1, 2, 1, 1, n))
        sl = ta * (math.sqrt(w0) * rng.normal(size=(R, 1, 1, 4, 1, n))
                   + math.sqrt(wL) * rng.normal(size=(R, 1, 2, 4, 1, n))
                   + math.sqrt(wm) * rng.normal(size=(R, M, 1, 4, 1, n))
                   + math.sqrt(wi) * rng.normal(size=(R, M, 2, 4, 1, n)))
        zc = rng.normal(size=(R, M, 1, 4, Smax, 1)); zl = rng.normal(size=(R, M, 2, 4, Smax, 1))
        v = s_run * (math.sqrt(rho_crn) * zc + math.sqrt(1 - rho_crn) * zl)
        p = 1 / (1 + np.exp(-(off.reshape(1, M, 2, 4, 1, 1) + u + e + sl + v)))
        y = rng.binomial(K, p).mean(axis=-1) / K                     # (R, M, L, A, S) run accuracy
        y = np.where(mask.reshape(1, M, 2, 1, Smax), y, np.nan)
        cell = np.nanmean(y, axis=-1)                                # (R, M, L, A)
        dY = cell[..., 0] - cell[..., 1]; dN = cell[..., 2] - cell[..., 3]
        VP = (dY + dN) / 2
        I2 = VP[:, :, 0] - VP[:, :, 1]
        I3 = (dY - dN)[:, :, 0] - (dY - dN)[:, :, 1]
        for mi, sz in enumerate(sizes):
            out[f'I2_{sz}'].append(I2[:, mi]); out[f'I3_{sz}'].append(I3[:, mi])
        out['I2_pool'].append(I2.mean(axis=1)); out['I3_pool'].append(I3.mean(axis=1))
        if 'cap' in out:
            out['cap'].append(VP[:, sizes.index('1.7B'), 0] - VP[:, sizes.index('0.6B'), 1])
    return {k: np.concatenate(vv) for k, vv in out.items()}


def mde(sd, a): return (Z[a] + Z80) * sd
def summarize(tag, res):
    for k in res:
        sd = res[k].std(ddof=1) * 100
        print(f'{tag:36s} {k:8s} mean={res[k].mean()*100:+5.2f} SD={sd:4.2f}pp '
              f'MDE80(.05)={mde(sd, .05):4.1f} MDE80(.0125)={mde(sd, .0125):4.1f}', flush=True)


def joint(size='1.7B', S=3, K=4, n=1000, reps=1500, chunk=50, s_item=1.5, s_lang=0.5, tau=0.5, rho_l=0.5,
          s_v=0.10, s_w=0.07):
    """Joint vi+en training. Runs = method(2) x vi-data verif(2) x en-data verif(2) x S; each run evaluated in
    both languages. vi eval responds only to the vi-data arm and en eval only to the en-data arm (no
    cross-lingual transfer: the most favourable case for this design)."""
    ta = tau / SQ2; sd_tot = math.sqrt(s_item**2 + s_lang**2 + ta**2 + s_v**2 + s_w**2)
    pv, pe = SIZES[size]; base = (solve_b(pv, sd_tot), solve_b(pe, sd_tot))
    res = []; done = 0
    while done < reps:
        R = min(chunk, reps - done); done += R
        u = rng.normal(0, s_item, (R, n))
        e = rng.normal(0, s_lang, (R, 2, n))
        sl = ta * (math.sqrt(rho_l) * rng.normal(size=(R, 1, 2, 2, n)) + math.sqrt(1 - rho_l) * rng.normal(size=(R, 2, 2, 2, n)))
        v = rng.normal(0, s_v, (R, 2, 2, 2, S)); w = rng.normal(0, s_w, (R, 2, 2, 2, 2, S))
        acc = np.zeros((R, 2, 2, 2, 2, S))      # [R, evalL, method, vi_verif, en_verif, S]; verif 0 = '+'
        for L in range(2):
            for mth in range(2):
                for a in range(2):
                    for c in range(2):
                        own = a if L == 0 else c
                        eta = (base[L] + u + e[:, L] + sl[:, L, mth, own])[:, None, :] + (v[:, mth, a, c] + w[:, L, mth, a, c])[:, :, None]
                        acc[:, L, mth, a, c] = rng.binomial(K, 1 / (1 + np.exp(-eta))).mean(-1) / K
        VPvi = (acc[:, 0, :, 0] - acc[:, 0, :, 1]).mean(axis=(1, 2, 3))
        VPen = (acc[:, 1, :, :, 0] - acc[:, 1, :, :, 1]).mean(axis=(1, 2, 3))
        res.append(VPvi - VPen)
    return np.concatenate(res)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'main'
    if what == 'main':
        for tag, sz, sv, se, kw in [
            ('O2  1.7B S5/5', ['1.7B'], 5, 5, {}),
            ('O3  0.6+1.7B S5/5', ['0.6B', '1.7B'], 5, 5, {}),
            ('O3b 0.6+1.7B S3/3', ['0.6B', '1.7B'], 3, 3, {}),
            ('O4  0.6+1.7+4B S3/3', ['0.6B', '1.7B', '4B'], 3, 3, {}),
            ('O5a 1.7B Svi5/Sen2 K4', ['1.7B'], 5, 2, {}),
            ('O5b 1.7B Svi5/Sen2 K8', ['1.7B'], 5, 2, {'K': 8}),
            ('O5c 0.6+1.7B Svi5/Sen3 K8', ['0.6B', '1.7B'], 5, 3, {'K': 8}),
            ('O6  0.6B S8/8', ['0.6B'], 8, 8, {}),
            ('O7  0.6B S8 + 1.7B S5', ['0.6B', '1.7B'], None, None, {'S_by_size': [(8, 8), (5, 5)]}),
            ('O1  full 0.6/1.7/4B S3 (S5@1.7B)', ['0.6B', '1.7B', '4B'], None, None, {'S_by_size': [(3, 3), (5, 5), (3, 3)]}),
        ]:
            summarize(tag, simulate(sz, sv, se, **kw))
    elif what == 'sens':
        for tag, kw in [('base', {}), ('s_run=0.06', {'s_run': 0.06}), ('s_run=0.20', {'s_run': 0.20}),
                        ('rho_l=0', {'rho_l': 0.0}), ('rho_l=1', {'rho_l': 1.0}), ('tau=0.8', {'tau': 0.8}),
                        ('rho_crn=0.3', {'rho_crn': 0.3}), ('n=600', {'n': 600}), ('K=1', {'K': 1}),
                        ('s_lang=1.0', {'s_lang': 1.0})]:
            r = simulate(['0.6B', '1.7B'], 5, 5, **kw)
            summarize('O3 ' + tag, {k: r[k] for k in ('I2_pool', 'I3_pool', 'I2_1.7B', 'cap')})
    elif what == 'power':      # check the normal approximation used for MDE
        r0 = simulate(['0.6B', '1.7B'], 5, 5)
        sd = r0['I2_pool'].std(ddof=1)
        for D2 in (0.02, 0.03, 0.04):
            r1 = simulate(['0.6B', '1.7B'], 5, 5, D2=D2)
            for a in (.05, .0125):
                pw = np.mean(np.abs(r1['I2_pool']) / sd > Z[a])
                approx = 0.5 * (1 + math.erf((r1['I2_pool'].mean() / sd - Z[a]) / SQ2))
                print(f'O3 pooled I2 D2={D2*100:.0f}pp a={a}: sim power={pw:.2f} normal-approx={approx:.2f} '
                      f'mean est={r1["I2_pool"].mean()*100:.2f}pp')
    elif what == 'joint':
        for S in (3, 5):
            r = joint('1.7B', S=S); sd = r.std(ddof=1) * 100
            print(f'JOINT 1.7B S={S} ({8*S} runs, 2x tokens each): I2 SD={sd:.2f}pp '
                  f'MDE80(.05)={mde(sd, .05):.1f} MDE80(.0125)={mde(sd, .0125):.1f}')
```

## Appendix B — `compute_cost.py` [R]

```python
"""E1 compute + precision-per-hour for each English-mirror option. Run: python -I compute_cost.py"""
# T3 §1.2 per-run T4-h for a 9M-token run (10k items x 450 tok x 2 ep), padding-free; ckpt off <=1B, on >=1.7B
T3 = {'0.6B': 0.42, '1.7B': 1.83, '4B': 4.27}
PAD = 1.3                      # padded / length-grouped batching (packing leak on T4, measured packcheck)
EN_RATIO = 1.951 / 2.236       # EN/VI tokens, Qwen3 tokenizer, parallel GSM8K+MATH alg/prealg train (fertility.py)
train = {('vi', s): T3[s] * PAD for s in T3}
train.update({('en', s): T3[s] * PAD * EN_RATIO for s in T3})
# eval: measured 300 4-shot generations in 87-92 s on one T4 (w2-pilot-v1 eval.json); linear extrapolation is an
# UPPER bound because vLLM batches better at 5k+ prompts.  EN: 1000 x (K4 + greedy) + ~900 greedy secondaries = 5.9k
# gens; VI (T5 §4.3): ~11k gens. 4B: x2.
per_gen_s = 90 / 300
def ev(ngen, size): return (ngen * per_gen_s + 150) / 3600 * (2 if size == '4B' else 1)
evalh = {('en', s): ev(5900, s) for s in T3}
evalh.update({('vi', s): ev(11000, s) for s in T3})
evalh_en_k8 = {s: ev(9900, s) for s in T3}
def run(lang, size, k8=False):
    return train[(lang, size)] + (evalh_en_k8[size] if k8 else evalh[(lang, size)])
print('per run (train + eval) T4-h:', {f'{l}-{s}': round(run(l, s), 2) for l in ('vi', 'en') for s in T3})
print('train only:', {f'{l}-{s}': round(v, 3) for (l, s), v in train.items()})
print('eval only:', {f'{l}-{s}': round(v, 3) for (l, s), v in evalh.items()})

# options: list of (lang, size, n_runs, k8)
O = {
 'O1 full mirror (7 EN arms x 3 sizes x S3, S5 @1.7B on 5 arms) + VI top-up': [('en','0.6B',21,0),('en','1.7B',31,0),('en','4B',21,0),('vi','1.7B',4,0)],
 'O2 Y/N +/- @1.7B S5':            [('en','1.7B',20,0),('vi','1.7B',4,0)],
 'O3 Y/N +/- @0.6B+1.7B S5':       [('en','0.6B',20,0),('en','1.7B',20,0),('vi','0.6B',8,0),('vi','1.7B',4,0)],
 'O3b Y/N +/- @0.6B+1.7B S3':      [('en','0.6B',12,0),('en','1.7B',12,0)],
 'O4 Y/N +/- @0.6/1.7/4B S3':      [('en','0.6B',12,0),('en','1.7B',12,0),('en','4B',12,0)],
 'O5a @1.7B Sen2 K4 (VI S5)':      [('en','1.7B',8,0),('vi','1.7B',4,0)],
 'O5b @1.7B Sen2 K8 (VI S5)':      [('en','1.7B',8,1),('vi','1.7B',4,0)],
 'O5c @0.6+1.7B Sen3 K8 (VI S5)':  [('en','0.6B',12,1),('en','1.7B',12,1),('vi','0.6B',8,0),('vi','1.7B',4,0)],
 'O6 @0.6B S8':                    [('en','0.6B',32,0),('vi','0.6B',20,0)],
 'O7 0.6B S8 + 1.7B S5':           [('en','0.6B',32,0),('en','1.7B',20,0),('vi','0.6B',20,0),('vi','1.7B',4,0)],
}
SD = {'O1':0.90,'O2':1.34,'O3':0.95,'O3b':1.19,'O4':1.00,'O5a':1.78,'O5b':1.68,'O5c':1.09,'O6':1.07,'O7':0.88}  # I2 pooled SD, power_mirror main
MDE = lambda sd: round(3.339 * sd, 1)
for k, rows in O.items():
    h = sum(n * run(l, s, k8) for l, s, n, k8 in rows)
    nen = sum(r[2] for r in rows if r[0] == 'en'); nvi = sum(r[2] for r in rows if r[0] == 'vi')
    sd = SD[k.split()[0]]
    print(f'{k:72s} EN runs={nen:3d} VI top-up={nvi:2d} T4-h={h:6.1f} I2 SD={sd:.2f} MDE80(.0125)={MDE(sd)} '
          f'precision/h={1/(sd**2*h):.4f} weeks@30={h/30:.1f} weeks@60={h/60:.1f}')
# joint training, 1.7B S3: 24 runs, each = VI tokens + EN tokens, evaluated in both languages
hj = 24 * (train[('vi','1.7B')] + train[('en','1.7B')] + evalh[('vi','1.7B')] + evalh[('en','1.7B')])
print(f'{"JOINT vi+en @1.7B S3 (24 runs, no reuse)":72s} T4-h={hj:6.1f} I2 SD=1.24 MDE80(.0125)={MDE(1.24)} precision/h={1/(1.24**2*hj):.4f}')

# chosen = O3 + S0-EN anchor (3 seeds x 2 sizes) + EN recipe-transfer check (S0-EN @1.7B, LR x0.5 and x2, 1 seed)
rows = O['O3 Y/N +/- @0.6B+1.7B S5'] + [('en','0.6B',3,0),('en','1.7B',3,0),('en','1.7B',2,0)]
tr = sum(n * train[(l, s)] for l, s, n, _ in rows); evh = sum(n * evalh[(l, s)] for l, s, n, _ in rows)
print(f'CHOSEN: train={tr:.1f} eval={evh:.1f} total={tr+evh:.1f} T4-h; weeks @30 T4-h/wk={(tr+evh)/30:.1f}, @60={(tr+evh)/60:.1f}; '
      f'EN runs={sum(r[2] for r in rows if r[0]=="en")}, VI top-up={sum(r[2] for r in rows if r[0]=="vi")}')

# NIM calls for the EN pools (T6 §4.2 placeholders: N 5 calls/accepted, Y 6 calls/accepted, 1 formalize/seed)
for pool in (20000, 12500):
    formalize = 11000 * pool / 20000
    calls = (formalize + 6 * pool + 5 * pool) * 1.10 + 15000      # +10 % retries, + ~15k S0-EN solution regen
    print(f'EN pool {pool}: NIM calls ~{calls/1000:.0f}k -> {calls/57600:.1f} days at 40 RPM')
```

Output:

```
per run (train + eval) T4-h: {'vi-0.6B': 1.5, 'vi-1.7B': 3.34, 'vi-4B': 7.47, 'en-0.6B': 1.01, 'en-1.7B': 2.61, 'en-4B': 5.91}
train only: {'vi-0.6B': 0.546, 'vi-1.7B': 2.379, 'vi-4B': 5.551, 'en-0.6B': 0.476, 'en-1.7B': 2.076, 'en-4B': 4.843}
eval only: {'en-0.6B': 0.533, 'en-1.7B': 0.533, 'en-4B': 1.067, 'vi-0.6B': 0.958, 'vi-1.7B': 0.958, 'vi-4B': 1.917}
O1 full mirror (7 EN arms x 3 sizes x S3, S5 @1.7B on 5 arms) + VI top-up EN runs= 73 VI top-up= 4 T4-h= 239.5 I2 SD=0.90 MDE80(.0125)=3.0 precision/h=0.0052 weeks@30=8.0 weeks@60=4.0
O2 Y/N +/- @1.7B S5                                                      EN runs= 20 VI top-up= 4 T4-h=  65.5 I2 SD=1.34 MDE80(.0125)=4.5 precision/h=0.0085 weeks@30=2.2 weeks@60=1.1
O3 Y/N +/- @0.6B+1.7B S5                                                 EN runs= 40 VI top-up=12 T4-h=  97.8 I2 SD=0.95 MDE80(.0125)=3.2 precision/h=0.0113 weeks@30=3.3 weeks@60=1.6
O3b Y/N +/- @0.6B+1.7B S3                                                EN runs= 24 VI top-up= 0 T4-h=  43.4 I2 SD=1.19 MDE80(.0125)=4.0 precision/h=0.0163 weeks@30=1.4 weeks@60=0.7
O4 Y/N +/- @0.6/1.7/4B S3                                                EN runs= 36 VI top-up= 0 T4-h= 114.3 I2 SD=1.00 MDE80(.0125)=3.3 precision/h=0.0087 weeks@30=3.8 weeks@60=1.9
O5a @1.7B Sen2 K4 (VI S5)                                                EN runs=  8 VI top-up= 4 T4-h=  34.2 I2 SD=1.78 MDE80(.0125)=5.9 precision/h=0.0092 weeks@30=1.1 weeks@60=0.6
O5b @1.7B Sen2 K8 (VI S5)                                                EN runs=  8 VI top-up= 4 T4-h=  36.9 I2 SD=1.68 MDE80(.0125)=5.6 precision/h=0.0096 weeks@30=1.2 weeks@60=0.6
O5c @0.6+1.7B Sen3 K8 (VI S5)                                            EN runs= 24 VI top-up=12 T4-h=  76.8 I2 SD=1.09 MDE80(.0125)=3.6 precision/h=0.0110 weeks@30=2.6 weeks@60=1.3
O6 @0.6B S8                                                              EN runs= 32 VI top-up=20 T4-h=  62.4 I2 SD=1.07 MDE80(.0125)=3.6 precision/h=0.0140 weeks@30=2.1 weeks@60=1.0
O7 0.6B S8 + 1.7B S5                                                     EN runs= 52 VI top-up=24 T4-h= 127.9 I2 SD=0.88 MDE80(.0125)=2.9 precision/h=0.0101 weeks@30=4.3 weeks@60=2.1
JOINT vi+en @1.7B S3 (24 runs, no reuse)                                 T4-h= 142.7 I2 SD=1.24 MDE80(.0125)=4.1 precision/h=0.0046
CHOSEN: train=76.7 eval=37.1 total=113.8 T4-h; weeks @30 T4-h/wk=3.8, @60=1.9; EN runs=48, VI top-up=12
EN pool 20000: NIM calls ~269k -> 4.7 days at 40 RPM
EN pool 12500: NIM calls ~174k -> 3.0 days at 40 RPM
```

## Appendix C — `fertility.py` (VI/EN token ratio, Qwen3 tokenizer from the local HF cache) [R]

Run from the repo root: `py -I fertility.py data`.

```python
# Token-count ratio VI/EN on parallel training data, Qwen3 tokenizer (cached tokenizer.json)
import json, glob, os, sys, statistics as st
from tokenizers import Tokenizer
root = sys.argv[1]
tok = Tokenizer.from_file(glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/*/tokenizer.json'))[0])
def n(s): return len(tok.encode(s).ids)
def load_jsonl(p): return [json.loads(l) for l in open(p, encoding='utf-8')]
# GSM8K train: aligned by index
vi = load_jsonl(f'{root}/translation/gsm8k/gsm8k_train_vietnamese.jsonl')
en = {r['index']: r for r in load_jsonl(f'{root}/generate/gsm8k/train.jsonl')}
P=[];S=[];T=[]
for r in vi:
    e = en.get(r['index'])
    if not e or e['ground_truth'] != r['ground_truth']: continue
    pv, pe = n(r['question']), n(e['question']); sv, se = n(r['answer']), n(e['answer'])
    P.append((pv,pe)); S.append((sv,se)); T.append((pv+sv, pe+se))
def rep(name, L):
    tv=sum(a for a,_ in L); te=sum(b for _,b in L)
    print(f'{name}: n={len(L)} VI tok={tv} EN tok={te} ratio VI/EN={tv/te:.3f} meanVI={tv/len(L):.0f} meanEN={te/len(L):.0f}')
rep('GSM8K-train problem', P); rep('GSM8K-train solution', S); rep('GSM8K-train total', T)
# MATH train alg+prealg: aligned by position within subject
for subj in ['algebra','prealgebra']:
    v = load_jsonl(f'{root}/translation/MATH/train/{subj}_vi.jsonl')
    e = json.load(open(f'{root}/generate/MATH/train/{subj}.json', encoding='utf-8'))
    if isinstance(e, dict): e = list(e.values())[0] if len(e)==1 else e
    L=[]
    for i,r in enumerate(v):
        if i >= len(e): break
        ee = e[i]
        if ee.get('level') != r.get('level'): continue
        L.append((n(r['problem'])+n(r['solution']), n(ee['problem'])+n(ee['solution'])))
    rep(f'MATH-train {subj} total', L)
```

Output:

```
GSM8K-train problem: n=7473 VI tok=563166 EN tok=445740 ratio VI/EN=1.263 meanVI=75 meanEN=60
GSM8K-train solution: n=7473 VI tok=1022924 EN tok=910161 ratio VI/EN=1.124 meanVI=137 meanEN=122
GSM8K-train total: n=7473 VI tok=1586090 EN tok=1355901 ratio VI/EN=1.170 meanVI=212 meanEN=181
MATH-train algebra total: n=1744 VI tok=379076 EN tok=350723 ratio VI/EN=1.081 meanVI=217 meanEN=201
MATH-train prealgebra total: n=1205 VI tok=270650 EN tok=244960 ratio VI/EN=1.105 meanVI=225 meanEN=203
```

## Appendix D — closed-form variance decomposition (`decomp.py`) and TOST power (`tost.py`) [R]

```python
import math, numpy as np
GH_x, GH_w = np.polynomial.hermite.hermgauss(60)
def Epq(b, sd):
    z = math.sqrt(2)*GH_x; p = 1/(1+np.exp(-(b+sd*z))); return float(np.sum(GH_w*p*(1-p))/math.sqrt(math.pi)), float(np.sum(GH_w*(p*(1-p))**2)/math.sqrt(math.pi))
sd = math.sqrt(1.5**2+0.5**2+(0.5/math.sqrt(2))**2+0.122**2)
for p0, b in [(0.31,-1.0),(0.49,-0.05),(0.63,0.75)]:
    # find b giving marginal p0
    lo,hi=-15,15
    for _ in range(80):
        m=(lo+hi)/2; z=math.sqrt(2)*GH_x; mp=float(np.sum(GH_w/(1+np.exp(-(m+sd*z))))/math.sqrt(math.pi))
        lo,hi=(m,hi) if mp<p0 else (lo,m)
    e1,e2 = Epq(m, sd)
    s_pp = 0.122*e1
    for S in (3,5):
        run = 2*s_pp**2/S
        bern = 8*0.25*e1/(1000*4*S)
        tot = math.sqrt(run+bern)
        print(f'p={p0} E[p(1-p)]={e1:.3f} sigma_run_pp={s_pp*100:.2f} S={S}: SD_run={math.sqrt(run)*100:.2f} SD_bern={math.sqrt(bern)*100:.2f} -> {tot*100:.2f} pp (run share of var {run/(run+bern):.0%})')

# P(TOST with margin +-3 pp at alpha=.05 declares equivalence | true I2 = 0) = 2*Phi(3/SD - 1.645) - 1
import math
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
SD = {'O1':0.90,'O2':1.34,'O3':0.95,'O3b':1.19,'O4':1.00,'O5a':1.78,'O5b':1.68,'O5c':1.09,'O6':1.07,'O7':0.88,'JOINT':1.24}
for k, sd in SD.items():
    print(f'{k:6s} SD={sd:.2f} TOST(+-3pp) power={max(0, 2*Phi(3/sd - 1.644854) - 1):.2f}  MDE80(.05)={2.801585*sd:.1f} MDE80(.0125)={3.339326*sd:.1f}')
```

Output:

```
p=0.31 E[p(1-p)]=0.149 sigma_run_pp=1.82 S=3: SD_run=1.49 SD_bern=0.50 -> 1.57 pp (run share of var 90%)
p=0.31 E[p(1-p)]=0.149 sigma_run_pp=1.82 S=5: SD_run=1.15 SD_bern=0.39 -> 1.21 pp (run share of var 90%)
p=0.49 E[p(1-p)]=0.170 sigma_run_pp=2.07 S=3: SD_run=1.69 SD_bern=0.53 -> 1.77 pp (run share of var 91%)
p=0.49 E[p(1-p)]=0.170 sigma_run_pp=2.07 S=5: SD_run=1.31 SD_bern=0.41 -> 1.37 pp (run share of var 91%)
p=0.63 E[p(1-p)]=0.160 sigma_run_pp=1.96 S=3: SD_run=1.60 SD_bern=0.52 -> 1.68 pp (run share of var 91%)
p=0.63 E[p(1-p)]=0.160 sigma_run_pp=1.96 S=5: SD_run=1.24 SD_bern=0.40 -> 1.30 pp (run share of var 91%)
O1     SD=0.90 TOST(+-3pp) power=0.91  MDE80(.05)=2.5 MDE80(.0125)=3.0
O2     SD=1.34 TOST(+-3pp) power=0.45  MDE80(.05)=3.8 MDE80(.0125)=4.5
O3     SD=0.95 TOST(+-3pp) power=0.87  MDE80(.05)=2.7 MDE80(.0125)=3.2
O3b    SD=1.19 TOST(+-3pp) power=0.62  MDE80(.05)=3.3 MDE80(.0125)=4.0
O4     SD=1.00 TOST(+-3pp) power=0.82  MDE80(.05)=2.8 MDE80(.0125)=3.3
O5a    SD=1.78 TOST(+-3pp) power=0.03  MDE80(.05)=5.0 MDE80(.0125)=5.9
O5b    SD=1.68 TOST(+-3pp) power=0.11  MDE80(.05)=4.7 MDE80(.0125)=5.6
O5c    SD=1.09 TOST(+-3pp) power=0.73  MDE80(.05)=3.1 MDE80(.0125)=3.6
O6     SD=1.07 TOST(+-3pp) power=0.75  MDE80(.05)=3.0 MDE80(.0125)=3.6
O7     SD=0.88 TOST(+-3pp) power=0.92  MDE80(.05)=2.5 MDE80(.0125)=2.9
JOINT  SD=1.24 TOST(+-3pp) power=0.56  MDE80(.05)=3.5 MDE80(.0125)=4.1
```

## Appendix E — raw simulation outputs [R]

`main` (2,000 replicates per design):

```
O2  1.7B S5/5                        I2_pool  mean=+0.01 SD=1.34pp MDE80(.05)= 3.7 MDE80(.0125)= 4.5
O2  1.7B S5/5                        I3_pool  mean=+0.02 SD=2.70pp MDE80(.05)= 7.6 MDE80(.0125)= 9.0
O2  1.7B S5/5                        I2_1.7B  mean=+0.01 SD=1.34pp MDE80(.05)= 3.7 MDE80(.0125)= 4.5
O2  1.7B S5/5                        I3_1.7B  mean=+0.02 SD=2.70pp MDE80(.05)= 7.6 MDE80(.0125)= 9.0
O3  0.6+1.7B S5/5                    I2_pool  mean=-0.01 SD=0.95pp MDE80(.05)= 2.6 MDE80(.0125)= 3.2
O3  0.6+1.7B S5/5                    I3_pool  mean=+0.04 SD=1.94pp MDE80(.05)= 5.4 MDE80(.0125)= 6.5
O3  0.6+1.7B S5/5                    I2_0.6B  mean=-0.02 SD=1.35pp MDE80(.05)= 3.8 MDE80(.0125)= 4.5
O3  0.6+1.7B S5/5                    I2_1.7B  mean=+0.01 SD=1.34pp MDE80(.05)= 3.7 MDE80(.0125)= 4.5
O3  0.6+1.7B S5/5                    I3_0.6B  mean=+0.07 SD=2.66pp MDE80(.05)= 7.5 MDE80(.0125)= 8.9
O3  0.6+1.7B S5/5                    I3_1.7B  mean=+0.01 SD=2.71pp MDE80(.05)= 7.6 MDE80(.0125)= 9.0
O3  0.6+1.7B S5/5                    cap      mean=-0.04 SD=1.38pp MDE80(.05)= 3.9 MDE80(.0125)= 4.6
O3b 0.6+1.7B S3/3                    I2_pool  mean=-0.01 SD=1.19pp MDE80(.05)= 3.3 MDE80(.0125)= 4.0
O3b 0.6+1.7B S3/3                    I3_pool  mean=+0.01 SD=2.41pp MDE80(.05)= 6.8 MDE80(.0125)= 8.1
O3b 0.6+1.7B S3/3                    I2_0.6B  mean=-0.03 SD=1.65pp MDE80(.05)= 4.6 MDE80(.0125)= 5.5
O3b 0.6+1.7B S3/3                    I2_1.7B  mean=+0.02 SD=1.73pp MDE80(.05)= 4.8 MDE80(.0125)= 5.8
O3b 0.6+1.7B S3/3                    I3_0.6B  mean=-0.03 SD=3.36pp MDE80(.05)= 9.4 MDE80(.0125)=11.2
O3b 0.6+1.7B S3/3                    I3_1.7B  mean=+0.06 SD=3.47pp MDE80(.05)= 9.7 MDE80(.0125)=11.6
O3b 0.6+1.7B S3/3                    cap      mean=+0.01 SD=1.78pp MDE80(.05)= 5.0 MDE80(.0125)= 6.0
O4  0.6+1.7+4B S3/3                  I2_pool  mean=+0.06 SD=1.00pp MDE80(.05)= 2.8 MDE80(.0125)= 3.3
O4  0.6+1.7+4B S3/3                  I3_pool  mean=-0.01 SD=1.93pp MDE80(.05)= 5.4 MDE80(.0125)= 6.4
O4  0.6+1.7+4B S3/3                  I2_0.6B  mean=+0.04 SD=1.75pp MDE80(.05)= 4.9 MDE80(.0125)= 5.9
O4  0.6+1.7+4B S3/3                  I2_1.7B  mean=+0.06 SD=1.78pp MDE80(.05)= 5.0 MDE80(.0125)= 5.9
O4  0.6+1.7+4B S3/3                  I2_4B    mean=+0.07 SD=1.59pp MDE80(.05)= 4.5 MDE80(.0125)= 5.3
O4  0.6+1.7+4B S3/3                  I3_0.6B  mean=+0.13 SD=3.39pp MDE80(.05)= 9.5 MDE80(.0125)=11.3
O4  0.6+1.7+4B S3/3                  I3_1.7B  mean=-0.11 SD=3.40pp MDE80(.05)= 9.5 MDE80(.0125)=11.4
O4  0.6+1.7+4B S3/3                  I3_4B    mean=-0.04 SD=3.13pp MDE80(.05)= 8.8 MDE80(.0125)=10.5
O4  0.6+1.7+4B S3/3                  cap      mean=+0.06 SD=1.86pp MDE80(.05)= 5.2 MDE80(.0125)= 6.2
O5a 1.7B Svi5/Sen2 K4                I2_pool  mean=-0.02 SD=1.78pp MDE80(.05)= 5.0 MDE80(.0125)= 5.9
O5a 1.7B Svi5/Sen2 K4                I3_pool  mean=+0.01 SD=3.48pp MDE80(.05)= 9.7 MDE80(.0125)=11.6
O5a 1.7B Svi5/Sen2 K4                I2_1.7B  mean=-0.02 SD=1.78pp MDE80(.05)= 5.0 MDE80(.0125)= 5.9
O5a 1.7B Svi5/Sen2 K4                I3_1.7B  mean=+0.01 SD=3.48pp MDE80(.05)= 9.7 MDE80(.0125)=11.6
O5b 1.7B Svi5/Sen2 K8                I2_pool  mean=+0.06 SD=1.68pp MDE80(.05)= 4.7 MDE80(.0125)= 5.6
O5b 1.7B Svi5/Sen2 K8                I3_pool  mean=-0.02 SD=3.50pp MDE80(.05)= 9.8 MDE80(.0125)=11.7
O5b 1.7B Svi5/Sen2 K8                I2_1.7B  mean=+0.06 SD=1.68pp MDE80(.05)= 4.7 MDE80(.0125)= 5.6
O5b 1.7B Svi5/Sen2 K8                I3_1.7B  mean=-0.02 SD=3.50pp MDE80(.05)= 9.8 MDE80(.0125)=11.7
O5c 0.6+1.7B Svi5/Sen3 K8            I2_pool  mean=+0.02 SD=1.09pp MDE80(.05)= 3.0 MDE80(.0125)= 3.6
O5c 0.6+1.7B Svi5/Sen3 K8            I3_pool  mean=-0.03 SD=2.12pp MDE80(.05)= 6.0 MDE80(.0125)= 7.1
O5c 0.6+1.7B Svi5/Sen3 K8            I2_0.6B  mean=+0.00 SD=1.51pp MDE80(.05)= 4.2 MDE80(.0125)= 5.0
O5c 0.6+1.7B Svi5/Sen3 K8            I2_1.7B  mean=+0.03 SD=1.52pp MDE80(.05)= 4.2 MDE80(.0125)= 5.1
O5c 0.6+1.7B Svi5/Sen3 K8            I3_0.6B  mean=-0.09 SD=2.99pp MDE80(.05)= 8.4 MDE80(.0125)=10.0
O5c 0.6+1.7B Svi5/Sen3 K8            I3_1.7B  mean=+0.04 SD=3.04pp MDE80(.05)= 8.5 MDE80(.0125)=10.1
O5c 0.6+1.7B Svi5/Sen3 K8            cap      mean=+0.04 SD=1.59pp MDE80(.05)= 4.4 MDE80(.0125)= 5.3
O6  0.6B S8/8                        I2_pool  mean=+0.03 SD=1.07pp MDE80(.05)= 3.0 MDE80(.0125)= 3.6
O6  0.6B S8/8                        I3_pool  mean=-0.04 SD=2.11pp MDE80(.05)= 5.9 MDE80(.0125)= 7.1
O6  0.6B S8/8                        I2_0.6B  mean=+0.03 SD=1.07pp MDE80(.05)= 3.0 MDE80(.0125)= 3.6
O6  0.6B S8/8                        I3_0.6B  mean=-0.04 SD=2.11pp MDE80(.05)= 5.9 MDE80(.0125)= 7.1
O7  0.6B S8 + 1.7B S5                I2_pool  mean=-0.01 SD=0.88pp MDE80(.05)= 2.5 MDE80(.0125)= 2.9
O7  0.6B S8 + 1.7B S5                I3_pool  mean=+0.01 SD=1.71pp MDE80(.05)= 4.8 MDE80(.0125)= 5.7
O7  0.6B S8 + 1.7B S5                I2_0.6B  mean=-0.00 SD=1.06pp MDE80(.05)= 3.0 MDE80(.0125)= 3.5
O7  0.6B S8 + 1.7B S5                I2_1.7B  mean=-0.01 SD=1.37pp MDE80(.05)= 3.8 MDE80(.0125)= 4.6
O7  0.6B S8 + 1.7B S5                I3_0.6B  mean=-0.02 SD=2.10pp MDE80(.05)= 5.9 MDE80(.0125)= 7.0
O7  0.6B S8 + 1.7B S5                I3_1.7B  mean=+0.03 SD=2.71pp MDE80(.05)= 7.6 MDE80(.0125)= 9.1
O7  0.6B S8 + 1.7B S5                cap      mean=-0.03 SD=1.25pp MDE80(.05)= 3.5 MDE80(.0125)= 4.2
O1  full 0.6/1.7/4B S3 (S5@1.7B)     I2_pool  mean=-0.01 SD=0.90pp MDE80(.05)= 2.5 MDE80(.0125)= 3.0
O1  full 0.6/1.7/4B S3 (S5@1.7B)     I3_pool  mean=+0.01 SD=1.80pp MDE80(.05)= 5.0 MDE80(.0125)= 6.0
O1  full 0.6/1.7/4B S3 (S5@1.7B)     I2_0.6B  mean=-0.01 SD=1.65pp MDE80(.05)= 4.6 MDE80(.0125)= 5.5
O1  full 0.6/1.7/4B S3 (S5@1.7B)     I2_1.7B  mean=+0.01 SD=1.35pp MDE80(.05)= 3.8 MDE80(.0125)= 4.5
O1  full 0.6/1.7/4B S3 (S5@1.7B)     I2_4B    mean=-0.05 SD=1.56pp MDE80(.05)= 4.4 MDE80(.0125)= 5.2
O1  full 0.6/1.7/4B S3 (S5@1.7B)     I3_0.6B  mean=-0.01 SD=3.38pp MDE80(.05)= 9.5 MDE80(.0125)=11.3
O1  full 0.6/1.7/4B S3 (S5@1.7B)     I3_1.7B  mean=-0.01 SD=2.70pp MDE80(.05)= 7.6 MDE80(.0125)= 9.0
O1  full 0.6/1.7/4B S3 (S5@1.7B)     I3_4B    mean=+0.05 SD=3.08pp MDE80(.05)= 8.6 MDE80(.0125)=10.3
O1  full 0.6/1.7/4B S3 (S5@1.7B)     cap      mean=-0.00 SD=1.58pp MDE80(.05)= 4.4 MDE80(.0125)= 5.3
```

`sens` (O3, 2,000 replicates):

```
O3 base                              I2_pool  mean=+0.00 SD=0.96pp MDE80(.05)= 2.7 MDE80(.0125)= 3.2
O3 base                              I3_pool  mean=-0.05 SD=1.85pp MDE80(.05)= 5.2 MDE80(.0125)= 6.2
O3 base                              I2_1.7B  mean=-0.01 SD=1.37pp MDE80(.05)= 3.8 MDE80(.0125)= 4.6
O3 base                              cap      mean=+0.01 SD=1.39pp MDE80(.05)= 3.9 MDE80(.0125)= 4.6
O3 s_run=0.06                        I2_pool  mean=-0.02 SD=0.56pp MDE80(.05)= 1.6 MDE80(.0125)= 1.9
O3 s_run=0.06                        I3_pool  mean=+0.01 SD=1.09pp MDE80(.05)= 3.1 MDE80(.0125)= 3.7
O3 s_run=0.06                        I2_1.7B  mean=-0.03 SD=0.77pp MDE80(.05)= 2.2 MDE80(.0125)= 2.6
O3 s_run=0.06                        cap      mean=-0.03 SD=0.80pp MDE80(.05)= 2.2 MDE80(.0125)= 2.7
O3 s_run=0.20                        I2_pool  mean=-0.00 SD=1.48pp MDE80(.05)= 4.2 MDE80(.0125)= 5.0
O3 s_run=0.20                        I3_pool  mean=+0.00 SD=3.03pp MDE80(.05)= 8.5 MDE80(.0125)=10.1
O3 s_run=0.20                        I2_1.7B  mean=-0.02 SD=2.09pp MDE80(.05)= 5.9 MDE80(.0125)= 7.0
O3 s_run=0.20                        cap      mean=-0.01 SD=2.21pp MDE80(.05)= 6.2 MDE80(.0125)= 7.4
O3 rho_l=0                           I2_pool  mean=+0.01 SD=0.99pp MDE80(.05)= 2.8 MDE80(.0125)= 3.3
O3 rho_l=0                           I3_pool  mean=-0.07 SD=1.95pp MDE80(.05)= 5.5 MDE80(.0125)= 6.5
O3 rho_l=0                           I2_1.7B  mean=+0.02 SD=1.38pp MDE80(.05)= 3.9 MDE80(.0125)= 4.6
O3 rho_l=0                           cap      mean=+0.02 SD=1.45pp MDE80(.05)= 4.1 MDE80(.0125)= 4.8
O3 rho_l=1                           I2_pool  mean=-0.01 SD=0.91pp MDE80(.05)= 2.5 MDE80(.0125)= 3.0
O3 rho_l=1                           I3_pool  mean=-0.05 SD=1.86pp MDE80(.05)= 5.2 MDE80(.0125)= 6.2
O3 rho_l=1                           I2_1.7B  mean=-0.01 SD=1.32pp MDE80(.05)= 3.7 MDE80(.0125)= 4.4
O3 rho_l=1                           cap      mean=-0.01 SD=1.35pp MDE80(.05)= 3.8 MDE80(.0125)= 4.5
O3 tau=0.8                           I2_pool  mean=-0.00 SD=0.95pp MDE80(.05)= 2.7 MDE80(.0125)= 3.2
O3 tau=0.8                           I3_pool  mean=+0.03 SD=1.94pp MDE80(.05)= 5.4 MDE80(.0125)= 6.5
O3 tau=0.8                           I2_1.7B  mean=+0.03 SD=1.35pp MDE80(.05)= 3.8 MDE80(.0125)= 4.5
O3 tau=0.8                           cap      mean=-0.00 SD=1.42pp MDE80(.05)= 4.0 MDE80(.0125)= 4.8
O3 rho_crn=0.3                       I2_pool  mean=+0.03 SD=0.83pp MDE80(.05)= 2.3 MDE80(.0125)= 2.8
O3 rho_crn=0.3                       I3_pool  mean=-0.04 SD=1.65pp MDE80(.05)= 4.6 MDE80(.0125)= 5.5
O3 rho_crn=0.3                       I2_1.7B  mean=+0.06 SD=1.18pp MDE80(.05)= 3.3 MDE80(.0125)= 4.0
O3 rho_crn=0.3                       cap      mean=+0.00 SD=1.40pp MDE80(.05)= 3.9 MDE80(.0125)= 4.7
O3 n=600                             I2_pool  mean=+0.06 SD=0.97pp MDE80(.05)= 2.7 MDE80(.0125)= 3.2
O3 n=600                             I3_pool  mean=+0.11 SD=1.98pp MDE80(.05)= 5.6 MDE80(.0125)= 6.6
O3 n=600                             I2_1.7B  mean=+0.02 SD=1.39pp MDE80(.05)= 3.9 MDE80(.0125)= 4.6
O3 n=600                             cap      mean=+0.11 SD=1.41pp MDE80(.05)= 3.9 MDE80(.0125)= 4.7
O3 K=1                               I2_pool  mean=-0.01 SD=1.07pp MDE80(.05)= 3.0 MDE80(.0125)= 3.6
O3 K=1                               I3_pool  mean=+0.04 SD=2.08pp MDE80(.05)= 5.8 MDE80(.0125)= 6.9
O3 K=1                               I2_1.7B  mean=+0.04 SD=1.51pp MDE80(.05)= 4.2 MDE80(.0125)= 5.1
O3 K=1                               cap      mean=+0.00 SD=1.58pp MDE80(.05)= 4.4 MDE80(.0125)= 5.3
O3 s_lang=1.0                        I2_pool  mean=-0.00 SD=0.91pp MDE80(.05)= 2.6 MDE80(.0125)= 3.0
O3 s_lang=1.0                        I3_pool  mean=-0.01 SD=1.73pp MDE80(.05)= 4.9 MDE80(.0125)= 5.8
O3 s_lang=1.0                        I2_1.7B  mean=+0.01 SD=1.27pp MDE80(.05)= 3.6 MDE80(.0125)= 4.2
O3 s_lang=1.0                        cap      mean=-0.03 SD=1.33pp MDE80(.05)= 3.7 MDE80(.0125)= 4.5
```

`power` (normal-approximation check) and `joint`:

```
O3 pooled I2 D2=2pp a=0.05: sim power=0.52 normal-approx=0.53 mean est=1.97pp
O3 pooled I2 D2=2pp a=0.0125: sim power=0.32 normal-approx=0.32 mean est=1.97pp
O3 pooled I2 D2=3pp a=0.05: sim power=0.88 normal-approx=0.87 mean est=3.00pp
O3 pooled I2 D2=3pp a=0.0125: sim power=0.73 normal-approx=0.73 mean est=3.00pp
O3 pooled I2 D2=4pp a=0.05: sim power=0.99 normal-approx=0.99 mean est=4.01pp
O3 pooled I2 D2=4pp a=0.0125: sim power=0.95 normal-approx=0.95 mean est=4.01pp
JOINT 1.7B S=3 (24 runs, 2x tokens each): I2 SD=1.24pp MDE80(.05)=3.5 MDE80(.0125)=4.2
JOINT 1.7B S=5 (40 runs, 2x tokens each): I2 SD=0.97pp MDE80(.05)=2.7 MDE80(.0125)=3.2
```
