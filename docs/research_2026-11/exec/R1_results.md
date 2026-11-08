# R1 — Results on the 0.6B training tranche (D5; FINAL, 2026-11-08 ~09:30)

**Final seed set:**
- VI: seeds 0–3 for S0, Y+, Y−, N+, N−.
- EN: seed 0 only. EN seed 1 was not run because Modal credit ran out; it is the first successor item, at about $0.7.
- Base: Qwen3-0.6B-Base, one run per language, 4-shot. A VI 0-shot base was added at the lead's request.
- All runs: Qwen3-0.6B-Base LoRA SFT, training sets from `select_v1` with n = 1,808 per arm × language (`D5_selection.md`).

**Code:**

| File | Role |
|---|---|
| `vi_math_verified_aug/analysis/c3_r1_results.py` | Y2 pipeline wrapper: contrast library, two-way cluster bootstrap over items × seeds (B = 4,000), Holm over H1–H4, TOST ±3 pp, decision rule |
| `c3_r1_glmm_kaggle.py` | Y2's `glmm_lme4.R` on a private Kaggle CPU kernel (`datuni/d5-r1-lme4`) |

- Tests: `tests/test_c3_r1.py`.
- Tables: `data/audit/r1/tables.md`, `results.json`. H4 sensitivity: `tables_yes_vs_no.md`. lme4 output: `data/audit/r1/glmm_out_final/`.
- Inputs: 50 complete eval files, 170,074 rows, in `jobs/outputs/t2-eval/ev-*.jsonl` (Y2 schema).
- Compute: local, peak about 70 MB. Kaggle CPU: about 25 min for both models. No Claude or Modal spend.

## 0. Executive summary

**Supported by this tranche:**

1. **Training on 1,808 generated items beats training on the translated seeds, on every dataset and in both languages.**
   - The contrast is arm − S0: the same 0-shot format and the same item count.
   - VI, all four arms, 4 seeds:

     | Dataset | Arm − S0 |
     |---|---|
     | gsm8k_platinum | **+6.1 to +9.5 pp** |
     | math_algprealg | **+10.2 to +11.2 pp** |
     | Vi-ExamMath Core | **+6.1 to +6.7 pp** |

   - EN (1 seed): +13 to +18 pp.
   - Every contrast has q < 0.001 (BH over 75 exploratory contrasts).
2. **H2 (symbolic vs LLM-native, Core) and H3 (verification, Core) are null, and equivalent within ±3 pp.**
   - Bootstrap and lme4 are both non-significant after Holm.
   - TOST at ±3 pp succeeds: H2 90% CI [−1.6, +1.2] pp; H3 [−1.8, +0.9] pp.
   - At 0.6B, n = 1,808, VI and Core, neither the generation method nor verification changes accuracy by 3 pp or more.
   - Effects of 1–2 pp cannot be excluded (§2).
3. **Exploratory, dataset-specific:** on VI gsm8k_platinum (translated GSM8K), N beats Y.
   - H2-type −2.5 [−4.0, −0.9] pp, q = 0.005; Y+ − N+ −3.0 [−5.0, −1.0] pp, q = 0.008.
   - This does not hold on VI math (−0.8, n.s.) or on Core (−0.2, n.s.).

**Not supported, or only provisional:**

4. **H1 (P2, language × verification) is provisional because EN has 1 seed.**
   - Estimate +0.3 [−1.4, +2.0] pp; lme4 logit +0.016, p = 0.81.
   - The bootstrap cannot include EN run-to-run variance, so the TOST "equivalent" verdict is **not claimed**.
   - EN seed 1 is the step that makes H1 testable.
   - The manipulation is present in all 4 arm × language cells (`D5_gate_audit.md` §3), so a null H1 would not be an artefact of a missing manipulation.
5. **H4 (P3, does certifiable training transfer?) is null and underpowered.**
   - Estimate +0.3 [−3.1, +3.7] pp; MDE80 4.8 pp.
   - With the yes-vs-no binary (sensitivity): +0.1 [−4.9, +5.4] pp.
   - TOST fails: the 90% CI is [−2.5, +3.1]. Inconclusive.
6. **Arm vs base is not a clean training contrast in either direction.**
   - 4-shot base: different prompt format. It beats every 0-shot arm on Core (33.8 vs 30.3–30.9).
   - 0-shot base: it does not stop. Of its Core decodes, 56.9% hit the 1,024-token cap, against 10.5–19.2% for the arms. It scores 17.8 on Core.

**Caveats that apply to everything above:**
- 0.6B only, and one model family (Qwen3).
- Seeds are GSM8K only (Y formalizer scope; MATH is 100% unreachable for Y).
- Training-set question length differs between Y and N (SMD 0.46–0.65). It is aliased with arm, so it is a mediator, not a covariate.
- The ± manipulation is much stronger in Y than in N: first-attempt verification rate 54–63% vs 80–87%.
- Core is evaluated greedy only (K = 1).
- Core truncation is high: S0 30%, arms 10–19%.

## 1. Design (fixed before arm results were seen; unchanged)

**Evaluation sets:**

| Dataset | Languages | Items | Decoding |
|---|---|---|---|
| `gsm8k_platinum` | VI + EN | 600 | pass@1 over K = 4 (T 0.6) + greedy |
| `math_algprealg` | VI + EN | 400 | pass@1 over K = 4 + greedy |
| `vi_exam_core_verified` | VI | **1,330** after dropping 37 contamination-flagged items | greedy only |

- The two primary sets share item ids across VI and EN.
- Core is greedy-only, so its greedy decode is used as one sample (K = 1) inside Y2's estimators.
- Results are reported per dataset, never pooled.

**Confirmatory family** (Y2 §2, Holm m = 4):

| H | Contrast | Data |
|---|---|---|
| H1 | VP_vi − VP_en on the parallel items, VP = ½[(Y+ − Y−) + (N+ − N−)] | lme4 model B |
| H2 | ½(Y±) − ½(N±) | Core; lme4 model A |
| H3 | ½(+) − ½(−) | Core; model A |
| H4 | (Y − N) on certifiable − on non-certifiable Core | model A; binary = yes vs (maybe ∪ no), lead pre-registered; yes vs no as sensitivity |

**Decision rule** (Y2): a hypothesis is significant only if all three hold: the bootstrap rejects after Holm, the lme4 logit test rejects after Holm, and the two signs agree. A non-significant result is followed by TOST at ±3 pp on the 90% bootstrap CI.

**GLMM fits** (lme4, full pre-registered random-effect structure, nAGQ = 1):
- Model A: (1|item) + (1|item:arm) + (1|run) + (1|run:evaltype). Converged (singular), 17 min.
  - Variance components (logit SD): item 2.39, item:arm 0.80, run 0.04, run:evaltype 0.
  - Run-to-run variance is tiny next to item variance.
- Model B: + item:lang, item:arm:lang, seed_key. Converged (singular), 8 min.

**Exploratory family** (BH over 75 contrasts):
- per dataset: Y+ − Y−, N+ − N−, Y+ − N+, Y− − N−;
- H2-, H3- and VxM-type contrasts;
- each arm − S0;
- S0 − base (4-shot and 0-shot).

## 2. What a 4 (VI) + 1 (EN) seed tranche can and cannot support

The power language is unchanged from the 08:18 refresh, as the lead requested. The numbers are updated to the final seed set.

**MDE80 at the final VI seeds** (= 2.8 × bootstrap SE, items × seeds):

| Contrast | MDE80 (pp) |
|---|---|
| H2 | 2.3 |
| H3 | 2.3 |
| H4 | 4.8 (7.4 with yes vs no) |
| Per-dataset simple effects | 2.7–3.8 |

- Item variance dominates: lme4 run SD is 0.04 logit vs item SD 2.39. Adding VI seeds beyond 4 buys little.
- Reaching 1–2 pp would need more items or larger models, not more seeds.
- **H1:** MDE80 is about 2.4 pp, but that figure omits EN seed variance (1 seed). E1 planned an MDE of 0.95–1.94 pp with the full design (5 seeds, 2 sizes).

| H | Status in this tranche |
|---|---|
| **H2, H3** | **Confirmatory, as pre-registered**, for effects ≥ 2.3 pp. **Both null and equivalent within ±3 pp.** They cannot exclude 1–2 pp effects. |
| **H1 (P2)** | **Confirmatory label kept; provisional (1 EN seed) and underpowered.** Not claimable until EN seed 1 runs. Even then, only interactions of about 2.5–3 pp or more are detectable. |
| **H4 (P3)** | **Effectively exploratory.** MDE 4.8–7.4 pp, and the estimate depends on the certifiable binary. |
| Per-dataset effects | **Exploratory (BH).** Arm − S0 (6–18 pp) is robust. The VI-gsm8k Y < N effect (−2.5 pp) needs replication on another model or size. |
| Not supported | Size dependence (only 0.6B); model-family generality; training-length mediation (aliased with arm); Core-difficulty/IRT breakdowns (K = 1, too few items per bin). |

## 3. Results per dataset

**Accuracy (%)**: mean over seeds, with the SD across seeds in brackets.
- VI: 4 seeds. EN: 1 seed. Base: 1 run.
- Primary sets show pass@1 over K = 4. Core shows greedy.

| Dataset | base 4-shot | base 0-shot | S0 | Y+ | Y− | N+ | N− |
|---|---|---|---|---|---|---|---|
| VI gsm8k_platinum (600) | 33.8 | 30.3 | 38.6 (0.5) | 45.1 (0.5) | 44.7 (1.1) | **48.1** (1.0) | 46.6 (0.6) |
| VI math_algprealg (400) | 34.8 | 26.8 | 26.3 (1.7) | 36.4 (1.7) | 36.9 (0.8) | **37.5** (1.1) | 37.3 (1.0) |
| **VI Vi-ExamMath Core (1,330)** | **33.8** | 17.8 | 24.2 (1.9) | 30.3 (1.3) | 30.8 (0.9) | 30.5 (0.5) | **30.9** (1.4) |
| EN gsm8k_platinum (600) | 43.8 | – | 46.8 | **61.7** | 60.0 | 61.0 | 60.3 |
| EN math_algprealg (400) | 40.7 | – | 35.1 | 52.4 | **53.5** | 50.9 | 52.4 |

**Truncation** (share of decodes that hit the 1,024-token cap):

| Dataset | base 4-shot | base 0-shot | S0 | Y+ | Y− | N+ | N− |
|---|---|---|---|---|---|---|---|
| VI gsm8k | 2.9 | 6.6 | 1.5 | 0.7 | 0.3 | 1.2 | 0.6 |
| VI math | 7.0 | 16.4 | 10.4 | 5.3 | 4.0 | 7.5 | 4.6 |
| **VI Core** | 32.9 | **56.9** | **30.0** | 14.2 | 10.8 | **19.2** | 10.5 |

**Output tokens on Core:** base-4 546, base-0 724, S0 472; Y+ 445, Y− 422, N+ 462, N− 408.

**S0 − base** (pp). **Neither is a training effect.** S0 − base-4shot mixes training with the prompt format. S0 − base-0shot compares against a base model that does not stop.

| Dataset | S0 − base-4shot | S0 − base-0shot |
|---|---|---|
| VI gsm8k | +4.8 [+2.5, +7.1] | +8.3 [+5.9, +10.8] |
| VI math | −8.6 [−11.3, −5.8] | −0.6 [−3.4, +2.4] |
| VI Core | −9.6 [−12.5, −6.6] | +6.4 [+3.5, +9.3] |
| EN gsm8k | +3.0 [+0.6, +5.4] | – |
| EN math | −5.6 [−8.3, −2.9] | – |

## 4. Confirmatory family: decision-rule outcomes

| H | Bootstrap estimate (pp), 95% CI | Holm p (boot) | lme4 logit estimate | Holm p (lme4) | 90% CI (TOST) | Outcome |
|---|---|---|---|---|---|---|
| H1 P2 | +0.31 [−1.36, +2.01] | 1.00 | +0.016 | 1.00 | [−1.10, +1.72] | **Provisional (1 EN seed).** Not significant; no equivalence claim |
| H2 Y vs N (Core) | −0.15 [−1.81, +1.46] | 1.00 | −0.016 | 1.00 | [−1.56, +1.19] | **Not significant; equivalent within ±3 pp** |
| H3 verification (Core) | −0.43 [−2.06, +1.16] | 1.00 | −0.047 | 1.00 | [−1.81, +0.88] | **Not significant; equivalent within ±3 pp** |
| H4 P3 (yes vs maybe ∪ no) | +0.32 [−3.08, +3.67] | 1.00 | +0.042 | 1.00 | [−2.52, +3.11] | **Not significant; TOST fails: inconclusive** |
| H4 sensitivity (yes vs no) | +0.10 [−4.94, +5.38] | 1.00 | (bootstrap only) | — | — | Not significant |

## 5. Exploratory findings (BH q < 0.05; 29 of 75 contrasts)

1. **Arm − S0 is positive on every dataset × language** (20 contrasts, all q < 0.001).

   | Set | Range (pp) |
   |---|---|
   | VI gsm8k | +6.1 to +9.5 |
   | VI math | +10.2 to +11.2 |
   | VI Core | +6.1 to +6.7 |
   | EN gsm8k | +13.3 to +14.9 |
   | EN math | +15.8 to +18.4 |

2. **VI gsm8k_platinum: N > Y.**
   - H2-type −2.47 [−4.04, −0.87], q = 0.005.
   - Y+ − N+ −3.01 [−5.04, −0.99], q = 0.008. Y− − N− −1.94, n.s.
   - Not seen on VI math or on Core; EN is flat, at 1 seed.
3. **S0 − base:** 7 contrasts with q < 0.05, all format- or truncation-confounded (§3).
4. **No verification (±) or VxM contrast survives BH on any dataset.**
   - The largest is N+ − N− on VI gsm8k, +1.5, q = 0.22.
   - At 2 seeds, VI gsm8k H3-type was +1.9 (q = 0.04). At 4 seeds it is +1.0 [−0.4, +2.3], n.s. This is regression to the null, which is why seed-0/1 snapshots were labelled provisional.
5. **Truncation, descriptive (no test):** on Core, the verified arms truncate more than the unverified ones (Y+ 14.2 vs Y− 10.8%; N+ 19.2 vs N− 10.5%) at equal accuracy. Worth a successor look: verified items have shorter traces in training (`D5_selection.md` §0), so this is unexpected.

## 6. Successor items (in order)

1. **EN seed 1** (≈ $0.7 Modal). It makes H1 testable.
2. **A 4-shot evaluation of the SFT adapters**, or a stop-sequence fix for the 0-shot base. Either gives a clean base comparison.
3. **A length-matched training subset** (`select_v1` can draw it), plus retraining. It identifies whether training length mediates H2.
4. **The 1.7B size and a second model family.** They test whether the null H2/H3 holds where items are less limiting.
5. **lme4 for the H4 yes-vs-no sensitivity**, with the same kernel and a re-export using `--binary yes_vs_no`.
