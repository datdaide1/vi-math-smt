# C3 — Faithfulness-gap attribution study with matched baselines (D5, 2026-11-07)

**Question (user):** the old error analysis was confounded, because an error could come from a weak method, model or prompt instead of the approach. Which error classes come from (i) the seed text, (ii) the mutation operator, (iii) the renderer and (iv) the generator model? Each one is measured against a matched baseline that holds the other factors fixed.

**Code (D5-owned):** `vi_math_verified_aug/analysis/`:

| File | Purpose |
|---|---|
| `c3_data.py` | loaders for every judged pilot item |
| `c3_stats.py` | Newcombe paired/unpaired RD, cluster bootstrap/permutation, PAF, Cochran-Q |
| `c3_attrib.py` | all contrasts (entry point) |
| `c3_absurd.py` | cause coding of every NL-ABSURD item, by reading |
| `c3_judge.py` | Opus Batches judge, ws D5 |
| `c3_lit.py` | literature baselines |
| `c3_report.py` | tables |

**Tests:** `tests/test_c3_stats.py` and `tests/test_c3_attrib.py` give **14 passed**. They check:
- Newcombe's published unpaired example (0.0524 to 0.3339);
- simulated coverage of the paired CI;
- Cochran's Q on a hand example;
- that the loaders reproduce the G3/G4 report counts;
- that every absurd item is coded;
- the G3→G4 identity control.

**Outputs:** `data/audit/c3/results.json` and `data/audit/c3/tables.md`. Every table cited below as T1–T9 is generated there. `data/audit/c3/judge_opus_R1b.jsonl` holds the new judge rows.

**Spend:** Claude **$0.58**: 88 Opus `validity_judge.v2` calls via the Batches API, R1b, logged in `logs/api/D5/claude.jsonl`. No Modal, Gemini or NIM. No test or benchmark item was sent anywhere.

**Reproduce:** `.venv\Scripts\python -m vi_math_verified_aug.analysis.c3_attrib`. It reads only, uses numpy (no torch) and takes about 40 s. The R1b judging is `E:\venv_tools\Scripts\python -m vi_math_verified_aug.analysis.c3_judge r1b go`; it is cached, so a re-run is free.

## 0. Answer

### Attribution per error class

Rates are on tag-passing items, judged by Opus 5.5 with `validity_judge.v2`. RD is in pp with 95% CI. "Y − N" is paired on the same 60 seeds / slots, with the same operator request, the same generator family (Haiku 5.5) and the same judge.

| Error class | Main cause(s), with the others held fixed | Key evidence (paired unless noted) | Approach residual: best Y − N, same slots | Verdict |
|---|---|---|---|---|
| **Tail** (NL-TACKED) | **(ii) operator only** | Legacy answer-expression operators: **1,008/1,008** rejected as answer tails (X4 test, re-run). R5: 84% tails. G3 M6 composition: +92 pp [+73, +100] over M1. Dropping/fixing M3/M4/M6 (G3→G4, same renderer): tails per produced item **14% → 0%** (T4d). Renderer, generator and judge cannot remove a tail: it is decided before rendering | E/R1 0 [−4, +4] | **Fixed by operator design.** Not intrinsic to the symbolic approach (the N arm also gets 12% tails with the v1 prompt, G2) |
| **Leak** | answer-exposure design (legacy), not any factor here | ≤ 1% in every G4 variant (E/R1/R2/R1b/N); G1 native generators 0–7% (n.s., omnibus p = 0.91). The legacy pipeline's 58% (R5) came from showing the informalizer the answer | 0 [−5, +5] | **Solved** in both arms |
| **Absurd** | **(iii) renderer/localization** and **(ii) operator value choice**; seed is minor; model and prompt are not significant | E→R1: **+16 [+7, +24]**, p = 5e-4. The causal test **R1→R1b** (only the localization changes, 1 USD = 1.000 → 10.000 đồng, no ounce/feet relabel) gives **−12 [−21, −4]**, p = 0.007, and **−22 [−38, −3]** on the localized items. Reading all 104 absurd items (T8): R1/R2 absurdity is mainly **renderer/localization** (primary cause 13/24 and 16/22, of which LOC-PRICE/LOC-UNIT 12 and 12); after R1b, **8/13** are operator-chosen values | **E − N +6 [−2, +14]** (n.s.). R1 − N +21 [+11, +31]. **R1b − N +9 [0, +18]**, p = 0.057 | Mostly **fixable pipeline defects**. A small residual (≈ +6 to +9 pp) is operator value choice that the IR does not constrain (no world-plausibility / integrality of group counts) |
| **Unanswerable** | **(i) seed** + **(ii) operator M3** | Seed unanswerable → E item 50% vs 8% (cluster perm p = 0.02). G3 M3 vs M1 **+28 [+8, +45]**. Renderer pairs are all n.s. under Opus (≤ +2); under Sonnet E→R2 is −10 [−20, −2] | R1 − N +6 [−3, +15] (Opus, n.s.); +13 [+4, +23] under Sonnet | **Judge-dependent, small.** Inherited ambiguity in seeds plus M3 templates |
| **Fluency / translationese** | **(i) seed × (iii) renderer**; also **(iv) model** | Seeds 88% fluency issues. The minimal-edit E inherits them (71%). Fluency pass E→R1: **−39 [−49, −27]**, p = 3e-9; replicated with Sonnet (−26). G3: items from a fluent seed are 53% vs 81% fluency issues, RD +27 [+14, +42]. Generator model (G1, native): Sonnet **+34 [+10, +54]** vs Haiku | E − N **+48**. R1 − N **+9 [−4, +21]** (n.s.; Sonnet +16) | **Renderer-fixable.** The translationese in arm Y was seed text carried through by a minimal-edit renderer |
| **Near-dup** | **(ii) operator** (by design) and the approach's span-grounded editing; also (iv) model | Per-operator near-dup in E: M3 72%, M4/M5 100%; N on the same requests: 39 / 25 / 33%. Operator effect is larger in Y than N (T4c: M3 R1 − N +39 [+7, +62]). Generator model: gemini-3.5-lite +35 [+9, +55] vs Haiku | R1 − N **+17 [+4, +29]**. R1b − N **+19 [+6, +31]** | **Intrinsic to minimal-edit symbolic rendering of M3–M5**, by construction. M2 is the exception (R1 − N −5) |
| Not teacher-acceptable (composite) | all of the above | E→R1 −17 [−27, −5]. R1→R1b −7 [−15, +2] | R1 − N +27 [+13, +39]. **R1b − N +20 [+8, +32]** | The remaining gap is near-dup (operators M3–M5) + operator values + inherited ambiguity |

### What the old analysis got wrong, and what holds up

1. **The R5 numbers mixed four causes, and they separate cleanly once the others are held fixed.**
   - The legacy 84% tail rate is an **operator** property.
   - The 58% leak rate is an **answer-exposure** property.
   - The 15.5% absurd rate is mostly operator value choice: tails on story quantities.
   - Generator model and prompt do not significantly change absurd, tail or leak on paired seeds: G1 generator omnibus p = 0.46 (absurd), 0.71 (tail), 0.91 (leak); G2 prompt omnibus p = 0.39 (absurd). One caveat: a weaker native prompt can produce tails in the N arm as well (G2 v1 12% [5–26] vs v3 0%).
2. **Most of the arm-Y vs arm-N quality gap is pipeline defects, not the symbolic approach.**
   - With the G3 renderer and the fixed operators (E), Y is **not** more absurd, unanswerable or leaky than N on the same slots: absurd +6 [−2, +14], unanswerable +3 [−6, +12].
   - Y's extra absurdity in R1/R2 is caused by the **localization step** (shown causally by R1→R1b).
   - Y's translationese is caused by the **seed + minimal-edit renderer** (shown causally by E→R1).
3. **What is intrinsic to the approach, with the best fixed Y variant (R1b) vs N:**
   - near-dup **+19 pp**, which comes from span-grounded edits of M3–M5;
   - a small **absurd +9 pp [0, +18]** residual from operator value choices that are certified but not world-plausible. 8 of the 13 R1b absurd items have an operator as primary cause (6 OP-VALUE, 2 OP-STRUCT); 7 slots are absurd in all of E, R1 and R1b.
   - Both are design-addressable: plausibility constraints in the IR and story-level operators. Both are reported as the faithfulness-gap residual, not hidden.
4. **Answer faithfulness is the approach's gain, and it matches the literature.**
   - **Before** the cycle check, 12% [7–18] of G3 renders are answer-unfaithful; MathCAMPS reports 12.2% unfaithful before its cycle check.
   - **After** the check, 98% [93–99] are faithful; MathCAMPS reports 97.7%.
   - With the G4 renderers it is 99–100% after the check and only 1–3% unfaithful before it (T9b).
   - Arm N has no certificate. Its generator answers agree with Opus on 97% of items, but there is no per-item guarantee (G4 §0).

## 1. Design

### Data

Everything already judged. One Opus judge (`validity_judge.v2`, effort medium) is used throughout, so the labels are comparable.

| Pilot | Matched design | n judged |
|---|---|---|
| G4 (`data/pilot/symbolic_v2/`) | 60 seeds → 180 slots. Same mutated programs rendered by **E** (G3 editor.v3), **R1** (E + fluency pass, 1 USD = 1.000 đ), **R2** (free render from IR) and **R1b** (fluency.v2, 1 USD = 10.000 đ, no imperial relabel; D5-judged). **N** = Haiku `native_generator.v3` on the **same slots and operator requests** | E 90, R1 90, R2 87, R1b 88, N 91 (+ Sonnet-v3 labels on E/R1/R2/N) |
| G3 (`data/pilot/symbolic/run_b/`) | Same 60 seeds and 180 slots, **G3 operators** (incl. M6), editor.v3 → paired with G4-E for the operator-fix contrast. Opus also judged the **60 untransformed seed texts** → the seed exposure | 139 + 60 seeds |
| G1 (`data/pilot/generator/`) | **4 generator models** × the same 30 seeds / operator / card (native, `native_v1`). **3 editor models** on the same 19 edit lists | 119 native, 70 editor |
| G2 (`data/pilot/g2/items_task3.jsonl`) | **4 native prompts** × the same 40 seeds (Haiku) | 160 |
| Legacy | X4 structural test (`tests/test_symbolic_rules.py::test_legacy_operators_are_answer_tails_on_every_fixture`; re-run by D5, 12 passed = 1,008 cases) and the R5 reading audit (n = 200) | — |

### Factors and their matched baselines

| Factor | Baseline that holds everything else fixed | Statistic |
|---|---|---|
| (i) seed | Exposure = the seed has the same defect (Opus on the seed text). Every variant, including N, sees the same seed | RD exposed − unexposed, cluster bootstrap over seeds; **cluster permutation p** (seed label shuffled across seeds); **PAF** |
| (ii) operator | (a) op vs M1 inside one renderer; (b) **Y − N on the same slot per operator** (N gets the same operator request, so a shared op effect = the transformation, a Y-only effect = the symbolic implementation); (c) **G3 vs G4 operators, same 180 slots, same renderer** | cluster-bootstrap RD; Newcombe paired + exact McNemar |
| (iii) renderer | E vs R1 vs R2 vs R1b on the **same mutated programs**; G1 editors on the same edit lists | Newcombe paired, McNemar, Holm over 7 classes |
| (iv) generator model / prompt | G1 4 models × 30 paired seeds; G2 4 prompts × 40 paired seeds | Cochran-Q (permutation) + pairwise vs reference (Holm) |
| approach | Best Y renderer vs N on the same slots | Newcombe paired, McNemar |
| judge | Renderer and approach contrasts re-run with Sonnet 5.5 + `validity_judge.v3` | same |

## 2. Results by factor

Full tables are in `data/audit/c3/tables.md`.

### (i) Seed (T3)

- The seeds are the bottleneck for fluency. **88% [78–94]** have fluency issues, and only **17%** are teacher-acceptable.
- Defects that are rarer in seeds still propagate:
  - **absurd seed → absurd G3 item 64% vs 18%**: RD +46 [+4, +85], perm p = 0.006.
  - **unanswerable seed → unanswerable E item 50% vs 8%**: perm p = 0.020.
- Few seeds are absurd (5/60), so the **population attributable fraction** of absurdity is small: G3 17% [0, 35], E 17% [−2, 47], R1 5%.
- Inheritance depends on the renderer. P(fluency issue | seed fluency issue) is 81% for G3 and 72% for E, against **32% for R1** and 25% for N.
  - The minimal-edit renderer transmits the seed's translationese.
  - A rewrite stage (R1, or N's free generation) breaks the link.
- **The seed causes errors in both arms**, through the same channel: N absurd 33% vs 5% when the seed is absurd, p = 0.045. This is not a Y-specific confound.

### (ii) Operator (T4a–d)

- **Operator fix, same renderer and same slots (T4d), G3 → G4-E:**
  - Per scheduled slot: absurd **−11 [−16, −7]**, tail **−11 [−16, −6]**, unanswerable **−11 [−16, −6]** pp.
  - On the changed operators M3/M4/M6: −20, −19 and −19.
  - Control: the **unchanged operators M1/M2/M5 show exactly 0 difference** (labels agree on 68/68 identical programs).
  - The cost is yield: changed-op slots go from 71% to 22% producing an item.
  - Per produced item: absurd 22% → 11% and tail 14% → 0%.
- **Inside one pipeline (T4b):**
  - G3 M6 vs M1: tail **+92**, absurd **+38**, unanswerable **+34**.
  - G3 M3 vs M1: absurd **+29**, unanswerable **+28**, tail **+12**.
  - With fixed operators (E), no operator differs significantly from M1 on absurd, unanswerable or tail.
- **Same operator request, Y − N (T4c):**
  - M2 (inverse): R1 − N absurd **+24 [+6, +41]**. The inversion request itself is easy for the N generator (N M2 absurd 8%), so the symbolic M2's **stated-value choice** is the cause. This is coded OP-VALUE in T8, e.g. "sau 15 tuần" (48/2^15 cans) and "75 ounce" making a cleared pipe no faster.
  - M3–M5: near-dup excess +33 to +75 pp. The N generator also produces near-dups on these requests (25–39%), so part of this is the transformation itself and part is span-grounded editing.

### (iii) Renderer (T5, T8b, T8c)

**Same programs (Opus):**

| Change | absurd | fluency | not teacher | near-dup |
|---|---|---|---|---|
| E→R1 | **+16** | **−39** | **−17** | — |
| E→R2 | **+15** | **−31** | −12 (n.s.) | **−11** |
| R1 vs R2 | n.s. on every class | n.s. | n.s. | n.s. |

Sonnet-v3 reproduces every significant renderer contrast.

**Localization is the cause of the renderer's absurdity:**
- Inside R1, localized items are absurd 39% vs 17% non-localized; R2 37% vs 16% (T8b).
- **R1 → R1b**, the same slots with only the localization fixed: absurd **−12 [−21, −4]**, p = 0.007. Discordant pairs are 13 vs 2.
- On the localized items alone the change is **−22 [−38, −3]**.
- Fluency and near-dup are unchanged.

**Editor model** (G1, the same 19 edit lists, T5b): no significant difference on absurd, unanswerable or near-dup (omnibus p ≥ 0.67). Fluency differs (p = 0.021; gemini-3.1-lite worst). G1 already concluded that *"the bottleneck is the edit design, not the editor"*, and this confirms it.

### (iv) Generator model and prompt (T6)

**G1 native generation, 4 models × 29 complete seeds:**
- Model does **not** significantly change absurd (omnibus p = 0.46), tail (0.71), leak (0.91) or unanswerable (0.35).
- It **does** change fluency (p = 0.009; Sonnet +34 [+10, +54] vs Haiku, Holm p = 0.039) and near-dup (p = 0.042; gemini-3.5-lite +35).

**G2 prompt, Haiku × 40 seeds:**
- The prompt changes not-teacher (omnibus p = 0.049; v3 −15 vs v1).
- It changes **tail in the N arm**: v1 12% [5–26] vs v3 0%, RD −12 [−26, −1], Holm p = 0.19. So a bolted-on tail is not unique to symbolic operators. A free generator does it too under a weaker prompt.
- Absurd: omnibus p = 0.39.

**Consequence:** the Y-vs-N differences in T7 are **not** a generator-model artifact.
- Both arms use Haiku 5.5.
- Model effects in G1 sit on fluency and near-dup, not on the absurd, tail or leak classes where Y and N differ most.

### Approach residual (T7, T8c)

| Y − N, same slots | absurd | unanswerable | fluency | near-dup | not teacher |
|---|---|---|---|---|---|
| E (tag-passing) | +6 [−2, +14] | +3 [−6, +12] | **+48** | **+23** | **+43** |
| R1 | **+21 [+11, +31]** | +6 [−3, +15] | +9 [−4, +21] | **+17** | **+27** |
| **R1b** (fixed localization) | **+9 [0, +18]**, p = .057 | +7 [−3, +17] | +9 [−3, +21] | **+19 [+6, +31]** | **+20 [+8, +32]** |
| R1, Y survivors only | **+17** | 0 | +8 | +13 | **+18** |
| R1, Sonnet-v3 labels | **+23** | **+13** | **+16** | +5 | **+28** |

### Absurd items read one by one (T8)

All 104 NL-ABSURD items across G3, E, R1, R2, R1b and N were coded from the Opus note, the text, and the paired E/N/seed verdicts:

| Variant | Absurd items | Primary cause |
|---|---|---|
| G3 | 30 | 23 operator (17 OP-STRUCT: M3 templates and M6 bridges), 7 seed |
| E | 10 | 8 operator, 2 seed |
| R1 | 24 | **13 localization/renderer**, 7 operator, 2 seed, 2 IR |
| R2 | 22 | **16 renderer/localization**, 4 operator, 2 IR |
| R1b | 13 | **8 operator**, 3 renderer, 2 seed |
| N | 5 | 4 generator, 1 seed |

**IR:** the "18 món hàng" seed, where a redundant total was encoded as a distractor, is the formalizer defect G3 already reported (fix 5).

**Single coder:** the coding is mine alone, not double-coded, so treat the cause shares as ±1–2 items.

## 3. Literature baselines (T9; exact numbers and where they come from)

| Source | What it measures | Value | Our closest number |
|---|---|---|---|
| **GSM-Plus** (Li et al., ACL 2024, [2402.19255](https://arxiv.org/html/2402.19255v2) §3.2, App. B.2 Table 7) | Human pass rate of GPT-4 question variations. "human annotators revised 18.85% of the variations produced by GPT-4" | Numerical Substitution **91.51%**, Adding Operation **75.75%**, Reversing Operation **29.76%** | Well-posed ∧ op-applied (Opus): M1 80–88%; **M2 62–89%** (far above GPT-4's 29.76%); **M3 44–67%** (below 75.75% for Y; N 61% CI includes it) |
| **MathCAMPS** (Mishra, Poesia, Goodman, [2407.00900](https://arxiv.org/html/2407.00900v3) App. F) | Manual faithfulness of GPT-4 word problems to the symbolic structure. "we identified 30 word problems (12.2%) that were not faithful"; "210 (97.7%) were judged as faithful" after the cycle check | 12.2% → 97.7% | G3: **12% [7–18] → 98% [93–99]**; G4 renderers 1–3% → 99–100% |
| **MATHWELL** (Christ et al., [2402.15861](https://arxiv.org/html/2402.15861v5) Table 3, 250 per model, experts) | Meets all criteria: solvable ∧ accurate ∧ appropriate | GPT-4 Turbo **78.8%**, MATHWELL 74.8%, Llama-2 62.4% | N teacher-acceptable 80% [71–87]; R1b 60%; R1 53% |
| **ElemMWP** (Ariyarathne et al., [2506.05950](https://arxiv.org/html/2506.05950v2) v2 Mar 2026, Tables 5/10) | Human error categories of 7B-model MWPs | unsolvable **6.92%**, unrealistic **4.62%** (Llama-2 0-shot); unsolvable 15.83% (fine-tuned 5-shot) | N unanswerable 8%, absurd 5%; R1b 15% / 15% |
| **MathQ-Verify** (Shen et al., [2505.13903](https://arxiv.org/html/2505.13903v2) §3.2) | GPT-4o raw synthetic questions incorrect: "(with fewer than 30% incorrect samples)". ValiMath 848/2,147 incorrect is curated to about 1:3, **not** a population rate | < 30% | not-valid: N 20% (18/91), R1b 38% (33/88) |
| **RV-Syn** ([2504.20426](https://arxiv.org/html/2504.20426) Table 4; GPT-4o judge, 1,000 per corpus) | Problem error rate | ScaleQuest 5.6%, Jiuzhang3.0 2.4%, NuminaMath 0.6%; solution error Mammoth2 10.6%, RV-Syn 1.4% | different rubric (no realism/naturalness); not directly comparable |
| **CoT-Self-Instruct** (Yu et al., [2507.23751](https://arxiv.org/html/2507.23751) Table 1) | Answer-Consistency filter kept 2,926/5,000 = **58.5%**; "not all generated synthetic instructions are well-defined and answerable" | filter proxy | N-arm verification yield (D5 Part B) |
| **OpenMathInstruct-2** (Toshniwal et al., [2410.01560](https://arxiv.org/html/2410.01560) §1, §2.2.3) | Filters flag "6% to 12%". SFT "robust to the presence of up-to 20% low-quality data" | — | context for the gate thresholds and the E1 ± effect size |
| **Yeo et al. 2602.19187** ([html](https://arxiv.org/html/2602.19187) §3.5, §4.4) | "Both pipelines produce similar rates of valid problems": **no rate reported**. SMT renderings "often results in unnatural-sounding word problems" (qualitative). Diversity Opt-Sym 0.37 vs Opt-NL 0.07 | not reported | Our renderer study puts the first numbers on the "unnatural" claim: E fluency 71% → R1 32% |
| **GSM-Symbolic** ([2410.05229](https://arxiv.org/html/2410.05229) §3.1) | "10 random samples per template are reviewed manually"; **no error rate reported** | not reported | — |
| **MetaMath** ([2309.12284](https://arxiv.org/abs/2309.12284)) | No invalid-question rate reported for SV/FOBAR (answer filter only). In-repo reading of MetaMathQA (R6, n = 150): **10.0%** of rows have problems inherited from the EN source (broken FOBAR templates, wrong AnsAug/SV reasoning) | not reported / 10% in-repo | — |
| **R5 legacy pipeline** (in-repo, n = 200 read) | tail 84.0%, leak 58.0%, absurd 15.5%, teacher 10.5%, new 0.0% | — | Matched-factor decomposition above |

**Reading:**
- The fixed arm-Y pipeline is at or above literature on answer faithfulness (MathCAMPS-level) and on inversion validity (vs GSM-Plus "Reversing Operation").
- It is below literature on M3-type "add a step" validity (GSM-Plus 75.75% vs our 44–67%).
- It is below literature on teacher-style acceptability (MATHWELL GPT-4 78.8% vs R1b 60%).
- Literature numbers come from other judges (human or GPT-4o), other languages (EN) and other rubrics. Only GSM-Plus Table 7 and MathCAMPS App. F are operator- or design-matched.

## 4. What this means for the thesis (C3 claim)

1. **The C3 headline can be stated with attribution:**
   - The certificate closes the *answer* faithfulness gap: 12% → 2% unfaithful pre-cycle with G4 renderers, 98–100% post-cycle, the same as MathCAMPS.
   - The *text* faithfulness gap of symbolic Vietnamese data decomposes into:
     - operator design: tail, near-dup, M2 value choice;
     - renderer/localization: absurd, fluency;
     - seed translation quality: fluency, inherited ambiguity.
   - **Not** into generator model or prompt.
   - Each part has a matched contrast with a CI (T2). The legacy pipeline's failure (R5) was operator + answer exposure, not the LLM.
2. **For the remaining Y-vs-N gap:**
   - Report the R1b-vs-N residual (near-dup +19, absurd +9 [0, +18], not-teacher +20) as the cost of span-grounded symbolic generation at this design point.
   - Do **not** report the R1 numbers, which include a localization bug.
3. **Arm-Y fixes ranked by attributable effect:**
   1. localization scale (done in R1b: −12 pp absurd);
   2. plausibility constraints on operator-chosen values (the 7 persistent OP-VALUE slots);
   3. story-level M3–M5 (near-dup);
   4. seed repair/post-edit (S3: unanswerable inheritance).
4. **Caveats:**
   - n = 60 seeds, so CIs are wide (±8–10 pp on paired RDs).
   - There is a single LLM judge. Opus re-test κ = 0.93 (G2) and no measured self-preference (G2 DiD −8 [−31, +13]). The Sonnet-v3 replicate agrees on every renderer contrast, but unanswerable is judge-sensitive.
   - The cause coding is single-coder.
   - R1b was compared before any cycle check, like every other tag-passing contrast; G4 could not cycle-check it (Gemini quota).
   - The seeds are translated GSM8K only.
   - G1 (generator) used different seeds from G4, so model effects are within-G1 contrasts and cannot be added to G4 effects.
   - Y v3 (`data/pilot/symbolic_v3/`, being built by another workstream) is not judged yet and is not included.
