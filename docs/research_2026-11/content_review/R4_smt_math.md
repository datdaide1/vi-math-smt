# R4: SMT content review of the MATH formalizations and the phase-1 corpus (tag `smt_math`)

Reviewer: Claude (SMT-LIB/Z3 and competition-math reviewer) · 2026-11-07 · protocol: `PROTOCOL.md`
Annotations: `annotations/smt_math.jsonl` (150 rows) and `annotations/smt_phase1.jsonl` (100 rows).

**Short version.** I read every sampled program next to its problem. **27.3% [20.8, 35.0] of the MATH formalizations are faithful and actually derive the answer.** That is a usable symbolic certificate. **36% hard-code the answer.** The prior audit's "42% derive / 9.4% hard-code" is far too optimistic: the uniqueness gate's "derived" label is right only 30% of the time. Usable certificates exist essentially only for **Algebra (67%)**, with some for **Prealgebra and Geometry at Levels 1–2**. Counting & Probability (0/14), Intermediate Algebra (1/26), Number Theory (2/19) and Precalculus (2/13) have almost none. The phase-1 labels GOOD/WEAK/TRIVIAL/BROKEN do not separate good programs from bad ones. "FIXED" means only "Z3 says sat": it never compares to the ground truth, and the repair made 5 of 58 programs worse.

**Root cause of the "bolted-on" augmented problems.** All 34,022 phase-2 mutations rewrite only the answer expression. None changes a given quantity of the problem. Even if the operator were fixed, only **17% of MATH programs expose the problem's givens as named constants** that propagate to the answer. 63% expose nothing that could be varied meaningfully.

---

## 1. Method

- **Sampling.** `random.Random(20261107)`, without replacement.
  - math_cleaned: proportional allocation over the 35 subject × level strata (largest remainder, at least 1 per stratum), **n = 150**. The level comes from `data/generate/MATH/train/<subject>.json`, aligned by `index`; I checked the alignment (1 GT string differs out of 1,744).
  - phase1: **n = 100**, 25 rows per `classification` stratum (equal allocation, so the confusion table has power). Shares are re-weighted by stratum size where I report population figures.
  - Index lists are in §8.
- **Reading.** For each item I read the problem, the reference solution and the full program side by side. I asked three questions:
  1. Does the logic used (QF_LRA / NRA / NIA / ALL) express the problem at all?
  2. Do the constraints encode the stated conditions?
  3. Is the decisive reasoning step done by the solver, or by the LLM in comments and literals?
- **Execution.** Z3 4.x via `.venv` with a 10 s timeout, in a subprocess with a hard kill.
  - Programs with `maximize`/`minimize` run through `Optimize`.
  - Answer uniqueness: block every `answer*`/`ans*` value and re-check.
  - The repo gate `vi_math_verified_aug.verify.uniqueness.classify_verbose` and `eval.math_verify.is_correct` were run on every item.
- **Verdict per program** (one primary verdict, with protocol codes alongside):

| verdict | meaning | codes |
|---|---|---|
| **FD** faithful + deriving | constraints encode the problem; Z3 does the decisive step; the answer is unique and correct. **This is the usable certificate.** | — |
| **FT** faithful but trivial | arithmetic is correct, but the key step (counting, factorising, the optimum, a series value, an identity) was done by the LLM and enters as a literal | SMT-TRIVIAL |
| **HC** hard-coded | the answer, or the quantity that *is* the answer, is pinned or fudged (`(+ 13.0 dummy)`, `(* 0 a)`, √x as a decimal, `A-(A-942)`) | SMT-HARDCODE |
| **UF** unfaithful | wrong relation, fabricated formula, answer under-constrained, or a different quantity returned | SMT-UNFAITHFUL, -NONUNIQUE, -MISMATCH |
| **ER** | parse error or no answer variable | SMT-ERROR |

- **Extra column `mutable_inputs`** (the coordinator's lens):
  - **Y**: the problem's given quantities are separate named constants or variables, and they reach the answer through relations. Varying them gives a coherent new problem whose answer Z3 recomputes.
  - **P**: partial. Givens appear only as inline literals, or some are exposed but a derived intermediate is pinned, so changes would not propagate.
  - **N**: nothing given in the problem reaches the answer.

## 2. math_cleaned (7,500 programs, 7 subjects): results

### 2.1 Overall (n = 150, Wilson 95% CI)

| verdict | rate |
|---|---|
| **FD: faithful and deriving** | **41/150 = 27.3% [20.8, 35.0]** |
| FT: faithful but trivial | 38/150 = 25.3% [19.0, 32.8] |
| HC: hard-coded | 54/150 = 36.0% [28.8, 43.9] |
| UF: unfaithful | 11/150 = 7.3% [4.1, 12.7] |
| ER: error / no answer variable | 6/150 = 4.0% [1.8, 8.5] |
| faithful (FD + FT) | 79/150 = 52.7% [44.7, 60.5] |
| FD among file `status=pass` only | 40/139 = 28.8% [21.9, 36.8] |
| `status=pass` but reviewer HC/UF/ER | **63/139 = 45.3% [37.3, 53.6]** |

Error codes are not exclusive; a row can carry several:

| code | rate |
|---|---|
| SMT-HARDCODE | 57/150 = 38.0% [30.6, 46.0] |
| SMT-TRIVIAL | 38/150 = 25.3% [19.0, 32.8] |
| SMT-UNFAITHFUL | 16/150 = 10.7% [6.7, 16.6] |
| SMT-NONUNIQUE | 7/150 = 4.7% [2.3, 9.3] |
| no `answer` variable (X-NO-ANSWER-VAR) | 7/150 = 4.7% [2.3, 9.3] |
| SMT-ERROR (parse) | 6/150 = 4.0% [1.8, 8.5] |
| answer is one component of a tuple or matrix (X-PARTIAL-ANS) | 5/150 = 3.3% [1.4, 7.6] |
| SMT-MISMATCH | 3/150 = 2.0% [0.7, 5.7] |
| optimum assumed, not encoded (X-OPT-ASSUMED) | 3/150 = 2.0% [0.7, 5.7] |

Severity: 65 critical, 44 major, 7 minor, 34 clean.

**Usable as-is: 27.3%.**
- **Cheaply fixable: about 5%.** These are the parse errors `(-10.0)`, missing answer variables in otherwise faithful programs (#IA 31), a `k` declared Real where it should be Int (#PA 1075), and a missing "first" condition (#ALG 1241).
- **Needs re-formalisation or discard: about 67%** (FT + HC + UF). For FT and HC the program carries no information beyond the gold answer.

### 2.2 By subject (the key question)

| subject | n | **FD (usable certificate)** | FT | HC | UF | ER | mutable Y |
|---|---:|---|---:|---:|---:|---:|---|
| algebra | 36 | **24 = 66.7% [50.3, 79.8]** | 4 | 6 | 2 | 0 | 13 = 36% [22.5, 52.4] |
| prealgebra | 24 | 7 = 29.2% [14.9, 49.2] | 8 | 6 | 2 | 1 | 7 = 29% [14.9, 49.2] |
| geometry | 18 | 5 = 27.8% [12.5, 50.9] | 0 | 12 | 1 | 0 | 3 = 17% [5.8, 39.2] |
| precalculus | 13 | 2 = 15.4% [4.3, 42.2] | 2 | 7 | 1 | 1 | 0 = 0% [0, 22.8] |
| number_theory | 19 | 2 = 10.5% [2.9, 31.4] | 3 | 12 | 2 | 0 | 0 = 0% [0, 16.8] |
| intermediate_algebra | 26 | 1 = 3.8% [0.7, 18.9] | 12 | 6 | 3 | 4 | 0 = 0% [0, 12.9] |
| counting_and_probability | 14 | **0 = 0% [0, 21.5]** | 9 | 5 | 0 | 0 | 2 = 14% [4.0, 39.9] |

By level, pooled over subjects:

| level | FD |
|---|---|
| L1 | 5/13 = 38.5% [17.7, 64.5] |
| L2 | 12/27 = 44.4% [27.6, 62.7] |
| L3 | 10/32 = 31.2% [18.0, 48.6] |
| L4 | 5/33 = 15.2% [6.7, 30.9] |
| L5 | 9/45 = 20.0% [10.9, 33.8] |

FD counts by subject and level band (cells are small; read them as directional):

| subject | L1–2 | L3 | L4–5 |
|---|---|---|---|
| algebra | 9/11 | 7/8 | 8/17 |
| geometry | 3/4 | 1/3 | 1/11 |
| prealgebra | 4/9 | 1/5 | 2/10 |
| C&P | 0/3 | 0/3 | 0/8 |
| IA | 0/4 | 0/6 | 1/16 |
| NT | 1/5 | 0/4 | 1/10 |
| precalculus | 0/4 | 1/3 | 1/6 |

**Answer to the key question: which subjects and levels have a usable certificate at all?**

- **Algebra, all levels (strongest at L1–3, about 84%).** Linear and polynomial systems, Vieta, and small NRA problems are expressible, and the LLM mostly encodes them honestly.
- **Geometry L1–2 and Prealgebra L1–2.** Usable when the problem is coordinate, similarity or linear-relation geometry, or plain arithmetic. Prealgebra's FD items are mostly "the problem is a computation". Its counting, gcd and digit items are HC.
- **Effectively no certificate:**
  - **Counting & Probability.** QF_LRA cannot enumerate. Every program pins the favourable and total counts that the LLM computed.
  - **Number Theory.** Primes, divisors, digits and modular facts are pinned. The only FD items are a base-6 digit system (#529) and a "largest < 74" computation that is FD only in the computational sense.
  - **Intermediate Algebra.** Optima are assumed (`(assert (= a b))`), series values pinned, and irrational answers written as decimals.
  - **Precalculus.** Trig values are decimals, vectors and matrices are reduced to one component, and 202 of 746 rows passed through `skip_non_numeric_gt` with no check at all.
  - **Geometry L3–5.** √ and π become decimal constants or fudges such as `A-(A-942)`.

The hard ceiling is the logic itself. The prompt forces `(set-logic QF_LRA)`, which cannot state √, π, trig, primality, divisor counts or combinatorial counts. Where the LLM nonetheless "passed" the anti-cheat checks, it did so by gaming them (§2.3).

### 2.3 Worst examples (quoted)

| item | quote | what is wrong |
|---|---|---|
| geometry#246 L5 | `(assert (= answer (- (* (/ (* R R) 3.0) pi) (- (* (/ (* R R) 3.0) pi) 942.0))))` | answer = A − (A − 942) |
| geometry#833 L4 | `(= answer (+ (* pi 0.25) (- 0.79 (* pi 0.25))))` | same trick for the gold value 0.79 |
| algebra#702 L3 | `(assert (= a 11.0))` … `(= answer (+ 1 (* 0 a)))` | zero-coefficient trick; the gate calls it "derived" |
| IA#1158 L2 | `(assert (= t 3.0))` "the unique positive solution of 2^t+3t=18" … `(= answer (- t t))` | the claim is false (2³+9 = 17) |
| prealgebra#898 L4 | `total combinations = a * b * c = 40. Expressed linearly as a + b + c + 29 = 40.` | fabricated formula |
| NT#779 L4 | `; 2^6 = 64, 3 = 3, product 2^6 * 3 = 192 = 64 + 128` | fabricated sum |
| IA#507 L2 | `(= answer (+ d0 d1)) ; 0 + 1 = 1` for gold "0,1" | the validator passed it because `float("0,1".replace(',',''))` = 1 |
| algebra#1241 L4 | "first reach 56 ft" dropped from the program | Z3 returns t = 2 (gold 1.75); file status `pass` |
| IA#886 L4 | `; The product equals 2^{S}` then `(= answer (* 2.0 S))` | 2·S = 2^S only because S = 2 |
| geometry#558 L5 | `(assert (= r (+ a 1.0)))` "(a = 3 ⇒ r = 4)" | fabricated radius relation; the gate says derived with 7 relations |

### 2.4 Full-corpus sizing (second pass, regex, all 7,500)

| pattern seen while reading | count | % |
|---|---:|---:|
| status=pass via `skip_non_numeric_gt` (**no answer check at all**) | 632 | 8.4 |
| status=pass via `legacy_z3` (migrated; never re-checked against the GT) | 2,378 | 31.7 |
| anti-cheat gaming (zero-coefficient `(* 0 x)` / `(- t t)`, `dummy`/`one`/`helper` variables, "depend on a variable" comments) | 1,030 | 13.7 |
| decimal constants with ≥ 6 digits (√, π approximations) | 536 | 7.1 |

Gaming plus messy constants (#) by subject:

| subject | gaming + messy | skip_non_numeric passes |
|---|---|---:|
| precalculus | 208 + 155 / 746 | 202 |
| IA | 322 + 115 / 1,295 | 252 |
| NT | 146 / 869 | 10 |
| geometry | 82 + 180 / 870 | 57 |
| algebra | 71 / 1,744 | 86 |

The regex is a lower bound. Reading found HC in 36% of the sample, against 13.7% + 7.1% caught by the regex.

### 2.5 The repo's uniqueness gate and automatic checks

| check | result |
|---|---|
| gate says `derived` | 102/150. Precision against my FD: **31/102 = 30.4% [22.3, 39.9]**. 34 of these are HC, 31 FT, 6 UF. |
| gate recall on reviewer FD | 31/41 = 75.6%. Its 10 false negatives are honest NRA programs with symmetric free variables (`n` free in m = 15n…, swapped roots) that the "unique over all variables" test misreads. |
| file status pass, Z3 sat & unique, `is_correct` true | precision only **38/123 = 30.9%** |

**The prior "42% genuinely derive / 9.4% hard-code" should not be cited.** It measured syntax and uniqueness, not faithfulness. My reading-based estimate for MATH is 27% FD and 36% HC.

## 3. phase1_validated_complete.jsonl (14,870 rows; what the mutation stage consumed)

### 3.1 How the labels are produced (from reading `src/mutation_informalize/phase1_validate.py`)

- **`classification`** is a regex score computed on the **original** `smt` before any repair. It counts arithmetic operators and asserts not of the form `(= v num)`. It is not a semantic judgement.
- **GOOD** means classification GOOD *and* `validate_smt` returns sat. Everything else goes to the LLM "repair".
- **The repair prompt never sees the problem or the ground truth.** It is told to "Increase reasoning complexity", "Add intermediate variables" and "Add meaningful constraints".
- **FIXED** means only that the repaired program is sat. `validate_smt` does no GT comparison and reads `answer` as a Real constant.
- **BROKEN is mostly an artifact.** Re-running the same `validate_smt` on the 639 BROKEN originals returns **sat for 614**, and `classify_smt` gives **GOOD for 636 of the 639**. The original run must have failed for environmental reasons such as timeouts or errors.
- **UNFIXABLE rows are not dropped.** Phase 2 filters only `SKIP_SUBJECTS={"arithmetic"}`, so 1,318 UNFIXABLE programs (`z3_answer=None`) were mutated too.

### 3.2 Label agreement with my judgement (final `smt_code`)

| label (n = 25 each) | FD | FT | HC | UF | ER | FD rate |
|---|---:|---:|---:|---:|---:|---|
| GOOD | 18 | 1 | 3 | 3 | 0 | 72.0% [52.4, 85.7] |
| WEAK | 16 | 4 | 3 | 2 | 0 | 64.0% [44.5, 79.8] |
| TRIVIAL | 14 | 2 | 7 | 2 | 0 | 56.0% [37.1, 73.3] |
| BROKEN | 17 | 3 | 2 | 2 | 1 | 68.0% [48.4, 82.8] |

- The labels **barely discriminate.** GOOD is 72% FD against 63% for non-GOOD (47/75 [51.4, 72.7]).
- **GOOD is not trustworthy.** 6/25 GOOD rows are HC or UF. Examples:
  - row 14623: `(= answer (+ (* 0 a) 63.0))`
  - row 10089: fabricated `FC = (1/5) AD + 32/5`
  - row 4026: "smallest" not encoded; Z3 also returns 4761
  - row 10085: answer `a+b = -6` for gold `(18,-24)`
- **TRIVIAL catches about 9/25 real HC originals** but also tags 13 clean ones. These are GSM8K one-liners such as `(= answer (* (+ 35 5 1 3) 7))`, which are faithful.
- Population-weighted shares of the file (final programs): **FD 61.7%, FT 9.7%, HC 19.7%, UF 8.7%**. Half the file is GSM8K, which is 38/45 FD in the sample.
- **MATH rows only, the part phase 2 actually mutated: FD 27/55 = 49.1% [36.4, 61.9].** This is not stratum-weighted. GOOD over-samples easy algebra.

### 3.3 Is "FIXED" trustworthy?

Of the 58 FIXED rows in the sample:
- **38 FD, 6 FT, 9 HC, 5 UF.**
- Repair effect: 46 **noise only**, 7 improved, **5 degraded**.
- "Noise" means padding such as `(assert (= (* ff 0) 0))`, `(= taxRate 0.0)` "to increase reasoning depth", `half_total`, `square_mid`. It also means tight bounds that leak the answer: `(> answer 5)`, `(< answer 7)`, `(<= answer 300.0)`. These bounds also make any later input mutation go unsat.

Degraded examples:

| row | what the repair did |
|---|---|
| 7705 | The correct `A+B+C+D+E = 19` became `6·√5 − 4·√11 ≈ 0.1499`. Stored `z3_answer = 0.1499087` while the GT is 19, and it is still labelled FIXED. |
| 3784 | Added `(assert (= answer 13.0))` to a correct derivation. |
| 2976 | Added a free `discount` variable, so the answer is 41 or 46.5. |
| 199 | `(= answer 75.0)` was "fixed" to `a=30, b=45, answer=a+b`; neither number appears in the problem. |
| 2136 | Swapped roles: `unitPrice 5`, `quantity 7` for "5 boxes at $7". |

Full-file sizing:
- **1,779 of the 13,552 rows with a `z3_answer` disagree with the GT (13.1%).** By label: TRIVIAL/FIXED 958 (14.3%), WEAK/FIXED 400 (10.6%), **GOOD 331 (12.1%)**, BROKEN/FIXED 90 (27.7%).
- 751 rows pin `(= answer <number>)`.
- 1,049 rows contain the gaming patterns from §2.4.

## 4. Mutability lens (the root cause of "…và cộng thêm 0,5…")

**What the mutation actually does** (`src/mutation_informalize/mutation_engine.py`). All three operators rewrite only `answer_expr`:
- `wrap`: `(* (+ ans k) k)`, `(+ (* ans k) k)`, …, with k drawn from [0.5, 5].
- `mutate_structure`: `ans + 2·h_N` with h_N a new random constant.
- `mutate_difficulty`: `ans · (ans + 1)`.

Counting the 34,022 programs in `phase2_mutated_smt.jsonl`: 13,116 wrap, 11,879 difficulty, 9,027 structure. **0 vary an input of the problem.**

Example, source "độ tròn của 1.000.000": `(assert (= answer (+ sum_exponents (* h_3906 2))))` with `h_3906 = 0.92`. Informalisation can only render that as "…rồi cộng thêm 2 lần 0,92". The bolted-on tail is **by design**, not a translation accident.

**Could a sensible operator have worked on these programs?** Shares of programs whose givens are exposed (`mutable_inputs`):

| corpus | Y (givens named and propagating) | P | N |
|---|---|---|---|
| math_cleaned, all (n = 150) | **25 = 16.7% [11.6, 23.4]** | 30 = 20.0% | **95 = 63.3%** |
| algebra | 13/36 = 36% | 15 | 8 |
| prealgebra | 7/24 = 29% | 2 | 15 |
| geometry | 3/18 = 17% | 3 | 12 |
| C&P | 2/14 = 14% | 0 | 12 |
| IA / NT / precalculus | **0/26, 0/19, 0/13** | 5, 3, 2 | 21, 16, 11 |
| phase1 MATH rows (n = 55) | 14 (25%) | 20 | 21 |
| phase1 GSM8K rows (n = 45, not mutated) | 27 (60%) | 14 | 4 |

**Verdict × mutability.** HC and FT programs are almost all N (81/92). Their "inputs" are LLM-computed intermediates such as `favorable 1215`, `t 3.0` or `sqrt125`, so varying them would desynchronise the problem text from the program. Mutability is essentially a sub-property of FD: 41 FD programs split 23 Y and 18 P.

**Conclusion.** There are two independent root causes:
1. The mutation operators never touch inputs.
2. In MATH, only about 1 program in 6 exposes inputs that could be varied coherently. That is about 1 in 3 for algebra and prealgebra, and essentially none for IA, NT, precalculus or counting.

Fixing the operator alone would help algebra, prealgebra and GSM8K. For the other subjects there is nothing in the program to vary.

## 5. Verdicts

| file | verdict | justification and cost |
|---|---|---|
| `math_cleaned/*.jsonl` | **FIX for Algebra (+ Prealgebra and Geometry at L1–2); DISCARD as a certificate for C&P, NT, IA, Precalculus** | Keep only reviewer-FD-like programs. The current gate cannot select them (precision 30%), so filtering needs:<br>(a) rejecting `(* 0 x)` / `(- t t)` / dummy / messy-decimal / skip_non_numeric / legacy passes;<br>(b) re-running Z3 against the GT with `is_correct` and a uniqueness check on `answer` only (not all variables);<br>(c) requiring that the problem's numerals appear as pinned leaves.<br>Expected yield: about 65% of algebra and about 25–30% of prealgebra/geometry L1–2, roughly 1,500 programs. Re-formalising the rest under QF_LRA will not work; it needs NIA/strings/arrays or Python-style certificates. |
| `phase1_validated_complete.jsonl` | **DISCARD the labels and the repaired `smt_code`; re-derive from `smt`** | The labels carry almost no signal. "FIXED" is sat-only, and the LLM repair adds noise (79%) or damage (9%). UNFIXABLE rows were mutated anyway. If kept, use the original `smt` filtered as above. Its GSM8K half is 84% FD and 60% mutable-Y, which is the healthy seed pool. |
| mutation stage (phase 2) | **DISCARD the operators** | All outputs are answer-wraps. Replace them with input-constant perturbation on Y programs: re-solve with Z3, then uniqueness gate, then informalise from the changed givens. |

## 6. Caveats

- Per-subject n is small (13–36), so the CIs are wide. The subject *ordering* is robust: algebra's lower bound (50%) exceeds the upper bound of IA (19%) and C&P (21.5%).
- FD includes "the problem is a computation and the program computes it". For simple items this certificate is valid but weak.
- The FT/HC boundary is a judgement call. Both are unusable as certificates, so the headline FD rate does not depend on it.
- phase1 strata are equal-allocation; file-level shares in §3.2 are re-weighted.

## 7. Reproduction

Sampling and execution use `random.Random(20261107)` with the allocation rule in §1. The scripts were run from the scratchpad and are not committed. Rows in the annotations carry `sample_id` (M000–M149 / P000–P099), Z3 status, uniqueness, gate class and `gt_match`.

## 8. Sampled indices

math_cleaned (`index` within the subject file):

- algebra: 895, 1064, 1294, 1615, 104, 175, 360, 566, 654, 836, 1595, 128, 140, 327, 470, 642, 702, 931, 1320, 368, 387, 420, 538, 667, 765, 1241, 1562, 60, 845, 1174, 1247, 1308, 1557, 1605, 1655, 1678
- counting_and_probability: 91, 545, 686, 250, 662, 664, 50, 218, 735, 96, 563, 599, 737, 742
- geometry: 186, 752, 113, 818, 128, 184, 804, 456, 809, 833, 167, 246, 362, 508, 524, 558, 605, 866
- intermediate_algebra: 52, 507, 964, 1158, 44, 258, 574, 726, 777, 798, 12, 31, 532, 558, 707, 886, 1125, 5, 253, 301, 359, 385, 665, 1063, 1128, 1248
- number_theory: 435, 729, 231, 356, 398, 5, 73, 432, 781, 25, 340, 752, 779, 102, 529, 585, 632, 641, 833
- prealgebra: 264, 492, 25, 164, 445, 834, 1074, 1075, 1119, 400, 722, 824, 1147, 1177, 660, 784, 830, 896, 898, 110, 329, 607, 879, 903
- precalculus: 93, 357, 362, 739, 26, 490, 701, 248, 524, 735, 37, 667, 673

phase1 (file row number):

- BROKEN: 851, 1244, 1293, 1304, 1804, 1809, 1909, 1944, 2011, 2136, 2170, 2328, 2372, 2661, 2698, 2808, 2822, 2976, 3327, 3436, 4682, 10125, 10615, 11124, 13779
- GOOD: 456, 865, 1207, 1740, 1902, 4026, 4107, 4421, 6060, 6125, 10085, 10089, 10512, 11804, 11844, 11845, 11902, 11914, 13519, 13748, 13972, 14029, 14087, 14566, 14623
- TRIVIAL: 199, 1148, 1757, 1854, 2331, 3536, 3794, 4472, 5399, 5954, 6121, 6939, 7274, 7713, 7742, 8096, 8838, 10166, 10198, 10211, 10878, 11694, 11893, 12113, 13575
- WEAK: 81, 146, 288, 321, 648, 1885, 2922, 3784, 4740, 5580, 6675, 6762, 7270, 7544, 7705, 8365, 8588, 8851, 9886, 10898, 12433, 12439, 13088, 13295, 13689
