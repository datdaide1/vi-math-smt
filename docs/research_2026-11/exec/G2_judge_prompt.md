# G2 — Self-preference check, judge cascade calibration, native-prompt iteration (2026-11-07)

**Follows:** `exec/G1_generator_pilot.md` (reuses its data in `data/pilot/generator/`).
**Code:** `vi_math_verified_aug/pilot_gen/g2_common.py` (shared `common/llm.py` client, budget cap, per-model admission control), `g2_judges.py` (tasks 1–2), `g2_native.py` (task 3), `g2_report.py` (all metrics), `g2_probe.py` (Gemini quota probe).
**Prompts (new, versioned):**
- `prompts/validity_judge.v2.md`: verbatim port of G1 `judge_v2` (the Opus judge of record).
- `prompts/validity_judge.v3.md`: new cheap-judge rubric. It solves first, then asks 11 yes/no questions with 6 few-shot anchors from G1 items Opus labelled. Labels are derived in code (`g2_common.lab_v3`).
- `prompts/native_generator.v2|v3|v4.md`: native-generation variants A, B and C.
- `prompts/solution_writer.v2.md`: port of G1 `solve_v1`.
- The `*.v1.md` templates were left untouched.

**Data:** `data/pilot/g2/` holds one JSONL per stage, plus `g2_metrics.json` and `items_task3.jsonl`.
**API log:** `logs/api/G2/{claude,gemini}.jsonl`.
- Three one-shot Gemini quota probes (3.6-flash, 3.5-flash, 3-flash-preview; 17+25 prompt) were sent without the logger and are therefore missing from it.

**Spend:** Claude **$6.50** of the $8 cap, from logged usage at list prices (direct calls; Batch was not used): Opus $3.92, Sonnet $1.94, Haiku $0.64. Gemini calls are free-tier.
**Data routing:** only Vi-GSM8K train dev seeds and items generated from them went to Gemini (`split="dev"` guard). No test or benchmark item was sent anywhere. No NIM.

## 0. Answers

| Question | Answer (evidence below) |
|---|---|
| **1. Does the Claude judge favour Claude items?** | **No measurable boost; if anything Opus is stricter on Claude items.** gemini-3-flash-preview re-judged 118/119 G1 subsample items. Opus − Gemini accept-rate gap: −32 pp on Claude items vs −24 pp on Gemini items. Difference-in-differences: **−8 pp [95% CI −31, +13]**. On absurdity, Opus flags Claude items more than Gemini does: DiD **+12 pp [+2, +23]**. Haiku ranks first under every judge (Kendall τ vs Opus: 0.67 on teacher, 0.83 on yield). |
| **2. Cheapest judge that agrees with Opus well enough to screen** | **Sonnet 5.5 + rubric v3** has the best agreement: κ(accept) 0.63, valid 0.74, precision 0.88, at $0.0061 per item. **Haiku 5.5 low + v3** reaches κ 0.52–0.57 and precision 0.82 (native) at $0.0005. gemini-3.1-flash-lite is too lenient to use alone even with v3: it accepts 86%, κ 0.19. Effort medium on Haiku gives no gain. **Recommended cascade:** Haiku-v3 ∧ 3.1-lite-v3 ∧ Haiku 2-sample answer screen; Opus (Batch) only on judge disagreements, plus a 5% audit. On the task-3 Haiku items: precision **87% [80–92]** on all arms and **95% [83–99]** on the chosen v3 prompt, recall 97–98%. Cost **≈ $31–36 per 10k accepted on Batch** (≈ $62–73 direct), vs **$123 Batch / $246 direct** for Opus on every item. |
| **3. Best native prompt (Haiku 5.5, effort low)** | **`native_generator.v3` ("self-check")**. On 40 new dev seeds under Opus: teacher **92% [80–97]**, new (M2–M6) **78% [61–89]**, absurd **0/40 [0–9]**, leak **0/40**, tail 0/40, near-dup (M2–M6) 19%, answer-correct 98%, strict yield 75%. **It meets every quality-gate threshold** in the point estimate; the G1 control prompt misses "new ≥ 60%" (59%). Against v1 on the same seeds: accept 9 vs 2 discordant pairs (exact McNemar p = .065). Cost +20% ($0.00115 per item). The contrastive-example variant (v4) is *worse*. Medium effort was not run (see §3.4). |
| 4. Arm-Y editor | Not touched. It waits for X4's symbolic library and real edit lists (G1 §4.3). |

**Two cautions that change how the G1 numbers should be read:**

**(a) Seed-sample variance is large.**
- The unchanged G1 prompt (v1) scores teacher 78% and absurd 2% (Opus) on the 40 new seeds, against 63% and 23% on G1's 30 seeds.
- Opus is stable on re-test: an independent re-judge of G1's 30 Haiku items gave κ = 0.93 (accept), 0.92 (valid) and 0.71 (absurd). The shift is therefore mostly about which seeds were drawn and which operator and card each got, not judge noise.
- n = 30–40 per arm cannot certify "absurd ≤ 3%": the CI upper bound at 0/40 is 9%. The T1 §2.5 gate (200 items per arm plus a human audit) is still needed.

**(b) "Absurd" depends on the judge.**
- On the v3 items, Sonnet with rubric v3 flags 18% as absurd; Opus with v2 flags 0%.
- I spot-read the 7 v3 items that Sonnet flags and Opus does not:
  - About 4 are real coherence defects inherited from the seed:
    - a weight-loss goal "vào tháng Sáu" with the question about tháng Năm;
    - 160 books stored in a suitcase;
    - a "cuối chương trình" leftover in a bakery story;
    - "bờ ruộng" in a home garden.
  - About 3 are matters of taste (a 60.000 đ lawn fee, 500 đ savings).
- The 0% therefore reflects the Opus-v2 judge's strictness; the realism problem itself is not solved. A human audit must settle the threshold.

## 1. Task 1 — self-preference check

**Judge availability (measured; `g2_probe.py` and the 429 bodies in `logs/api/G2/gemini.jsonl`):**

| Model | Free-tier quota |
|---|---|
| gemini-3.1-pro-preview | limit 0 (no free tier) |
| gemini-3.8-flash | **20 requests/day/project**. gemini-3.7-flash calls count against the 3.8-flash quota (alias). |
| gemini-3.6-flash | 20 RPD; gemini-3.5-flash counts against it. Already exhausted when I reached it. |
| gemini-3-flash-preview | 20 RPD |
| gemini-3.1-flash-lite | no daily cap hit (≈ 490 calls in G2 today, after G1's load) |

- With 6 projects this gives ≈ 100–120 strong-judge calls per model per day. Timeouts and in-flight cut-offs also consume quota: about 10–16 successful calls per key were observed on 3.8-flash.
- 3.8-flash latency is 30–440 s per call (high-demand 503s, two 600 s timeouts).
- Outcome: **gemini-3.8-flash judged 63 items** (44 native, 19 render) before its quota ran out. **gemini-3-flash-preview judged 118/119 native items.** It is the task-1 judge; 3.8-flash is a partial cross-check.
- Both used rubric `validity_judge.v2`, identical to Opus's.

**Per-generator rates on the same items (G1 Opus-subsample native items, train-side only):**

| Generator | Opus teacher | Gemini-3-flash teacher | Opus accept* | Gem accept* | Opus absurd | Gem absurd | Opus near-dup (M2–M6) | Gem near-dup (M2–M6) | yield (judge-valid ∧ Opus-solve agrees ∧ no leak) Opus / Gem |
|---|---|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 63% (19/30) | 97% (29/30) | 63% | 97% | 23% | 3% | 25% | 0% | 67% / 93% |
| Claude Sonnet 5.5 | 48% (14/29) | 79% (23/29) | 48% | 79% | 31% | 17% | 39% | 0% | 41% / 62% |
| gemini-3.5-flash-lite | 57% (17/30) | 73% (22/30) | 53% | 73% | 20% | 10% | 58% | 25% | 37% / 50% |
| gemini-3.1-flash-lite | 59% (17/29) | 83% (24/29) | 55% | 83% | 10% | 10% | 43% | 9% | 41% / 55% |

\*accept = valid ∧ teacher ∧ no leak ∧ no tail (the judge-side acceptance used throughout G2).

**Rank agreement.**
- Haiku is first under both judges on valid, teacher, accept and yield.
- Kendall τ (4 generators): valid 0.67, teacher 0.67, accept 0.67, yield 0.83. Disagreement is only in the order of the three trailing generators.
- On 3.8-flash's partial set (n = 44, unbalanced: Sonnet n = 8), τ(yield) = 0.67.

**Per-field κ, Opus vs non-Claude judge:**

| field | κ vs gemini-3-flash-preview (n=118) | κ vs gemini-3.8-flash (n=44, partial) | κ Opus vs Opus re-test (n=30 Haiku items) |
|---|---|---|---|
| valid | 0.42 | 0.46 | 0.92 |
| teacher | 0.35 | 0.46 | 0.93 |
| new | 0.20 | 0.69 | 0.86 |
| absurd | 0.59 | 0.69 | 0.71 |
| near-dup | 0.18 | 0.57 | 0.86 |
| leak | 0.32 | 0.79 | — |
| natural ≥ 4 | 0.12 | 0.40 | — |
| accept | 0.33 | 0.46 | 0.93 |

**Self-preference (difference-in-differences; seed-cluster bootstrap, 2,000 reps):**
- **accept:** Opus − Gem gap is −32 pp on Claude items and −24 pp on Gemini items, so DiD = −8 pp [−31, +13].
- **teacher:** −12 pp [−35, +9].
- **absurd:** +12 pp [+2, +23].
- **Discordant pairs on accept:**
  - Claude items: Opus-only 0, Gemini-only 19.
  - Gemini items: Opus-only 2, Gemini-only 16.
- 3.8-flash partial: DiD accept −9 pp [−42, +21].

**Conclusion.**
- The Claude judge does **not** inflate Claude items. Every point estimate goes the other way: Opus is relatively harsher on Claude items, especially on absurdity.
- G1's "Haiku is the best native generator" survives a non-Claude judge.
- **Limits:**
  - (i) Gemini-3-flash is a lenient, weak judge (κ 0.2–0.4 with Opus vs 0.93 Opus–Opus). The stronger 3.8-flash agrees with Opus much better (κ 0.46–0.79) but covers only 44 items.
  - (ii) The DiD mixes Claude-judges-Claude and Gemini-judges-Gemini preference.
  - (iii) No third family is available (NIM banned).

## 2. Task 2 — judge cascade calibration

**Calibration set.**
- 189 G1 items Opus judged (119 native + 70 rendered texts).
- Hold-out: all items on the 4 seeds that supply the v3 few-shot anchors (G1 idx 6204, 1569, 2120, 5630) are dropped, leaving **n = 163 (103 native)**.
- Out-of-sample check: the 160 task-3 items, which Opus also judged and blind-solved.

**κ against Opus (hold-out, all 163 items unless stated):**

| Judge (rubric, effort) | valid | teacher | new | absurd | near-dup | leak | unansw | nat≥4 | **accept** | accept precision vs Opus | recall | $/item (direct) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Haiku 5.5, v2, low | 0.56 | 0.52 | 0.60 | 0.41 | 0.52 | 0.28 | 0.71 | 0.54 | **0.57** | 69% (59/85); native 81% | 87% | 0.00043 |
| Haiku 5.5, **v3**, low | 0.64 | 0.53 | 0.36 | 0.55 | 0.39 | 0.66 | 0.82 | 0.56 | **0.52** | 69% (52/75); native 82% | 76% | 0.00051 |
| Haiku 5.5, v3, medium | 0.57 | 0.56 | 0.36 | 0.55 | 0.35 | 0.66 | 0.66 | 0.54 | 0.53 | 67%; native 79% | 85% | 0.00066 |
| **Sonnet 5.5, v3, low** | **0.74** | **0.65** | **0.62** | **0.60** | **0.63** | 0.59 | 0.71 | **0.74** | **0.63** | **88% (45/51)**; native 91% | 67% | 0.0061 |
| gemini-3.1-flash-lite, v2 | 0.06 | 0.07 | 0.15 | 0.04 | 0.23 | 0.28 | 0.14 | 0.08 | 0.07 | 44% (accepts 94%) | 99% | free |
| gemini-3.1-flash-lite, **v3** | 0.17 | 0.20 | 0.50 | 0.31 | 0.52 | 0.00 | 0.16 | 0.05 | 0.19 | 48% (accepts 86%) | 99% | free |
| gemini-3.8-flash, v2 (n=54) | 0.53 | 0.50 | 0.68 | 0.73 | 0.48 | 0.79 | 0.68 | 0.49 | 0.50 | 70% | 73% | free, 20 RPD/project |
| gemini-3-flash-preview, v2 (native n=102) | 0.37 | 0.35 | 0.17 | 0.57 | 0.16 | 0.32 | 0.48 | 0.09 | 0.33 | 66% | 98% | free, 20 RPD/project |
| *Opus re-test (ceiling, n=26)* | 0.91 | 0.91 | 0.92 | 0.65 | 0.92 | 1.00 | 1.00 | 0.88 | 0.91 | 94% | 100% | 0.0154 |

**What the improved rubric did.**
- **For flash-lite**, v3 helps where decomposition gives it something concrete to check: new 0.15 → 0.50, absurd 0.04 → 0.31, near-dup 0.23 → 0.52. Accept κ is still only 0.19, because it marks almost everything natural 5 and teacher-acceptable.
- **For Haiku**, v3 improves valid, absurd, leak and unanswerable, but *hurts* new (0.60 → 0.36). Haiku over-applies `q_same_structure`. Net accept κ is flat (0.57 vs 0.52, within noise).
- **Effort medium buys nothing** for Haiku as a judge.
- v3 also costs more input (4.9k-token cached system prompt). With prompt caching, Haiku pays $0.0005 per item.

**Answer screen (cheap replacement for the Opus blind-solve).** The screen passes when Haiku 5.5 low solves 2/2 and both agree with the generator's answer. Against "Opus blind-solve agrees" on 103 native hold-out items:
- κ = **0.89**
- precision 69/70
- recall 69/73
- $0.00047 per item

**Same-family risk (generator = Haiku).**

| Judge | Leniency vs Opus on Haiku items | κ(accept), Haiku items | Leniency on Gemini items | κ(accept), Gemini items |
|---|---|---|---|---|
| Haiku-v3 | +10 pp (57% vs 48%) | 0.43 | +1 pp (40% vs 39%) | 0.50 |
| Sonnet-v3 | −5 pp (stricter) | 0.70 | — | — |

- Haiku-v3 is **more lenient on Haiku-generated items** (n = 42). This is a measurable same-family effect, though small-n.
- Sonnet-v3 shows no lenience on Haiku items.
- Opus is also Claude, but task 1 shows no Claude-favouring bias in Opus.
- Remaining risk: a Haiku-generated, Haiku-judged pipeline can share blind spots that no Claude judge catches. Examples are seed-inherited incoherence and taste in "realistic" values; see caution (b) in §0.
- That is why the cascade keeps a cross-family element (flash-lite) and a non-Claude audit (below).

### 2.1 Cascades (final accept = judge accept ∧ answer screen ∧ regex leak check)

- **Truth label:** Opus accept ∧ Opus blind-solve agrees ∧ no regex leak.
- **Cost model:**
  - generation $0.00115 per raw item (v3)
  - Haiku-v3 $0.00051, Sonnet-v3 $0.0061
  - answer screen $0.00047
  - Opus judge + solve $0.0216 (measured in task 3: $0.0146 + $0.0070)
  - plus a 5% random Opus audit
  - Batch halves every Claude term

| Policy | Calib native (n=102): precision / recall | Task-3 all arms (n=160): precision / recall | **Task-3 v3 prompt (n=40)**: precision / recall | Opus share (task 3) | $ per 10k accepted, direct / **Batch** (task-3 v3) |
|---|---|---|---|---|---|
| Opus on everything | 100 / 100 | 100 / 100 | 100 / 100 | 100% | 246 / **123** |
| Haiku-v3 + screen | 82 [68–90] / 78 | 85 [78–90] / 90 | 94 [81–98] / 86 | 5% | 38 / **19** |
| Sonnet-v3 + screen | 88 [73–95] / 65 | 90 [82–94] / 77 | 96 [82–99] / 70 | 5% | 130 / **65** |
| **Haiku-v3 ∧ 3.1-lite-v3, disagreements → Opus** | 85 [73–92] / 98 | **87 [80–92] / 98** | **95 [83–99] / 97** | 19–22% | **62 / 31** |
| Haiku-v3 ∧ Sonnet-v3, disagreements → Opus | 93 [81–98] / 85 | 92 [86–96] / 96 | 97 [85–99] / 92 | 30–33% | 171 / **85** |
| Haiku-v3 → Sonnet-v3 on Haiku-accepts, disagreements → Opus | 92 [80–97] / 78 | 92 [85–95] / 90 | 97 [85–99] / 86 | 25–28% | 161 / **80** |
| Haiku-v3 → Opus confirms every accept | 100 / 78 | 100 / 90 | 100 / 86 | 86–90% | 270 / **135** |

**Recommendation.**
- **Default (cost-optimal):** Haiku-v3 + flash-lite-v3 + Haiku answer screen. Opus on disagreements (≈ 20% of raw items, as Batch with the cached rubric) plus a 5% random audit.
  - Estimated precision ≈ 87% on mixed prompts and ≈ 95% on the v3 prompt.
  - ≈ **$31 per 10k accepted on Batch**, 4× cheaper than Opus-on-all.
  - flash-lite's free quota held 31 calls/min sustained in G1, so 10k raw items take ≈ 5.5 h on 6 keys.
  - In this cascade flash-lite mainly *rescues* Haiku false rejects (recall 98% vs 86–90%) and seldom vetoes. The cross-family protection therefore comes from the audit, not the screen.
- **If ≥ 95% precision must hold on mixed or harder prompts:** use the Haiku → Sonnet → Opus chain (precision 92–97%, ≈ $80–85 Batch).
- **Audit:** 5% Opus plus **a fixed 120 items/day from a non-Claude strong judge** (gemini-3.8-flash, 20 RPD × 6 projects). This tracks same-family drift: if the Haiku-vs-Gemini-3.8 accept gap on accepted items grows by > 10 pp, re-calibrate.
- **Gemini role:** Gemini output is used here as a screen only, not as training text.

**O1 practices (coordinator).**
- *Prompt caching* is used: the rubric sits in a cached system block for every Claude judge; v3 is a 4.9k-token cached system prompt with per-item user text.
- *One item per call*: there is no multi-item packing anywhere in G2.
- *Batch* was not used in this interactive pilot. The cascade prices above give both columns, and Batch is recommended for every scale run (Opus disagreement and audit calls are latency-insensitive).
- *Structured outputs* are recommended at scale. The v2/v3 verdicts are flat boolean/int JSON, a natural fit for a JSON schema. In G2, free-form JSON parsing lost only 1 of 1,700+ judge replies (one Sonnet), and generation lost 8/160 to thinking cut-offs at max_tokens = 4000. A retry at 2× max_tokens recovered all 8 (`retried` flag in `n_gen.jsonl`). Structured output plus a higher max_tokens removes both failure modes.

## 3. Task 3 — native-generation prompt iteration (Haiku 5.5, effort low)

**Seeds.**
- 40 new dev seeds: G1's dev pool (Vi-GSM8K train, hash ≡ 0 mod 20, R1 corrupt and half-English rows excluded) minus G1's 100 seeds, `random.Random(20261108)`; overlap with G1 = 0.
- Operators stratified to π: M1 8, M2 8, M3 10, M4 6, M5 4, M6 4. One card and an M6 partner (also outside G1) per seed.
- Every arm gets identical (seed, operator, card) triples (`data/pilot/g2/seeds40.jsonl`).

**Arms.**
- **v1:** G1 `native_v1` (control).
- **v2 "rules":** v1 plus:
  - realistic-value constraints (integer counts at every step, đồng in multiples of 500 inside listed real price ranges, plausible sizes, speeds and ages, coherent events);
  - a per-operator *structural-change* recipe (M2 invert, M3 mid-story event that changes an intermediate quantity and never a tail, M4 change the *type* of relation, M5 a different quantity, M6 a causal join);
  - a soft localization card: names and đồng always, the card's setting only if the numbers stay plausible.
- **v3 "self-check":** v2 plus `solution` (every intermediate value) and a 7-item `self_check` block, with an instruction to rewrite before answering if any check fails.
- **v4 "contrastive":** v3 plus 2 good and 2 bad G1 examples (G1 seeds, disjoint from these).

**Judging.**
- Opus 5.5 with `validity_judge.v2` (= G1 judge, effort medium, cached), plus an Opus blind-solve.
- **Non-Claude judge: only gemini-3.1-flash-lite-v3** could be run on all 160 items. The quotas of gemini-3.8/3.7/3.6/3.5/3-flash were exhausted by task 1, and only 6 gemini-3.6 verdicts landed (not used).
- Sonnet-v3 and Haiku-v3 verdicts are also reported.

### 3.1 Opus 5.5 judge (G1 Table B rubric). Gate: teacher ≥ 70%, new ≥ 60%, leak 0%, absurd ≤ 3%

| Arm | teacher | new (M2–M6) | absurd | leak | near-dup M2–M6 (all) | tail | answer-correct (Opus) | yield | strict yield | gate | gen $/item (out tok) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (G1 control) | 78% [62–88] | 59% [42–74] ✗ | 2% [0–13] | 0% | 34% (40%) | 12% | 100% | 90% | 60% | 3/4 | 0.00096 (1,745) |
| v2 rules | 70% [55–82] | 78% [61–89] | 8% [3–20] ✗ | 0% | 22% (35%) | 8% | 95% | 78% | 60% | 3/4 | 0.00105 (1,945) |
| **v3 self-check** | **92% [80–97]** | **78% [61–89]** | **0% [0–9]** | **0% [0–9]** | **19% (32%)** | **0%** | 98% | **92%** | **75%** | **4/4** | 0.00115 (2,067) |
| v4 contrastive | 75% [60–86] | 59% [42–74] ✗ | 8% [3–20] ✗ | 2% ✗ | 38% (50%) | 5% | 92% | 78% | 50% | 1/4 | 0.00125 (2,232) |

Paired exact McNemar vs v1, same seeds, Opus labels:

| Arm | teacher (arm-only / v1-only, p) | accept (arm-only / v1-only, p) |
|---|---|---|
| v3 | 8 / 2, p = .11 | **9 / 2, p = .065** |
| v2 | 3 / 6, p = .51 | 3 / 5, p = .73 |
| v4 | 7 / 8, p = 1.0 | 7 / 9, p = .80 |

### 3.2 Other judges on the same items (teacher / new M2–M6 / absurd / near-dup M2–M6)

| Arm | Sonnet-v3 | Haiku-v3 | gemini-3.1-lite-v3 (non-Claude) |
|---|---|---|---|
| v1 | 62% / 69% / 12% / 31% | 88% / 56% / 2% / 41% | 92% / 66% / 0% / 25% |
| v2 | 68% / 97% / 20% / 6% | 78% / 81% / 5% / 22% | 95% / 84% / 5% / 19% |
| **v3** | 68% / 94% / 18% / 6% | 88% / 81% / 2% / 19% | 98% / 78% / 2% / 22% |
| v4 | 70% / 91% / 18% / 9% | 80% / 69% / 2% / 31% | 98% / 91% / 2% / 9% |

### 3.3 Reading

- **v3 wins on the judge of record.**
  - Under every judge, v2 and v3 clearly improve novelty: near-dup M2–M6 falls from 25–41% (v1) to 6–22%, and new M2–M6 rises.
  - Under Opus and Haiku, only v3 does it without raising absurdity. Under Sonnet-v3, v3's absurd rate is higher than v1's (18% vs 12%; see the next bullets).
- Its self-check fields are always reported all-true (40/40). They are not informative as a filter; the gain comes from making the model solve and re-check before committing.
- **v2's realism rules alone raise absurd** (Opus 8%). Asking for structural change without a solve-and-check step produces more strained scenarios.
- **v4's examples hurt.** It regresses toward reskins (near-dup 50% overall) and copies example structures. Few-shot exemplars are not recommended for the generator.
- **The non-Claude judge ranks the arms the same way on novelty** but cannot separate them on teacher (92–98%, ceiling). Its κ with Opus is low (§2), so it is not evidence for the gate.
- Sonnet-v3 says v3's absurd rate is 18%, not 0% (caution (b) in §0). The decision to scale v3 should therefore come with the 200-item human/Opus gate audit specified in T1 §2.5.

### 3.4 Effort

- Medium effort was not run. v3 at low effort already meets every gate threshold under Opus, and the remaining Claude budget ($1.40) is better kept for the gate audit.
- Haiku at low effort already spends ~2.1k output tokens per v3 item (thinking included).
- Medium-effort Haiku as a *judge* gave no gain (§2), which is weak evidence that medium would not pay for generation either.
- Recommendation: keep low effort, and retry at 2× max_tokens on a thinking cut-off (8/160 needed it).

## 4. Recommendations for scale-up (arm N)

1. **Generator:** Haiku 5.5, effort low, `prompts/native_generator.v3.md`.
   - max_tokens 8000, or 4000 with one 2× retry on a cut-off.
   - Batch API with the system prompt cached.
   - Generation ≈ $11.5 direct / $5.8 Batch per 10k raw items.
2. **Gate before scale:** 200 v3 items, judged by Opus-v2 and Sonnet-v3, plus a human sample of ≥ 50.
   - The human sample settles the realism disagreement in §0(b).
   - ≈ $4.5 Opus direct / $2.3 Batch.
3. **Verification at scale:** the §2.1 default cascade, ≈ $31 per 10k accepted on Batch. Add a 120/day gemini-3.8-flash audit for same-family drift.
4. **Arm-Y editor:** untouched. It waits for X4's symbolic library and real (typed) edit lists.

## 5. Caveats

- Task 3 has n = 40 per arm, so CI half-widths are ±10–15 pp. Only v3 vs v1 on accept approaches significance (p = .065).
- Calibration items are G1 items whose generators and seeds differ from the task-3 distribution. That is why the cascade is also evaluated on the task-3 items (out-of-sample: Opus labels never informed the cheap judges there).
- The v3 anchors are hand-rewritten from Opus verdicts on 4 G1 seeds. Every item on those seeds is excluded from calibration.
- All judges are LLMs; there are no human labels (user decision). "Precision" means agreement with Opus, not truth.
- Gemini quotas: strong-judge coverage in task 1 is gemini-3-flash-preview (weak, lenient), not 3.8-flash. Re-running task 1 with 3.8-flash needs about 2 days of its 20-RPD/project quota.

## 6. Reproduce

```
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_gen.g2_probe
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_gen.g2_judges --holdout --judges g38_v2@all,g31l_v2,g31l_v3,haiku_v2,haiku_v3,haikum_v3,sonnet_v3 --solve haiku --samples 2
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_gen.g2_native seeds
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_gen.g2_native run --arms v1,v2,v3,v4 --judges opus_v2,haiku_v3,g31l_v3,sonnet_v3 --haiku-solve 2
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_gen.g2_native run --arms "" --judges "" --no-opus-solve --haiku-solve 0 --calib "g3fp_v2@all,opusR_v2@all:haiku"
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_gen.g2_report      # -> data/pilot/g2/g2_metrics.json
```

All calls are cached (`.cache/vimath/llm_cache.sqlite`), so re-runs are free unless prompts or params change. Nothing has been committed.
