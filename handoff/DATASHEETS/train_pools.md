# Datasheet — training data for arms Y± and N± × {VI, EN}: pools and the final matched sets

Builders and reports: **arm Y** — G4 → G5 (`vi_math_verified_aug/arm_y/`;
[G4](../../docs/research_2026-11/exec/G4_symbolic_v2.md), [G5](../../docs/research_2026-11/exec/G5_Y_scale.md));
**arm N** — D1 (`vi_math_verified_aug/scale_gen/`; [D1](../../docs/research_2026-11/exec/D1_N_scale.md));
**final selection** — D5 (`analysis/select_v1.py`; [D5_selection](../../docs/research_2026-11/exec/D5_selection.md));
**quality audit of the final sets** — D5 ([D5_gate_audit](../../docs/research_2026-11/exec/D5_gate_audit.md));
**attribution of the Y−N quality gap** — [C3](../../docs/research_2026-11/exec/C3_attribution.md).
Status: **FROZEN** (pools 05:53–07:04, final sets published 07:03, 2026-11-08). The S0 anchor is in `seed_pool_v1.md` §4b.

## 0. Files and hashes (all `SHA256SUMS` re-verified OK on 2026-11-08 08:40)

**Final matched sets — what the 11-08 tranche trained on** (`data/train/final/`, n = **1,808** per file;
`manifest_{vi,en}.json` → `n_effective = 1808`, version `select_v1`, created 07:02:45):

| Cell | `train_plus.jsonl` (verified, "+") | `train_minus.jsonl` (unverified, "−") |
|---|---|---|
| Y VI | `4bb481930ebd788cfec592b3f2291c98dda28e14968b57d61cf3592626acfdb9` | `0b51bebf01343984abc6717675f6ef6d3cab135aaedcf567a111a7ab4fb2f749` |
| N VI | `9c3e61f691b02a805fa68b2f51a72059f651d6bb3cf1c5bed9f134eb5235a381` | `fc35ac449a115bde9d4c7afcdd6f53d38b25f51eb55ebf9ee56634489232c3c2` |
| Y EN | `c0def46f4c1ed67cde47f6f6f0711a9a11fcd4f555747b86aa94efb9f2cb833d` | `fd5481ba9a68379e79ec4245c298b1f5372db853a8250f5347851ef8809646e2` |
| N EN | `242419d25c2dda5d1920ed73b9e90241c49a69c1f4ddc5a83966c7d2a0e58edb` | `f3e5d8325874c8266be77995d9a6fb7fb64dae81b53423bcafc712dfca8796d8` |

**Secondary sets (sensitivity / successor; same layout, own `SHA256SUMS`):**

| Set | n (VI / EN) | Rule | plus / minus sha256 (first 16), Y-VI · N-VI · Y-EN · N-EN |
|---|---|---|---|
| `final_shared/` | 1,679 / 1,679 | one seed set S = S+ for all four arms; minus = first items of the S+ seeds. Its minus items are 92–100 % verified, so it **does not identify H3** | plus `c1dfeda0…` · `16c6d13b…` · `d5ebdca0…` · `3a9b344b…`; minus `f420c865…` · `da358b1d…` · `e05c8d69…` · `97ae7e40…` |
| `final_natural/` | 2,734 / 2,325 | each arm independently from all its accepted items, coverage-maximising; N keeps MATH seeds and M6 | plus `ed7a8455…` · `ba3894d6…` · `24f527f6…` · `706416c2…`; minus `fa2bf7ed…` · `9f472207…` · `9b088ba1…` · `1f0f043f…` |
| `final_opmatched/` | 1,387 / 1,404 | largest seed subset whose Y and N edit-class histograms are equal within each sign; selected per language (VI/EN differ by 17 seeds — intersect before use) | plus `3901b54b…` · `287eaf18…` · `08464798…` · `7bb26d7c…`; minus `be9a8d40…` · `a014a715…` · `e7bddb61…` · `a7c313f8…` |

**Generated pools** (`data/train/{Y,N}/{vi,en}/`: `pool.jsonl` = every generated item with its labels;
`train_10k_{plus,minus}.jsonl` = provisional rank-order prefixes, **superseded by `final/`**; `manifest.json`, `RELEASE`/`SHA256SUMS`):

| Pool | Rows | Verified ("+") | Minus-eligible | pool.jsonl sha256 | Release |
|---|---|---|---|---|---|
| Y VI | 5,106 (wave 1, 1 per seed) | **2,782** | 5,106 | `5be027b69cf9eacdb81489b96cf085033ffd551c52acaf71278c994f8c6560aa` | r4 final 05:53 |
| Y EN | 4,352 (3,921 attempt 1 + 431 EN2 retries) | **2,649** (74 EN2 rows unchecked at the Modal cap, `plus=None`) | 3,921 | `72c344b8188631b4d7a45e654f2f83fa8eba04a0771c18f46cb31fd5fc601676` | r2 final 06:00 |
| N VI | 13,500 slots (ranks 0–13,499) | **7,195** (3,633 agreeing items never screened) | 12,968 | `73c466cd7ea81b02ddb116c400fbc43f18cb202690d03db80b70fd7a9a8b0fff` | r5 final |
| N EN | 4,395 slots (incl. 244 retries) | **2,325** | 4,254 | `087372f988312d948e67b3b69c24359120f8a19e8c7a55674d9a51524fed8120` | r4 final |

Provenance note: `final/manifest_vi.json` records the N-VI pool input as `cde1e8a3…` (read at 07:02); D1 rewrote the
file at 07:04 (now `73c466cd…`). Checked 2026-11-08: all 3,616 selected N-VI rows are present in the current pool with
identical question, solution, answer and `verified` (KNOWN_ISSUES P-1).

## 1. Motivation
Matched training sets for the controlled comparison C2 and H1–H4: **same seeds** (S+ for both "+" arms, S− for both
"−" arms, the same seeds in VI and EN), **same n** (1,808), **identical strata** (seed step-count terciles t0 ≤ 3 steps
1,061 · t1 = 4 steps 356 · t2 ≥ 5 steps 391), **one shared solution writer and one shared "+" screen for both arms**,
so arms differ in how the *problem* was made (symbolic mutation + render vs native LLM generation) and in whether
items were verified.

## 2. Composition (final sets)
- **Row** (both arms, N layout): `id`, `seed_id`, `slot_id`, `op` (M1–M5), `arm`, `lang`, `question`, `solution`,
  `answer` (= the trace's own last `\boxed{}`), `generator_answer`, `verified`, `minus_eligible`, `attempt_no`,
  `plus_only`, `flags` (per gate), `seed_screens`, `card`, `partner`, `rank`, `source`, `provenance`, `selection`.
- **All seeds are GSM8K** (G5 ran the Y formalizer on GSM8K seeds only; MATH unreachable by Y → P3 coverage result).
- **Operator mix (VI):** Y+ M1 643 / M2 916 / M3 215 / M4 2 / M5 32; Y− 621 / 903 / 219 / 0 / 65; N+ 451 / 375 / 483 / 314 / 185;
  N− 416 / 401 / 502 / 303 / 186. Y is value/premise-heavy, N structure-heavy (reported, not matched; see `final_opmatched`).
- **R8 manipulation check (must be carried into the analysis):** first-attempt verification rate Y **54.5 %** VI /
  **63.4 %** EN vs N **80.1 %** VI / **87.1 %** EN; selected minus items that would pass verification: Y− 56.5 % / 61.1 %,
  N− 84.4 % / 87.6 %. Verification removes ~45 % of Y items but ~15 % of N items.
- **Length is not balanced:** question-word SMD Y vs N 0.46–0.65 (KS p < 1e-27); solution words SMD 0.05–0.22.
  Aliased with arm in this design (every run of an arm trains on the same 1,808 items) → a **length-matched retrain is
  a successor task** (KNOWN_ISSUES S-LEN).
- **Selection effect:** verification keeps easier seeds (verification rate t0 0.60, t1 0.52, t2 0.46; TV distance S+
  vs unconstrained S− 0.050); the matched quotas remove it, so H3 = "correctness at matched difficulty".
- **Diversity (bge-m3 Vendi, VI / EN):** Y+ 40.0 / 48.5, Y− 41.8 / 50.4, N+ 35.0 / 41.6, N− 36.3 / 42.5; cosine to own
  seed Y ≈ 0.76 / 0.83 vs N ≈ 0.66 / 0.68 (Y keeps the seed scenario).
- **Quality of the final sets (Opus 5.5 `validity_judge.v2`/`_en.v1`, VI n = 100 per cell, EN n = 50):**

  | Cell | teacher | absurd | leak | T1 §2.5 gate |
  |---|---|---|---|---|
  | Y+ vi / en | 66 % / 74 % | **12 % / 6 %** | 1 % / 0 % | FAIL / FAIL |
  | Y− vi / en | 50 % / 52 % | 21 % / 10 % | 1 % / 6 % | FAIL / FAIL |
  | N+ vi / en | **89 % / 96 %** | 5 % / 2 % | 1 % / 0 % | FAIL / **PASS** |
  | N− vi / en | 75 % / 82 % | 9 % / 2 % | 1 % / 4 % | FAIL / FAIL |

  The gate is **reported, not enforced** (training had started). Paired on the same seeds, Y is worse than N on
  validity (+20 pp), teacher-acceptability (+23 pp) and fluency (+21 pp) in the "+" sign. C3 attributes the remaining
  gap to operator value choices, near-duplicates and translationese inherited from the seed text.

## 3. Collection process (generation)
**Arm Y (symbolic).** Seed (pool_v1, GSM8K) → formalizer `claude-haiku-5-5` (`formalizer.v2`) → typed quantity-DAG IR
(accepted only if it executes to the seed gold with span-anchored givens) → v3 operators (M6 dropped; M3/M4
constrained at mutation time; π M1 .25 / M2 .40 / M3 .25 / M4 .05 / M5 .05, J = 3) → render: VI **R2b** =
`ir_render.v1` from the localized IR (1 USD = 10.000 đồng); EN = `ir_render_en.v1` from the same IR → tag, leak,
answer-outside-tags and chrF gates → shared solution writer → **Y+ only:** trace answer == program answer (X1
grader), common Haiku-v3 screen, **blind back-formalization by Qwen3-32B-AWQ** (self-hosted vLLM 0.33 on Modal; 94/100
agreement with the Gemini verdicts on the validation pilot; 93 % [91–95] agreement with glm-5.3-flash via NIM on
1,095 VI items), decontamination. **Y−:** wave-1 first-attempt items with tag + leak gates + a boxed trace +
decontamination, no verification. Funnel per VI slot: mutated 75 % → tag 72 % → trace = program 67 % → screen 41 % →
cycle 39 %. Pilot quality under the final screen: teacher 66 % [50–79], absurd 8 % [3–21]; Opus blind solve agrees
with the program answer on 64/64 survivors.

**Arm N (LLM-native).** Same seed + operator slot (`slots_v1`: 9,649 seeds → 28,947 slots, π M1 .20 / M2 .20 / M3 .25 /
M4 .15 / M5 .10 / M6 .10) → `claude-haiku-5-5`, `native_generator.v3` / `_en.v3` (effort low, adaptive thinking,
max_tokens 6,000, one retry at 12,000; Message Batches) → parse/format gate (= N−) → shared solution writer → **N+
only:** trace agrees with the generator answer + common Haiku-v3 screen, decontamination. Opus audit of 80 VI N+ pool
items: absurd 2.5 % [0.7–8.7], teacher 78.8 %.

**Shared components (lead rule 23:10):** training row = {question, solution = `clean_trace(trace)` from ONE writer
(`scale_gen.solver`, Haiku 5.5, `solution_writer.v2` / `_en.v2`, effort low), answer = the trace's own boxed answer};
common "+" screen = `validity_judge.v3` / `_en.v3` on Haiku 5.5 (pass = ¬absurd ∧ ¬unanswerable ∧ teacher_would_write).
Decontamination (X5 G4 rule) against all `data/eval/*` v1.1 (VI + EN) + Vi-ExamMath (13,655 targets): any shared
13-gram, templated 13-gram coverage ≥ 50 %, bge-m3 cos ≥ 0.85 same-language / ≥ 0.80 cross-language, MinHash J ≥ 0.8
within arm.

**Deviations from the plan (each lead-approved, recorded in the manifests):**
1. **Gemini quota → screen = Haiku-v3 only.** The free tier is 500 requests/day/project/**model**; both flash-lite
   models on all 6 keys were exhausted at 02:10 (≈13.6 h cooldown). The planned Haiku-v3 ∧ flash-lite screen became
   Haiku-v3 only for both arms (`screen_lite: "not applied"` per row). Calibrated absurd among accepts 8 % [5–12] vs
   6 % [3–11] with lite (n.s.).
2. **Y cycle-check model:** planned Gemini flash-lite (cross-family) → NIM candidates → **Qwen3-32B-AWQ on Modal** for
   every Y item, VI and EN (one model; a Haiku cycle is an ablation label only). Non-executing IRs (59 VI, 118 EN + 4
   EN2) count as cycle fail; 74 EN2 rows stayed unchecked at G5's $9.5 Modal cap.
3. **Size:** n_train 10k → 4k → 3k → **1,808** (Claude budget, then Modal budget; VI Y wave 2 stopped at 05:13 with
   2,400 slots queued, 1,831 mutated, 453 rendered).
4. **Selection:** seed-matched instead of slot-paired (Y and N use different slot schedules); two seed sets (S+, S−);
   "−" = the non-adaptive first item only; "+"-only retry rows admitted to S+ (EN: Y+ 158, N+ 177); 148 S− seeds refilled
   from the reserve within stratum.
5. **No Opus on cascade disagreements** (budget); Opus used for audits only.

**Data routing:** only training seeds and their generated items went to Gemini (dev/train guard); no test or
benchmark item went anywhere but Claude; NIM was used only for G5's cycle-model validation (glm-5.3-flash, gemma,
nemotron candidates) on generated training items.

**Spend:** Claude D1 $17.24, G4 $4.15 + G5 $18.46, D5 $4.40 (audit incl. C3 $0.58); Modal G5 $9.50, D1 $0.19.
Ledgers `logs/api/{D1,G4,G5,D5}/` (G5 also `nim.jsonl`); batch ledgers `data/train/N/_work/batches.jsonl`,
G5 pending files under `data/train/Y/_work/vi_scale/`.

## 4. Preprocessing
Items longer than the trainer's `max_length` (1,024 tokens) are dropped on the pool before the seeded subsample (none
dropped: 0/1,808 in every cell). The launcher validates each file (malformed > 1 % → refused) and reads `n_effective`
from `manifest_<lang>.json`. Selection used no outcome information from student models or eval sets.

## 5. Uses
SFT for the arm comparison (0.6B tranche of 11-08: VI seeds 0–3, EN seed 0; see `MODEL_CARDS/`). Not for evaluation.
Analysis must carry the R8 asymmetry and the length confound (R1, D5 §0).

## 6. Distribution, licenses and provenance of Claude outputs
- **Claude-generated text is the content of every Y and N row:** problems (N: Haiku 5.5 generator; Y: Haiku 5.5
  formalizer + renderer over the seed), solutions (Haiku 5.5 shared writer); screens by Haiku 5.5; audits by
  `claude-opus-5-5`. Every call is logged with model requested/returned, prompt id/version/sha256, full request and
  response, usage and cost in `logs/api/<WS>/claude.jsonl` (private bundle); prompt hashes in `handoff/PROMPTS.md` and in
  each pool `manifest.json`.
- Gemini outputs are judgements/solve-checks only (no training text). Qwen3-32B-AWQ (Apache-2.0, self-hosted) produced
  cycle-check verdicts only. NIM outputs are not training text.
- Upstream seeds: GSM8K (MIT). Proposed license for a release: CC BY 4.0 with an "AI-generated" disclosure naming the
  Claude models above (06 §4.5). Internal: bundle `data`.

## 7. Maintenance and costed growth (successor)
Pools and final sets are frozen with `SHA256SUMS`; a new selection gets a new version (`select_v2`) and its own
manifests; changing a file after a run used it invalidates that run. Cheapest paths to larger n (D5 §2, measured unit
costs): finish G5's 74 unchecked EN2 rows (+≈60, ≈$0.1 Modal); EN N 2nd retry (+≈50, ≈$0.07); VI N retry on Y+ seeds
(+≈150, ≈$0.3); **VI Y wave 2 + cycle** on 2,239 seeds (+≈735, ≈$1.6 Batch VI + Qwen cycle on Modal + ≈$2 Batch EN) →
n ≈ 2,800. 10k/20k targets: see `RUNBOOK.md` §Successor queue.
