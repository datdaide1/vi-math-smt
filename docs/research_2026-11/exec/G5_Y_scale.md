# G5: arm Y v3 and scale-up (VI, then EN mirror), 2026-11-07/08

**Owner:** G5 (follows `exec/G4_symbolic_v2.md`). The lead's decisions arrived by message; each one is quoted where it applies.

**Code:** `vi_math_verified_aug/arm_y/`
- `ops` holds the v3 operators.
- `pipeline` has every stage, each resumable.
- `common` holds the client, `call_many` (with Batch pending-file resume) and `call_many_chunks`.
- `screen` holds the calibrated rule, superseded by D1's frozen `scale_gen.screen`.
- `calib` and `evalrun` cover calibration and the Opus evaluation.
- The cycle check is in `nim_cycle`, `vllm_cycle`, `vllm_cycle_modal` and `vllm_cycle_modal2`.
- Decontamination is in `decontam_modal`; the pool export is in `export`.
- Run scripts: `run_scale.sh`, `run_rest.sh`, `watchdog.sh`.

**Prompts (new):**
- `ir_render_en.v1` (EN renderer from the IR)
- `back_formalizer_en.v1`
- `fluency.v2` (from G4)

Reused unchanged:
- `ir_render.v1`, `story_check.v1`, `formalizer.v2`, `back_formalizer.v1`
- D1's `solution_writer[_en].v2`, `validity_judge[_en].v3`, `validity_lite.v1`

**Tests:** `tests/test_arm_y_export.py` runs a Y row through `train.launcher.check_data` and `train.data.prepare_pool` with 0 malformed rows, in the N layout. Together with `tests/test_pilot_symbolic_v2.py`, all tests pass.

**Data:**
- Work directories: `data/train/Y/_work/vi_scale/` (all stores) and `data/pilot/symbolic_v3/` (the v3 iteration and calibration).
- Pools: `data/train/Y/{vi,en}/`, containing `pool.jsonl`, `train_10k_{plus,minus}.jsonl`, `SHA256SUMS`, `manifest.json` and `RELEASE`.

## 0. Answer

**Arm-Y pools are frozen** (`data/train/Y/{vi,en}/`, sha256 in each `SHA256SUMS`).

| pool | rows | **Y+ (verified)** | Y− (`minus_eligible`) | release |
|---|---|---|---|---|
| VI | 5,106 wave-1 rows, 1 per seed | **2,782** distinct seeds | 5,106 | r4, final |
| EN | 4,352: 3,921 attempt-1 rows + 431 EN2 retries | **2,649** distinct seeds (2,486 + 163 from retries) | 3,921 (attempt 1 only) | r2, final; 74 EN2 rows left unchecked (plus = None) at the Modal cap |

- **2,026 seeds have Y+ in both languages.** D5 selects S+ from these, intersected with N.
- The provisional `train_10k_{plus,minus}` files are superseded by D5's `data/train/final/`.

**Yields and definitions**
- **Y+ per VI slot = 0.39.** Funnel for 7,100 slots:
  - mutated: 75%
  - tag check: 72%
  - trace = program answer: 67%
  - common screen: 41%
  - cycle: 39%
- **p_EN(Y) = 66%** of EN pool rows.
- **Y+ definition:**
  - tag check;
  - leak, answer-outside-tags and chrF gates;
  - D1's shared solver trace, whose own boxed answer == program answer (the X1 grader);
  - common screen (Haiku-v3; the flash-lite leg was not applied, because the Gemini quota was gone);
  - **blind back-formalization by Qwen3-32B-AWQ executes to the program answer**;
  - decontamination.
- **Y− definition:** wave-1, first-attempt items with tag check + leak gates + a boxed trace + decontamination. No verification. This mirrors N−.

**Quality, by Opus 5.5 on the v3 pilot:**
- Under the final common screen, Y+ items are teacher-acceptable at 66% [50–79] and **absurd at 8% [3–21]** (n = 38).
- On G4's 358 items the screen alone was calibrated at absurd 8% [5–12].
- **The T1 gate (absurd ≤ 3%, teacher ≥ 70%) is NOT met by Y+.** Only the Sonnet leg (1% absurd) would meet it, and the lead excluded it so that both "+" arms share one screen. D5's joint audit measures the final rates.
- Answer faithfulness, by contrast, is near-perfect:
  - The Opus blind solve agrees with the program answer on 100% of v3 survivors (64/64).
  - The Qwen and GLM cycle verdicts agree on 93% [91–95] (n = 1,095).

**Spend**
- Claude G5: **$18.46**. With G4's $4.15 that is $22.6 of the $27 cap.
- Modal G5: **$9.50** of the $9.5 cap. All apps are stopped.
- **Lesson:** the single Qwen call that timed out cost **$2.89** (image build plus 62 min of L40S), and its results were lost. Long GPU jobs must be chunked, with results written per chunk (`vllm_cycle_modal2`).

**Not done (costed in §6)**
- VI wave 2: 2,400 slots queued, 1,831 mutated, 453 rendered. It was stopped at 05:13: no Modal budget was left for its cycle check.
- 10k / 20k scale-ups.

## 1. Arm-Y v3 iteration: 34 fresh dev seeds, 102 slots (`data/pilot/symbolic_v3/`)

**What changed relative to G4** (`arm_y/ops.py`):
- **M6 dropped.**
- **M3/M4 constrained at mutation time,** not only filtered afterwards:
  - entity role: count, person and measure steps act only on a *stated* stock, never on a computed aggregate, a per-item quantity or a difference;
  - price templates apply only to a price or bill;
  - "chia đều" and "tăng gấp" removed;
  - M4 is restricted to direction-preserving changes (factor k→k′, pct_inc↔pct_dec); add↔sub and gấp↔1/k swaps are banned.
- **Domain checks on every node.** A node that also exists in the seed must stay within ¼–4× of its seed value. Givens must be clean literals: no 400/7. Ages ≤ 100, counts per unit ≤ 200, hours per day ≤ 24.
- **Re-weighted π_v3** over the surviving operators: M1 .25, M2 .40, M3 .25, M4 .05, M5 .05. The basis is G4's per-operator accepted yield per slot (M2 is the only operator that is both new and certified at volume).
- **Localization = R1b** (1 USD = 10.000 đồng; no ounce/feet relabel). D5's C3 causal test supports this: absurd −12 pp [−21, −4].

**Renderers:** R1b (edit + fluency.v2) and R2b (`ir_render.v1` from the localized IR).

| (Opus 5.5 `validity_judge.v2` + Opus blind solve) | R1b | R2b |
|---|---|---|
| tag pass / cycle pass (Haiku back-formalizer) | 68/68, 60 (88%) | 68/68, **66 (97%)** |
| survivors (cycle + gates), teacher / absurd | 57: 49% / 23% | 64: 45% / 22% |
| after Haiku→Sonnet cascade (rejected later as arm-specific, see §2) | 15: teacher 93% [70–99], absurd 0% [0–20] | 22: teacher 91% [72–97], absurd 0% [0–15], new 89% |
| **after Haiku-v3 only = the final common screen** | 37: teacher 70% [54–83], absurd 11% [4–25] | 38: teacher 66% [50–79], **absurd 8% [3–21]**, accept 63% |
| Opus blind solve = program answer (survivors) | 57/57 | 64/64 |

**Decision: R2b** (lead, 23:00).
- It has the higher yield (Y+ per slot).
- Its cycle pass is 97%.
- It is a single call.
- English renders from the same IR, which E1 R3 requires.

v3 gate verdict, stated plainly: absurd is at or below 3% only behind the Sonnet leg. The common screen (§2) leaves about 8% [3–21] absurd on Opus.

## 2. The common screen of the "+" arms (calibration on G4's 358 Opus-labelled items)

Files: `data/pilot/symbolic_v3/calib.json` and `calib_common_screen.json`.

| screen | pass | precision vs Opus-accept | recall | absurd among accepts |
|---|---|---|---|---|
| D1 flash-lite `lite_ok` alone | 232 | 60% [54–66] | 77% | 10% [7–15] |
| Haiku-v3 (¬absurd ∧ ¬unansw ∧ teacher) | 239 | 67% [61–73] | 88% | 8% [5–12] |
| Haiku-v3 ∧ flash-lite | 180 | 71% [64–77] | 71% | 6% [3–11] |
| Sonnet-v3 (same rule) | 182 | 82% [76–87] | 83% | 1% [0–4] |
| Haiku-v3 ∧ Sonnet-v3 | 164 | 84% [77–88] | 76% | 1% [0–4] |

κ against Opus:
- flash-lite: absurd 0.19, unansw 0.04
- Haiku: absurd 0.45
- Sonnet: absurd 0.53

Lead decisions, in order:
1. **No Sonnet for Y.** Y+ and N+ must share one screen, so that Y+ vs N+ isolates the arm's verification (program certificate vs solve agreement).
2. **Common screen = Haiku-v3 ∧ lite** (D1 added it to N+; code `scale_gen.screen`).
3. At 02:10 the Gemini free tier was exhausted on both flash-lite models: `GenerateRequestsPerDayPerProjectPerModel`, about 13.6 h cooldown, all 6 keys. **Deviation: common screen = Haiku-v3 only**, recorded per row as `screen_lite: "not applied"`, for both arms. The calibrated cost of that deviation is absurd among accepts 8% [5–12] instead of 6% [3–11] (not significant).

## 3. The cycle check: one cross-family model for the whole arm

The design requires a blind back-formalizer from a different family than the Haiku renderer. Gemini was gone. A same-family Haiku cycle was started and then cancelled (lead: weaker, correlated certificate); none of its verdicts are used.

Candidates were validated on 100 G4 items with Gemini verdicts (`data/pilot/symbolic_v3/nim_validation.jsonl`):

| model (host) | agreement with Gemini | items/min | verdict |
|---|---|---|---|
| nemotron-3.5-bolt-e4b (NIM) | 79% | 27.5 | fails agreement |
| gemma-4-31b-it (NIM) | 28/28 | ≈ 6 | fails speed |
| glm-5.3-flash (NIM) | 91/92 = 99% | ≈ 13 (p50 latency 57 s) | fails speed; kept as the **secondary label** |
| **Qwen3-32B-AWQ** (self-hosted vLLM 0.33 on Modal) | **94/100** | 62 (A100-80GB), 48.7 (L40S) | **THE cycle model** for every Y item, VI and EN |

- Items per dollar are about equal on the two GPUs: about 1,490 per $ on both.
- Qwen3 is the student's model family. It is cross-family with respect to the renderer (Claude Haiku), which is what the certificate needs: the back-formalizer must not share the renderer's errors.
- The student never sees the cycle check's outputs. Only a pass/fail bit selects items, so there is no train/eval leakage. The family overlap is recorded as a caveat for P1.
- Robustness: on wave-1 items judged by both models, Qwen and GLM agree on 251/279 = 90%.
- Order (lead 03:40): render → free gates → solver → trace-match → screen → **cycle last**, on screen passes only. This halves the paid cycle volume.

## 4. Scale run (`data/train/Y/_work/vi_scale/`, tables in `scale_tables.md` and `scale_metrics.json`)

**Seeds and slots**
- Seeds: S3 `pool_v1` (frozen 02:05). All 7,100 GSM8K seeds; no dev or pilot seeds.
- 164 seeds whose `pool_v1` text differs from the text formalized at 23:10 are flagged per row (`seed_text_changed_in_pool_v1`) and excluded from wave 2.
- Slots follow D1's `slots_v1` rank order (all j = 1 first; sha1 order within j), with Y's own per-slot π_v3 operator draw.
- **Wave 1 = J = 1 on every seed.** Y− uses wave-1 items only (lead): they are non-adaptive, one per seed, and generated before any verification.

**Pipeline per item (cheapest first)**
1. R2b render (Haiku)
2. Tag check
3. Leak, answer-outside-tags and chrF gates
4. D1's shared solver; the training answer is the trace's own boxed answer
5. Trace answer == program answer
6. Common screen (Haiku-v3)
7. **Qwen3-32B blind cycle check, last**

| VI wave 1 | slots | mutated | tag ok | gates | trace = program | screen pass | **Y+** |
|---|---|---|---|---|---|---|---|
| M1 | 1,833 | 96% | 95% | 95% | 88% | 56% | 53% |
| M2 | 2,883 | 94% | 89% | 89% | 84% | 51% | 46% |
| M3 | 1,684 | 37% | 37% | 37% | 33% | 22% | 20% |
| M4 | 349 | 2% | 2% | 2% | 2% | 1% | 1% |
| M5 | 351 | 54% | 54% | 54% | 51% | 19% | 18% |
| **all** | **7,100** | 5,300 (75%) | 5,126 (72%) | 5,114 | 4,787 (67%) | 2,945 (41%) | **2,783 (39%)** after the feedback retry (71 of 133 recovered; 59 count as cycle fails) |

- **2,783 distinct VI seeds have a Y+ item**; 2,782 remain after decontamination.
- The cycle check passes 94.5% of screen passes, so the screen is now the binding filter, not the certificate.
- Operator mix of Y+: M2 1,328, M1 979, M3 341, M5 62, M4 2.
- **Robustness of the certificate:** Qwen3-32B and glm-5.3-flash agree on 93% [91–95] (1,020/1,095) of the VI items that both judged.

**EN mirror** (E1 R1–R3: same seeds, same slot IDs, same mutated IR; only the prompt language changes: `ir_render_en.v1`, `solution_writer_en.v2`, `validity_judge_en.v3`, `back_formalizer_en.v1`):

| EN slots (D5 part 1 = 3,300 S−/reserve seeds; part 2 = 702 needs_EN.Y; wave 1) | mutated | tag ok | gates | trace = program | screen pass | **cycle pass (Y+)** |
|---|---|---|---|---|---|---|
| 3,983 | 3,983 | 3,937 | 3,923 | 3,771 | 2,663 (70.6% of eligible) | **2,487** |

- **EN2 retry** (lead 05:35): one fresh EN render, `attempt_no` 2 (same IR and slot, `sample_idx` = 1), for 443 S+ candidate seeds.
  - Seed list in `data/train/Y/_work/vi_scale/en_retry_seeds.json`. Base = VI Y+ ∩ N-vi verified True. Groups:
    - A: Y-en False ∧ N-en True (350)
    - B: no Y-en item ∧ N-en True (21)
    - C: Y-en False ∧ N-en False (72; D5 counted 56, and the lead kept 72)
  - Y-en False means plus False in the EN pool at 05:36 (screen, trace or gate fail; no cycle verdicts existed yet).
  - Results:
    - render: 435 tag ok
    - screen: 244 pass
    - Qwen cycle: run on the first 170 in slots_v1 rank order (the Modal cap), 163 Y+
    - the remaining 74 are unchecked (plus = None)
  - EN2 rows are "+"-only (`minus_eligible` False). First-attempt flags are unchanged (R8).
- IRs that would not execute (VI 59, EN 118, EN2 4) count as cycle failures: there is no certificate. Only VI got a feedback retry round; EN had no Modal budget for one.

**Wave 2 (stopped 05:13 by the lead, because there is no Modal budget for its cycle check)**
- 2,400 slots were scheduled, seeds whose wave-1 item failed stochastically first.
- 1,831 were mutated and 453 rendered before the stop.
- Every stage is resumable from the stores; it is costed in §6.

## How to resume after any interruption

1. **Every LLM call is cached** in `.cache/vimath/llm_cache.sqlite`, keyed by the request. A re-run never pays twice for a finished call.
2. **Every stage appends per item** to its JSONL store (last row wins), so a restart skips items that are already done:
   - `formalize.jsonl`, `mutate.jsonl`, `render_<V>.jsonl`, `solve_<V>.jsonl`;
   - `gates.jsonl` (recomputed);
   - `screen_haiku.jsonl`, `vgate.jsonl`;
   - `cycle_<V>.jsonl` (primary Qwen) and `cycle_glm_<V>.jsonl` (secondary).
   - render, solve and the screen write per chunk of 200 (`call_many_chunks`).
3. **Batch IDs are recorded** in `data/train/Y/_work/pending_batches.jsonl` before waiting. On restart, `common.recover_pending` collects them, filling the cache, instead of resubmitting.
4. **Qwen chunks** are written one file per chunk (`vllm2_out_*.json.partN.json`). `vllm_cycle ingest_parts` ingests whatever exists, and `prep` re-asks only items that have no Qwen verdict.

Commands, with `PYTHONPATH=. OPENBLAS_NUM_THREADS=1`, run ONE heavy stage at a time (laptop limit ≤ 300 MB):

```
python -m vi_math_verified_aug.arm_y.pipeline data/train/Y/_work/vi_scale status            # Y+ counts
python -m vi_math_verified_aug.arm_y.pipeline <run> mutate|render|solve|gates|vgate           # resumable
python -m vi_math_verified_aug.arm_y.pipeline <run> extend 2 [max_new]                       # wave 2 slots
python -m vi_math_verified_aug.arm_y.pipeline <run> en_selected data/train/final/selection_vi_v1.json
python -m vi_math_verified_aug.arm_y.vllm_cycle prep <run> R2b <k>; modal run -m vi_math_verified_aug.arm_y.vllm_cycle_modal2 --inp <run>/vllm_in_<k>.json --out <run>/vllm2_out_<k>.json; python -m vi_math_verified_aug.arm_y.vllm_cycle ingest_parts <run> R2b vllm2_out_<k>.json
python -m vi_math_verified_aug.arm_y.export <run> R2b vi items|write ; same for EN/en
```

Set `ARMY_DIRECT=1` to use the real-time API (the Batch queue was unusable on 2026-11-08 02:00–03:00). `ARMY_CLAUDE_INFLIGHT` sets the org-shared concurrency, which is capped at 14. `ARMY_RENDER_VARIANTS=R2b` keeps EN from rendering on every slot. `ARMY_SCREEN_EN=1` screens the EN/EN2 items. Further stages: `en_retry <seeds.txt>`; `vllm_cycle prep_multi <run> R2b,EN <k>`.

**Resuming VI wave 2** (the successor item): `pipeline <run> mutate` (the 2,400 wave-2 slots are already in `slots.jsonl`; 1,831 are mutated), then `ARMY_RENDER_VARIANTS=R2b render`, `solve R2b`, `gates`, `vgate`, then a Qwen job on the screen passes (`vllm_cycle prep <run> R2b <k>`).

## 5. Spend

**Claude, G5: $18.46** (logs/api/G5/claude.jsonl, live calls only; the EN2 retry added $0.44). Together with G4's $4.15 that is **$22.6 of the $27 G4+G5 cap**.

| item | $ |
|---|---|
| R2b render | 3.63 |
| formalize | 3.06 |
| screen VI | 2.54 |
| EN render | 1.84 |
| EN screen | 1.72 |
| story checks | 1.51 |
| shared solver (VI 1.20, EN 0.37) | 1.57 |
| v3 Opus evaluation | 1.30 |
| Sonnet test | 0.49 |
| calibration and the rest | 0.37 |

Batch was used for formalize, story checks and the first renders. From about 02:00 the Batch queue processed nothing in 45 min, so everything after that ran on the real-time API at ≤ 14 requests in flight; the concurrency limit is org-wide and shared with D1.

**Modal, G5 apps: $9.50 of the $9.5 cap** (`modal billing report --for today`, per app). All apps stopped.

| app | $ |
|---|---|
| g5-vllm-cycle: single call, timed out, results lost | **2.89** |
| chunked round 1 (A100, 2 chunks incl. validation) | 0.97 |
| VI round 2 (L40S, 2,558 items) | 2.62 |
| VI feedback retry (133 items) | 0.34 |
| EN (2,663 items) | 2.38 |
| EN2 retry (170 items) | 0.26 |
| bge-m3 decontamination × 2 | 0.05 |

**NIM:** free. GLM was the secondary cycle label, about 1,560 items.

## 6. Costed items for the successor (measured unit costs, real-time prices unless noted)

**Per VI slot ≈ $0.0018.**
- Calls per slot: formalize 1/seed, story check about 0.9, render 0.75, solve 0.72, screen 0.67 at $0.00053 each, Qwen cycle 0.41 at $0.00067 each on L40S.
- About 3.8 LLM calls per slot.
- Yield: 0.40 VI Y+ per slot in wave 1, so **≈ $0.0044 per VI Y+**.

**Per EN slot on an existing IR ≈ $0.0014.**
- Calls per slot: render, solve, screen at 0.96, Qwen at 0.68.
- About 3.6 calls per slot.
- Yield: 0.66 EN Y+ per slot, so **≈ $0.0021 per EN Y+**.

| target per language | VI slots needed | VI $ (Claude + Modal) | EN slots | EN $ | note |
|---|---|---|---|---|---|
| finish wave 2 (2,400 slots queued) | 2,400 | ≈ $4.3 | for the new VI Y+ seeds ≈ 700 | ≈ $1.0 | ≈ +600–700 VI Y+ seeds; every stage resumes from the stores |
| **10k Y+** | ≈ 25k (J = 3 on all 7.1k GSM8K seeds gives 21.3k; needs J = 4 or MATH seeds) | ≈ $44 (≈ 95k calls) | ≈ 15k | ≈ $21 (≈ 55k calls) | wave yields fall after wave 1 (only stochastic failures re-qualify); budget +20% |
| **20k Y+** | ≈ 50k (J ≈ 7 on GSM8K, or new seed sources; MATH seeds untested with this formalizer) | ≈ $88 (≈ 190k calls) | ≈ 30k | ≈ $43 (≈ 110k calls) | the GSM8K seed pool is the hard limit; a new seed source is needed |

Halve the Claude share if the Batch API is responsive. The Qwen share stays the same.
