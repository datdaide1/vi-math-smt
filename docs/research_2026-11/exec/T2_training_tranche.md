# T2 — Training tranche 2026-11-08 + the whole remaining grid as a runnable queue

Owner: T1r/T2. Updated 2026-11-08 ~03:00 +07. Brief: `DELIVER_BRIEF.md` + the lead's decisions of 2026-11-08
(platform FINAL, n = 3,000, 5 seeds at 0.6B, fail-fast guards, QA-T fixes).
Code (commits `195a6bd`…`2c9d2e4` on `feat/sprint0`): `train/grid.py` (queue), `train/launcher.py` (dispatch + eval +
gates), `train/guards.py` (fail-fast), `train/check_runs.py` (post-run checker), `train/sft.py`, `train/build_dev.py`,
`evalrun/runner.py`, `evalrun/modal_eval.py`. Queue `jobs/queue/t2_queue.jsonl` (184 rows) + `.totals.json`; state
`jobs/queue/t2_state.json`; logs `jobs/logs/T2/`; per-run checks `jobs/outputs/T2_checks/<run>/`.
Pilot evidence: `exec/T1r_first_runs.md`.

## 0. Decisions in force

| Item | Setting [evidence] |
|---|---|
| Platform (all comparison blocks: S0 + arms, VI + EN, 0.6B; 1.7B too) | **Modal `gpu="L40S"`, no fallback (wait instead)**, bf16, Unsloth 2026.11.3 packed, micro-batch 32 (no accumulation), no grad-ckpt, ONE lane per container, one container per (size, lang, seed) **cell** with its 5 runs in order [R-GPU; `exec/R_GPU_platform.md`]. Kaggle T4 = nothing in comparison blocks |
| Measured L40S throughput | **35.4k tok/s** (G6 sweep, 5 runs: 34.2–36.3k); 3,000 × 2 epochs = **33 s of training**, 150–200 s per run incl. load, gate, 4 dev/quick-eval checks [`jobs/outputs/t2-g6/*/manifest.json`] |
| n_train | **3,000 per arm per language, S0 included** (S0 = seeded subsample of the 6,788-row S0 pool). Exact n from D5's `data/train/final/manifest.json`; the run tolerates ≤ 1 % shortfall (QA-T B3), the launcher blocks below 2,400 |
| Seeds | **5 per arm at 0.6B, VI and EN** (S0 included); order VI s0 → EN s0 → seeds 1–4 interleaved VI/EN (grid P1–P3) |
| Recipe | LoRA r 32 / α 32, all-linear, 32 seq/step, WSD (3 % warm-up, 10 % decay), **LR 1e-4, 3 epochs (G6, §2.2)**, fresh main run; epoch-1 and epoch-2 decay branches saved as hedges (~+7 % compute); primary row = final adapter |
| Dev set | **dev_v1** (S3; 512 DEV-bucket seeds = 369 GSM8K + 143 MATH, never generation seeds), VI `data/t1r/dev_seed_vi.jsonl`, EN mirror on the same ids; sha256 recorded in every run's `preflight.json`; 0 overlap with 11,245 eval/benchmark questions |
| Eval | **Modal L40S pinned, no fallback**, bf16, vLLM 0.33, **`VLLM_BATCH_INVARIANT=1`**, K = 4 (T 0.6, top-p 0.95, top-k 20, one seed per request) + greedy, cap 1,024; VI runs → `vi_primary_1000`, EN runs → `en_primary_1000` (same item ids). Rows in the Y2 schema with `dataset` ∈ {gsm8k_platinum, math_algprealg}: **report per dataset, never only the pooled number** (§4) |
| Training data paths | `data/train/final/{Y,N}/{vi,en}/train_{plus,minus}.jsonl` (D5 selector); never the provisional pools. A language is READY only when all 4 files + SHA256SUMS exist and validate |
| Caps | Modal $18 total for T1r/T2 (spent **$3.88** to ~04:00 +07 incl. G6, `jobs/logs/T2/modal_billing_report_after_g6.json`), 15 % reserve; Kaggle 0.91 quota-h used (pilot) |

## 1. Fail-fast guards (lead's rule: nothing runs on full data and is found wrong later)

Implemented in `train/guards.py` + `train/sft.py`, unit-tested on mocks (`tests/test_train_guards.py`, 17 tests; all
train tests: 83 pass), exercised on GPU by the canaries and the G6 sweep.

1. **Pre-flight, before step 1** (abort `ABORTED_PREFLIGHT_*`, exit 7): 3 decoded label examples must equal the
   completion text + EOS (no prompt, no padding, answer present); trained-token fraction in [0.15, 0.95]; template
   round-trip; EOS last; dataset sha256 / rows = manifest; 0 eval-question overlap; step-0 loss in band
   ([0.3, 1.6] at 0.6B; measured 0.69–0.80). Data/setup errors exit 8 (`FAILED_DATA`) and never trigger an engine
   fallback.
2. **In-loop** (exit 7, manifest `ABORTED_<reason>`): non-finite loss / gnorm (an fp16 skipped step is allowed),
   no progress (mean of steps 50–100 not 3 % below 1–20), spike (> 3× running median), collapse (< 0.05 in the first
   25 %), throughput < 50 % of the platform's measured rate. **Demonstrated:** a canary on 200 items repeated
   13 epochs aborted at step 18 with `LOSS_COLLAPSE` (`jobs/logs/T2/canary_test_v1.log`).
3. **Mid-run** at 25 %, 50 % and the end: dev_v1 loss vs train loss (abort if > 2×), greedy quick-eval on 64 fixed
   dev items (batched `generate`, Unsloth-safe; 8–15 s on L40S). QA-T #2: accuracy drop = warning only; hard aborts
   only on format collapse (repetition > 0.5, no-answer > 0.5, accuracy 0). The end check runs **after** the
   adapter and final checkpoint are saved and never aborts.
4. **Unsloth pack gate**, precision-aware: ΔNLL < 0.02/token (primary) and argmax agreement ≥ min(0.97, measured
   floor − 0.02), the floor being separate-vs-padded rows in the same precision. Measured: fp16 T4 floor 1.000
   (packed 0.992–1.0); **bf16 L40S floor 0.962** (packed 0.968–0.985, max ΔNLL 0.0046) → effective 0.942. R-GPU's
   0.90 is not needed; the measured floor supports 0.94.
5. **Visibility / canary:** every new config (arm, size, lang, recipe) gets an 80-step canary first; its logs are
   on the Volume within 60 s (`launcher tail <run>`). On L40S a whole run is ~3 min, so the canary is the run's own
   first 3 minutes plus the guards; the launcher still requires a clean canary per config key.
6. **Post-run checker** `train/check_runs.py`: flags (status, missing pre-flight, skips, no learning, dev up, dev
   gap, format collapse, throughput, missing quick stats) **block** the next batch until `launcher ack RUN`;
   warnings (quick-acc drop, repetition 0.2–0.5) do not. Writes `train_log.csv`, `dev_log.csv`, `loss.svg`,
   `check.json` per run.

**Incident 2026-11-08 07:10 (seed-0 cells) and the resulting rule change.** In both VI s0 and EN s0 the S0 run
finished cleanly; the Y+ run aborted at step 42 with `DEV_DIVERGED` (dev 0.90 > 2 × train 0.32 VI; 0.97 > 2 × 0.39
EN), and stop-on-fail skipped Y−/N+/N− (≈ $0.2 spent). Quick-eval accuracy was RISING (VI 0.36 → 0.52, EN 0.61 →
0.63): dev_v1's references are seed-style, Y+ trains another solution style, so dev NLL measured style distance, not
divergence (KNOWN_ISSUES M-1). Lead-approved fix: dev-vs-train gap → warning; abort only on dev NLL > 2 × its own
step-0 value or non-finite; NEW accuracy guard: quick-eval accuracy ≤ 50 % of its step-0 value at 2 consecutive
checks aborts (`guards.quick_collapse`); format collapse unchanged. Arm-level monitoring uses accuracy. The 4 arms of
both seed-0 cells were relaunched; the finished S0 adapters were kept.

**Persistence (user rule):** checkpoints every ≤ 15 min and at every epoch boundary, written to the Volume and
committed every 60 s; adapters written the moment a run ends; `t2_state.json` rewritten atomically at every logged
transition; eval preds appended + fsynced per chunk and resumed; `test_crash_mid_run_then_rerun_resumes_from_the_
last_checkpoint` (a run killed at step 13 resumes from its last checkpoint and ends with the identical model).

## 2. G6 LR × epoch sweep (quality gate; L40S, n = 3,000, S0-VI seed 1, dev_v1)

Training (all COMPLETE, 0 flags; `jobs/outputs/T2_checks/t2-g6-*`): dev_v1 loss at 25 % / 50 % / end of the
3-epoch main run (steps 69 / 139 / 279):

| LR | dev loss 69 | 139 | 279 (ep 3) | note |
|---|---|---|---|---|
| 1e-4 | 0.5417 | 0.5237 | 0.5157 | still improving at epoch 3 |
| 2e-4 | 0.5304 | 0.5170 | 0.5230 | minimum near epoch 2 |
| 4e-4 | 0.5273 | 0.5182 | 0.5377 | over-fits in epoch 3 |
| 8e-4 | 0.5375 | 0.5426 | 0.5595 | too high |
| 2e-4 fresh 2-epoch (G6 fidelity) | 46: 0.5431 | 93: 0.5209 | 186: **0.5072** | |

### 2.1 dev_v1 accuracy (512 items, K = 4 + greedy, L40S, batch-invariant; `jobs/t2/g6_selection.json`)

Common sampling seeds across candidates; 95 % item-bootstrap CIs; "vs best" = paired difference to the top row.

| Candidate (LR, epochs) | pass@1 [95 % CI] | greedy | GSM8K seeds (369) | MATH seeds (143) | vs best [CI] |
|---|---|---|---|---|---|
| **4e-4, 3** | **37.3** [34.0, 40.7] | 37.7 | 40.5 | 29.0 | — |
| 1e-4, 3 | 37.0 [33.8, 40.4] | **41.0** | 39.4 | 30.6 | −0.3 [−2.8, +2.3] |
| 4e-4, 2 (branch) | 36.5 [33.2, 39.8] | 39.3 | 38.8 | 30.4 | −0.8 [−3.1, +1.4] |
| 1e-4, 2 (branch) | 36.1 [33.0, 39.4] | 40.6 | 38.0 | 31.1 | −1.2 [−3.6, +1.2] |
| 2e-4, 3 | 36.1 [33.0, 39.3] | 38.5 | 37.9 | 31.5 | −1.2 [−3.7, +1.3] |
| 2e-4, 2 (branch) | 35.9 [32.8, 39.2] | 38.7 | 37.7 | 31.5 | −1.4 [−3.7, +0.9] |
| 2e-4, 2 (fresh run) | 34.7 [31.5, 38.0] | 39.8 | 37.4 | 27.6 | −2.6 [−5.0, −0.4] |
| any LR, 1 epoch | 33.2–34.1 | 37.1–39.8 | 34.8–36.5 | 27.3–29.0 | −3.2 to −4.2 (CIs exclude 0) |
| 8e-4, 1–3 | 30.4–32.1 | 32.4–33.8 | 32.8–36.1 | 19.4–24.1 | −5.2 to −6.9 (CIs exclude 0) |
| base 4-shot | 33.0 [29.9, 36.2] | 38.9 | 31.8 | 36.2 | −4.3 [−7.1, −1.4] |
| (Kaggle pilot S0, n = 7,157; trained on 353 dev_v1 GSM8K seeds — NOT comparable) | 42.4–42.5 | 44.9–46.9 | 47.5–47.9 | 28.2–29.7 | |

### 2.2 Confirmation and FREEZE: **LR 1e-4, 3 epochs** (fresh main run)

Two more seeds (2, 3) of the two leading LRs, 3-epoch main runs with an epoch-2 branch (4 runs, all clean,
`jobs/outputs/T2_checks/t2-g6c-*`), evaluated like §2.1. Decision metric (lead): composite = (pass@1 + greedy) / 2,
mean over seeds 1–3, paired item-bootstrap CI vs the top (`jobs/t2/g6_selection.json` → `confirmation`):

| Candidate | composite [95 % CI] | pass@1 (s1, s2, s3) | greedy (s1, s2, s3) | GSM8K-seed composite | MATH-seed pass@1 / greedy | vs top [CI] |
|---|---|---|---|---|---|---|
| **1e-4, 3 epochs (main run)** | **38.7** [35.7, 41.8] | 37.0 (37.0, 37.3, 36.6) | 40.4 (41.0, 40.4, 39.7) | 41.2 | 30.5 / 33.8 | — |
| 1e-4, 2 epochs (branch) | 38.1 [35.1, 41.3] | 36.3 | 40.0 | 39.9 | 30.7 / 36.1 | −0.5 [−1.9, +0.9] |
| 4e-4, 2 epochs (branch) | 37.3 [34.3, 40.4] | 36.1 | 38.5 | 39.5 | 29.1 / 33.8 | −1.4 [−3.0, +0.1] |
| 4e-4, 3 epochs (main run) | 36.7 [33.9, 39.8] | 35.5 (37.3, 34.0, 35.2) | 38.0 (37.7, 35.9, 40.2) | 39.5 | 26.3 / 32.6 | **−1.9 [−3.5, −0.3]** |

- 4e-4/ep3 led on one seed (§2.1) but is significantly behind on 3 seeds; **1e-4 wins** (also the lower LR, the
  lead's tie-break, and the lowest-variance across seeds).
- Epochs: 3 vs 2 at 1e-4 is −0.5 pt [−1.9, +0.9], a tie; **3 epochs is chosen because it was measured as a fresh
  main run on 3 seeds** (the 2-epoch numbers are branches, and the G6 fidelity point estimate is over 1 pt, §2.1).
  Arms therefore train a fresh 3-epoch WSD main run; the epoch-1 and epoch-2 branches are stored as hedges only.
- **MATH-seed dev stays below the 4-shot base** for every S0 config (30.5 vs 36.2 pass@1): S0 is GSM8K-only. Hence
  per-dataset reporting (§4).
- Frozen in `train/grid.py` (`LR, EPOCHS = 1.0e-4, 3`) and in every launcher config (`branch_epochs = [1, 2]`).
  EN: the S0-EN LR ×½ / ×2 check (5e-5, 2e-4) runs as P2b before P3.

**Reading of §2.1.** 1 epoch and LR 8e-4 are clearly worse. LR ∈ {1e-4, 2e-4, 4e-4} × {2, 3} epochs are statistically
tied (all within −1.4 pt of the top, CIs straddle 0), with 3 epochs ≥ 2 epochs at every LR. The single-seed top (4e-4 × 3) did not survive confirmation (§2.2).
**G6 fidelity** (2e-4: epoch-2 branch − fresh 2-epoch run): +1.3 pt [−0.4, +2.9] pass@1, −1.2 pt greedy — at the
±1 pt rule's edge with a CI that includes 0; the main recommendation is a 3-epoch **main** run (no branch involved),
so the branch only serves the hedge adapters, which are stored, not reported.

## 3. Eval repeatability (before any arm eval)

S0-s1 adapter evaluated twice on `vi_primary_1000`, identical config, **two separate L40S containers**,
`VLLM_BATCH_INVARIANT=1`: **5,000/5,000 identical responses**, pass@1 35.93 % vs 35.93 %, greedy 38.5 % vs 38.5 %,
paired difference 0.0 [0.0, 0.0] (`jobs/outputs/t2-rep/`). Without the flag (A10, 128 items, chunk 128 vs 16):
87/128 identical greedy responses (`jobs/logs/T2/det-*.log`). Arm differences are therefore readable at the item
level; the remaining noise is training-seed noise, which the 5 seeds address.

**vLLM-adapter vs PEFT parity** (QA-T; `modal_eval.py::parity_peft`, run on the S0-s1 pilot adapter; to be repeated on
the first P1 adapter): 100 vi_primary items, greedy, HF transformers + PEFT (bf16, batched generate) vs the vLLM
LoRA greedy rows: same verdict **94 %**, same extracted answer 84 %, identical first 200 chars 67 %; accuracy 52 % (PEFT)
vs 50 % (vLLM), 6 discordant items in both directions — kernel-level divergence, no sign of a wrong adapter load
(`/vol/eval/_parity/pilot-S0-s1.json`).

## 4. What the pilot says about reporting

S0 (GSM8K seeds only) helps GSM8K-Platinum (35.0 → 44.9 / 43.1 % pass@1) but **hurts MATH alg/prealg
(33.2 → 21.4 / 23.8 %)** versus the 4-shot base (T1r §0). The arms' seed sets are mostly GSM8K too (D5 restricts S+ to
the seeds Y supports), so every table reports gsm8k_platinum and math_algprealg separately (Y2 `dataset` field) and
never only the pooled number.

## 5. Queue and cost (grid.py, L40S rates measured tonight; eval $0.059 per 1k items)

| Block | Runs | Train $ | Eval $ | Status |
|---|---|---|---|---|
| P1 0.6B VI seed 0 (S0 + Y± N±) | 5 | 0.71 | 0.30 | waits for D5 VI files + lead's go (G6 frozen) |
| P2 0.6B EN seed 0 | 5 | 0.71 | 0.30 | waits for D5 EN files |
| P2b 0.6B S0-EN LR ×½ / ×2 (recipe transfer) | 2 | 0.29 | 0.12 | with P2 |
| P3 0.6B VI/EN seeds 1–4, interleaved | 40 | 5.72 | 2.40 | after P1/P2 checks are clean |
| P4 GRPO 0.6B smoke (fp16) | 2 | 0.16 | — | **done** (§6) |
| P5 1.7B VI seed 0 | 5 | 0.85 | 0.45 | if budget/time allow (1.7B tok/s [EST] 14k) |
| P6 1.7B VI seeds 1–2 | 10 | 1.68 | 0.90 | successor |
| P7 E1-O3 EN mirror at 1.7B × 5 + S0-EN anchors | 23 | 3.84 | 2.0 | successor |
| P8 EN recipe check, VI 1.7B seeds 3–4 | 10 | 1.68 | 0.9 | successor |
| P9 T5-D3 T / T-off / M arms, extra ~2B seeds | 24 | 3.72 | 2.2 | successor (data not built) |
| P10 Qwen3-4B × 8 arms × 3 (L40S HF); LFM2.5-2.6B × 8 × 3 (T4 HF) | 48 | 6.87 + 20.5 T4 quota-h | 4.8 | successor |
| P11 GRPO 1.7B (S0/Y+/N+ × 3 + random × 3), fp16, Modal L4 | 12 | ≈ 53 [EST, O1] | 1.7 | successor |

P1–P3 (the tranche, 3-epoch recipe, 5 seeds): **≈ $7.4 train + ≈ $3.1 eval ≈ $10.5**. Spent so far ≈ $3.9 (sweep,
confirmation and their evals included; `modal billing report`), so ≈ $14.4 of the $18 cap (15 % reserve = $15.3).

### 5.1 Exact cost table at measured rates (lead, 2026-11-08: ≤ $9.5 for all training + eval)

Rates [M] from `modal billing report` per app (`jobs/logs/T2/modal_billing_report_costprobe.json`): an L40S container
costs **$0.00056/s** ($2.0/h incl. CPU/RAM).

| Item | $ | Basis |
|---|---|---|
| Train 1 run, n = 3,000 × 3 epochs (+ epoch-1/2 branches, all guards) | **0.105** | [M] 4 runs $0.422, 5 runs $0.505; per run ≈ 190 s = 95 s setup (Unsloth import, load, pre-flight, gate, probe) + 55 s training + 30 s mid-run checks + saves |
| Eval container start (vLLM engine load, once per container) | **0.10** | [M] 178–186 s |
| primary_1000, K = 4 + greedy, cap 1,024 (VI) | **0.071** / adapter | [M] 127 s (rep-a, batch-invariant) |
| primary_1000 EN | ≈ 0.06 / adapter | [EST] EN outputs ≈ 0.8× the VI tokens |
| **Vi-ExamMath VERIFIED (1,367), K = 4 + greedy** | **0.357** / adapter | [M] 200-item probe: 94 s, mean 408 tokens/sample, 20 % of samples hit 1,024 (19 % repetition loops) → × 6.84 |
| Vi-ExamMath VERIFIED, greedy only | ≈ 0.09 / adapter | [EST] 1/5 of the sequences, greedy truncation 28.5 % |
| dev_v1 (512), K = 4 + greedy | 0.034 / adapter | [M] G6 eval |

| Plan block | Runs | Train | Eval | Total |
|---|---|---|---|---|
| 1. VI 0.6B, 5 arms × 5 seeds; primary K4+g; **ExamMath K4+g** | 25 | 2.63 | 1.78 + 8.93 + 0.10 | **13.4** (does not fit) |
| 1'. same, **ExamMath greedy-only** | 25 | 2.63 | 1.78 + 2.25 + 0.10 | **6.76** |
| 2. EN 0.6B, 5 arms × 3 seeds; EN primary K4+g | 15 | 1.58 | 0.90 + 0.10 | **2.58** |
| 4. Base rows on L40S, batch-invariant (VI primary, ExamMath greedy, EN primary), in the same containers | — | — | 0.22 | **0.22** |
| **1' + 2 + 4** | 40 | 4.21 | 5.35 | **9.56** (≈ the cap) |
| 3. EN seeds 4–5 | 10 | 1.05 | 0.60 | 1.65 (does not fit without L5) |

Levers (quality-neutral unless stated):
- L1 one vLLM container per block evaluates all its adapters (one engine, LoRA switched per config, never mixed in a
  batch, batch-invariant): included above ($0.10 per block instead of per cell).
- L2 branch adapters stored, not evaluated: included.
- L3 K = 4 + greedy only on primary_1000; **ExamMath greedy-only** (deterministic, 1,367 items): included in 1' —
  this is the only design change, and it is the lead's call (T5 names K = 4 for the primary analyses).
- L4 dropping `VLLM_BATCH_INVARIANT` would cut eval time ≈ 16 % (107 vs 127 s) but loses exact repeatability: **not
  recommended**.
- L5 (not built yet, ~1 h + tests) one training process per cell (load the base once, re-init LoRA per arm): saves
  ≈ 60 s ≈ $0.034 per run → ≈ $1.35 over 40 runs, which would make EN seeds 4–5 fit (1' + 2 + 4 + 3 ≈ $9.9 − 1.7 =
  **≈ $8.2 + $1.3 ≈ $9.5**).

### 5.2 Decisions after the cost table (lead, 2026-11-08 ~04:30)

- **Vi-ExamMath VERIFIED is evaluated greedy-only** for every adapter; primary_1000 keeps K = 4 + greedy.
  **Deviation from T5** (K = 4 on primary sets), reason: cost $0.357 vs ≈ $0.09 per adapter at a $9.6 cap. The
  GLMM on Vi-ExamMath uses one binary per item × run, with 5 seeds per arm.
- **Cap 1,024 stays for Vi-ExamMath:** of the 216/1,000 probe samples that hit the cap, 202 are word n-gram loops,
  12 character-level loops (`\,\,\,`, `+0+1+2…`), 1 a degenerate enumeration and **1** a genuine long derivation
  (0.5 %); all 57 capped greedy samples are loops (`jobs/outputs/t2-costprobe-exam200-trunc.json`).
- **Single process per cell: built and PASSED its GPU parity test** (`jobs/t2/cell_parity.json`, L40S, commit
  `46d6321`): arm B trained in-process after arm A vs the same config in a fresh process — identical LoRA-init sha256
  (`482728c6…`, A's differs), identical step-0 loss (0.702199), first-50-step loss max |Δ| 0.0012 (mean 0.00037; bf16
  kernel nondeterminism), dev_v1 loss at step 60 0.548661 vs 0.548666, same throughput (35.4k vs 35.1k tok/s). Mock
  tests: `tests/test_train_cell.py` (arms equal fresh runs; a failing arm does not stop the cell). Saving measured:
  73 s per additional arm → per-run train ≈ **$0.074** in a 5-arm cell (was $0.105).
- `VLLM_BATCH_INVARIANT=1` stays.

| Plan at the new rates | Runs | Train | Eval | Total |
|---|---|---|---|---|
| 1'. VI 0.6B 5 arms × 5 seeds (primary K4+g, ExamMath greedy) | 25 | 1.85 | 4.13 | 5.98 |
| 2. EN 0.6B 5 arms × 3 seeds | 15 | 1.11 | 1.00 | 2.11 |
| 4. Base rows (L40S, batch-invariant) | — | — | 0.22 | 0.22 |
| **1' + 2 + 4** | 40 | 2.96 | 5.35 | **8.31** |
| 3. EN seeds 4–5 | 10 | 0.74 | 0.60 | 1.34 → 9.65 total (decide on measured actuals after 1'+2) |

### 5.3 Re-estimate at n ≈ 2,400 (lead: final n ≈ 2,200–2,600 per arm; launcher blocks below 2,000)

Training time scales only in its 55 s of steps; setup (95 s first arm, 22 s each further arm in the cell process),
mid-run checks (30 s) and saves (10 s) do not. Per 5-arm cell at $0.00056/s: n = 3,000 $0.368, **n = 2,400 $0.338
($0.068/run)**, n = 2,200 $0.327, n = 2,600 $0.348. Eval cost does not depend on n.

| Block (n = 2,400) | Runs | Train | Eval | Total |
|---|---|---|---|---|
| 1'. VI 0.6B 5 arms × 5 seeds (primary K4+g, ExamMath greedy) | 25 | 1.69 | 4.13 | 5.82 |
| 2. EN 0.6B 5 arms × 3 seeds | 15 | 1.01 | 1.00 | 2.01 |
| 4. Base rows | — | — | 0.22 | 0.22 |
| **1' + 2 + 4** | 40 | 2.70 | 5.35 | **8.05** |
| 3. EN seeds 4–5 (P12, on hold) | 10 | 0.68 | 0.60 | 1.28 → **9.33** (fits the $9.6 cap with $0.27 to spare, on estimates) |

### 5.4 FINAL tranche under the Modal budget (lead 2026-11-08 ~05:00: $6.5 for ALL training + eval)

D5 projects the final n at **≈ 1,600–1,800** per arm (VI wave 2 stopped); the launcher blocks below **1,500** and
otherwise trains every arm and S0 of a language on D5's `manifest_<lang>.json` `n_effective`. Per-run estimates at
n = 1,700 (grid.py, measured components): train **$0.060** (5-arm cell process), eval VI **$0.167** (primary K4+g
$0.071 [M] + Vi-ExamMath greedy-only $0.076 [from the probe's token share] + engine share $0.02), EN **$0.08**.

| Order | Cell(s) | Runs | Train | Eval | Total |
|---|---|---|---|---|---|
| 0 | base rows (VI primary K4+g + ExamMath greedy, EN primary; 4-shot; first eval container) | — | — | 0.22 | 0.22 |
| 1 | VI seed 0 (S0 + Y± N±) | 5 | 0.30 | 0.85 | 1.15 |
| 2 | EN seed 0 | 5 | 0.30 | 0.40 | 0.70 |
| 3 | VI seeds 1–3 | 15 | 0.90 | 2.55 | 3.45 |
| 4 | EN seed 1 | 5 | 0.30 | 0.40 | 0.70 |
| | **Tranche** | 30 | 1.80 | 4.42 | **6.22** of $6.3 (lead: cap 6.3; EN seed 1 is the first cut) |
| HOLD | VI seed 4 | 5 | | | 1.15 |
| HOLD | EN seeds 2–4 | 15 | | | 2.10 |
| HOLD | S0-EN LR ×½ / ×2 recipe check | 2 | | | 0.28 |

**Cap enforcement** (`--modal-cap-usd 6.3`): before each cell the launcher computes spend = billed month-to-date
for app `t1r-eval` since its first live tick + the estimates of every cell not yet settled (trained, evaluated and
billed), and refuses the cell if spend + its estimate (+ $0.22 for the base rows with the first cell) exceeds $6.3;
the order above means the cap can only cut EN seed 1. The state file records `committed_usd`, `modal_spent_usd`.

**Canary = the seed-0 cell** (`--canary-mode seed0`, default): on L40S a separate 80-step canary costs ≈ 60 % of a
3-minute run (≈ $0.7 for 10 configs — not in the budget). Seed 0 of every config therefore runs first with every
guard live and its logs on the Volume within 60 s; seeds 1–3 of that config launch only after the seed-0 run is
TRAINED and its post-run check is clean (`--canary-mode separate` restores the 80-step canaries).
Approved by the lead with two conditions, both implemented (commit after `aa3a15b`):
1. In a seed-0 cell the first failing arm (pre-flight, step-0 band, in-loop abort, gate, data error) **stops the
   cell's remaining arms** (`sft-cell --stop-on-fail`, exit 9 `CELL_STOPPED` in their manifests) and **blocks that
   language** in the launcher (`blocking["lang:<xx>"]`; the other language continues; `launcher ack lang:<xx>`).
   Test: `test_seed0_cell_stops_at_the_first_guard_failure`.
2. When the VI seed-0 cell is trained and evaluated the launcher writes **`jobs/logs/T2/vi_s0_summary.md`**: per arm
   status / post-run check, pre-flight verdict (decoded labels), dev_v1 loss start → end and quick-eval metrics,
   primary pass@1 and greedy with 95 % CIs per dataset, Vi-ExamMath greedy, loss-curve file; base rows as reference.
   VI seeds 1–3 start automatically once the seed-0 post-run checks are clean. Test:
   `test_eval_metrics_and_vi_s0_summary`.

Successor items (QA-T): matched-tokens control (T3 §3.2 B); LFM2.5 BOS handling check; GRPO with a difficulty-
filtered prompt pool, dev eval during GRPO, and eval loading the merged fp16 SFT model.

### 5.5 Tranche close (2026-11-08 09:03)

**Guards (commits `ebbdc4a`, `98620bb`).**
- The tranche cap is $6.9.
- HARD workspace guard: before each cell the launcher reads `modal billing report --for "this month" --json`, summed over ALL apps, and adds this tranche's in-flight estimates that are not yet billed. If that total + the cell estimate exceeds **$29.5**, it refuses the cell.
- The guard re-checks every tick. The loop exits only when no cell is training and no eval is still unbilled.

**What ran.**
- EN s1 was refused at 08:14, 08:15 and 08:18: $29.72–29.95 projected + $0.70 > $29.5.
- The lead's 08:19 priority call (EN s1 ahead of VI s3) came after VI s3 had already started training (quick-eval at step 168 was logged before 08:19). So VI s3 finished, and **EN s1 is the first successor item (P11 HOLD; ≈ $0.55 measured, $0.70 estimated)**, commit `4f5f2b9`.
- **Evaluated:** VI 0.6B seeds 0–3 and EN 0.6B seed 0. All 25 arm runs are TRAINED and their post-run checks are clean.

**VI 0.6B, 4 seeds, mean ± sd** (L40S, batch-invariant). GSM8K and MATH are pass@1 on vi_primary_1000 (K = 4); ExamMath is greedy on Vi-ExamMath VERIFIED. The two base rows are single runs.

| Arm | GSM8K | MATH | ExamMath |
|---|---|---|---|
| S0 | 38.6 ± 0.5 | 26.3 ± 1.7 | 24.4 ± 2.1 |
| Y+ | 45.1 ± 0.5 | 36.4 ± 1.7 | 30.7 ± 1.3 |
| Y− | 44.7 ± 1.1 | 36.9 ± 0.8 | 31.3 ± 0.7 |
| N+ | 48.1 ± 1.0 | 37.5 ± 1.1 | 31.1 ± 0.5 |
| N− | 46.6 ± 0.6 | 37.3 ± 1.0 | 31.5 ± 1.2 |
| base 4-shot (`base-qwen3-0.6b-4shot-vi`) | 33.8 | 34.8 | 34.2 |
| **base 0-shot, arms' template** (`ev-base-qwen3-0.6b-0shot-vi-*`) | 30.3 [27.8, 32.6] | 26.8 [23.6, 30.0] | 18.1 [16.0, 20.0] |

**Base 0-shot eval.**
- Launched under the lead's rule: run only if the exact month-to-date + $0.30 ≤ $29.5. At 08:49–08:54 the exact figure was $29.18, so it ran on L40S (`jobs/t2/run_base0shot.py`).
- Month-to-date after it: $29.43 (read at 09:05; billing lags by a few minutes).
- Reading: in the arms' own 0-shot format, the base model scores 18.1 on ExamMath vs ≈ 31 for every arm. The arms-vs-base gap on ExamMath therefore depends on the prompt format: the 4-shot base (34.2) is above all arms, while the 0-shot base is far below.
- The Kaggle T4 fallback (`jobs/t2/make_base0shot_kaggle.py`, CROSS-PLATFORM, INDICATIVE ONLY) was built but not pushed.

**Successor items with cost.**
- EN s1 cell: ≈ $0.55–0.70.
- Holds unchanged: VI s4 (≈ $1.12), EN s2–s4 (≈ $0.55 each), and the S0-EN LR ×½/×2 check.

## 6. GRPO smoke (fp16 everywhere; T3 §4.1, 2510.26788)

Merged the S0-s1 pilot adapter in fp16, then 10 steps each on Modal T4 fp16 with vLLM colocate
(`vllm_max_model_length` 1024, memory 0.35; 8 prompts × G = 8): **math_verify** reward mean 0.22–0.56 per step
(never constant; zero-std groups 0–38 %), grad-norm 0.025–0.046; **random** reward mean 0.36–0.52 with
`true_accuracy` logged (0.20–0.52). Both COMPLETE (`jobs/outputs/t2-grpo/`), 15–32 s/step. GRPO trains through
plain HF + PEFT (not Unsloth), so the fp16 activation monitor works there; on Unsloth SFT it reads 0.0 because
Unsloth's fused LoRA kernels bypass module hooks — irrelevant for bf16 SFT, documented.

## 7. Runbook for P1 (after the lead's go)

```
.venv/Scripts/python.exe -m vi_math_verified_aug.train.launcher approve s0-review      # done by the lead's review
.venv/Scripts/python.exe -m vi_math_verified_aug.train.launcher loop --every 120 --max-prio 3 \
     --backend modal --platform modal_l40s --modal-cap-usd 18 --stop-at 17:00
```
The launcher refuses a dirty tree (`code_stamp`: git sha + dirty flag in every manifest), validates the D5 files,
runs the canaries, launches one container per cell, pulls logs, runs the checker (blocking on flags), uploads
nothing for Modal runs (adapters already on the Volume) and evaluates each finished cell on L40S.

## 8. Open / blocking

- D5's `data/train/final/Y/vi/train_plus.jsonl` currently holds placeholder rows ("w w w …", no answer field):
  QA's contract test fails on it, and the launcher would refuse it. Waiting for the real files.
- The lead's go before P1 (G6 is frozen, §2.2).
