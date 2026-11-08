# T1r — First real GPU runs + the S0 review (2026-11-07/08)

Owner: T1r (then T2). Code: `vi_math_verified_aug/train/` (sft, guards, launcher, check_runs, build_s0, build_dev),
`vi_math_verified_aug/evalrun/` (runner, modal_eval). Commits `195a6bd`, `86a2dd5`, `e62bdaa`, `0328cb6`.
Raw outputs: `jobs/outputs/t1r-runs-v1/` (Kaggle), `jobs/outputs/t1r-eval-l40s/` + `jobs/outputs/t1r-eval/` (Modal
eval), `jobs/outputs/t2-canary/`, `jobs/outputs/T2_checks/<run>/` (CSV + SVG loss curves, check.json).

> **Status of these runs (lead, 2026-11-08):** the Kaggle T4 fp16 runs are a **PILOT**. The comparison blocks run on
> Modal L40S bf16 (R-GPU: the same seed on another GPU flips 11–15 % of dev items), and S0 is re-run inside every
> cell. Use the numbers below for the review (labels, loss sanity, parity), never as a comparison row.

## 0. Answers

| Question | Answer [evidence] |
|---|---|
| One lock | **transformers 5.20.2** (vLLM 0.33.0 resolves it): torch 2.14.0+cu130, peft 0.21.2, accelerate 1.15.0, triton 3.8.0, datasets 4.8.5; Unsloth 2026.11.3 + unsloth_zoo 2026.11.3 + trl 1.17.1 + cut_cross_entropy 25.1.1 + bitsandbytes 0.50.2 `--no-deps` in an isolated dir; `torchao` removed. Kaggle ENVCHECK: 0 violations, Unsloth import OK; the Unsloth packing gate passes on 5.20.2 [`results/env_check.json`, `*/pack_gate.json`]. `train/requirements-kaggle.txt` |
| Parity gate (300 steps, Qwen3-0.6B, same seed) | **Loss parity: yes.** Train loss by 50 steps Unsloth packed vs HF padded: 0.5934/0.5930, 0.5356/0.5330, 0.5151/0.5124, 0.5049/0.5022, 0.4588/0.4565; dev300 loss 0.4867 vs 0.4847 (step 100), 0.4642 vs 0.4614 (step 200). Logits gate: max ΔNLL 0.00098/token, argmax 1.0, fused loss = logits path to 1.6e-4. **Accuracy parity at step 300: not measured** — the HF run's OOM probe cascaded to micro-batch 1 (810 tok/s) and it hit its budget at step 257 (bug fixed, `sft._probe_once`). Moot now: HF padded is only the fallback |
| tok/s (real tokens) | Kaggle T4 fp16: Unsloth packed mb 8 **3,814–3,833**; HF padded (mb 1 after the probe bug) 810; Modal T4 smoke HF mb 8 2,630, Unsloth padded 5,302. LFM2.5-2.6B HF 679. **Modal L40S bf16 Unsloth packed mb 32: 35,412** (G6 sweep, = R-GPU 35.5k) |
| $ / quota | Kaggle t1r-runs-v1: **0.91 quota-h** (2.52 → 3.43 h; 30 min QUEUED free, 54.7 min RUNNING). Modal so far: see §5 |
| Base 4-shot vs S0 on vi_primary_1000 (L40S eval) | base **34.3 %** pass@1 [32.0, 36.4] (greedy 39.2); S0-s1 **35.5 %** [33.1, 38.0] (greedy 38.7); S0-s2 **35.4 %** [33.1, 37.6] (greedy 39.7). By subset: GSM8K-Platinum 35.0 → 44.9 / 43.1; MATH alg/prealg 33.2 → 21.4 / 23.8 (S0 is GSM8K-only: it hurts MATH) |
| LFM2.5-2.6B fp16 smoke (100 steps, T4) | **Pass**: 0 skipped steps, 0 non-finite, GradScaler 65,536 throughout, max \|Linear output\| 16.1 (4,000× fp16 headroom), loss 0.95 → 0.50, 679 tok/s, 9.0 GB. Second family gated in (Unsloth for LFM2 unverified → HF loop) |
| Seed variance (2 S0 seeds) | s1 − s2 = **+0.15 pt [−1.4, +1.7]** (paired item bootstrap); Y2 `seed_variance`: σ_run(mirror) = 0.0 (raw −0.002), 95 % CI [0, 0.81] logit — **df = 1, uninformative**; the Y2 pilot rule needs ≥ 3 seeds per cell |
| Cap 1,024 | Keep. In the 2,048-token reruns of 200 items, 108 of 2,000 samples ran past 1,024 tokens and **1** of them was correct; 105 of the 108 also hit 2,048 (repetition loops). The pass@1 differences between the 1,024 and 2,048 runs (base −2.8, S0 +0.6 pt) are batch nondeterminism, not the cap (§4) |

## 1. Data (S0 seed set, `vi_math_verified_aug/train/build_s0.py`, `data/t1r/MANIFEST.json`)

- Source `data/translation/gsm8k/gsm8k_train_vietnamese.jsonl` (7,473). Excluded: R1's 14 corrupt + 2 half-English
  rows. Calculator annotations `<<…>>` stripped; the answer line is rebuilt from the integer `ground_truth`.
- **The Kaggle pilot trained on 7,157 rows** (clean − dev300). Since 2026-11-08 the S0 pool also **excludes the DEV
  seed bucket** (S3's dev_v1, QA-T HIGH 1): **6,788 rows** (VI and EN mirror, same ids). The pilot adapters therefore
  saw ~353 dev_v1 GSM8K seeds — another reason they are pilot-only.
- dev300 (S0-only selection), dev_v1 (arm dev set, 512 = 369 GSM8K + 143 MATH DEV-bucket seeds,
  sha256 `0109576a…`), EN mirrors, eval-question hashes (11,245) for the overlap guard: `data/t1r/`.

## 2. S0 review (the lead's checklist)

**Loss curves.** `jobs/outputs/T2_checks/t1r-S0-uns-0p6B-s{1,2}/{loss.svg,train_log.csv,dev_log.csv}`.
S0-s1 (446 steps, LR 2e-4 WSD, 7,157 × 2 epochs): train loss per 50 steps 0.593, 0.536, 0.515, 0.505, 0.459, 0.433,
0.431, 0.440, 0.429; dev300 loss 0.752 → 0.477 (150) → 0.458 (300) → 0.444 (446). S0-s2: 0.590 … 0.434; dev 0.752 →
0.476 → 0.459 → 0.446. No skipped steps, no non-finite loss, packed (pad fraction 0), peak 6.2 GB.

**Dev metrics.** dev300 loss above; primary_1000 accuracy in §0; dev_v1 accuracy of the pilot adapters: §3 table
(reference rows).

**Three decoded label examples** — the tokens with label ≠ −100, decoded with the Qwen3 tokenizer by the pre-flight
guard (`preflight.json` of the guarded S0 canary on the same data/template; and again in every G6 run):

```
vi_gsm8k_train/0  trained 64 / 138 tokens   (prompt ends "...vào tháng 4 và tháng 5?\n\nLời giải:\n")
Natalia đã bán 48/2 = 24 kẹp tóc vào tháng 5.
Natalia đã bán tổng cộng 48+24 = 72 kẹp tóc vào tháng 4 và tháng 5.
Đáp số: \boxed{72}<|endoftext|>
vi_gsm8k_train/1  trained 55 / 101
Weng kiếm được 12/60 = $0.2 mỗi phút.
Làm việc 50 phút, cô ấy kiếm được 0.2 x 50 = $10.
Đáp số: \boxed{10}<|endoftext|>
vi_gsm8k_train/2  trained 79 / 170
Ban đầu, Betty chỉ có 100 / 2 = $50.
Ông bà của Betty đã cho cô ấy 15 * 2 = $30.
Điều này có nghĩa là Betty cần thêm 100 - 50 - 30 - 15 = $5 nữa.
Đáp số: \boxed{5}<|endoftext|>
```
Asserted: exactly the completion (no prompt, no padding), answer present, EOS last and trained, trained fraction
0.255–0.831 (mean 0.564), dataset sha256 = manifest, 0 eval-question overlap.

**Parity gate:** §0. **LFM2.5 smoke:** §0.

## 3. G6 LR × epoch sweep (Modal L40S bf16, n = 3,000, selected on dev_v1 accuracy)

Done and frozen in T2 (`exec/T2_training_tranche.md` §2): **LR 1e-4, 3 epochs**, chosen on dev_v1 composite
(pass@1 + greedy)/2 over 3 seeds: 38.7 [35.7, 41.8]; 4e-4 × 3 is −1.9 [−3.5, −0.3]; 1e-4 × 2 ties (−0.5 [−1.9, +0.9]).

## 4. Eval determinism (QA-T E-det)

Same 128 dev items, greedy, base 4-shot, A10, chunk 128 vs chunk 16:
- default vLLM 0.33: **87/128** identical responses (90/128 in the first 200 chars), 117/128 same verdict;
- `VLLM_BATCH_INVARIANT=1`: **128/128** identical; cost +13 % at chunk 128 (+70 % at chunk 16).
Now set in the Modal image (`e62bdaa`). The §0 accuracy rows above ran **before** it (L40S, default kernels); the
comparison blocks all run with it.

## 5. Spend

- Kaggle: 0.91 quota-h (t1r-runs-v1). Cap 4 h.
- Modal (app `t1r-eval`, `modal billing report --for today`): **$3.88 in total** through the G6 confirmation
  (smokes, canaries, determinism + repeatability checks, base/S0 evals, GRPO smoke, G6 sweep + confirmation + evals).
