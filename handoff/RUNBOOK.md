# RUNBOOK — running the rest of the grid (Modal L40S) and the successor queue

**Sources of truth:** [`exec/T2_training_tranche.md`](../docs/research_2026-11/exec/T2_training_tranche.md) (queue,
launcher, guards, measured costs), [`exec/R_GPU_platform.md`](../docs/research_2026-11/exec/R_GPU_platform.md)
(platform choice), [`exec/QA_T_training_audit.md`](../docs/research_2026-11/exec/QA_T_training_audit.md) (stack
audit, GRPO prerequisites), [`exec/R1_results.md`](../docs/research_2026-11/exec/R1_results.md) (results so far).
When this runbook and T2 disagree, T2 wins.

## 0. State at hand-off (FINAL FREEZE 2026-11-08, HEAD `ffa8ae6`, from `jobs/queue/t2_state.json`)

| Block | Runs | Trained | Evaluated | Notes |
|---|---|---|---|---|
| Base Qwen3-0.6B, **4-shot** (VI primary K4+g, Vi-ExamMath greedy, EN primary) | — | — | ✓ | `base_metrics` in the state |
| Base Qwen3-0.6B, **0-shot in the arms' template** (VI primary K4+g, Vi-ExamMath greedy) | — | — | ✓ | `jobs/t2/ev-base-qwen3-0.6b-0shot-vi.metrics.json`; it does not stop (56.9 % of Core decodes hit the 1,024 cap) |
| P1 VI seed 0 (S0, Y+, Y−, N+, N−) | 5 | ✓ | ✓ | Y+ arms re-launched after the M-1 incident |
| P2 EN seed 0 | 5 | ✓ | ✓ | |
| P3 VI seeds 1–3 | 15 | ✓ | ✓ | |
| P11 HOLD **EN seed 1** | 5 | — | — | **first successor item** (Modal credit ran out) |
| P12 HOLD VI seed 4 · EN seeds 2–4 · S0-EN recipe check | 22 | — | — | successor |

Adapters (25): **local copies** in `jobs/outputs/t2-modal/<run_id>/adapters/epoch3/{adapter_model.safetensors,
adapter_config.json}` (1.9 GB, git-ignored, verified 25/25 and hashed in `handoff/ADAPTERS.md` / `ADAPTERS.sha256`,
shipped in the `runs` bundle); originals on the Modal Volume **`t1r-eval`**, `/train/<run_id>/adapters/epoch3`
(epoch-1/2 hedge branches beside them, stored, not reported). Item-level predictions:
`jobs/outputs/t2-eval/ev-<run_id>-{primary,exam-greedy}.jsonl` (Y2 schema). Run manifests, configs, logs, pre-flight,
pack gate: `jobs/outputs/t2-modal/<run_id>/`; post-run checks + loss curves: `jobs/outputs/T2_checks/<run_id>/`.
Cards: `handoff/MODEL_CARDS/`. Results: `docs/research_2026-11/exec/R1_results.md` (lme4 GLMM + cluster bootstrap).

Re-pull an adapter from the Volume (read-only):
```
E:\venv_tools\Scripts\modal.exe volume get t1r-eval /train/<run_id>/adapters/epoch3 jobs/outputs/t2-modal/<run_id>/adapters/epoch3
```

## 1. Platform and recipe (frozen; do not change inside a block)

- **Training:** Modal `gpu="L40S"`, no fallback (wait instead), bf16, Unsloth 2026.11.3 packed, micro-batch 32, one
  container per (size, lang, seed) **cell** that trains its 5 arms in one process (parity-tested). LoRA r 32 / α 32
  all-linear, 32 seq/step, WSD (3 % warm-up, 10 % decay), **LR 1e-4, 3 epochs** (G6 sweep + 3-seed confirmation),
  n = `n_effective` from `data/train/final/manifest_<lang>.json` (1,808).
- **Eval:** Modal L40S pinned, vLLM 0.33 bf16, `VLLM_BATCH_INVARIANT=1` (5,000/5,000 identical responses across two
  containers), K = 4 (T 0.6, top-p 0.95, top-k 20) + greedy, cap 1,024 on `vi_primary_1000` / `en_primary_1000`;
  **Vi-ExamMath Core VERIFIED greedy-only** (deviation from T5, cost). Report gsm8k_platinum and math_algprealg
  separately, never only pooled.
- **Guards** (`train/guards.py`): pre-flight label decode, step-0 loss band, in-loop non-finite/no-progress/spike/
  collapse, mid-run dev **accuracy** (quick-eval) — dev NLL is a warning only (M-1); post-run checker blocks the next
  batch until `launcher ack RUN`.
- Kaggle T4 is **not** used for comparison blocks (R-GPU: the same seed on another GPU flips 11–15 % of dev items).

## 2. Measured cost per run (L40S, $0.00056/s incl. CPU/RAM)

| Item | $ |
|---|---|
| Train 1 arm, n ≈ 1,800, 3 epochs, in a 5-arm cell | ≈ 0.06 |
| Eval container start (engine load, once per block) | 0.10 |
| VI primary K4+g | 0.071 / adapter |
| Vi-ExamMath VERIFIED greedy | ≈ 0.076 / adapter (K4+g would be 0.357) |
| EN primary K4+g | ≈ 0.06 / adapter |
| **Per 5-arm cell** | **VI ≈ $1.15 · EN ≈ $0.70** |

## 3. Pre-flight (every session)

1. `git pull`; `uv sync --locked --all-extras` (laptop env, no torch) — `ENV.md`.
2. Credentials present (never print them): Modal (`modal token new` → user profile), Kaggle (`~/.kaggle`), `.env`
   (only needed for data generation).
3. **Budget:** `E:\venv_tools\Scripts\modal.exe billing report --for "this month" --json > jobs/logs/D4/modal_billing_mtd_<ts>.json`
   then `python handoff/tools/ledger.py`. The launcher refuses a cell when workspace month-to-date + in-flight
   estimates would exceed `--workspace-cap-usd` (default 29.5) and stops the loop. **The November credit is spent;
   new work needs December's credit or a new budget.**
4. `python -m vi_math_verified_aug.train.launcher status` and `tick --dry-run` (validates the D5 files and renders
   the cells; pushes nothing). The launcher refuses a dirty git tree (`code_stamp` in every manifest).
5. `python handoff/tools/run_light_tests.py` green (except the known B-1).

## 4. Run

```
# release HOLD rows: edit their "status" from "hold" to "queued" in jobs/queue/t2_queue.jsonl (or change
# vi_math_verified_aug/train/grid.py and rebuild the queue), then:
.venv/Scripts/python.exe -m vi_math_verified_aug.train.launcher loop --every 120 --max-prio 12 \
    --backend modal --platform modal_l40s --modal-cap-usd <this block's $> --workspace-cap-usd <credit − margin> \
    --stop-at 23:59
.venv/Scripts/python.exe -m vi_math_verified_aug.train.launcher status
.venv/Scripts/python.exe -m vi_math_verified_aug.train.launcher tail <run_id>     # live train/dev log from the Volume
.venv/Scripts/python.exe -m vi_math_verified_aug.train.launcher ack <run_id|lang:xx>   # after reading a checker flag
```
A new config (arm, size, lang, recipe) runs its seed-0 cell first with every guard live (`--canary-mode seed0`); the
first failing arm stops that cell and blocks the language until `ack lang:<xx>`.

## 5. Successor queue, in priority order, with costs

Costs: Modal at the rates of §2 unless stated; Claude at Batch prices from D1/G5/D5 measurements.

| # | Item | What / why | Cost | How |
|---|---|---|---|---|
| 1 | **EN seed 1** (P11 HOLD) | H1 (language × verification) is provisional with 1 EN seed; seed 1 makes it testable | ≈ $0.70 | release P11 (status `hold` → `queued`), `launcher loop` |
| 2 | **4-shot evaluation of the 25 SFT adapters** (or a stop-sequence fix for the 0-shot base and re-run it) | the only clean arm-vs-base comparison: the 4-shot base uses another format, the 0-shot base does not stop | ≈ $3–4 [EST: VI 20 adapters × (primary + Core greedy) ≈ $0.15 each + EN 5 × $0.06 + engine] ; stop-sequence route ≈ $0.30 | eval config with `shots: 4` for adapters (evalrun supports few-shot rows); or add a stop on the next "Bài toán:" to the base row and re-run `jobs/t2/run_base0shot.py` |
| 3 | **Length-matched retrain** | question length is aliased with arm (SMD 0.46–0.65); only a length-matched subset tells whether length mediates H2 | VI 4 arms × 3 seeds ≈ $2.6 (+ EN ≈ $1.6) | `select_v1` draws equal question-length histograms Y vs N within strata (D5 §0) → train → R1 |
| 4 | **1.7B size and a second model family** (P5 VI s0, P6 VI s1–2, P8 VI s3–4, P7 EN mirror + S0-EN anchors, P8 EN recipe; P10 LFM2.5-2.6B / Qwen3-4B) | does the null H2/H3 hold where items are less limiting; size dependence for E1 O3 | 1.7B: P5 $1.95 · P6 $3.90 · P8 VI $3.12 · P7 $4.60 + $0.69 · P8 EN $0.46. 4B (L40S HF) $6.24 + $14.4 eval; LFM2.5 (T4 HF) 7.44 Kaggle quota-h + $10.8 eval | release by `--max-prio`; before LFM2.5: BOS check (QA-T 1.3) |
| 5 | Other HOLD rows: VI seed 4; EN seeds 2–4; S0-EN recipe check (P12) | 5 seeds/arm (E1 O3, T5 D3); recipe transfer to EN | ≈ $1.15 + $2.10 + $0.28 | release P12 |
| 6 | lme4 for the H4 yes-vs-no sensitivity | R1 §6 item 5 | Kaggle CPU (free) | re-export with `--binary yes_vs_no`, same kernel `datuni/d5-r1-lme4` |
| 7 | **VI Y wave 2 + Qwen cycle** (+ cheap retries) | 2,400 queued slots on the 2,239 VI-Y+-failed seeds → n ≈ 2,800 per arm | ≈ $4.3 VI + $1.0 EN (G5 §6); D5's path ≈ $1.6 Batch VI + Qwen cycle + ≈ $2 Batch EN; 74 EN2 cycle ≈ $0.1, EN N 2nd retry ≈ $0.07, VI N retry ≈ $0.3 | `arm_y.pipeline` (resumes from stores) → `analysis.select_v1 part2 vi,en` → `select_v2` |
| 8 | 10k / 20k top-ups | the original n targets | **N:** 10k VI ≈ $1.15, EN ≈ $5.8; 20k VI ≈ $14.5, EN ≈ $14 (D1 §8). **Y:** 10k VI ≈ $44, EN ≈ $21; 20k VI ≈ $88, EN ≈ $43 (G5 §6; the GSM8K seed pool is the hard limit). Halve the Claude share on Batch | `scale_gen.run gen …`, `arm_y.pipeline …`; re-select |
| 9 | **GRPO** (C4): P11 1.7B from S0/Y+/N+ SFT × 3 + random-reward × 3 | does the advantage survive RLVR | ≈ $52.8 train + $3.6 eval [EST, O1] | **fp16 everywhere**; prerequisites below |
| 10 | T5-D3 baselines (P9: T / T-off / M arms, extra ~2B seeds) | translated-data baselines | $5.58 + $2.34 + data building (not built) | build the arms first |
| 11 | **EN population-level R8 estimate** | EN was screened only on S+ seeds (biased toward VI Y+ passes) | ≈ $0.40 Haiku-v3 Batch on 1,559 unscreened EN N− items | `python -m vi_math_verified_aug.scale_gen.run <stage> --hv3-only en:data/train/N/_work/en_slots_part1.txt` (D1 §7) |
| 12 | **Grader gaps** G-1…G-4 (X1) | named points, compound conditions, prose last-value FP/FN on Vi-ExamMath | code + re-score (no GPU): `python -m vi_math_verified_aug.evalrun.cli rescore PREDS.jsonl --config EVAL.yaml --out RESCORED.jsonl` | KNOWN_ISSUES §B |
| 13 | **X2 segmentation: sibling sub-questions** | item text carries "a) … b) …" while the key covers one part | code fix in `vi_exam/segment.py`, rebuild, Q1 re-key of affected items (6 known, LOW_CONF) → `core_v1.2` | KNOWN_ISSUES X2-SEG |
| 14 | **M-1 dev loss as a style artefact** | never compare dev NLL across arms; monitoring already uses quick-eval accuracy | none (paper text) | KNOWN_ISSUES M-1 |
| 15 | Truncation look: verified arms truncate more on Core (Y+ 14.2 vs Y− 10.8 %; N+ 19.2 vs N− 10.5 %) at equal accuracy | unexpected (verified items have shorter traces) | none (analysis of stored preds) | R1 §5 |

Items 1–4 follow R1 §6 / D5's successor order (EN s1 → 4-shot eval of the adapters → length-matched retrain → 1.7B /
second family). **The November Modal credit is spent ($29.43 of $30 billed month to date); items 1–5 need December's
credit or a new budget.**

**GRPO prerequisites (QA-T §5, T2 §6):** fp16 (the smoke ran fp16 on T4 and passed: reward never constant, random
control logs `true_accuracy`); `vllm_max_model_length` set (1,024 in the smoke); a **difficulty-filtered prompt pool**;
dev eval during GRPO; evaluation must load the **merged fp16 SFT model** + GRPO adapter; replace O1's 14–17 T4-h
estimate by the smoke's measured 15–32 s/step before budgeting P11.

**Analysis after every block:** `python -m vi_math_verified_aug.analysis.c3_r1_results --pred "jobs/outputs/t2-eval/ev-*.jsonl"` (R1 tables; local, ~70 MB) and
the lme4 GLMM on a Kaggle CPU kernel (Modal is out of credit); then `handoff/tools/fill_model_cards.py`,
`ledger.py`, `make_bundle.py --hash`.

## 6. Failure handling

| Symptom | Action |
|---|---|
| `hard_stop` in `t2_state.json` (workspace MTD guard) | nothing runs until credit is available; raise `--workspace-cap-usd` only with a new budget |
| A seed-0 arm aborts (`ABORTED_*`, exit 7) | read `jobs/outputs/T2_checks/<run>/` + `launcher tail`; fix; `ack lang:<xx>` |
| `DEV_DIVERGED`-style dev NLL rise with rising accuracy | expected for style-shifted arms (M-1); the guard now only warns |
| Post-run checker flag | inspect; `launcher ack <run>` to release the next batch |
| Exit 75 (budget) | resumable: re-launch; checkpoints every ≤ 15 min and per epoch on the Volume |
| Modal app crash-looping at startup | `modal app stop <app-id>` at once (startup failures retry even with `retries=0`) |
| A long single GPU call | chunk it and write results per chunk (G5 lost $2.89 on one timed-out call) |

## 7. Hard rules
Kaggle kernels/datasets private · laptop ≤ 300 MB/process, no torch · test/benchmark items only to Claude · never
print keys · one platform/precision per block · report per dataset · stop and report before exceeding a cap.
