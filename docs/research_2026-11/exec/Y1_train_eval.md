# Y1 — Student training + evaluation code (2026-11-07)

Brief: `EXEC_BRIEF2.md`, workstream Y1. Spec: `tech/T3_training.md` §3.1 (SFT) and §4.1 (GRPO), `tech/T5_evaluation.md` §4.
All code below was written from scratch. Nothing imports `notebooks/`, `src/`, `scripts/sft.py` or other legacy code.
Kaggle was **not** used, because S2 owns it.

## 1. What was built

| Path | What |
|---|---|
| `vi_math_verified_aug/train/config.py` | One YAML file per run. It carries `kind/arm/model/size/seed/budget`. Defaults follow T3/T5. Unknown keys are an error. `run_id = <kind>-<arm>-<size>-s<seed>-<hash8>`. The hash excludes per-session wall-clock fields, so a Kaggle relay can resume its own checkpoints. |
| `train/template.py` | `Bài toán:\n{q}\n\nLời giải:\n{sol}` ending in `Đáp số: \boxed{ans}` + EOS. Any trailing `####`/`Đáp số`/`Đáp án` line is replaced. The same prompt is used by SFT, GRPO and eval. |
| `train/data.py` | Tokenise. Items longer than `max_length` are **dropped (never truncated) on the pool, before** the seeded subsample, and the drop rate is logged. This keeps item counts matched across arms. Exactly 32 sequences per step from a seeded per-epoch permutation. Inside a step, items are sorted by length and cut into micro-batches. Rows are **right-padded, one sequence per row, with an explicit attention mask; there is no packing code path.** Setting `sft.packing: true` is a config error that cites the packcheck evidence. |
| `train/schedule.py` | WSD schedule: 3% warm-up, constant, then linear decay over the last 10%. Plus the epoch-branch plan. |
| `train/sft.py` | Custom PyTorch loop: completion-only token-mean NLL over the step's 32 sequences. CE runs in fp32, and lm_head is applied only at completion positions, so no [B,L,V] logits tensor is built. fp16 autocast, fp32 LoRA r=32/α=32/dropout 0 on all text-decoder Linear layers, torch AdamW with fp32 states and wd 0. GradScaler with skip counting; **exit 4 if more than 1% of steps are skipped**. Grad-clip 1.0 after unscale. The micro-batch is halved on OOM, found by a probe on the longest items. Resumable checkpoints (adapter + optimiser + scaler + RNG + step + branch bookkeeping) every 30 min, at each fork, and when the budget ends (then **exit 75**). Gradient checkpointing turns on automatically at ≥1.5B parameters. |
| `train/grpo.py`, `train/reward.py` | TRL `GRPOTrainer` with `loss_type=dr_grpo`, `beta=0`, `scale_rewards="none"`, G=8, 32 prompts/step (micro × accumulation = 256 rollouts), 200 steps, ≤512 tokens, T=1.0, LR 2e-5 constant after 10 warm-up steps, ε 0.2/0.28, `mask_truncated_completions`, fp16, and a fresh LoRA r=32/α=64 on the merged SFT model. Reward is binary `math_verify` with no format reward. `reward: random` is a seeded Bernoulli(0.5) keyed by (seed, step, row) that also logs `true_accuracy`. The wall-clock callback saves and stops, giving exit 75, and the next session resumes from `checkpoint-N`. Unknown GRPOConfig fields are refused. |
| `evalrun/runner.py`, `evalrun/cli.py` | Uses vLLM fp16 when it is importable and CUDA is present, otherwise HF `generate`. T=0.6, top-p 0.95, top-k 20, **K=4 + greedy**, max_new_tokens 2048, the training template (4-shot in the same template for base rows). Item-level `preds.jsonl` is appended and fsynced per chunk, and a torn tail line is repaired on resume. Raw responses are kept, and `rescore` re-grades without re-generating. `metrics.json` holds, per dataset: mean pass@1, greedy accuracy, maj@4, truncation, repetition-loop and no-answer rates. |
| `train/kaggle/job_template.py`, `train/kaggle/render.py` | Kernel template plus laptop renderer (details in §3). |
| `train/requirements-kaggle.txt` | Pins (details in §4). |
| `train/configs/*.yaml` | Examples: SFT sweep run with branches [1,2]; GRPO; GRPO random-reward control; eval of an SFT model; 4-shot base row. |
| `tests/test_train_{core,sft_loop,grpo,evalrun,kaggle}.py` | 56 tests, all mock-based. |

**Prediction row (GLMM fields first):**
- GLMM fields: `run_id, arm, model, size, seed, lang, dataset, item_id, sample_k, correct, tokens`.
- Then: `decoding, temperature, top_p, sampling_seed, finish_reason, truncated, repetition_loop, pred_span, grader_flag, grader_version, backend, response`.
- `sample_k` is 0..3 for sampled decodes and **−1 for greedy**.
- `run_id` is the *training* run's id (`eval.source_run_id`).

## 2. Evidence

Tests were run in the documented env `E:\venv_y1`. It is an Anaconda 3.11.7 base with torch 2.14.0+cpu, transformers 5.20.2, peft 0.21.2, trl 1.17.1 and accelerate 1.15.0. It has a `sitecustomize.py` that preloads the system VC++ runtime, because Anaconda's bundled msvcp140 14.27 makes `import torch` fail with WinError 1114.

```
> E:\venv_y1\Scripts\python.exe <scratchpad>\y1\run_y1_tests.py    (pytest on the 5 files)
........................................................                 [100%]
TORCH_IMPORTED: False | TRANSFORMERS_IMPORTED: False
PEAK_WORKING_SET_MB: 48
> pytest --collect-only: core 21, sft_loop 8, grpo 10, evalrun 8, kaggle 9  (= 56 passed)
```

What the tests establish (each one uses a mocked model, trainer, backend or grader):
- **Branches do not perturb the main run.** The final adapter is identical with and without branches [1,2]. The epoch-1 branch equals a fresh 1-epoch WSD run, which is gate G6 in miniature.
- **Budget stop then resume gives identical models.** This holds for cuts before a fork, inside branch 1 and inside branch 2.
- **The skip rule works:** 1 skip in 7 steps aborts the run; a fraction under the threshold passes. Exit codes are 0, 75 and 4.
- **Data rules:** over-long items are dropped and the rest are untouched; the subsample is seeded; every step has 32 items; micro-batches are length-sorted; rows are padded and never packed.
- **GRPO:**
  - GRPOConfig carries the T3 values, and unknown TRL fields are refused.
  - The random reward is seeded and balanced, it ignores correctness, and it logs `true_accuracy`.
  - A mocked `run()` completes, and it resumes from the newest `checkpoint-N`.
- **Eval:**
  - All GLMM fields are present; there are 5 rows per item; the decoding parameters are right.
  - It resumes with no duplicate rows after a torn line.
  - `rescore` flips the labels without calling the generator.
- **Kaggle template:** the GPU lane, budget + grace timeout, `{input}`/`{results}` substitution, dependency and deadline skips, and the rc→status ledger.

**Real CPU smoke run, done once before the RAM rule arrived:** a tiny random 2-layer Qwen3 (`model: tiny-random`, offline). SFT and merge:

```
[sft] pool 84 eligible 80 dropped>max_len 4 (rate 0.047619) -> train 64     rc=0, 10.5 s
plan total 24, branches epoch1 fork 7->8, epoch2 fork 14->16; exec_steps 27; adapters epoch1/2/3
lora_param_dtypes [torch.float32]; 14 LoRA modules (q,k,v,o,gate,up,down)
```

That run's log showed a decay window of 1 step running at LR multiplier 1.0. I fixed this: the decay multiplier is now (T−s)/(D+1).

GRPO ran on the merged model with `math_verify` and `random` rewards, both rc=0. The random run showed reward mean 0.5, std 0.53 and `true_accuracy` 0. With 16-token completions every sample is truncated and masked, so the loss was 0; with masking off, grad_norm was 0.31.

**Incident:** the coordinator killed a GRPO smoke process of mine at about 270 MB when free RAM reached 0.02 GB. After that no torch or TRL process was run on the laptop, and the tests were made fully torch-free. Three lazy imports (`probe_micro`, `setup`, `pick_device`) were fixed so that torch is not loaded by the tests.

## 3. Kaggle interface (coordinated with S2's `infra/`, which I did not edit)

- **Header injection.** The kernel gets a `JOB = {...}` header the same way `infra/pilot/submit_pilot.py` adds one.
- **Outputs.** It writes `results/meta.json` with `job.run_id` first, so `KaggleRunner.verify_output_run_id` works. It appends a `results/tasks.jsonl` ledger.
- **ENVCHECK** runs before any task. It checks:
  - the GPU is T4;
  - the Python version, if pinned;
  - every pin in `requirements-kaggle.txt`;
  - an fp16 LoRA forward/backward + GradScaler smoke test on GPU;
  - that trl is present when GRPO is queued.

  Any violation fails fast with exit 5. Like S2, it removes `torchao` before the check.
- **Lanes.** One lane per GPU (`CUDA_VISIBLE_DEVICES`), with at most 2 lanes.
- **Budgets.** Each task runs as a subprocess with a per-task `budget_s`. The child checkpoints and exits 75 at `budget − reserve`. The parent hard-kills the process group at `budget + grace`. There is also a kernel deadline.
- **Dependencies.** A task with `requires: <run_id>` is skipped unless that run's `DONE` exists, so SFT → merge → eval can run in one lane.
- **Resume relay** (T6 §1.3 A): mount the previous `ckpt/` as a dataset listed in the config's `resume_dirs`.
- **Rendering.** `render.py` writes `run.py` + `job.json`. Pushing uses S2's `KaggleRunner.render_kernel/push` unchanged.

## 4. Deviations from T3/T5 and decisions taken

1. **Padded, length-grouped batching instead of padding-free.** This follows the brief and `w2-pilot-v1/*/packcheck.json`. Position-id packing gave max |Δlogit| of 17–34 (Qwen3) and 9–13 (Qwen3.5); padding gave ≤0.22. `pad_fraction` is logged so the cost can be measured.
2. **Branch fork point.** A branch forks at `e·spe − decay(e·spe)` and decays over the rest of epoch e. The branch model therefore sees exactly e epochs with the decay length of a fresh e-epoch run. Extra compute is about 0.1·e epochs per branch, i.e. 3.3 epochs per sweep run instead of T3's 3.2. The warm-up is 3% of the main run, not of e epochs; G6 must check this.
3. **When the skip rule applies.** It is evaluated from 50 executed steps on (`skip_check_after`), so the GradScaler's first calibration skips cannot abort short runs. Skips in the first 10 steps are also reported separately.
4. **GRPO prompts.** TRL 1.17.1 has no `max_prompt_length`, so prompts longer than 512 tokens are **dropped** from the pool and the count is logged.
5. **GRPO dev eval.** T3's "dev eval every 25 steps" is not run inside the trainer. Checkpoints are saved every 25 steps (`save_steps`) so that `evalrun` can score them.
6. **HF-backend seeding.** HF `generate` cannot seed each row, so it seeds per batch. vLLM, the primary backend, seeds per request.

## 5. Blocked / TODO

- **Kaggle smoke runs (owner: S2's queue).** One job is needed: SFT with real Qwen3-0.6B, about 50 steps with branch [1], then merge, eval (vLLM) and GRPO (about 5 steps, with and without vLLM). Measure tok/s, skipped steps, peak memory, and vLLM-vs-HF agreement (G5). None of this has been run on GPU.
- **Pins (owner: S2).** `docs/research_2026-11/exec/S2_student_pilot.md` did not exist yet, so `requirements-kaggle.txt` has TODO markers. Still to confirm on the Kaggle image: trl 1.17.1 with vllm 0.33.0, datasets, PyYAML, and math-verify 0.9.0.
- **Grader (owner: X1).** The code relies only on `grade(pred, gold, locale)` returning `{correct, span, flag}`, with an `is_correct` fallback. A grader exception is scored as incorrect and flagged; it never crashes a run. Re-check the API when X1 lands (the current snapshot reports `GRADER_VERSION 3.0.0`).
- **Model coverage.** The Gemma-4 multimodal loader is not implemented; only `AutoModelForCausalLM` is supported, and S2's `load_model` has the text-only path. This is only needed if Gemma passes G1.
- **vLLM attention backend.** It is not forced to `TRITON_ATTN`. Set `VLLM_ATTENTION_BACKEND` if S2's pilot shows that `auto` picks a bad backend on sm_75.
- **Before the grid.** `max_length` in the configs is a placeholder (1024). Set it to the p99 of the pooled arm lengths (`data.suggest_max_length`) and freeze it.
- **Running the tests.** They need `E:\venv_y1`. The project `.venv` has no PyYAML, so the tests skip there.
