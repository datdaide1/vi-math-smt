# R-GPU — Platform, precision and engine for the 0.6B blocks

**Questions:** FP8 or bf16? Which GPU? How many runs per GPU? Where does eval run?

**Context.** Date 2026-11-07 23:00 → 2026-11-08 02:30 (+07). Owner: R-GPU. The brief is the lead's message of 23:05 plus
the follow-ups (a)–(d) and the 02:01 message. The report was due 01:30. It is late because the usage limit paused work
from about 00:30 to 02:01.

**Code and outputs.**
- `vi_math_verified_aug/bench_gpu/sft_bench.py` runs `train.sft.run` **unchanged** (same template, masking, loss,
  packing and schedule) and applies one recorded in-process variant: Unsloth `load_in_fp8`, torchao float8, or
  `torch.compile`.
- `vi_math_verified_aug/bench_gpu/eval_bench.py` runs `evalrun.runner.run` unchanged with an optional vLLM `quantization`.
- `bench/r_gpu/modal_rgpu.py` is the Modal app `r-gpu-bench`. `bench/r_gpu/summarize.py` does the steady-state tok/s and
  the paired McNemar/CI analysis.
- Specs are in `bench/r_gpu/specs/*.json`. Raw results are in `bench/r_gpu/results/*.json`; per-item predictions are in
  `bench/r_gpu/results/preds/`. The Modal Volume `r-gpu` holds train dirs, adapters and eval outputs. Launcher logs and
  billing before/after are in `jobs/logs/R_GPU/`.

**Tags:** **[M]** measured here · **[DOC]** vendor docs read tonight · **[EST]** arithmetic from [M].

**Spend:** Modal **$4.34** for R-GPU (cap $5), from `jobs/logs/R_GPU/billing_before.json` (r-gpu $0) and
`billing_after.json` (r-gpu-bench apps $4.343). The dev_v1 eval (§5) adds about $0.15. No Kaggle quota and no API
spend.

**Setup of every training measurement:**
- Model Qwen3-0.6B-Base. LoRA r=32, α=32, all-linear, fp32 adapters, AdamW, LR 2e-4 with WSD, 32 sequences per step,
  seq ≤1024.
- Data: the real T1r S0 file (`t1r-eval:/data/s0_train.jsonl`, 7,157 items, about 195 tokens/item).
- Throughput is real (non-pad) tokens/s over steps ≥5 (steady state). Peak memory is `torch.cuda.max_memory_allocated`.
- Quality has three measures: the loss curve, dev loss on dev300 every 100 steps, and greedy plus K=4 sampled accuracy
  through the frozen eval runner. Training-precision effects are always evaluated on one bf16 vLLM engine (L40S).

## 0. Answers in one screen

| Question | Answer (evidence §) |
|---|---|
| "L40S has FP8, right?" | **The hardware does; the usable stack does not.** Ada sm_89 (L4, L40S) has FP8 tensor cores. L40S lists 733 dense FP8 TFLOPS vs 362 dense BF16 TFLOPS [DOC, vendor spec sheets; dense vs sparse labelling varies by source]. But Unsloth 2026.11.3 refuses on-the-fly FP8 on Ada: `ValueError: Unsloth: On the fly load_in_fp8 requires H100 GPUs or after` [M, §1]. torchao float8 runs on L40S but **0.35× slower** than bf16 (§1) |
| FP8 LoRA SFT at 0.6B: faster? | **No, slower everywhere.** H100 Unsloth FP8 runs at **0.62×** bf16 (8,679 vs 14,027 tok/s, mb 8) and **0.49×** (32,759 vs 67,302, mb 32). L40S torchao float8 (HF engine) runs at 0.35× eager and 0.78× with `torch.compile` vs the compiled bf16 control. At these GEMM sizes (K,N ≤ 3,072; about 6k tokens per micro-batch) the quantise/scale overhead exceeds the FP8 matmul gain [M, §1] |
| FP8 SFT quality | Small penalty. Dev loss at step 0 is 0.7565 vs 0.7523 (the FP8 base is already worse), and at step 300 0.4557 vs 0.4526. Pack-gate ΔNLL rises from 0.002 to 0.015, against a limit of 0.02. Dev300 greedy accuracy shows no detectable difference (46.0 vs 45.3, p=0.90) [M, §2] |
| FP8 inference in vLLM (L4/L40S) | **No speed gain** (L40S 8,592 vs 8,794 gen tok/s), and **a measurable accuracy loss**. On vi_primary_1000 with the same adapter, pass@1 is **−1.77 pt [−3.15, −0.40]** and greedy −1.8 pt [−4.3, +0.7], paired. L4 on a partial run of 512 items gives pass@1 −2.05 pt [−4.0, −0.1]. Multi-LoRA on an FP8 base works: 4 adapters were swapped on one FP8 engine [M, §3] |
| Fastest $-efficient engine | **Unsloth packed, micro-batch 32** (one packed micro-batch per 32-sequence step): **35.5k tok/s on L40S**, against 20.5k at mb 8, 7.9k for the HF loop and 16.6k for HF+compile. Dev loss is equal to mb 8 (Δ +0.0002) [M, §1–2] |
| Several runs per GPU | Only to hide per-run overhead. GPU throughput is already saturated by one mb-32 process (3 concurrent: 36.4k aggregate vs 35.5k single). Three concurrent mb-32 processes on L40S are **memory-unsafe**: 2 of 3 crashed with `CUBLAS_STATUS_ALLOC_FAILED` once, and peaked at 42.6 of 46 GB when they did pass. Two concurrent are safe for S0-length items [M, §4] |
| GPU per $ (0.6B, mb 32, steady) | **L40S $0.054 per 3.9M-token run (GPU time only)**; H100 $0.064, but 1.9× faster; L4 $0.079, 3.2× slower, and its queue exceeded 60 min tonight; Kaggle T4 is $0 but needs about 15 min per run plus queue and install. The real per-run cost including overhead is in §4 |
| Precision consistency | One block = one GPU type (**L40S, no fallback**), bf16, Unsloth mb 32, one image, one layout. Put **all 5 arms of a (size, language, seed) cell in the same container**, so they share the same physical GPU. Same seed on a different GPU (L40S vs H100 bf16) gives dev loss within 0.0009 but **13.7 % of dev items flip** (greedy 45.3 vs 49.0, p=0.12). So never mix GPU types inside a comparison [M, §2] |
| Plan for tomorrow (0.6B VI+EN, about 45 runs) | **Modal L40S, bf16, Unsloth packed mb 32, 1 container per cell (5 arms sequential), all cells in parallel, eval on L40S.** At n_train=3,000 (the current setting) that is about **$0.06 per run trained and $0.07 per run evaluated, so about $6 for 45 runs**. Wall-clock is about 9 min of training per cell after its data lands (§6) |
| LR × epoch sweep (Kaggle T4 fp16) → L40S? | **Transfers.** The numerics differences measured here (fp16↔bf16 losses equal to 3 decimals in S2; GPU spread 0.0009 dev loss; mb 8↔32 0.0002) are 3–15× smaller than the seed-to-seed dev-loss SD (0.003). A 2-run confirmation (the winner and the runner-up, S0 seed 0, on L40S, about $0.25 and 15 min) is cheap. It is a tie-breaker, not a requirement (§7) |

## 1. Throughput (Qwen3-0.6B-Base unless stated; steady tok/s; 1 process unless k>1)

| GPU | Engine / variant | mb | tok/s | Peak GB | Loss last 10 / dev@300 | Notes |
|---|---|---|---|---|---|---|
| L40S | Unsloth bf16 packed (T1r recipe) | 8 | 20,471 | 5.78 | 0.4286 / 0.45176 | `train_v2_l40s_uns_bf16_mb8.json` |
| L40S | **Unsloth bf16 packed** | **32** | **33,847–36,256** (5 runs) | 16.2–17.1 | 0.4282 / 0.45194 | 300-step quality run + 3 full runs + 2 short |
| L40S | Unsloth bf16, 2 processes | 8 | 30,462 aggregate | 5.78 each | — | |
| L40S | Unsloth bf16, 3 processes | 8 | 32,226 aggregate | 5.78 each | — | |
| L40S | Unsloth bf16, 3 processes | 32 | 36,409 aggregate (full runs) | 9.3/16.6/16.4 (GPU 42.6 GB used) | dev 0.4629/0.4597/0.4655 | the 60-step repeat **failed 2/3** (cuBLAS alloc) |
| L40S | HF loop bf16 padded, length-grouped | 8 | 7,871 (8,031 pad-16) | 15.1 | 0.5546@50 | |
| L40S | HF + `torch.compile` per decoder layer | 8 | 16,777 (16,622 pad-16) | 12.4 | 0.5538@50 | |
| L40S | HF + torchao float8 tensorwise, eager, pad-16 | 8 | **2,789** | 16.2 | 0.5588@50 | unpadded: `_scaled_mm` needs dims %16 (1024×1464) |
| L40S | HF + torchao float8 rowwise, eager, pad-16 | 8 | 2,642 | 16.2 | 0.5579@50 | |
| L40S | HF + torchao float8 + compile, pad-16 | 8 | 13,013 (**0.78×** of HF+compile) | 12.4 | 0.5585@50 | unpadded + compile: dynamo error (symbolic dim %16) |
| L40S | Unsloth `load_in_fp8=True` / `"block"` | 8 | — | — | — | refused on Ada (see §0) |
| H100 | Unsloth bf16 packed | 8 | 14,027 | 5.8 | 0.4283 / 0.45261 | overhead-bound (slower than L40S) |
| H100 | Unsloth bf16, 3 processes | 8 | 38,562 aggregate | 5.8 each | — | 2.75× of 1 process |
| H100 | **Unsloth bf16 packed** | **32** | **67,302** | 17.2 | 0.5543@50 | |
| H100 | Unsloth FP8 (`load_in_fp8=True`, row; base `Float8Tensor`, LoRA fp32) | 8 | 8,679 (**0.62×**) | 5.39 | 0.4320 / 0.45575 | pack gate ΔNLL 0.015, argmax 0.935 |
| H100 | Unsloth FP8 | 32 | 32,759 (**0.49×**) | 17.0 | 0.5574@50 | |
| H100 | Unsloth FP8 `"block"` | 8 | — | — | — | load fails: `TorchAoDeserialize … v_proj.weight` |
| H100 | HF loop bf16 | 8 | 6,639 | 15.1 | — | the host CPU is slower than L40S's |
| L4 | Unsloth bf16 packed | 8 | 7,380 | 5.67 | 0.4282 / 0.45209 | |
| L4 | Unsloth bf16 packed | 32 | 10,946 | 17.1 | — | |
| **1.7B** L40S | Unsloth bf16, mb 32, grad-ckpt **on** (recipe `auto`) | 32 | 14,331 | 4.7 | — | |
| 1.7B L40S | Unsloth bf16, mb 32, **grad-ckpt off** | 32 | **19,643** | 27.6 | — | same math; recompute avoided |
| 1.7B L40S | Unsloth bf16, mb 8, grad-ckpt on | 8 | 7,704 | 4.5 | — | |
| 1.7B L40S | 2 processes, mb 32, grad-ckpt on | 32 | 13,139 aggregate | 4.7 each | — | worse than 1 process with GC off |

**Why FP8 loses here.**
- The 0.6B GEMMs are small: hidden 1,024, MLP 3,072, about 6k tokens per mb-32 micro-batch.
- The LoRA side path stays bf16, and every frozen-weight matmul pays dynamic activation quantisation and scale
  computation.
- torchao's own training docs say end-to-end speedups "increase as M,K,N (GEMM dimensions) increase". Its published
  numbers are for Llama-3-8B and larger on H100/B200, and it recommends `torch.compile` for competitive performance
  [DOC: docs.pytorch.org/ao/main/workflows/training.html, updated 2026-11-06].
- Unsloth's FP8 page targets RL (vLLM inference is about 1.4× faster). It reports only that bf16 and FP8 SFT loss curves
  "closely track", with no SFT speed claim [DOC: unsloth.ai/docs/new/fp8-reinforcement-learning].
- FALQON (arXiv 2510.24061) names the same effect: FP8 quantisation overhead dominates for small LoRA dimensions.
- Our measurement matches all of these: FP8 is 0.35–0.78× of bf16 at 0.6B.

**Why mb 32 wins.** The step objective is unchanged: sum of NLL over the step's completion tokens divided by their
count. mb 32 runs it as 1 forward/backward instead of 4, so there are 4× fewer kernel launches and Python round-trips.
At 0.6B the loop is launch-bound:
- L40S mb 8 shows 0% GPU utilisation in the first `nvidia-smi` samples of every run.
- The H100 single process is slower than the L40S one, and scales 2.75× with 3 processes.

The cost is memory: 17 GB per process at S0 lengths, against 5.8 GB at mb 8.

## 2. Quality: training numerics (all evaluated on the same L40S bf16 vLLM engine, dev300, 0-shot; greedy + K=4)

| Training run (300 steps, same seed 0) | dev loss @0 / @300 | dev300 greedy | dev300 pass@1 (K=4) | Paired vs L40S bf16 mb 8 (greedy Δ [95% CI], McNemar p; items flipped) |
|---|---|---|---|---|
| L40S bf16 mb 8 (reference) | 0.75203 / 0.45176 | 45.3 | 43.0 | — |
| **L40S bf16 mb 32** | 0.75213 / **0.45194** | 44.7 | 42.9 | −0.7 [−4.6, +3.3], p=0.87; 12.0 % flipped; pass@1 −0.1 [−3.2, +3.0] |
| H100 bf16 mb 8 | 0.75226 / 0.45261 | 49.0 | 41.2 | +3.7 [−0.5, +7.8], p=0.12; 13.7 % flipped; pass@1 −1.8 [−4.7, +1.1] |
| L4 bf16 mb 8 | 0.75230 / 0.45209 | (dev_v1 only, §5: 42.8 greedy) | | |
| H100 **FP8** mb 8 | **0.75652 / 0.45575** | 46.0 | 42.7 | +0.7 [−4.5, +5.8], p=0.90; 20.7 % flipped; pass@1 −0.3 [−3.4, +2.8] |

**What n=300 can resolve:**
- The paired 95% CI half-width is about ±4 pt for greedy and about ±3 pt for pass@1. **Differences below about 3–4 pt
  are invisible.**
- Dev loss is the sensitive instrument. Seed-to-seed SD at n=4,000 is **0.003**: seeds 0/1/2 give 0.4625/0.4597/0.4655.
- On that scale:
  - mb 32 vs mb 8 (Δ 0.0002) and GPU type (spread 0.0009) are ≪ 1 seed SD.
  - FP8 (Δ 0.003–0.004) is about 1 seed SD, i.e. a real if small penalty, because the FP8 base model itself is worse
    at step 0 (+0.004).
- Even when the loss agrees, a different GPU yields a different model at the item level: 12–14 % of greedy answers
  flip. This is why a comparison block must not mix hardware.

## 3. Quality and speed: FP8 inference in vLLM 0.33 (`quantization="fp8"`: dynamic W8A8; LoRA stays bf16)

Same adapter (L40S bf16 mb 8, 300 steps), frozen eval config (0-shot, greedy + K=4, T=0.6, cap 1,024, per-request
seeds). Paired by item.

| Set / GPU | bf16 engine | FP8 engine | Greedy Δ (bf16−fp8) [95% CI], p | pass@1 Δ [95% CI] | Gen tok/s bf16 → fp8 | Engine load s |
|---|---|---|---|---|---|---|
| vi_primary_1000 / L40S | greedy 39.9, pass@1 37.2 | 38.1 / 35.4 | +1.8 [−0.7, +4.3], p=0.18 | **+1.77 [+0.40, +3.15]** | 8,794 → 8,592 | 64 → 121 |
| vi_primary_1000 / L4 (FP8 killed at 512 items; paired on 512) | 44.7 / 45.3 | 43.2 / 43.2 | +1.6 [−1.7, +4.8], p=0.41 | **+2.05 [+0.08, +4.02]** | 3,553 → n/a | 118 |
| dev300 / L40S, 4 adapters | — | — | +2.7, +1.7, +5.0, +4.0 (one with p=0.04) | +1.9, +0.8, +2.2, 0.0 | 7.6–7.8k → 8.1–9.5k | 114 → 41 (warm) |

**Verdict:**
- FP8 serving costs about **1.5–2 pt of pass@1** on math. It is not faster for a 0.6B model on L40S, because the 0.6B
  decode is not weight-bandwidth-bound at batch 256+.
- **Do not use FP8 for reported eval.**
- Multi-LoRA serving on an FP8 base works. The 4 adapters were loaded into one FP8 engine (`max_loras=1`, swapped per
  config), with `quant_method=fp8` recorded.

**Cross-GPU inference parity (bf16).** On vi_primary_1000 with the same adapter, L40S and L4 give greedy 39.9 vs 39.9
(Δ 0.0 [−1.5, +1.5], 94 % item agreement) and pass@1 37.2 vs 38.0 (Δ −0.8 [−1.7, +0.1]).

## 4. Real per-run wall-clock and $, layout, availability

**Measured full runs.** Setup: L40S, n_train=4,000 × 2 epochs = 250 steps, 1.57M real tokens, Unsloth mb 32, dev300
loss at step 0 and at the end, adapter save. File: `train_v5_l40s_full_n4k_mb32_s0.json`.

| Item | Seconds |
|---|---|
| Spawn → function entry (queue + container cold start; image cached) | **6.2 / 9.6 / 14.6** (3 containers) |
| Run 1 in a fresh container (process start, Unsloth import/patch, model load from Volume cache, data prep, pack gate, micro-batch probe, 2 dev passes, training, save) | **141.0** |
| Runs 2 and 3 in the same container | **110.6, 110.6** |
| of which training steps | 44.8 / 44.0 / 43.7 (35.4–36.0k tok/s) |
| → fixed overhead per run | **about 66 s**, GPU idle for most of it |
| 3 runs concurrent in one container | 236.8 total = **79 s per run** (1.4× better), peak GPU memory 42.6 of 46 GB |

**Per-run cost on L40S at $1.95/h (cost = $1.95 × (66 s + tokens / 35.5k tok/s) / 3600):**

| Run size | Tokens [EST at 195 tok/item] | Sequential (1 lane) | 2 lanes [EST] |
|---|---|---|---|
| **n=3,000 × 2 ep (current T2 setting, lead 02:15)** | 1.17M | ≈ 99 s · **$0.054** | ≈ 70 s · $0.038 |
| n=4,000 × 2 ep (measured) | 1.56M | 110 s · **$0.060** | ≈ 80 s · $0.043 |
| n=10,000 × 2 ep (Y±, N± arms) | 3.9M | ≈ 176 s · **$0.095** | ≈ 125 s · $0.068 |
| 9M tokens (the brief's figure) | 9M | ≈ 320 s · $0.17 | — |
| **1.7B**, n=10k × 2 ep, mb 32, GC off | 3.9M at 19.6k tok/s | ≈ 290 s (≈ 90 s overhead) · **$0.16** | not useful (2 lanes with GC on: 13.1k) |

**Scaling caveat.** If an arm's items are longer than S0's 195 tokens, the training part scales linearly with tokens.
Memory scales with packed tokens per micro-batch: about 2.3 GB per 1k tokens at 0.6B. S0 peaks at 16–17 GB at mb 32.

**Availability tonight.** Times are from Volume file timestamps and launcher logs:
- **L40S:** started within 1–3 min every time (8 containers). Once it logged "waiting to be scheduled on a GPU_L40S
  worker" for about 2 min.
- **H100:** started within a few min (2 containers).
- **L4:** the training container spawned at about 22:59 and started at about 00:05, so **≈ 65 min queued**. The eval
  container spawned at about 23:10 and started at 00:09, so **≈ 60 min queued**. It was then killed by my 65-min local
  timeout during its second block.
- So T1r's current eval default `gpu=["L4","A10G"]` faces the same queue risk. S2 had already seen "L4" served as A10G.

**Container layout for about 45 runs at 0.6B (VI+EN, 5 arms × up to 5 seeds):**
- **One container per (size, language, seed) cell: 5 arms sequential, 1 lane, mb 32, gpu="L40S", no fallback.**
- All arms of a cell share one physical GPU, one image and one process layout, so the cell's numerics are as
  identical as hardware allows.
- Cost at **n_train=3,000** (the current T2 setting): cold start ≈ 15 s, then 1 × ≈ 130 s + 4 × ≈ 99 s ≈ 9 min,
  **≈ $0.30 per cell**. 9 cells (45 runs) ≈ **$2.7 train**.
- At n=10k arms the cell would take 1 × 141 s + 4 × ≈ 176 s ≈ 14–16 min, ≈ $0.50 per cell, ≈ $4.5 for 9 cells. Launch all cells whose data are ready in parallel. Wall-clock is about 16 min
  after the data lands.
- 2 lanes would save about $1.3 in total. It is not recommended for tomorrow because:
  - the memory headroom depends on the unseen Y/N item lengths;
  - the 3-lane case crashed once;
  - concurrency changes the step-0 dev loss in the 5th decimal (0.752082 vs 0.752132), a different cuBLAS/Unsloth
    path.
- If a later block uses lanes, every cell of that block must use the same lane count.

**1.7B layout.**
- Same layout: one container per cell, 5 arms sequential, 1 lane.
- Settings: mb 32, `grad_ckpt: false` explicitly (the recipe's `auto` turns it on at ≥1.5B and costs 27 % throughput
  here). The micro-batch probe still halves mb on OOM.
- Cost: ≈ 25 min and **≈ $0.85 per cell**, so 9 cells ≈ $7.5 train [EST from 60-step tok/s].

## 5. mb 32 vs mb 8 on S3's frozen DEV bucket (`data/seeds/dev_v1.jsonl`, 512 items, sha256 0109576a…)

**Setup.**
- Same 300-step adapters as §2.
- One L40S bf16 vLLM engine, 0-shot, greedy + K=4, frozen eval config.
- 512 VI items: 369 Vi-GSM8K and 143 Vi-MATH algebra/prealgebra, `split=dev`.
- Paired against L40S bf16 mb 8.
- Files: `bench/r_gpu/results/eval_l40s_devv1-bf16.json`; preds `bench/r_gpu/results/preds/devv1.*` (local only).
- Cost about $0.15; this was the last Modal spend.

| Training run | Greedy | pass@1 (K=4) | Greedy Δ vs mb 8 [95% CI], McNemar p (flipped) | pass@1 Δ [95% CI] |
|---|---|---|---|---|
| L40S bf16 **mb 8** (reference) | 45.7 | 41.75 | — | — |
| L40S bf16 **mb 32** | 44.7 | 40.87 | **−1.0 [−3.8, +1.9], p=0.59** (10.7 %) | **−0.9 [−3.1, +1.3]** |
| H100 bf16 mb 8 | 44.3 | 40.77 | −1.4 [−4.3, +1.5], p=0.43 (11.1 %) | −1.0 [−3.1, +1.2] |
| L4 bf16 mb 8 | 42.8 | 40.72 | −2.9 [−6.3, +0.5], p=0.11 (15.4 %) | −1.0 [−3.3, +1.2] |
| H100 FP8 mb 8 | 44.1 | 42.68 | −1.6 [−5.0, +1.9], p=0.44 (16.0 %) | +0.9 [−1.2, +3.1] |

**Reading.**
- mb 32 vs mb 8 shows no detectable difference. The CI excludes a pass@1 loss larger than 3.1 pt. Dev loss agrees to
  0.0002 (§2).
- The mb-32 delta is the same size as moving the identical recipe to another GPU (H100 −1.0, L4 −1.0 pass@1). That is
  hardware-level noise of a single run, about ±1–3 pt at n=512.
- **Consequence:** mb 32 is safe to adopt. What matters is that **every arm of a block uses the same mb and the same
  GPU type**. Either change on its own moves an individual run by as much as a 1–3 pt arm effect could.
- The pilot runs already done at mb 8 on Kaggle T4 must not be pooled with mb-32 L40S runs inside one comparison block.

## 6. Recommended plan for 2026-11-08 (0.6B VI+EN blocks)

| Item | Decision |
|---|---|
| Platform | **Modal `gpu="L40S"`, no fallback list**, for every 0.6B comparison block. Kaggle T4 stays for the sweep and as a whole-block alternative only |
| Precision | **bf16** (no FP8 in training or eval) |
| Engine | Unsloth 2026.11.3 packed, the T1r `train/sft.py` loop, **micro_batch 32**, grad_ckpt false. `pack_gate_min_argmax: 0.90` for bf16 (the default 0.97 fails on L40S at 0.964; ΔNLL 0.0058 < 0.02). Keep `pack_gate_max_dnll: 0.02` |
| Image | Identical to `evalrun/modal_eval.py`. pip `vllm==0.33.0 transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 math-verify==0.9.0 pyyaml datasets==4.8.5`; `pip uninstall -y torchao`; `pip install --no-deps --target /uns_site unsloth==2026.11.3 unsloth_zoo==2026.11.3 trl==1.17.1 cut_cross_entropy==25.1.1 bitsandbytes==0.50.2`. Resolves to torch 2.14.0+cu130, triton 3.8.0, driver 610.57.04. Env `VLLM_USE_FLASHINFER_SAMPLER=0`, `UNSLOTH_RETURN_LOGITS=0`, `PYTHONPATH=/root:/uns_site` for SFT only. **Never torchao on the path** |
| Concurrency | 1 container per (size, language, seed) cell, 5 arms sequential, 1 process |
| Eval | **Move to L40S, bf16**: one engine per cell block, LoRA swapped per adapter, never mixed in a batch. **$0.059 per 1,000 items** (greedy + K=4: 108 s gen) vs L4 $0.060 (269 s), with a 2.5× shorter wall and no 60-min queue. Re-run the base 4-shot rows on L40S too (they were served on A10G; about $0.05 each), so every eval row of a comparison shares one GPU type |
| Cost for 45 × 0.6B runs | At n_train=3,000: train ≈ $2.7 + eval ≈ $3.3 (45 × 1,000 items × $0.059 + 9 engine loads) ≈ **$6**. At n=10k it would be ≈ $7.8, or ≈ $8.5 with longer Y/N items. This fits the $18 T1r/T2 cap with the $10 reserve kept |
| Wall-clock | ≈ 9 min per cell train at n=3,000 (≈ 16 min at 10k) + ≈ 11 min per cell eval, all cells in parallel. VI cells are done about 30 min after 06:00 data, EN about 30 min after 09:00 |
| Quality risks and their size | (1) Hardware mixing: same seed, different GPU flips 12–14 % of dev items (±4 pt per run on n=300). **Mitigation: one GPU type per block, enforced (no fallback)**. (2) mb 32 vs the mb 8 pilot: Δ dev loss +0.0002; dev_v1 pass@1 −0.9 [−3.1, +1.3] (n.s., the same size as GPU-to-GPU noise). Never pool mb-8 T4 runs with mb-32 L40S runs in one block. (3) Unsloth nondeterminism under concurrency: 5e-5 dev loss, negligible, avoided by 1 lane. (4) An OOM-probe halving mb on long-item arms: recorded per run in `manifest.micro_batch`; same objective |

## 7. Does tonight's LR × epoch sweep (Kaggle T4 fp16) transfer to L40S bf16 mb 32?

Yes, with high confidence. The numerics differences between platforms are an order of magnitude below the effect being
selected:

| Change | Dev-loss effect | Source |
|---|---|---|
| fp16 T4 → bf16 Ampere | losses equal to 3 decimals; dev300 Δ −1.7 pt / +0.3 pt, n.s. | S2 §4 |
| GPU type (L4 / L40S / H100) | spread **0.0009** at step 300 | §2 |
| mb 8 → 32 | **0.0002** | §2 |
| Seed (n=4k) | SD **0.003** | §4 |

An LR step of ×2 or an extra epoch moves dev loss by far more than 0.003, otherwise the sweep could not discriminate
anyway.

**Optional confirmation, ≈ $0.25 and 15 min:** run the sweep's winner and runner-up (S0, seed 0) once on L40S with this
recipe, and check that their order by dev loss is preserved. If the sweep's margin is < 0.003, treat the two as tied
and keep the default.

## 8. Caveats

- All quality numbers are from 300-step runs (1.9M tokens, 1.3 epochs of S0), not full 2-epoch 10k runs.
- n=300/512 dev sets resolve only about 3–4 pt.
- Token lengths of the Y/N arms are not yet known. Cost scales linearly with tokens.
- "1.7B" numbers are 60-step throughput only, with no quality check.
- The concurrency-throughput figures for 2 lanes at full-run scale are [EST] from the 3-lane measurement.
- H100 at mb 32 is the fastest GPU (67k tok/s, ≈ 1.9× L40S) at about 1.15× the $ per run. Use it only if the wall-clock
  matters more than about $1 for the whole tranche, and then for a whole block.
