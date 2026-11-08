# S2 — Student pilot v2 (T4 fp16 + Modal parity/throughput)

Date 2026-11-07. Owner: S2. Status: **done**. Spend: Kaggle **1.82 GPU-h** (cap 3; quota 0.70 → 2.52 h),
Modal **$1.68** (cap $5).
Raw outputs: `jobs/outputs/s2-pilot-v1/`, `jobs/outputs/s2-pilot-v2/` (Kaggle), `jobs/outputs/s2-modal/<plan>/` (Modal).
Code: `vi_math_verified_aug/infra/pilot/s2_pilot.py` (kernel; also runs on Modal unchanged), `.../pilot/summarize_s2.py`,
`vi_math_verified_aug/infra/modal_s2.py`, and hardened `kaggle_runner.py` and `pilot/submit_pilot.py`.
Job specs: `jobs/kernels/s2-pilot-v1/`, `jobs/kernels/s2-pilot-v2/`. Status timelines: `jobs/logs/s2-pilot-v*_timeline.jsonl`.

Measured = [M]. Everything is 100 LoRA steps (r=32, α=32, all-linear text decoder, fp32 adapters, AdamW fp32, LR 2e-4,
32 seq/step, completion-only NLL, grad-clip 1). Batching is padded and length-grouped within each step. Throughput is
real (non-pad) tokens/s on un-hooked steps only. Data is Vi-GSM8K train (items >512 tokens dropped).

## 0. Answers in one screen

| Question | Answer |
|---|---|
| Environment fix | `pip uninstall -y torchao` after installing vLLM. peft 0.21.2 then imports, and the fp16 LoRA smoke test passes [M]. Pins are in §5 |
| Qwen3 T4 fp16 SFT throughput (HF loop) [M] | **0.6B 2,352 · 1.7B 1,358–1,384 · 4B 426 tok/s** → **1.06 / 1.84 / 5.87 T4-h per 9M-token run** |
| fp16 safety | No overflow in any model. 0 skipped steps after step 10. Tightest is **Qwen3-1.7B, max \|act\| 14,360 (4.6× headroom, `down_proj` input)** |
| Padding-free | HF `position_ids` + `use_cache=False` is **exact** (fp32 Δlogit 3.4e-4) but **slower on T4** (2,167 vs 2,352 tok/s). Unsloth via TRL `SFTTrainer(padding_free=True)` silently stays padded (§3) |
| fp16 (T4) vs bf16 (A10) parity [M] | Same seed, 0-shot dev300: **1.7B 63.0 vs 62.7 (+0.3 pt)**; **0.6B 42.3 vs 44.0 (−1.7 pt, McNemar p=0.41)**. Losses agree to 3 decimals. No evidence of an fp16 penalty, but n=300 cannot resolve ±1 pt (paired 95% CI ±3–4 pt) |
| Qwen3-4B platform [M] | **L40S bf16: 3,969 tok/s, $1.23/run** · H100: 5,856 tok/s, $1.69/run · A10G: 1,039 tok/s, $2.65/run · Kaggle T4: 426 tok/s, 2.9 quota-h/run (2 lanes). **4B → L40S** if Modal money is available |
| Second family | **Strong: LFM2.5-2.6B-Base** (60.3 % 4-shot, vLLM ✓ on T4, fp16 headroom 2,400×, 3.3 T4-h/run). **Light: none qualifies.** LFM2.5-1.2B scores 17.3 % with 1.6× token fertility. **Gemma-4-E2B: reject on T4** (22.3 %, no vLLM on sm_75, 5.5 T4-h/run). Licence-safe fallback: SmolLM3-3B-Base (52.0 %, Apache-2.0, no Vietnamese claimed) |
| T4×2 billing | **1 quota-hour per wall-hour** (not 2). QUEUED time is not billed [M] |
| SFT grid (HF loop, measured) | Qwen3 blocks **377 T4-h** + LFM2.5-2.6B block 257 = **634 T4-h ≈ 317 Kaggle quota-h** with 2 lanes (≈ 10.6 weeks). With Unsloth (O1: ×1.3–1.7) ≈ 400–490 T4-h. Moving 4B to L40S saves 176 T4-h for ≈ $37 (§6) |

## 1. Per-model table (T4 fp16 unless stated)

4-shot = Vi-GSM8K fixed 300-item subset, greedy, frozen `math_verify.py`. "T4-h/run" = 9M **Qwen-equivalent** tokens:
tok/s is divided by the model's token fertility relative to Qwen3 on the same items, so it compares runs over the same
data.

| Model | 4-shot acc [95% CI] | Eval engine · gen tok/s | Train tok/s (raw / Qwen-eq) | T4-h/run | micro×accum, GC · peak GB | Loss first5→last5 | Skipped (after 10) | Max \|act\| (headroom) | Tooling status |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 38.3 [33.0, 44.0]¹ | vLLM/TRITON_ATTN · 398¹ | 2,352 / 2,352 (v2) | **1.06** | 4×8, off · 8.4 | 0.695→0.529 | 0 (0) | 6,484 (10×) | ✅ vLLM, HF, Unsloth (O1) |
| Qwen3-1.7B-Base | 58.7 [53.0, 64.1]¹ | vLLM · 365¹ | 1,358 / 1,358 (v2) | **1.84** | 4×8, off · 13.1 | 0.616→0.462 | 0 (0) | **14,360 (4.6×)** | ✅ |
| Qwen3-4B-Base | **78.3** [73.3, 82.6] | vLLM · 222 | 426 / 426 | **5.87** | 8×4, on · 14.2 | 0.539→0.400 | 0 (0) | 7,112 (9.2×) | ✅ |
| **LFM2.5-2.6B-Base** | **60.3** [54.7, 65.7] | vLLM/TRITON_ATTN · **953** | 692 / 758 | **3.30** | 8×4, on · 9.9 | 0.784→0.509 | 1 (0) | **27 (2,433×)** | ✅ vLLM + HF on T4. Unsloth for LFM2 [UNV]. Conv layers: keep padded batching (packing not tested) |
| SmolLM3-3B-Base | 52.0 [46.4, 57.6] | vLLM · 416 | 599 / 672 | 3.72 | 8×4, on · 10.9 | 0.836→0.547 | 0 (0) | 593 (110×) | ✅ vLLM + HF |
| LFM2.5-1.2B-Base | 17.3 [13.5, 22.0] | vLLM · 1,126 | 2,355 / 1,443 | 1.73 | 8×4, off · 13.1 | 1.037→0.370 | 0 (0) | 13 | ✅ but **fertility 1.63× Qwen** (4-shot prompt 1,280 vs 681 tokens; 540 of 3,740 items >512 dropped) |
| Gemma-4-E2B (base) | 22.3 [18.0, 27.4] | **HF generate · 52** | 438 / 459 | 5.45 | 2×16, on · 13.2 | 1.757→0.512 | 4 (0)² | 1,554 (42×) | ⚠️ vLLM fails on T4 (§2). HF text-only load works (`Gemma4ForCausalLM` + `key_mapping ^model.language_model.→model.`). LoRA also hits `per_layer_*` projections |
| Llama-3.2-1B | — | — | — | — | — | — | — | — | skipped: gated (HTTP 401, no access) |

¹ W2 pilot v1, same subset and engine (not re-run). ² GradScaler calibration (65,536 → 4,096) in steps 0–9; none after.

**Loss curves** (mean per 10 steps) all fall monotonically and then flatten. For example, Qwen3-1.7B goes
0.557, 0.493, 0.482, …, 0.460 and Gemma goes 1.254, 0.592, …, 0.517. None are non-finite.

**Old T3 estimates vs measured (tok/s, T4):** 0.6B 5,930 → **2,352**; 1.7B 2,050 (GC off) → **1,358**; 4B 590 (GC on)
→ **426**; Gemma 820 → **438**. MFU on T4 is ≈ 40–70 % of what T3 assumed.

## 2. Second-family candidates: details

- **Gemma-4-E2B on T4.**
  - vLLM 0.33 fails in two ways:
    - TRITON_ATTN is forced by the heterogeneous head dims (256/512). It needs 98 KB of shared memory and sm_75 has 64 KB.
    - FlexAttention raises "does not support kv sharing yet".
  - HF generate gives 52 tok/s, so 300 items take 9.5 min against about 1.5 min for vLLM models. That also blocks
    vLLM-based RL.
  - fp16 SFT is safe over 100 steps. This is shorter than T3 G1's 300 steps, but the max activation is 1,554 and there
    are no skips after calibration.
  - The conditional "Gemma bf16 on L4" job was therefore **not run**; it was not needed.
- **LFM2.5-2.6B** is the only candidate above Qwen3-1.7B (60.3 vs 58.7). It has the fastest vLLM eval of any
  model ≥1.7B and the largest fp16 margin. Risks:
  - Licence: LFM Open 1.0, with a $10M revenue cap (02_models).
  - A non-Transformer hybrid (short conv + GQA), so Unsloth support and packing are unverified.
- **SmolLM3-3B**: Apache-2.0 and very fp16-safe, but 8 points below LFM2.5-2.6B and it has no Vietnamese in its
  language list.
- **LFM2.5-1.2B** is below a useful floor (17.3 %), and its tokenizer makes Vietnamese 1.6× longer.

**Recommendation (light for speed, strong for quality).**
- **Strong / second family = LFM2.5-2.6B-Base.** It is 1.65× cheaper per run than Gemma-4-E2B on T4 (3.3 vs 5.45
  T4-h), +38 pt on 4-shot accuracy, and vLLM works.
- **Light:** no second-family model is both light and above the floor, so the light role stays with **Qwen3-0.6B**
  (primary family).
- **Licence fallback:** if the LFM licence is unacceptable, use SmolLM3-3B-Base (3.7 T4-h/run, 52 %).
- Gate before committing LFM2.5-2.6B: one 300-step fp16 run plus a dev eval. Also check whether Unsloth supports
  `Lfm2ForCausalLM` (if not, use the HF loop at 758 Qwen-eq tok/s).

## 3. Packing / padding-free: reconciled with O1

| Variant (Qwen3-0.6B, 4 items) | fp32 max Δlogit | fp16 max Δlogit | NLL item 2 (fp16) sep → variant |
|---|---|---|---|
| `position_ids`, model-default `use_cache` (W2 v1 call) | **22.95 ✗** | 22.98 | 0.875 → **3.703** (leak) |
| `position_ids`, `use_cache=False`, no mask | **3.4e-4 ✓** | 0.119 | 0.875 → 0.875 |
| explicit 4-D block-diagonal mask | 3.4e-4 ✓ | 0.119 | 0.875 → 0.875 |
| right-padded batch (control) | 3.6e-4 ✓ | 0.185 | 0.875 → 0.875 |

- **The v1 brief's "padding-free leaks under SDPA" was a misattribution.** It leaks even in fp32 and is fixed by
  `use_cache=False`. A DynamicCache disables transformers' packed-sequence detection. **This agrees with O1.**
- **The 1e-2 rule is only meaningful in fp32.** In fp16 the padded control itself differs from separate runs by
  0.12–0.18. Apply the gate in fp32, or on per-item ΔNLL (O1 uses <0.02).
- **HF packed is slower on T4** [M]: v2 gives 2,167 tok/s packed vs 2,352 padded and grouped (micro 4). Losses
  match per 10 steps to 3 decimals (0.6375/0.5780… vs 0.6375/0.5751…). O1 reports the same (937 vs 1,071 at 1.7B).
- **Unsloth packing.**
  - O1 drove `FastLanguageModel` from its own loop (`position_ids` + `packed_seq_lengths`) and measured it exact
    (ΔNLL ≤ 0.00065 on T4).
  - My test went through **TRL `SFTTrainer(padding_free=True)` (unsloth 2026.11.3, trl 1.17.1, SDPA on T4)**. Its
    collator **silently produced padded `[4, 337]` batches with an attention_mask**, so the packed path never ran.
  - Accordingly, the "padding_free" and "padded" 100-step runs are identical: 3,055 vs 3,025 tok/s, loss_by_10 equal
    to 4 decimals.
  - This is **not a contradiction** of O1. It is a tooling trap: do not rely on the trainer flag. Use O1's direct
    `FastLanguageModel` loop and assert that the batch is 1×L before training.
  - The by-product is that **Unsloth padded on T4 runs 3,040 tok/s at 0.6B**, 1.29× the HF loop (2,352). That is
    consistent with O1's 1.7×, which O1 measured at 1.7B.

## 4. fp16 (T4) vs bf16 (A10G) parity: Modal, same code, same seed

| Base | T4 fp16 dev300 | A10G bf16 dev300 | Δ (fp16−bf16) [paired 95% CI] | Discordant (fp16-only / bf16-only) | Loss last5 fp16 / bf16 |
|---|---|---|---|---|---|
| Qwen3-0.6B | 42.3 % | 44.0 % | −1.7 [−4.8, +1.5], McNemar p=0.41 | 9 / 14 | 0.5282 / 0.5289 |
| Qwen3-1.7B | 63.0 % | 62.7 % | +0.3 [−3.2, +3.9], p=1.0 | 15 / 14 | 0.4616 / 0.4624 |

- dev300 is the 300 Vi-GSM8K train items right after the 3,200 SFT items in the same shuffle. It is disjoint from
  the training data. The eval is 0-shot, greedy, on the merged adapter, with vLLM in the training precision.
- **Verdict.**
  - 1.7B passes the ±1 pt rule.
  - 0.6B's point estimate is −1.7, but that is not significant. The training losses are identical to 3 decimals.
  - **n=300 cannot decide a ±1 pt gate**, because the paired CI is ±3–4 pt. Y1's parity gate should use ≥1,000 dev
    items or 3 seeds.
- The "L4" request was served by **A10G** both times (L4 capacity was unavailable; fallback list). Both are Ampere-class
  bf16.

## 5. Confirmed package pins (for Y1)

**Kaggle T4×2 image (2026-11-07), verified in v1 and v2 (fail-fast pins on in v2):**

```
python 3.13.16 · driver 580.178.04 · CUDA 13.0 · torch 2.14.0+cu130 · triton 3.8.0
pip install 'vllm==0.33.0'            # 7.6 min per session; pulls the stack below
transformers 5.20.2 · peft 0.21.2 · accelerate 1.15.0 · datasets 4.8.5
pip uninstall -y torchao              # REQUIRED: image torchao 0.10.0 makes peft>=0.18 raise at LoRA dispatch
```

**Unsloth, isolated so torch and vLLM stay untouched:**

```
pip install --no-deps --target /tmp/uns_site unsloth==2026.11.3 unsloth_zoo==2026.11.3 trl==1.17.1 \
    cut_cross_entropy==25.1.1 bitsandbytes==0.50.2      # bitsandbytes is REQUIRED (raised inside from_pretrained)
PYTHONPATH=/tmp/uns_site  (only for the Unsloth process)
```

- This combination imports and trains on transformers 5.20.2 [M].
- O1's Unsloth image instead resolved torch 2.14.1 + transformers 5.17.0. **Y1 must pick one lock.** O1's exactness
  result is on 5.17.0, so either use that lock or rerun O1's logits gate on 5.20.2. That is 1 min.

**Modal image** (`modal_s2.py`): python 3.12 with the same four pins. Set env `VLLM_USE_FLASHINFER_SAMPLER=0` (§7).
Gemma-4 in transformers needs the text-only class with `key_mapping`. vLLM ≥ 0.33 does not support Gemma-4 on sm_75.

## 6. Platform cost per 9M-token run, and the re-estimated SFT grid

Measured with the HF loop (Unsloth would improve all rows). $/run excludes the ≈1–2 min cold start (≈ $0.03–0.07).

| Base | Kaggle T4 (fp16) | Modal T4 $0.59 | A10G $1.10 (bf16) | L40S $1.95 (bf16, GC off) | H100 $3.95 |
|---|---|---|---|---|---|
| 0.6B | 2,352 tok/s → 1.06 T4-h (0.53 quota-h on 2 lanes), $0 | — | 4,789 → $0.57 | 7,802 → $0.62 | — |
| 1.7B | 1,358 → 1.84 T4-h (0.92 quota-h), $0 | 1,534 → $0.96 | 3,032 → $0.91 | 6,999 → $0.70 | — |
| 4B | 426 → 5.87 T4-h (2.94 quota-h), $0 | — | 1,039 (GC on) → $2.65 | **3,969 → $1.23** | 5,856 → $1.69 |

**SFT grid (T3 §5.2 run counts; a run = 9M Qwen-eq tokens):**

| Block | Runs | T4-h/run [M] | T4-h | Note |
|---|---|---|---|---|
| Qwen3-0.6B | 30 | 1.06 | 32 | T3 had 11 (optimised) |
| Qwen3-1.7B | 92 | 1.84 | 169 | T3 had 136 |
| Qwen3-4B | 30 | 5.87 | 176 | T3 had 110. **Alternative: L40S at 30 × $1.23 ≈ $37** (≈ 2.2 months of the $17/month training cap; with Unsloth ≈ $20–25 [EST]) |
| Second family LFM2.5-2.6B | 78 | 3.30 | 257 | Gemma-4-E2B would be 425 (plus 9× slower eval). SmolLM3 290 |
| **Total (HF loop)** | | | **634 T4-h ≈ 317 Kaggle quota-h ≈ 10.6 weeks** at 30 h/week, since T4×2 is billed 1× | T3 had 440 "optimised" |
| With Unsloth on Qwen3 (×1.3–1.7) and 4B on L40S | | | **≈ 370–400 T4-h ≈ 6–7 weeks + ≈ $25–37** | [EST] |

## 7. Kaggle reliability (lessons for the orchestrator)

**Measured timeline.**

| Kernel | QUEUED | RUNNING | Quota Δ | Notes |
|---|---|---|---|---|
| s2-pilot-v1 | 21.5 min | 82 min | 0.70 → 2.07 h (**1.37 h = run time**) | pushed 09:23:52Z, RUNNING ≈ 09:45Z, COMPLETE 11:07:34Z |
| s2-pilot-v2 | 12.4 min | 27.7 min | 2.07 → 2.52 h (0.45 h) | `run_id` `s2-pilot-v2-20261107T111006Z` verified on pull |

**Rules, all implemented in the code:**

1. **Billing: T4×2 costs 1 quota-hour per wall-hour.** QUEUED time is free. Two lanes therefore halve the quota cost
   of every per-GPU job. Settled by the 90 s quota timeline: 0.79 → 1.77 h over 59.2 min RUNNING.
2. **Queues are long and variable** (12–22 min here). `KaggleRunner.wait`:
   - polls every 30–90 s;
   - writes every transition plus quota snapshots to `jobs/logs/<slug>_timeline.jsonl`;
   - stops waiting after `max_queued_s` (default 60 min) or `max_wait_s`, returning `QUEUED_TOO_LONG:`/
     `TIMEOUT_WAITING:` rather than busy-waiting. It never cancels by itself.
3. **Stale outputs.** `kernels output` returns the latest *completed* version. `submit_pilot` injects a unique `run_id`
   into JOB, and the kernel echoes it in `results/meta.json`. `verify_output_run_id` checks it after the pull, with 4
   retries, and marks the run `STALE_OUTPUT` otherwise. Each push records `kernel_version` + `run_id` in
   `jobs/kernels/<slug>/last_push.json`. Use a new slug for a materially different job (s2-pilot-v1 → v2).
4. **Live logs are unreliable.** `kernels logs` returned nothing during the 8-min install phase, then lagged. Do not
   build progress or decisions on it. Use status, the timeline and the incremental files instead.
5. **Incremental results.**
   - Every task writes its own JSON on completion.
   - `results/tasks.jsonl` is appended after every task.
   - SFT writes `*.partial.json` every 10 steps.
   - An `ENVCHECK` block (GPUs, driver, python, torch/transformers/peft/vllm/…) goes into `meta.json` and the log
     before any GPU task.
   - A killed or timed-out kernel therefore still leaves data.
6. **Per-task time budgets enforced in the script.**
   - Each task is a subprocess in its own **process group**.
   - On timeout the whole group is SIGKILLed. vLLM's EngineCore is a grandchild and would otherwise keep the GPU.
   - Budgets are capped by the kernel deadline (`deadline_s` + grace).
   - The SFT loop also checks its budget (`S2_TASK_BUDGET_S`), stops early and still writes results.
7. **Pins + fail-fast.**
   - The JOB carries `python` and `pins`.
   - After install, a version mismatch, a failed install, or a failed peft LoRA smoke test makes `meta.fatal` explain it
     and exit 5 before any GPU task. v2 passed.
   - The image is Python 3.13 and its preinstalled packages drift (torchao 0.10.0 was the v1 blocker), so re-pin per
     week.
8. **Install time is a tax.** The vLLM install takes 7.6 min per session (≈ 0.13 quota-h). Caching wheels in a private
   dataset is the next saving [EST: 6–7 min per session].
9. **OOM probe bug (fixed).**
   - The micro-batch probe ran inline, so the failed attempt's graph stayed referenced. This caused cascading OOMs:
     Qwen3-0.6B fell to micro 1, 735 tok/s, in v1.
   - The probe now runs in its own function, with gc and `empty_cache` between sizes.
   - v1's 0.6B throughput is void; v2 is the valid figure. v1's 1.7B, Gemma, 4B and other numbers were not affected.

**Modal lessons (same session):**

1. **Crash loop, app `ap-9fjZI5SEmtD7QOrF1qMTF9`.**
   - `REPO = Path(__file__).resolve().parents[2]` ran at import time *inside the container*. There the file is
     `/root/modal_s2.py`, so this raised `IndexError: 2` during startup.
   - Startup failures are **not** covered by `retries=0`, so Modal kept rescheduling. It cost $0.013 and was stopped.
   - Fix: resolve local paths only under `modal.is_local()`. Pre-flight-test the import with
     `_TaskLifecycleManager._singleton` set.
2. **GPU fallback lists.** Modal 1.6.3 accepts `gpu=["L4","A10G","L40S"]` only in the decorator.
   `with_options(gpu=list)` raises `InvalidError`, so there is one decorated function per GPU spec. Request "L4"
   resolved to **A10G both times**, so record the actual GPU from `nvidia-smi` (each `ledger.json` does).
3. **vLLM bf16 on A10/L4 in `debian_slim`** died on "Could not find nvcc": the FlashInfer top-k/top-p sampler JIT
   compiles at warm-up. Set `VLLM_USE_FLASHINFER_SAMPLER=0` (or use a CUDA devel image).
4. **Persist intermediate artefacts.** Merged SFT models now go to a Volume (`s2-merged`), so a failed eval no longer
   forces retraining (that cost one rerun here).
5. **Bounded waits.** `call.get(timeout = plan timeout + 20 min)`, then `cancel()`. Every function has `retries=0` and a
   hard `timeout`.
6. **Log spend per app.** Use `modal billing report --for today --json`, filtered by app description. S2 total was
   $1.675.

## 8. Caveats and open items

- 100 steps is a smoke test, not G1's 300-step gate. Gemma, LFM2.5-2.6B and the Qwen3 parity should get one 300-step
  run each before the grid freezes.
- The HF-loop activation monitor covers decoder Linear inputs/outputs and layer outputs, not norms or attention
  scores. A clean run here is necessary but not sufficient.
- Unsloth with LFM2 is [UNV]. So is the second-family bf16 throughput on L40S (only Qwen3 was measured on Modal).
- The packcheck was done on Qwen3-0.6B only. Do not pack LFM2 (short-conv state) without its own check.
