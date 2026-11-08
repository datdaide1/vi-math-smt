# O1 — Throughput optimisation: measured levers, configs, revised GPU-hours (2026-11-07)

**Scope.** Optimisation is not a contribution of the paper. Its only job is to let the planned grid finish on free
compute without changing any result. So this report covers only the few levers that matter, and measures the top
ones on Modal. Tags:
- **[M]** measured here (command and file listed in §6);
- **[DOC]** vendor docs opened in this session;
- **[EST]** arithmetic from [M] numbers (±30 %);
- **[UNV]** not verified.

## 0. Bottom line

1. **Evaluation, not SFT, is now the biggest GPU cost, and T3 under-estimated it by ~10×.**
   - vLLM on T4 (fp16, `TRITON_ATTN`; FA2 is unavailable on sm 7.5) generates **341 tok/s** for Qwen3-1.7B under our eval
     workload. On L4 (bf16, `FLASH_ATTN`) it generates **2,151 tok/s**: 6.3× faster at 1.36× the price [M].
   - One checkpoint is ~2.1 M generated tokens (2k items × greedy + K=4 × ~210 tok). That costs **1.7 T4-h, versus
     0.32 L4-h ≈ $0.26**.
   - **→ Move the reported checkpoint evaluation to Modal L4** (≈$64 for 260 checkpoints, about 2 months of the whole
     $30 credit), or cut tokens per checkpoint (§3.2).
2. **SFT: switch the training forward/backward to Unsloth (2026.11.3). It is the largest SFT lever, and the result
   is exact.**
   - On T4 fp16, Unsloth packed runs at **1,865 tok/s** and Unsloth padded (mb 4) at 1,792 tok/s, against **1,071**
     for the best HF configuration. That is **1.74×** [M].
   - On L4 bf16 it runs at **3,587 tok/s** against 1,504 (**2.4×**) [M].
   - Peak memory is 4–5 GB instead of 10–17 GB.
   - Unsloth packing is exact: per-item ΔNLL ≤ 0.00065 on T4 fp16 and 0.0047 on L4. Its packed and padded losses
     agree to 4 decimals at every step [M].
3. **Packing in plain HF on T4: the pilot-v1 leak is explained and fixed, but it is not worth it on T4.**
   - The leak came from the cache, not from SDPA. With `use_cache=False`, `position_ids` reset per sequence and **no
     `attention_mask`**, transformers 5.20.2 builds a block-diagonal mask under SDPA. The result is exact: per-item
     NLL differs by ≤ 0.0006 (fp16 T4) and ≤ 0.004 (bf16 L4); argmax agreement is ≥ 99.5 % [M].
   - On T4, though, packed SDPA (937 tok/s) is **slower** than Y1's padded, length-grouped micro-batches of 4
     (1,071 tok/s) [M].
   - Without Unsloth: keep Y1's padded + within-step length grouping on T4, with micro-batch 4.
4. **HF-only fallback on L4/L40S (bf16): packing with FlexAttention.**
   - It runs at 2,131 tok/s, versus 1,504 tok/s padded-grouped (1.42×) and 981 tok/s padded-random (2.17×). The
     logits check shows it is exact [M].
   - FlexAttention does not run on T4 (Triton shared-memory limit) [M].
5. **Multi-LoRA in one vLLM engine is not a throughput lever.**
   - Four adapters batched together give 1,988 tok/s, the same as one adapter (1,961–1,979). One LoRA adapter
     costs ~8 % against the base model [M].
   - The real saving is **no engine reload per checkpoint**: 92–106 s cold and 44 s warm per load, for 1.28× on 4
     adapters [M].
   - Batch composition changes bf16 numerics. Greedy outputs of the same adapter were identical in only 54–60 % of
     items between the multi-adapter and one-adapter runs [M].
   - **→ Fix eval batch composition per run** (deterministic chunking, adapters never mixed in a batch) so that
     re-runs reproduce.
6. **API layer.** Use Claude **Batch** (−50 % on every token, and it stacks with caching) and **prompt caching** on the
   frozen system prompt (reads 0.1× on Haiku/Sonnet 5.5; 512-token minimum on the 5.5 models) [DOC]. Do **not** pack
   several items per call; the quality risk is documented (§2).

## 1. Measurements (Modal; Qwen3-1.7B-Base; train-split data only)

### 1.1 Setup

- Script: `bench/o1/` (`modal_bench*.py`, `sft_worker.py`, `eval_worker.py`).
- Libraries: torch 2.14.0, transformers 5.20.2, peft 0.21.2, vLLM 0.33.0. These are the same pins S2 uses.
- **SFT data:** 2,048 Vi-GSM8K-train + Vi-MATH-train items in the plain template: mean 275 tok, p50 232, p99 887.
- **SFT settings:** LoRA r=32/α=32, all-linear, fp32 adapters, 32 seq/step, grad-ckpt on, 6 timed steps after 1
  warm-up.
- **Eval workload:** the runner's own:
  - 256 train-split questions, 4-shot;
  - greedy n=1 + sampling n=4 (T=0.6, top-p 0.95, top-k 20, per-request seed);
  - max_tokens 1024, stop on `Bài toán:`;
  - the same as `evalrun/runner.py` except the token cap.

### 1.2 SFT throughput (real, non-pad tokens/s; fwd+bwd+AdamW)

| GPU / dtype | Variant | tok/s | Pad fraction | Peak GB | Packing exact? (max per-item ΔNLL) |
|---|---|---|---|---|---|
| L4 bf16 | padded, random order, mb 8 | 981 | 0.48 | 16.5 | — |
| L4 bf16 | **padded, length-grouped within step, mb 8 (Y1 default)** | **1,504** | 0.23 | 16.5 | — |
| L4 bf16 | packed, SDPA, `position_ids`, `use_cache=False`, no mask | 1,641 | 0 | 12.8 | ✅ 0.0039 |
| L4 bf16 | **packed, `flex_attention`** | **2,131** | 0 | 12.8 | ✅ 0.0052 |
| L4 bf16 | packed, `kernels-community/flash-attn2` | — (needs `kernels>=0.17.2,<0.18`; not retried) | | | |
| L4 bf16 | mb 16, or grad-ckpt off | OOM (full-logit HF loss) | | | |
| T4 fp16 | padded, grouped, mb 8 / random mb 8 / ckpt off | OOM (full-logit HF loss)¹ | | | |
| T4 fp16 | **padded, grouped, mb 4** | **1,071** | 0.11 | 10.1 | — |
| T4 fp16 | padded, grouped, mb 2 | 1,031 | 0.04 | 6.9 | — |
| T4 fp16 | packed SDPA, mb 8 | 937 | 0 | 12.8 | ✅ 0.00055 |
| T4 fp16 | packed SDPA, mb 16 | OOM | | | ✅ 0.00055 |
| T4 fp16 | packed `flex_attention` | fails: Triton "out of resource … Required 65536, Hardware limit 65536" | | | |
| L4 bf16 | **Unsloth 2026.11.3, padded grouped mb 8** | 1,948 (≈ 3,100 after compile²) | 0.23 | 4.7 | — |
| L4 bf16 | **Unsloth, packed mb 8** (`position_ids` + `packed_seq_lengths`) | **3,587** | 0 | 5.4 | ✅ 0.0047 |
| T4 fp16 | **Unsloth, padded grouped mb 4** | 1,792 | 0.11 | 4.2 | — |
| T4 fp16 | Unsloth, padded grouped mb 8 | 1,651 | 0.23 | 4.6 | — |
| T4 fp16 | **Unsloth, packed mb 8** | **1,865** | 0 | 5.4 | ✅ 0.00065 |

² Unsloth compiles each decoder block on the first padded shapes, so step 1 is slow. Packed rows run uncompiled.
The Unsloth image resolved torch 2.14.1 and transformers 5.17.0. xformers was ABI-mismatched, so PyTorch SDPA was used.

¹ The benchmark used HF's full-logit loss. Y1's `completion_nll_sum` projects only completion positions, so Y1 may
fit mb 8 on T4. Y1's OOM probe settles it, and the mb 4 vs mb 2 difference is only 4 %.

Losses are identical across padded and packed variants at each step to about 3 decimals (e.g. step 0: 0.5586 /
0.5597 / 0.5583). That is a second check that packing does not change the objective.

### 1.3 Evaluation throughput (vLLM 0.33, 256 prompts × 5 sequences)

| GPU | Mode | Gen tok/s | Gen s | Engine load s | Truncated at 1024 |
|---|---|---|---|---|---|
| T4 fp16 (`TRITON_ATTN`) | base, seeded | 341 | 792 | 104 | 80 / 1280 |
| T4 fp16 | base, no per-request seed | 338 | 810 | 122 | 87 / 1280 |
| T4 fp16 | `VLLM_ATTENTION_BACKEND=FLEX_ATTENTION` | 340 (env var ignored in 0.33: "Unknown vLLM environment variable") | 785 | 133 | 76 / 1280 |
| T4 fp16 | 4 adapters, one engine | > 1,100 s budget: killed | | | |
| L4 bf16 (`FLASH_ATTN`) | base (no LoRA) | **2,151** | 123 | 92 | 76 / 1280 |
| L4 bf16 | 1 adapter per engine (today's runner), × 2 | 1,979 / 1,961 | 138 / 134 | 106 cold / **44 warm** | 75–77 |
| L4 bf16 | 4 adapters in one engine (`max_loras=4`) | 1,988 | 551 | 106 | 347 / 5120 |

**What the eval numbers show:**
- **Per-request seeds cost nothing** (341 vs 338 tok/s), so keep them.
- **6 % of sequences hit the cap and account for ~29 % of generated tokens** (76 × 1024 of 264,888).
  - These are degenerate runs: few-shot continuations or repetition.
  - The runner's `max_new_tokens=2048` doubles the waste.
- **L4 needs `VLLM_USE_FLASHINFER_SAMPLER=0`** on a plain `debian_slim` image. Otherwise engine start fails with "Could
  not find nvcc" (FlashInfer JIT for top-k/top-p) [M].

## 2. Ranked levers

| # | Lever | Evidence | Speed-up for us | Quality risk | Adopt |
|---|---|---|---|---|---|
| 0 | **Unsloth kernels for SFT (and GRPO training)**, padded mb 4 or packed | [M] T4 1,865 vs 1,071; L4 3,587 vs 1,504; packing exact | **1.7× T4, 2.4× L4**; memory −60 % | low: exact packing; step-0 loss matches HF (0.5583–0.5592 vs 0.5597). Needs a 300-step same-init parity gate (dev acc ±1 pt) and a pinned stack | **yes, after the parity gate** |
| 1 | **Eval on Modal L4 instead of T4** | [M] 2,151 vs 341 tok/s | 6.3× per GPU-hour; 4.6× per $ vs Modal T4 | none (bf16 vs fp16 inference; one platform for all eval, per T6) | **yes** |
| 2 | **Eval token cap = 1024** (from 2048) + repetition flag | [M] 29 % of tokens are in capped sequences | ~1.3–1.5× fewer tokens | ~0: capped runaways have no final `\boxed{}`. Check once: accuracy@1024 = accuracy@2048 on 1 checkpoint | **yes**, after the 1-checkpoint check |
| 3 | **One engine per process, loop adapters/checkpoints via `LoRARequest`** (or `max_loras=4`), deterministic chunks | [M] load 92–106 s cold, 44 s warm; multi = same tok/s | 1.28× on 4 adapters; saves ~1.5 min per checkpoint | reproducibility only: fix composition | **yes** |
| 4 | **Padded + within-step length grouping** (T4 SFT) | [M] 1,504 vs 981 tok/s (L4, mb 8) | 1.53× vs random padding | none (same rows per step) | **yes** (already Y1) |
| 5 | Packed FlexAttention on L4/L40S/H100 SFT (HF fallback) | [M] 2,131 vs 1,504; logits exact | 1.42× | none if the logits gate passes per model | only if Unsloth fails its gate |
| 6 | Packed SDPA on T4 | [M] 937 vs 1,071 (mb 4 padded) | 0.87× (slower) | none | **no** |
| 7 | Claude Message Batches | [DOC] "50% off every token … including cache reads and writes" | −50 % $ | none (same model/params; ≤24 h latency) | **yes** for all non-interactive generation and judging |
| 8 | Claude prompt caching of the frozen system prompt + few-shot block | [DOC] read 0.1× (Haiku/Sonnet 5.5), 0.05× (Opus 5.5); write 1.25× (5 min) / 2× (1 h); minimum 512 tokens on 5.5 models; inside Batch, hits are "best-effort" | −(0.9 × prefix share) of input $ | none | **yes** |
| 9 | Structured outputs (`output_config.format`) for judges and formalisers | [DOC] schema-valid JSON | removes parse-failure retries (rate seen in G1 logs) | none | **yes** |
| 10 | Explicit `effort` per route (Haiku/Opus 5.5 default `medium`, Sonnet 5.5 `high`) | [DOC] | output tokens ↓ on judge routes | possible, so only after a judge-vs-human κ check | **yes, measured per route** |
| 11 | Several items per API call | Batch Prompting (2301.08721): up to 5× fewer tokens, "better or comparable" on short QA with Codex. BatchPrompt (2309.00384): longer batches "will inevitably lead to worse performance"; accuracy "significantly correlated with the positions" | only the shared prefix is saved, and caching already gives 0.1× on it | position bias in judges; cross-item contamination; one failure kills k items; breaks per-item cache and ledger | **no** |
| 12 | Gemini per-project token-rate shaping (Y3 AIMD lanes) | G1: 429s behave like a TPM limit | avoids 429 storms | none | **yes** (already in Y3) |
| 13 | Gemini Batch / context caching | [DOC] Batch 50 % price, 24 h, supports caching; free-tier availability not stated | $0 tier: no gain [UNV] | — | **no** |
| 14 | Multi-LoRA *training* (mLoRA/PLoRA), DDP over T4×2 | T3 §5.1 E2; adapters do not share base compute | ≤1.1× | none | **no**: 2 independent lanes instead |
| 15 | Liger / fused CE | Y1 already projects only completion positions | memory only | none | **no** |
| 16 | FP8/AWQ for eval | changes logits | — | **changes accuracy** | **no** |
| 17 | `torch.compile` of the training model, FA2 on T4 | T4 has no FA2 (sm 7.5) [M: vLLM log]; compile gain at small batch unmeasured | — | — | **no** |

## 3. Recommended configs

### 3.1 SFT

**Primary (all platforms):** Unsloth `FastLanguageModel`, driving Y1's own loop. This was measured exactly so.
```
pip: unsloth==2026.11.3 unsloth_zoo==2026.11.3 (resolves torch 2.14.1, transformers 5.17.0; freeze the lock)
FastLanguageModel.from_pretrained(name, max_seq_length=8192, dtype=fp16|bf16, load_in_4bit=False, full_finetuning=False)
FastLanguageModel.get_peft_model(m, r=32, lora_alpha=32, lora_dropout=0, bias="none", random_state=seed,
    target_modules=[q,k,v,o,gate,up,down]_proj, use_gradient_checkpointing="unsloth")
packed rows: input_ids (1,sum L), position_ids reset per row, labels -100 at each row start, packed_seq_lengths,
use_cache=False; micro-batch 8 rows (T4 peak 5.4 GB); 32 seq/step unchanged; start-of-job logits gate (dNLL<0.02)
```

**HF-only fallback, Kaggle T4 fp16 (Qwen3-0.6B/1.7B; 4B in DDP-2 only if a run would exceed 12 h):**
```
attn_implementation=sdpa   packing=False   length_grouping=within_step   micro_batch=4 (OOM-probe may halve)
grad_ckpt: off for 0.6B, on for >=1.5B   seq_per_step=32   fp16 autocast + GradScaler   use_cache=False
two independent jobs per T4x2 session (CUDA_VISIBLE_DEVICES=0 / 1)
```

**HF-only fallback, Modal L40S/L4/H100, bf16 (4B, Gemma-bf16, full-FT check):**
```
attn_implementation=flex_attention   packing=varlen (rows of one micro-batch concatenated, position_ids reset,
attention_mask=None, use_cache=False)   micro_batch=8 rows   gate: per-model logits check (max per-item ΔNLL<0.02,
argmax agree>0.97) at job start, abort on fail
image env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ; weights on a Volume ; save adapters only
```

### 3.2 Evaluation (all checkpoints; Modal L4, bf16)

```python
LLM(model=base, dtype="bfloat16", enable_lora=True, max_lora_rank=32, max_loras=1,   # loop adapters, 1 engine
    gpu_memory_utilization=0.85, max_model_len=4096, seed=0, enable_prefix_caching=True)  # V1 default
SamplingParams(n=4, temperature=0.6, top_p=0.95, top_k=20, max_tokens=1024, seed=<per item>, stop=STOP_STRINGS)
# greedy pass: keep only if the analysis uses it (T5 primary metric is K=4 sampling)
# env: VLLM_USE_FLASHINFER_SAMPLER=0 (plain image) ; one container evaluates a whole block of checkpoints
```

**Settings:**
- Chunks are fixed per (run, dataset): `chunk_items=256` and the same ordering, which keeps re-runs bit-stable on the
  same hardware.
- Never mix two adapters in one batch for reported numbers.
- Merged-weights evaluation costs the same as LoRA (base 2,151 vs LoRA ~1,970 tok/s, plus merge time). Use LoRA and
  skip the merge.

### 3.3 GRPO (TRL GRPOTrainer, from T3 §4.1; unchanged science)

```
use_vllm=True  vllm_mode="colocate"  vllm_gpu_memory_utilization=0.35  vllm_enable_sleep_mode=True (memory only)
num_generations=8  max_completion_length=512  mask_truncated_completions=True  generation_batch_size=256
vllm_importance_sampling_correction=True (default, sequence_mask)   precision fp16 on T4 / bf16 on L4 per block
```

**Where to run it:**
- Rollout generation dominates.
  - At the measured T4 generation rate, a 200-step run (51,200 completions × ~250 tok ≈ 12.8 M tokens) needs ≈ 10–12
    T4-h of generation plus ≈ 5 T4-h of training (≈ 3 with Unsloth) ≈ **14–17 T4-h/run** [EST]. T3 had 7.
  - On L4 it needs ≈ 2 h generation + 3.6 h training ≈ **5.5 L4-h ≈ $4.4/run** [EST].
- **Run GRPO at 0.6B on T4, or reduce the number of RL runs.** That decision belongs to the lead, not O1.

### 3.4 API layer (generation, judging)

**Claude (all bulk work):**
- Use `messages.batches` and set `cache_control` on the frozen system prompt and few-shot block (≥ 512 tokens on 5.5
  models).
- Set `output_config.format` (JSON schema) on every judge, formaliser and validator.
- Set `output_config.effort` explicitly per route: `low` for binary judges, `medium` for generators. Keep a route at
  that level only if κ against the human sample holds.
- Use streaming for interactive calls with `max_tokens` > 16k only.
- One item per request.

**Gemini (train/dev only):**
- One lane per project, with the AIMD TPM that Y3 already has.
- Set `maxOutputTokens` to the route's real need. Long outputs are what triggered the 429s.
- Send long-output generation to Claude Batch and keep Gemini for short judge votes.

## 4. Revised GPU-hours (Qwen3 0.6/1.7/4B; same run counts as T3 §5.2; [EST] from [M])

**Inputs:**
- Mean training item 275 tok (measured mix; T3 assumed 450).
- 1.7B on T4: 1,071 tok/s with HF [M], 1,800 with Unsloth [M].
- 0.6B and 4B are scaled by FLOPs: 0.6B ≈ 2,500 (HF) / 4,300 (Unsloth) tok/s, 4B ≈ 430 / 770 tok/s. S2's
  measurements should replace these.
- Eval: 260 checkpoints × 2k items.

| Block | Tokens | T4-h (HF → **Unsloth**) | Modal alternative |
|---|---|---|---|
| SFT 0.6B (30 runs) | 143 M | 16 → **9** | — |
| SFT 1.7B (92 runs) | 410 M | 106 → **63** | — |
| SFT 4B (30 runs) | 143 M | 92 → **52** | L40S packed-flex ≈ 17–25 h ≈ $34–49 (over the $17 cap: keep on T4, DDP-2 for long runs) |
| SFT Gemma-4-E2B (78 runs, if kept) | 334 M | ≈ 140 → ≈ 70 (fp16 gate) | L4 bf16 packed ≈ 60 h ≈ $48 |
| **Eval**, today's runner (greedy + K4, cap 2048) | ≈ 2.4 M/ckpt | 510 | L4 ≈ 90 h ≈ $72 |
| **Eval**, recommended (K4 + greedy, cap 1024, 1 engine) | 2.1 M/ckpt | 445 | **L4 ≈ 80 h ≈ $64** |
| Eval, if the greedy pass is dropped | 1.7 M/ckpt | 360 | L4 ≈ 62 h ≈ $50 |
| GRPO 30 runs at 1.7B | 12.8 M gen + 20 M train each | ≈ 510 → ≈ 410 | L4 ≈ 165 h ≈ $130 |
| GRPO 30 runs at 0.6B | same | ≈ 200 | — |

**Calendar** (Kaggle 2 lanes × 30 h/week ≈ 60 T4-h/week; Modal ≈ $24/month usable after reserve):
- **SFT, Qwen3 only:** 214 T4-h with HF ≈ 3.6 weeks, or **124 T4-h with Unsloth ≈ 2.1 weeks**. Gemma on T4 adds
  ~1.2 weeks.
- **Eval on L4:** $64 ≈ 2 months of the full credit. On Kaggle it would be 445 T4-h ≈ 7.4 weeks, more than all of
  SFT.
  - **Combine levers:** cap 1024, a single greedy-or-sampled design, and evaluating only the final adapter of
    data-efficiency runs (T3 already reuses the core 10k points).
  - **Then split eval:** the ~120 core checkpoints on L4 (~$31), the rest on Kaggle (~200 T4-h).
- **GRPO at 0.6B on Kaggle:** ≈ 3.3 weeks.
- **Total with Unsloth ≈ 8.7 Kaggle weeks** (Qwen3 only: 124 SFT + ~200 eval + ~195 GRPO-0.6B T4-h), ≈ 9.9 with
  Gemma on T4, **+ ≈ $30 Modal for the core eval**. T3 said 9–11 weeks but missed ~10× of the eval cost. GRPO at
  1.7B on T4 would add ~3.5 weeks. The difference is almost entirely eval
  and GRPO generation on T4.
- The English mirror axis is **not** included. Every mirror run adds the same per-run costs.

## 5. Hand-offs (O1 did not edit these files)

### Y1 (`vi_math_verified_aug/train/`, `evalrun/`)

0. **`modeling.load_model`:** add an `engine="unsloth"` path (`FastLanguageModel` + `get_peft_model` as in §3.1) and
   keep the custom loop.
   - Packed micro-batches pass `position_ids` and `packed_seq_lengths`, with `attention_mask=None`.
   - Run the S2-style parity gate once per base: same seed, 300 steps, HF vs Unsloth, dev accuracy ±1 pt and loss
     curves overlapping.
   - Pin the Unsloth-resolved stack in the Kaggle wheels dataset.
   - `completion_nll_sum` can stay. Unsloth's patched decoder returns `last_hidden_state` as usual. **[UNV]**: the
     benchmark used `out.loss`.

1. `config.py`: keep `packing=False` as the T4 default. Fix the comment: the leak is caused by `use_cache` defaulting
   to True (DynamicCache disables packed-sequence detection), not by SDPA.
2. Add `packing="varlen"` for bf16 platforms only:
   - `attn_implementation="flex_attention"`, `position_ids` reset per row, `attention_mask=None`, `use_cache=False`;
   - in `modeling.completion_nll_sum`, pass `position_ids` and `attention_mask=None` to `dec(...)`;
   - a mandatory start-of-job logits gate (separate vs packed, 4 items; ΔNLL < 0.02).
3. `sft.micro_batch`: default 4 on T4 (8 OOMs with full logits; Y1's selective projection may allow 8, so let the
   probe decide).
4. `evalrun/runner.py`:
   - (a) `max_new_tokens` 2048 → 1024, after a 1-checkpoint equality check;
   - (b) evaluate many adapters per engine: accept a list of `adapter_path`s and reuse the `LLM` with one
     `LoRARequest(name, i, path)` per adapter;
   - (c) record vLLM version, backend, and `chunk_items` in the manifest (batch composition affects greedy text);
   - (d) the platform for reported eval = Modal L4 bf16, with `VLLM_USE_FLASHINFER_SAMPLER=0` in the image.
5. `grpo.py`: `vllm_enable_sleep_mode=True` on T4, and a measured tok/s log line per step for re-planning.

### Y3 (`common/llm.py`)

1. **Batch:**
   - `batches.run` should be the default path for any request list with `split in {train, dev}` and no latency need.
   - Group requests by identical `system` so that cache prefixes match.
2. **Caching:** add `cache_control={"type":"ephemeral"}` to the system block when it is ≥ 512 tokens (5.5 models).
   Log `cache_read_input_tokens` and `cache_creation_input_tokens` (already in usage). Alert if the read share is
   < 50 % on a route with a fixed system prompt.
3. **Structured outputs:** pass `output_config.format` through `LLMRequest.extra`. Make `parse_json` failures count as
   a `content` error class, so retries become visible.
4. **Effort:** require an explicit `output_config.effort` per route. Defaults differ by model (Haiku/Opus 5.5
   `medium`).
5. **No multi-item packing** in prompts or requests.
6. **Gemini:** cap `maxOutputTokens` per route. Keep one lane per project.

## 6. Evidence

**Spend** (`jobs/logs/O1_modal_billing_before.txt`, `O1_modal_billing_after.txt`, `O1_modal_billing_report_today.json`):
- The workspace summary went from $0.06 to $4.77 metered, but it includes S2/Y2/X5 apps that ran at the same time.
- Per-app report, O1 apps (`o1-throughput-bench*`): **$1.62** of the $4 cap.
- Lost runs: L4 capacity waits (~15 min queue), one FlashInfer failure, and the T4 multi-adapter run killed at its
  budget.

**Runs:**
```
modal run bench/o1/modal_bench.py --plans sft_l4,eval_l4,sft_t4,eval_t4,unsloth_l4   -> bench/o1/run1.log
modal run bench/o1/modal_bench_eval2.py --plans eval_l4      (FlashInfer sampler off) -> bench/o1/run2.log
modal run bench/o1/modal_bench_eval3.py --plans seedtest_t4 | sft_t4b | flex_t4 | unsloth_t4 -> run3/4/5/6.log
```

**Results:**
- Per-variant JSON and logs: `bench/o1/results/<plan>[_v2|_v3]/`.
- The T4 base-eval JSON was pulled from the live container before its function timed out:
  `bench/o1/eval_t4_base_pulled.json` (summary) and `modal container exec … summ.py`.
- Data bundle: `bench/o1/make_data.py` → `bench/o1/data/` (train splits only: Vi-GSM8K train, Vi-MATH train). No
  test item left the laptop.

**Caveats:**
- These are single runs, with 6 timed steps per SFT variant (step time CV ≈ 10–15 %).
- Throughput was measured on Modal T4, which may differ from Kaggle T4 (different CPU, same GPU). The pilot-v1 Kaggle
  eval (greedy, 400-token cap) implies ~400–500 tok/s, so it is consistent.
- The multi-adapter vs one-adapter greedy agreement (54–60 %) uses random-init adapters. Accuracy equality was not
  measured (random adapters have no meaningful accuracy).
