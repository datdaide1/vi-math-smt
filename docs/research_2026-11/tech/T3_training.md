# T3 — Training algorithms and compute efficiency for 0.5–4B Vietnamese math students (2026-11-07)

**Scope.** This report covers the SFT objective, distillation, RLVR, LoRA variants, merging, and the techniques that cut
wall-clock time without changing the science. It ends with ONE SFT recipe, ONE RL stage, and an efficiency plan for
~230 SFT + ~30 RL runs.

**Evidence rules (BRIEF.md).**
- Every claim carries a source that was opened (arXiv abstract/HTML or docs page) or a script that was run.
- **[SEARCH]** means the number comes from a search-result snippet and the primary page was *not* opened.
- **[UNVERIFIED]** means the claim is from memory or was not checked.
- **[ESTIMATE]** means arithmetic. The script is `scratchpad/t3/t4_estimate.py` and its output is `t4_estimate_out.txt`.
  Treat every estimate as ±2×.

---

## 0. Bottom line

### 0.1 Hardware correction

The coordinator measured `modal billing rates` on the user's account. The real price is **L4 $0.80/h**, not the
$0.222/h in `03_training_infra.md`, so **$30/month buys only about 37 L4-hours**.

**The main training GPU is therefore the Kaggle T4.** Its constraints are:
- **fp16 only.** It has no bf16, no FP8 and no FlashAttention-2. FA2 needs sm ≥ 8.0
  ([vLLM backend table](https://docs.vllm.ai/en/latest/design/attention_backends/)).
- **16 GB** of memory.
- About **30 h/week** of quota and **12-h sessions** [SEARCH].

Everything below is optimised for that. Modal is reserved for bf16 parity checks and overflow work.

### 0.2 SFT recipe

**Completion-only token-mean NLL** on a plain-text base-model template, with these settings:
- **fp16** LoRA r=32, α=32, all linear layers. Adapters are kept in fp32.
- LR swept over {1e-4, 2e-4, 4e-4} on arm S0 using **WSD epoch branches**, then frozen.
- 32 sequences per step.
- **Padding-free varlen batching**, not bin-packing.
- **Matched items** across arms, with token counts reported.

**Why not DFT or the other reweighted objectives (DFT/InfoSFT/ASFT).**
- They help strong priors: Qwen2.5-Math-1.5B MATH-500 went from 43.8 to 64.9.
- They lose at the **model-weak end** ([2510.00526](https://arxiv.org/abs/2510.00526)). Small bases on Vietnamese
  are at that end.
- They down-weight exactly the low-probability tokens that carry the differences between arms. That compresses the
  very effect C2 measures.
- DFT is kept as one **ablation**.

### 0.3 RL stage

**GRPO with the Dr.GRPO loss** (TRL/Unsloth), set up as follows:
- `beta=0`, `scale_rewards=False`.
- G=8, 32 prompts per step, 200 steps, completions ≤512 tokens.
- Binary `math_verify` reward with **no format reward**.
- A fixed mid-difficulty prompt pool.
- **fp16 end-to-end.** The 2025 FP16-vs-BF16 RL paper shows fp16 *removes* the train/inference mismatch and
  stabilises LoRA-GRPO ([2510.26788](https://arxiv.org/html/2510.26788)), so the T4's fp16-only limitation becomes
  a strength here.
- A **random-reward control** × 3 seeds is mandatory.

### 0.4 Students on T4

- **Qwen3-Base (0.6/1.7/4B) is the only family whose whole pipeline is known to run on T4:** fp16 training,
  padding-free attention, and vLLM for evaluation and RL.
- **Qwen3.5** (hybrid Gated-DeltaNet) has three problems:
  - The native GDN layer **does not reset state at packed boundaries**
    ([Twinkle docs](https://twinkle-kit.readthedocs.io/en/latest/Components/Training%20Middleware/Padding-Free.html)).
  - Its Triton kernels compile slowly "especially on T4".
  - Unsloth RL runs it without vLLM ([Unsloth guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)).
- **Gemma-4-E2B** has **documented fp16 overflows** on T4 ([Unsloth Gemma-4 guide](https://unsloth.ai/docs/models/gemma-4/train)).
  It is also ~3 T4-h per run.
- **Both are gated by the day-1 tests in §6.** They are not assumed.

### 0.5 Compute and time

The optimised plan needs **≈ 440 T4-h of SFT + ≈ 30 T4-h of evaluation + 85–210 T4-h of RL**. At ~60 GPU-h per week
(T4×2 run as two job lanes) that is **8–11 Kaggle weeks**.

- The optimisations in §5 save **≈ 1.9×** versus a naive padded setup.
- **"One day" is not reachable on free compute.** The single biggest lever is the choice of second family.
  Gemma-4-E2B alone is ~45% of the SFT hours.

---

## 1. Hardware reality: what each candidate student can do on a T4

### 1.1 T4 platform facts

| Item | Fact | Source |
|---|---|---|
| Precision | fp16 tensor cores (65 TFLOPS dense). No bf16, no FP8 | NVIDIA T4 page (via 03); vLLM backend table |
| Attention kernels | **FA2 is excluded** (needs ≥ 8.0). `TRITON_ATTN` and `FLEX_ATTENTION` are listed for "Any" compute capability with fp16 | [vLLM attention backends](https://docs.vllm.ai/en/latest/design/attention_backends/) |
| Unsloth | Minimum compute capability 7.0, T4 listed. Padding-free and packing are claimed to work with SDPA/xFormers on T4 | [Unsloth packing blog](https://unsloth.ai/docs/blog/3x-faster-training-packing); [SEARCH] README |
| fp16 overflow risk | Gemma 3 activations exceed 65504 in fp16 (Unsloth). Gemma 4's audio mask uses `-1e9`, which overflows on T4. Unsloth staging PRs clamp the Gemma-4 MLP for fp16 GRPO [SEARCH]. Qwen3-Embedding-8B gives NaN in fp16 (hidden states > fp16 max) [SEARCH] | [Unsloth Gemma-4 guide](https://unsloth.ai/docs/models/gemma-4/train), search results |
| Kaggle quota | ~30 h/week, floating; 12-h sessions; whether T4×2 is billed as one hour or two is **[UNVERIFIED]**. Check the sidebar counter in the first session | [SEARCH] gpuperhour 2026-09 review |

### 1.2 Throughput per T4 per student [ESTIMATE]

**Inputs.** LoRA forward+backward costs 4N per token, or 6N with gradient checkpointing, ×1.15 for attention. MFU is
0.25 for dense Qwen3/Gemma and 0.18 for the GDN hybrids (Triton on sm_75, untuned). Qwen3.5 counts include the ViT, so
they are conservative. A "run" is 10k items × 450 tokens × 2 epochs = 9M tokens.

| Student | tok/s (ckpt on / off) | M tokens per T4-h | T4-h per 9M-token run (ckpt on / off) | Fits 2 jobs per 16 GB? | T4 blockers | Verdict for a T4 grid |
|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 3,950 / 5,930 | 14.2 / 21.3 | 0.63 / 0.42 | yes | none known | ✅ |
| Qwen3.5-0.8B-Base | 1,940 / 2,910 | 7.0 / 10.5 | 1.29 / 0.86 | yes (3 GB LoRA, Unsloth) | GDN packing boundaries; Triton sm_75 untested; RL without vLLM | gate G2/G3 |
| Qwen3-1.7B-Base | 1,370 / 2,050 | 4.9 / 7.4 | 1.83 / 1.22 | marginal | none known | ✅ |
| Qwen3.5-2B-Base | 750 / 1,120 | 2.7 / 4.0 | 3.35 / 2.23 | no | as for 0.8B | gate |
| Gemma-4-E2B | 820 / 1,230 | 2.9 / 4.4 | 3.05 / 2.03 | no (8–10 GB LoRA) | **fp16 overflow** (documented); no vLLM in Unsloth RL | gate G1 |
| Qwen3-4B-Base | 590 / 880 | 2.1 / 3.2 | 4.27 / 2.85 | no | none known; use DDP-2 to stay inside a 12-h session | ✅ |
| Qwen3.5-4B-Base | 360 / 550 | 1.3 / 2.0 | 6.87 / 4.58 | no (10 GB) | as for 0.8B | gate; expensive |

**Consequences.**
- On a T4, Qwen3.5-2B costs about **1.8× Qwen3-1.7B** per run, and Gemma-4-E2B about **1.7×**.
- Measure real tok/s in the first session and replace this table. MFU on T4 is the largest unknown.

---

## 2. Evidence table: methods for ≤4B math reasoning (2025–2026)

| # | Method | Paper / date | Gain at small scale (as reported) | Compute | Relevance |
|---|---|---|---|---|---|
| S1 | Completion-only NLL SFT on teacher CoT | R1-Distill-1.5B (via 03); [TRL SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer) | The strongest single lever ≤2B (MATH-500 83.9 with 800k samples) | linear in tokens | **Baseline objective** |
| S2 | **DFT**: loss = −sg(p)·log p per token | [2508.05629](https://arxiv.org/html/2508.05629v3), ICLR 2026 | Qwen2.5-Math-1.5B avg 15.9 → SFT 18.0 → **DFT 31.6**; MATH-500 43.8 → **64.9**. LLaMA-3.2-3B only 3.2 → 4.7. **Hurts knowledge:** NQ SFT 36.6 vs DFT 30.1. Full-FT LR 5e-5, batch 256 | same as SFT | Big gains *only* for math-pretrained Qwen. See S3–S4 |
| S3 | Probability-based objectives across capability | "Beyond Log Likelihood" [2510.00526](https://arxiv.org/abs/2510.00526), ICML 2026 | "toward the model-weak end, NLL dominates". Prior-leaning objectives (−p, DFT-like) win only at the strong end (8 backbones, 27 benchmarks) | — | **Vietnamese small bases are weak-end, so keep NLL** |
| S4 | DFT limits on hard data | [SEARCH] 2605.12906 | DFT "saturates more quickly as data size increases and is eventually surpassed by SFT" | — | Supports S3 |
| S5 | Newer token reweighting (InfoSFT, RankTuner, VCORE, ASFT) | [2605.14967](https://arxiv.org/abs/2605.14967); [SEARCH] 2602.01745, 2510.27462, 2509.23753 | InfoSFT +6 AMC over SFT/DFT on Qwen-1.5B [SEARCH]. RankTuner +1.5 MATH over DFT on Qwen3-8B [SEARCH] | one-line changes | Not stable across settings. Ablation only |
| S6 | **Critique FT (CFT)** | [2501.17703](https://arxiv.org/abs/2501.17703); one-shot CFT [2506.03295](https://arxiv.org/html/2506.03295v2) | +4–10% over SFT (Qwen2.5/-Math, DeepSeek-Math) [SEARCH]. One-shot CFT: Qwen2.5-Math-7B 27 → 42% in 5 GPU-h | needs teacher critiques | It is a *data* format, so it would be an arm, not the objective |
| S7 | **NFT** (negative-aware FT) | [2505.18116](https://arxiv.org/abs/2505.18116) | ≈ GRPO/DAPO, beats RFT. Equivalent to GRPO under strict on-policy training. 7B/32B only | needs on-policy sampling, so RL-cost | Same cost as RL; no ≤4B evidence |
| S8 | Short vs long CoT for ≤3B | learnability gap [2502.12143](https://arxiv.org/abs/2502.12143) (via 03) | ≤3B gain more from short CoT; mix if long is used | long CoT is ~5× tokens | **Short CoT** |
| S9 | Oracle-refined vs self-generated traces | "Quality-Utility Paradox" [2606.16152](https://arxiv.org/abs/2606.16152), ICML 2026 | Oracle-refined traces "consistently underperform" SLM self-generated rejection-sampled traces (distribution drift). No numbers on the abstract page | — | A threat to the teacher-generated arms. Report it as a finding, not a recipe change |
| D1 | Off-policy SFT on API teacher (SeqKD) | standard; GAD paper uses it as baseline | — | cheapest | **The only option with a NIM/API teacher**, which returns no logits |
| D2 | Black-box on-policy distillation (GAD) | [2511.10643](https://arxiv.org/abs/2511.10643) | Qwen2.5-3B (GAD) ≈ 7B (SeqKD) on **chat** evals (GPT-4o judge). No math reported | discriminator + GRPO, so RL-cost | Not proven for math → reject |
| D3 | On-policy distillation (reverse KL, logits) | [Thinking Machines OPD](https://thinkingmachines.ai/blog/on-policy-distillation/) (via 03); EOPD [2603.07079](https://arxiv.org/abs/2603.07079) | EOPD over OPD: Qwen3-0.6B-Base +1.16 Avg@8, 1.7B +0.99 (six math benchmarks) [SEARCH] | teacher forward per student token; same tokenizer, or GOLD/ULD (experimental) | Possible with an **open same-family teacher** (e.g. Qwen3-4B → 0.6B on one T4). Hand-off "best model" only |
| R1 | GRPO / Dr.GRPO / DAPO / SAPO / CISPO / GSPO | [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) | TRL default `loss_type="dapo"`, `beta=0`, G=8, TIS/MIS vLLM correction on by default | — | All are one flag; pick by bias argument |
| R2 | **FP16 for RL** | [2510.26788](https://arxiv.org/html/2510.26788) | Mismatch ~24× smaller. **LoRA-GRPO on Qwen2.5-Math-1.5B: bf16 collapses ≈ step 600, fp16 stable** (r=32, α=64, LR 4e-5). Sanity set: 95 → 99% training accuracy, AIME24 34 → 39 | free | **Exactly the T4 regime** |
| R3 | Zero-RL on small bases | SimpleRL-Zoo [2503.18892](https://arxiv.org/abs/2503.18892) | Qwen2.5-0.5B/1.5B needed *simpler prompts*. **Format rewards hurt weak bases.** Data difficulty must match capability | — | No format reward; prompt pool matched to the SFT model |
| R4 | Difficulty for ≤3B | [2604.06298](https://arxiv.org/abs/2604.06298) | GRPO+LoRA ≤3B: lower-difficulty subset "matches full-dataset accuracy" in ~45% of steps; capacity plateau on hard tiers | — | Cheap pool |
| R5 | Online difficulty filtering | [2504.03380](https://arxiv.org/abs/2504.03380) | Intermediate-difficulty prompts: "up to +12% in less than half the training steps" | — | Filter the pool to pass rate ∈ (0,1) |
| R6 | REINFORCE++ / RLOO / group size | [SEARCH] 2501.03262; 2-GRPO study | REINFORCE++ ≈ GRPO with rule rewards (its authors). G=2…16 shows no monotonic trend at 1.5B [SEARCH] | — | No reason to leave GRPO |
| R7 | LoRA for RL | LoRA Without Regret (via 03); TinyLoRA [2602.04118](https://arxiv.org/abs/2602.04118) [SEARCH] | RL matches full FT even at rank 1. 13 parameters lift Qwen2.5-7B-Instruct to 91% GSM8K, *but* RL mostly elicits | — | LoRA r=32 is ample. Elicitation confound → controls |
| R8 | Spurious rewards / contamination | [2506.10947](https://arxiv.org/abs/2506.10947) (ICML 2026); [2507.10532](https://arxiv.org/html/2507.10532v3); [SEARCH] 2512.16912 | Random reward +21.4 on Qwen2.5-Math-7B MATH-500, not on Llama/OLMo. Qwen2.5 MATH-500 is contaminated (60% prefix → 54.6% exact completion). Mechanism disputed (clipping vs entropy) | — | **Random-reward control** + temporal-split Vi-ExamMath. No Qwen3 replication exists [SEARCH] |
| R9 | Evaluation variance | "Sober look" [2504.07086](https://arxiv.org/abs/2504.07086) | RL gains "far below prior claims". SFT generalises more consistently. Results are sensitive to seeds, decoding and hardware | — | Fix the decoding config; ≥3 seeds; bootstrap CIs |
| L1 | LoRA variants (DoRA, PiSSA, MiLoRA, …) | [SEARCH] "LR Matters" 2602.04998 (ICML 2026) | After per-method LR tuning, all within 0.43% at r=128. PiSSA's optimum LR is ~10× lower than LoRA's | DoRA is slower [UNVERIFIED] | **Vanilla LoRA + LR sweep** |
| L2 | LoRA vs full FT | LoRA Without Regret (via 03) | Matches full FT on small/medium SFT. All-linear ≫ attention-only. LR ≈ 10× full-FT. Less tolerant of big batches | 2/3 of full-FT FLOPs | Sets targets, LR range and batch |
| M1 | Checkpoint / weight averaging | [SEARCH] 2504.18580; model soups 2203.05482; Model Stock 2403.19522 | LoRA math checkpoint merge +1.8 pts over the final checkpoint (GSM8K). Seed soups are vision-only evidence | eval-only cost | Hand-off model only, never in arm comparisons |
| E1 | Padding-free / packing | [Unsloth blog](https://unsloth.ai/docs/blog/3x-faster-training-packing) | Padding-free 1.1–2× with "0% accuracy change". Packing 1.7–3× but changes the loss and row count | — | Padding-free yes; bin-packing no (§5) |
| E2 | Multi-LoRA concurrent training | PLoRA [2508.02932](https://arxiv.org/html/2508.02932v1); LoRAFusion [SEARCH] 2510.00206 | PLoRA 5.94× (3B, A10) vs a *Min-GPU* baseline, gains from **underutilisation at batch 1**; code not released. LoRAFusion 1.29× vs mLoRA (multi-GPU, ≥8B) | — | Little gain for a well-batched T4 job (§5) |
| E3 | Liger kernels | [SEARCH] linkedin/Liger-Kernel README; PyTorch torchtune blog | Llama-3.2-1B on A100: −47% peak memory, "marginal" throughput | — | Redundant with Unsloth's fused CE |

---

## 3. The recommended SFT recipe (frozen for every arm and base)

### 3.1 Recipe

| Item | Setting | Why |
|---|---|---|
| Objective | **Completion-only, token-mean NLL**. Prompt tokens are masked; the EOS token is trained | S1, S3. NLL wins at the weak end. It does not shrink arm differences (unlike DFT's sg(p) weight, which mutes unlikely, i.e. novel, tokens) |
| Ablation (1 run-set) | DFT on S0 and on the best arm, mid size, 3 seeds | Cheap (one line). Answers a reviewer's "would a better objective change the ranking?" |
| Format (base models) | Plain text, no chat template: `Bài toán:\n{problem}\n\nLời giải:\n{solution}` with the solution ending `Đáp số: \boxed{ans}` + `<eos>`. The same template is used at eval (0-shot after SFT; 4-shot with the same template for base rows) | LoRA does not train embeddings, so chat-template special tokens that the base never trained are a silent risk [UNVERIFIED per model; check G4]. SimpleRL-Zoo: weak bases do better with simple prompts. `\boxed{}` matches `eval/math_verify.py` |
| CoT length | Short CoT. `max_length` = p99 of the pooled training lengths (expected ≤1024). **Drop, don't truncate** longer items, and log the drop rate per arm | S8. Truncation would remove answers |
| Precision (T4) | Base weights fp16. **LoRA weights fp32** (PEFT `autocast_adapter_dtype=True`). Autocast fp16, dynamic GradScaler, CE logits upcast to fp32 (Unsloth does this), grad-clip 1.0 after unscale. **Abort a run if >1% of steps are skipped for overflow** | T4 has no bf16. R2 shows fp16 is not inherently worse, *if* activations stay in range (gate G1) |
| Adapter | **Vanilla LoRA, r=32, α=32, dropout 0, all linear layers of the text decoder.** For Qwen3: q,k,v,o,gate,up,down. For Qwen3.5: also the GDN projections (`target_modules="all-linear"`, excluding the vision tower and `lm_head`; module names **[UNVERIFIED]**, list them in G2). No DoRA, PiSSA or rsLoRA | L1 (variants give no gain once LR is tuned; DoRA adds cost). L2 (all-linear; rank barely shifts the optimal LR; r=32 is ample for 4–9M tokens). r=64 adds VRAM with no expected gain [ESTIMATE] |
| Optimiser | AdamW (torch, **fp32 states**, not 8-bit), β=(0.9,0.999), weight decay 0 | Removes the optimiser-quantisation confound. Memory is trivial for adapters |
| LR | **Sweep {1e-4, 2e-4, 4e-4} on arm S0, per base**, then freeze. Prior: 2e-4 (≈10× a full-FT LR of 2e-5) | L1, L2. The tuned LR is the single biggest factor |
| Schedule | **WSD**: 3% warm-up, constant, then a linear decay to 0 over the last 10% of steps. For the sweep only: one 3-epoch run per LR, saving the stable-phase state at the end of epochs 1 and 2, and branching a 10%-of-epoch decay from each. This gives {1,2,3}-epoch models for ~3.2 epochs of compute instead of 6 | WSD/decay-branch idea from MiniCPM **[UNVERIFIED, 2404.06395 not opened]**. If G6 shows a branch differs from a fresh run by >1 pt, fall back to separate runs |
| Epochs | Chosen once on S0 from {1,2,3} by dev accuracy (Vi-GSM8K dev + Vi-ExamMath dev), then frozen | Fair across arms |
| Batch | **32 sequences per optimiser step** (micro-batch × accumulation; the micro-batch fits memory) | L2 (LoRA dislikes large batches). Fixed sequences per step makes the step count identical across arms |
| Batching | **Padding-free varlen** (Unsloth default): sequences concatenated per micro-batch with `cu_seqlens`/position resets; same rows per step | E1. Keeps the loss identical to padded training while removing pad compute |
| Gradient checkpointing | **Off** for ≤1B (fits in 16 GB, so 4N instead of 6N); Unsloth checkpointing **on** for ≥1.7B | 1.5× fewer FLOPs where memory allows [ESTIMATE] |
| Seeds | 3 per cell. A seed controls LoRA init, data order and dropout (=0) | Q2 norm |
| Checkpoint | End of training only (plus resumable state every 30 min for the 12-h session limit) | — |

### 3.2 Matching the arms fairly: matched items vs matched tokens

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **A. Matched items** (N items, same epochs, same 32 seq/step, hence identical step count) | The natural unit of a dataset. Matches the data-efficiency x-axis (items). Optimiser steps identical | An arm with longer solutions gets more gradient tokens | **Primary** |
| B. Matched tokens (subsample to equal total completion tokens) | Removes the length confound | Arms get different item counts and step counts. Arbitrary which items are dropped | **Control**: one run-set (mid size, S0 vs the best and worst arm, 3 seeds) |
| C. Matched steps with packing (fixed tokens per step) | Equal compute | Item count per step varies with length. Bin-packing changes the loss | Reject |

**Reporting rules.**
- Report per arm: mean and p99 completion tokens, total tokens, drop rate.
- If the A and B rankings disagree, say so. That is a finding.

### 3.3 SFT objective alternatives (one choice)

| Option | Evidence at ≤4B | Effect on C2 | Cost | Choice |
|---|---|---|---|---|
| **NLL, completion-only** | S1, S3 (weak end) | neutral instrument | 1× | ✅ |
| DFT | +13.6 avg on Qwen2.5-*Math*-1.5B; tiny gain on LLaMA-3.2-3B; hurts knowledge | compresses arm differences (down-weights novel tokens) | 1× | ablation |
| InfoSFT / RankTuner / ASFT | single-paper, mixed | same issue | 1× | no |
| CFT | +4–10% over SFT (7B mostly) | it is a data arm | teacher calls | could be a future arm |
| NFT / RFT-with-negatives | ≈ GRPO (7B+) | needs on-policy samples | RL-cost | no |

---

## 4. The recommended RL stage (RQ-F: "does the arm advantage survive RLVR?")

### 4.1 Recipe

| Item | Setting | Why |
|---|---|---|
| Framework | Unsloth (`fast_inference=True`, vLLM colocate) over TRL `GRPOTrainer`, fp16. Fallback if vLLM-on-T4 fails gate G5: TRL + HF `generate` (≈ several-× slower [ESTIMATE]) | Only maintained single-GPU LoRA-GRPO stack (03) |
| Model / start | Arm SFT checkpoint, **merged to fp16** (adapter folded in), with a fresh LoRA r=32, α=64, all-linear | R2 used exactly r=32, α=64 for 1.5B LoRA-GRPO. R7: rank is not the bottleneck |
| Algorithm / loss | GRPO, `loss_type="dr_grpo"`, `scale_rewards=False`, `beta=0`, ε=0.2, `epsilon_high=0.28`, `num_iterations=1`, `mask_truncated_completions=True`, vLLM importance-sampling correction on (TRL default `sequence_mask`) | Dr.GRPO removes the length and std biases (03, R1). R2: in fp16 the variants converge to near-identical results, so choose the one with the fewest biases |
| Precision | **fp16 for both trainer and vLLM** | R2: ~24× smaller mismatch; bf16 LoRA-GRPO collapses |
| Group size G | **8** | TRL default. R6: no gain from other G at 1.5B. G=8 keeps the zero-variance-group fraction low |
| Prompts per step | 32 (256 rollouts per step) | Half of 03's 64, within the published range |
| Steps | **200**, fixed for every arm (no early stopping inside the comparison). Dev eval every 25 steps is logged | Fixed budget = fair. R4: small models plateau early |
| Completion length | ≤512 tokens, temperature 1.0, top_p 1.0 | Short CoT. Sampling stays on-policy |
| LR | 2e-5, constant after 10 warm-up steps. A pilot on S0 picks from {1e-5, 4e-5} (100 steps) | R2 used 4e-5 for LoRA 1.5B. [SEARCH] 2607.12640: GRPO outcomes at small scale are "learning-rate-gated" |
| Reward | `math_verify` on the last `\boxed{}`: 1 if equivalent to gold, else 0. **No format reward, no length penalty** | R3: format rewards hurt weak bases. Binary reward is the RLVR standard |
| Prompt pool | ~4k Vietnamese training problems, disjoint from all eval sets, **one fixed pool for all arms**. Filtered once by the **S0-SFT model's pass@8 ∈ [1/8, 7/8]** | R4, R5: intermediate difficulty. A fixed pool avoids an arm-dependent curriculum. Log each arm's zero-variance-group fraction |
| Controls | (1) **Random reward**: Bernoulli(0.5) independent of correctness, ×3 seeds per family. (2) The no-RL SFT checkpoint under the same decoding. (3) Evaluation emphasis on the **Vi-ExamMath temporal split** (post-cutoff, contamination-safe) | R8, R9 |
| Evaluation | Greedy pass@1 (primary) + pass@8 at T=1.0. 3 seeds. Paired bootstrap | Separates elicitation (pass@1 ↑, pass@k flat) from expansion **[UNVERIFIED claim from Yue et al. 2504.13837, not opened]** |
| Size | **Qwen3-1.7B** (mid). Second family only if it passes G1+G5 | Mature vLLM path on T4. 0.6B is the cheap fallback (2.9 vs 7.0 T4-h per run) |

### 4.2 RL alternatives (one choice)

| Option | For | Against | Choice |
|---|---|---|---|
| **GRPO + Dr.GRPO loss, fp16** | bias-free normaliser; stable in fp16; one flag | clipping bias (spurious rewards), handled by the control | ✅ |
| TRL default DAPO loss (token-level) | standard, ok | token-mean still weights long outputs more | equivalent in fp16 (R2); fine as a fallback |
| GSPO (`importance_sampling_level="sequence"`) | stable for MoE | dense ≤2B gains nothing (03) | no |
| REINFORCE++ / RLOO | global or leave-one-out baseline | no ≤2B advantage shown; not in Unsloth's fast path | no |
| NFT (supervised on positives and negatives) | ≈ GRPO | no ≤4B evidence; needs custom code | no |
| On-policy distillation (open teacher) | densest signal; 9–30× cheaper than SFT continuation (03) | answers a different question (not RLVR); needs a same-tokenizer teacher on GPU | hand-off "best model" only |
| GAD (black-box OPD) | API teacher | chat-only evidence; discriminator + RL cost | no |

### 4.3 Distillation: API-only teacher vs open teacher

| Teacher access | Possible | Recommended use |
|---|---|---|
| **API only** (NIM: DeepSeek-V4.1-Flash, Nemotron-3-Ultra; no logits) | SeqKD (SFT on samples), rejection-sampled multi-sample SFT, GAD (unproven for math) | **The C2 arms themselves** (SFT on verified teacher solutions). That is the scope of this thesis |
| **Open, same tokenizer, fits a T4** (e.g. Qwen3-4B → Qwen3-0.6B/1.7B; fp16 4B ≈ 8 GB + student) | OPD / GKD with λ=1 (reverse KL); EOPD variant | Optional final hand-off model after SFT: ~150 steps [ESTIMATE, from Thinking Machines' "~150 steps" anchor] |
| Open, different tokenizer | TRL GOLD/ULD (experimental; token-alignment bugs reported) | Avoid |

---

## 5. Efficiency plan for ~230 SFT + ~30 RL runs on Kaggle T4×2

### 5.1 Techniques ranked by T4-hours saved (none changes the scientific design)

| # | Technique | Speed-up / saving | Evidence | Adopt |
|---|---|---|---|---|
| 1 | **Two job lanes per T4×2 session** (one independent run per GPU via `CUDA_VISIBLE_DEVICES`; DDP-2 only for 4B runs that would exceed 12 h) | 2× per session-hour **if** quota bills session-hours [UNVERIFIED] | Kaggle T4×2 hardware | ✅ check the quota counter in session 1 |
| 2 | **Padding-free varlen batching** (not bin-packing) | ~1.3–1.5× at 450-token items with mixed lengths [ESTIMATE]. Unsloth: 1.1–2×, "0% accuracy change" | E1 | ✅ (Qwen3). Qwen3.5 only after G2 |
| 3 | **Reuse the core 10k runs as the 10k point of the data-efficiency curves** (identical config) | −27% of the data-efficiency block ≈ −60 T4-h [ESTIMATE] | arithmetic | ✅ |
| 4 | **WSD epoch-branch LR sweep** (3 runs × 3.2 epochs instead of 9 runs × 2 epochs on average) | sweep −47% [ESTIMATE] | MiniCPM WSD [UNVERIFIED] | ✅ after G6 |
| 5 | **No gradient checkpointing for ≤1B** | 1.5× on those runs (4N vs 6N) [ESTIMATE] | FLOP count | ✅ |
| 6 | **Queue runner**: many runs per 12-h session, model loaded once per base, data pre-tokenised to Arrow on a Kaggle Dataset | saves ~10 min setup per run, ≈ 35 T4-h over 230 runs [ESTIMATE] | — | ✅ |
| 7 | **vLLM (fp16, `TRITON_ATTN`) for all evaluation and RL rollouts** | generation is the RL bottleneck; HF `generate` is several-× slower [ESTIMATE] | vLLM backend table (fp16 "Any") | ✅ after G5 |
| 8 | **RL shape**: 32 prompts × G8 × ≤400–512 tokens × 200 steps (vs 64 × 8 × 500 × 300 in 03) | 3.8× fewer rollout tokens: 7.0 vs 26.3 T4-h per 1.7B run [ESTIMATE] | R4, R5 | ✅ |
| 9 | Offline difficulty filtering of the RL pool | fewer zero-advantage groups; "up to +12% in <½ the steps" | R5 | ✅ |
| 10 | Evaluate all checkpoints of a session in one vLLM process (`enable_lora` or merged) | removes model reloads | — | ✅ |
| 11 | Multi-LoRA concurrent training (PLoRA/mLoRA style) | ≤1.1–1.2× when the T4 is already batch-saturated [ESTIMATE]. PLoRA's 5.9× is vs Min-GPU at batch 1; code unreleased. Adapters do not share base compute, because each adapter alters the hidden states | E2 | ❌ build. ✅ only "2 processes per GPU for 0.6B" if `nvidia-smi` util < 70% |
| 12 | Liger kernels | memory only, "marginal" throughput; Unsloth already fuses CE | E3 | ❌ |
| 13 | FP8 | not available on T4 (Turing) | T4 spec | ❌ |
| 14 | 8-bit optimiser / QLoRA | small saving; adds a confound; Unsloth discourages QLoRA on Qwen3.5 | Unsloth Qwen3.5 guide | ❌ |
| 15 | Early stopping | **Not inside comparisons** (fixed epochs/steps = fairness). Used only in pilots: stop the LR-sweep branch if dev accuracy is not best after epoch 2 | R9 | pilots only |

### 5.2 GPU-hours [ESTIMATE, `t4_estimate_out.txt`, ±2×]

**Assumed base set:** Qwen3 0.6B / 1.7B / 4B + Gemma-4-E2B. This is the 03 §6.6 grid with the same run counts. Swapping
to Qwen3.5 adds about 1.8× on the mid block.

| Block | Naive (padded, ckpt on, 9-config sweep) | Optimised (#2–#6) |
|---|---|---|
| Qwen3-0.6B (30 runs) | 29 | 11 |
| Qwen3-1.7B (92 runs: sweep, core, data-efficiency, English, full-FT check) | 259 | 160 − 24 (#3) = **136** |
| Gemma-4-E2B (78 runs) | 366 | 225 − 40 (#3) = **185** |
| Qwen3-4B (30 runs) | 197 | **110** |
| **SFT total** | **851 T4-h** | **≈ 440 T4-h** (1.9×) |
| Evaluation (≈ 260 checkpoints × ~7 min on vLLM) | — | ≈ 30 |
| RL (30 runs, Qwen3-1.7B, 32×8×400×200) | 788 (03 shape) | **≈ 210** (or ≈ 86 at 0.6B) |
| **Total** | ≈ 1,640 | **≈ 560–680 T4-h** |

**Calendar.**
- At 2 lanes × 30 h/week ≈ 60 GPU-h/week, this is **≈ 9–11 weeks**. If T4×2 is billed per GPU, it is about 2× longer.
- **Biggest single lever:** the Gemma-4-E2B block is ≈ 42% of the optimised SFT hours.
  - A ~1B fp16-safe second family would save ≈ 120 T4-h.
  - The choice belongs to the model report. From a training standpoint, Gemma-4-E2B is also the riskiest model on T4.
- **Modal (37 L4-h/month)** is kept for:
  - bf16 parity checks (G1);
  - the full-FT-vs-LoRA sanity run;
  - and, if Gemma fails G1 but is kept, the Gemma block in bf16 on L4: ≈ 2× T4 speed, ~95 L4-h ≈ $76 ≈ 2.5 months of credit.
- The laptop 3050 Ti (bf16-capable, 4 GB) can run the fp16-vs-bf16 parity check for 0.6B/0.8B at $0.

---

## 6. Day-1 gates (≈ 3 T4-h + ≈ 1 L4-h; all must pass before the grid starts)

| Gate | Test | Pass rule [thresholds are ESTIMATES] | If it fails |
|---|---|---|---|
| G1 fp16 safety, per base | 300 SFT steps on S0 in fp16 (T4) vs bf16 (laptop or L4), same seed; inf/NaN forward hooks; GradScaler skip count | skipped steps <1%, no inf activations, dev accuracy within ±1 pt of bf16 | keep sensitive ops (norms, residual) in fp32 (Unsloth path), retest; else move that base to L4 bf16 or drop it |
| G2 packing boundaries (Qwen3.5) | Run the same 4 short items as (a) separate padded rows and (b) one padding-free row; compare per-token logits | max abs logit diff <1e-2 (fp16), loss identical to 3 decimals | Qwen3.5: length-grouped padded batching (`group_by_length`), accepting ~1.3× cost, or use Qwen3 |
| G3 GDN kernels on sm_75 | Import FLA kernels and run fwd+bwd on T4; record compile time and tok/s | runs, and tok/s ≥ 60% of the §1.2 estimate | Qwen3 primary (02's fallback) |
| G4 template tokens | Check that every token in the template/EOS has a trained embedding (norm not ≈ init) | all trained | change the template |
| G5 vLLM on T4 | vLLM ≥0.17, `--dtype float16`, model loads; log the chosen backend; 200-item eval matches HF `generate` greedy answers ≥98% | pass | HF `generate` for that base (eval OK; RL too slow, so RL only on Qwen3) |
| G6 WSD branch fidelity | S0, mid size: epoch-2 branch vs a fresh 2-epoch WSD run | dev accuracy within ±1 pt | run the sweep as separate runs (+~50 T4-h) |
| G7 throughput | log tok/s per base | — | re-plan §5.2 |
| G8 quota accounting | read the Kaggle quota counter after 1 h on T4×2 | — | halve the lane assumption |

---

## 7. Model merging: what it is good for here

- **Not inside any comparison.** A merged model is a different estimator. Averaging seeds would also remove the seed
  variance that the CIs need.
- **Hand-off model (free, evaluation cost only).** For the winning arm, average the 3 seeds' merged weight deltas:
  W = W₀ + mean(BᵢAᵢ).
  - Do **not** average the A and B matrices separately. Seeds have unaligned LoRA bases.
  - Keep the averaged model only if dev accuracy ≥ the best single seed. Evidence is weak (M1: +1.8 pts for LoRA
    checkpoint merging on GSM8K [SEARCH]), so treat it as optional.

---

## 8. Open items / unverified

- **T4 MFU** and every throughput number in §1.2 and §5.2 are estimates. Measure them in session 1.
- **Kaggle quota accounting** for T4×2 is unverified (one hour or two per session-hour).
- **Qwen3.5 GDN** on sm_75 (FLA Triton kernels) and padding-free correctness in Unsloth on Qwen3.5: no source found
  either way. Gates G2/G3.
- **Gemma-4-E2B fp16 training stability on T4** beyond the documented audio-mask overflow and the GRPO MLP clamp
  (both [SEARCH]/Unsloth). Gate G1.
- **WSD decay-branch equivalence** (MiniCPM, 2404.06395) was not opened. Gate G6.
- **Yue et al. (2504.13837)** pass@k elicitation argument was not opened. It is used only to motivate reporting pass@8.
- **Numbers marked [SEARCH]** come from search snippets: InfoSFT, RankTuner, 2605.12906, LR-Matters, TinyLoRA,
  REINFORCE++, the 2-GRPO study, Liger and PLoRA-adjacent claims.

## Sources (opened)

- [arxiv.org/html/2508.05629v3](https://arxiv.org/html/2508.05629v3) (DFT)
- [arxiv.org/abs/2510.00526](https://arxiv.org/abs/2510.00526) (Beyond Log Likelihood)
- [arxiv.org/abs/2606.16152](https://arxiv.org/abs/2606.16152) (Quality-Utility Paradox)
- [arxiv.org/abs/2605.14967](https://arxiv.org/abs/2605.14967) (InfoSFT)
- [arxiv.org/html/2510.26788](https://arxiv.org/html/2510.26788) (FP16 RL)
- [arxiv.org/abs/2604.06298](https://arxiv.org/abs/2604.06298) (difficulty limits ≤3B)
- [arxiv.org/abs/2504.03380](https://arxiv.org/abs/2504.03380) (online difficulty filtering)
- [arxiv.org/abs/2504.07086](https://arxiv.org/abs/2504.07086) (Sober look)
- [arxiv.org/html/2508.02932v1](https://arxiv.org/html/2508.02932v1) (PLoRA)
- [huggingface.co/docs/trl/main/en/grpo_trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [unsloth.ai/docs/models/qwen3.5/fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [unsloth.ai/docs/models/gemma-4/train](https://unsloth.ai/docs/models/gemma-4/train)
- [unsloth.ai/docs/blog/3x-faster-training-packing](https://unsloth.ai/docs/blog/3x-faster-training-packing)
- [twinkle-kit Padding-Free](https://twinkle-kit.readthedocs.io/en/latest/Components/Training%20Middleware/Padding-Free.html)
- [docs.vllm.ai attention backends](https://docs.vllm.ai/en/latest/design/attention_backends/)
- Via `03_training_infra.md` (opened there): LoRA Without Regret, Thinking Machines OPD, 2502.12143, 2506.10947,
  2503.20783, 2504.15777.
- Search-snippet only ([SEARCH]): 2503.18892, 2507.10532 (v3 HTML listed), 2511.10643, 2603.07079, 2501.17703,
  2506.03295, 2505.18116, 2602.04118, 2602.04998, 2504.18580, 2510.00206, 2501.03262, 2605.12906, 2607.12640,
  2512.16912, Kaggle quota pages, Liger README.

Scratch: `C:\Users\tranh\AppData\Local\Temp\claude\E--MATH-REASONING-THESIS-vi-math-smt\884a7b4b-700c-4c10-ad97-d56477e30ce1\scratchpad\t3\`
(`t4_estimate.py`, `t4_estimate_out.txt`).
