# 03 — Training method & compute infrastructure: re-validation (2026-11-07)

**Scope.** Re-checks `RESEARCH_PLAN.md` §6.3 (fine-tuning protocol), §7 (compute) and §10 (decisions)
against evidence available on 2026-11-07. Every claim carries a source (URL, PyPI JSON, or a command run on
the researcher's laptop). Claims from memory are marked **[UNVERIFIED]**. Estimates are marked
**[ESTIMATE]** and the arithmetic is in `scratchpad/infra/estimate.py` (output: `estimate_out.txt`).

**Bottom line.** The Q2-sized plan the user asked for is in **§6.6**. It supersedes the smaller grid in §6.5.

0. **Q2 scope, no cuts (§6.6):**
   - **230 SFT runs + 30 RLVR runs**: 4 bases (Qwen3-0.6B/1.7B/4B + **Gemma-3-1B** as a second family), 7
     arms × 3 seeds, 4-point data-efficiency curves on 2 families, an English control, LoRA-vs-full checks,
     and an RLVR stage with random-reward controls.
   - **It fits $0 only by combining sources:** laptop (whole 0.6B block, bf16) + Kaggle T4×2 (data-efficiency
     block, *if the user re-allows Kaggle*) + Modal $30/month for **4–5 consecutive cycles**.
   - **Binding bottleneck: Modal monthly credit, i.e. calendar months.** About 45–60% of it goes to the RLVR
     stage. A Modal academic grant ("up to $10k") is the single lever that removes it.

1. **Keep SFT as the C2 instrument. Do not add RL to the main grid.** It is still the right choice for a
   *data-comparison* study. Change the recipe from "QLoRA r=16 on the laptop + LoRA on L4" to
   **one bf16 LoRA recipe (r=64, all linear layers incl. MLP, LR ≈ 10× full-FT LR) on the same cloud GPU
   for every model size.** This removes the precision confound between sizes in the old plan (4-bit at
   0.6B, bf16 at 1.7B).
2. **The compute picture has changed a lot.** Modal now charges **L4 $0.222/h** (the old plan assumed
   $0.80/h), **L40S $0.542/h** and **A100-40GB $0.583/h**. The $30/month Starter credit therefore buys about
   **135 L4-hours or 55 L40S-hours a month at $0**. All ~60 grid runs at both sizes, plus an optional
   **Qwen3-4B third size** and an optional **GRPO probe**, fit in about $15–25 of one month's credit.
3. **The laptop is a dev box, not a training box.** The old estimate for a 0.6B run on the laptop
   (30–50 min) is about 3–7× too optimistic (it is more like 1.7–3.4 h per 10–20k-sample run). vLLM
   has no Windows wheels, so it would need WSL2, and WSL2 currently sees only about 3 GB of RAM.
4. **Fallback: Kaggle (T4×2, about 30 h/week).** ⚠ Kaggle was **excluded by a previous user decision.**
   The evidence says it is a workable *fallback* and a good *reproduction path for third parties*, but it
   is not a drop-in replacement. The T4 has no BF16, and the P100 is below vLLM's minimum compute
   capability. Re-including Kaggle needs the user's explicit OK.

---

## 1. Local machine (measured 2026-11-07, commands run in this session)

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop, **4096 MiB**, compute cap **8.6** (Ampere → BF16 OK), WDDM, power limit 35 W (max 40 W), max SM clock 2100 MHz, mem clock 5501 MHz | `nvidia-smi --query-gpu=…` |
| Driver / CUDA | driver **610.78**, CUDA UMD **13.3**; Windows `nvcc` = CUDA **11.8** (old toolkit; irrelevant if torch wheels bring their own CUDA) | `nvidia-smi`, `nvcc --version` |
| CPU | Intel i5-11400H, 6C/12T | `Win32_Processor` |
| RAM | **7.7 GB** total (Windows). **WSL2 sees ~3 GB + 1 GB swap** | `Win32_ComputerSystem`; `free -g` in WSL |
| Disk free | C: 66 GB, E: 100 GB (WSL ext4 VHD reports 950 GB virtual, but it is backed by a host drive) | `Get-PSDrive`; `df -h` |
| WSL2 | **Installed**: Ubuntu 24.04.4 LTS, kernel 6.6.87.2, Python 3.12.3, systemd on, **GPU visible inside WSL** (`nvidia-smi` works). No torch/vllm/unsloth/trl in system pip | `wsl -l -v`, `wsl -- …` |
| Windows Pythons | 3.13 (`C:\Python313`), 3.12.3 (`G:\`), 3.11 Anaconda (default) | `py -0p` |
| Project `.venv` | Python 3.11.7, 44 packages: sympy, z3-solver, openai, pandas, pytest… **No torch / unsloth / vllm / trl / peft / transformers** | `.venv\Scripts\python -m pip list` |
| Anaconda torch | torch **2.12.0** is installed but **`import torch` fails**: `OSError: [WinError 1114] … c10.dll` | `python -c "import torch"` |

**What follows:**

- Training is not set up anywhere on the machine yet, so any training stack is a fresh install. The broken
  Anaconda torch must not be reused.
- 4 GB of VRAM rules out bf16 training of anything ≥1.7B, and 7.7 GB of host RAM is tight for WSL2 + vLLM.
  WSL's limit can be raised in `%UserProfile%\.wslconfig` (`memory=6GB`), but then Windows has about 1.7 GB
  left. **[ESTIMATE]**

---

## 2. Algorithms for ≤4B math reasoning (2025–2026): what actually yields gains

### 2.1 Evidence table

| Method | Key evidence (numbers) | Model / compute | Relevance to this thesis |
|---|---|---|---|
| **SFT on distilled CoT** (large → small) | DeepSeek-R1-Distill-Qwen-1.5B: **AIME24 28.9, MATH-500 83.9** after SFT on **800k** R1-curated samples. DeepSeek states that distilling a larger model's patterns beats running RL on the small model. [HF card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 1.5B; 800k samples | Confirms SFT-on-teacher-data is *the* strongest lever for ≤2B. This is exactly what the S2/S5 arms are. |
| Strong-to-weak distillation vs RL (Qwen3 report) | Qwen3-8B: distillation **74.4% AIME24 at ~1,800 GPU-h** vs RL **67.6% at 17,920 GPU-h** (as cited in [Thinking Machines, On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)). The [Qwen3 abstract](https://arxiv.org/abs/2505.09388) says distillation "significantly reduce[s] the computational resources required to build smaller-scale models." | 8B | Distillation is about 10× more compute-efficient than RL at a better result. |
| **Small-model learnability gap** | Models **≤3B** "don't consistently gain" from long CoT or from distilling large reasoners; they do better on **shorter, simpler chains**. *Mix Distillation* (long+short) helps. [arXiv:2502.12143](https://arxiv.org/abs/2502.12143) (v3 Nov 2025) | ≤3B | **Actionable:** S2 generations should be *short CoT*, not long thinking traces. If long traces are ever used, mix them with short ones. |
| **Rejection-sampling FT (RFT)** | LLaMA-7B GSM8K: SFT **35.9% → 49.3%** with rejection samples pooled from several models. "RFT brings more improvement for less performant LLMs." [arXiv:2308.01825](https://arxiv.org/abs/2308.01825) | 7B | Cheap self-improvement baseline. It overlaps conceptually with arm S2 (LLM samples + answer check). |
| **RLVR / GRPO on 1.5B** | Open-RS: R1-Distill-1.5B, **7k samples, 4×A40 for 24 h, $42**, AMC23 **63→80%**, AIME24 46.7%. Caveat: "optimization instability and length constraints … with prolonged training." [arXiv:2503.16219](https://arxiv.org/abs/2503.16219) | 1.5B (already distilled) | RL works on a *distilled* 1.5B. Gains are on hard competition math, with stability caveats. |
| DeepScaleR | R1-Distill-1.5B + GRPO with 8k→16k→24k context lengthening, ~40k problems: AIME24 **28.8→43.1**, MATH-500 **82.8→87.8**. Used 8–32 A100-80GB. [HF card](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview) | 1.5B | Large compute (multi-node A100). Out of a $0 budget at full scale. |
| **RL with LoRA, tiny budget** | Tina: LoRA-GRPO on R1-Distill-Qwen-1.5B reaches **AIME24 43.3%** for **$9** (training $8 + eval $1, at ~$1/GPU-h on 2× L40S). Best checkpoint at step 450 of 875. Rank-16 ≥ rank-32. [arXiv:2504.15777](https://arxiv.org/abs/2504.15777), [HTML](https://arxiv.org/html/2504.15777) | 1.5B, 2×L40S | **RL+LoRA is feasible on ≤$10.** Note that the gain is largely *format/structure* adaptation (the authors' hypothesis). |
| **1-shot RLVR** | Qwen2.5-Math-1.5B MATH-500 **36.0 → 73.6%** with **one** training example (74.8% with 2). This matches the 1.2k DeepScaleR subset. 8.6 pts of the gain is *beyond format correction*. [arXiv:2504.20571](https://arxiv.org/abs/2504.20571) | 1.5B | RL on Qwen mostly *elicits* latent ability. That is good for a demo, but **it confounds a data comparison.** |
| **Spurious rewards** | Qwen2.5-Math-7B MATH-500: **random reward +21.4 pts** vs ground truth +29.1. The gains **do not transfer** to Llama3/OLMo2, and the authors attribute part of the effect to GRPO's clipping bias. [arXiv:2506.10947](https://arxiv.org/abs/2506.10947) | 7B | **Any RL claim on Qwen needs a random-reward control.** RL gains on Qwen bases are partly pretraining artefacts. |
| GRPO variants | **Dr. GRPO** removes a GRPO bias that "artificially increases response length, especially for incorrect outputs" ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)). **DAPO**: decoupled clip + dynamic sampling, Qwen2.5-32B AIME **50** ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)). **GSPO**: sequence-level importance ratio, stabilises MoE RL, used for Qwen3 ([arXiv:2507.18071](https://arxiv.org/abs/2507.18071)) | various | All are one-flag switches in TRL/Unsloth (§3). For a dense ≤2B model, use `dr_grpo` or TRL's default `dapo` loss. GSPO matters mainly for MoE. |
| **LoRA vs full FT** ("LoRA Without Regret") | SFT: LoRA **matches full FT on small-to-medium datasets** (Tulu3, OpenThoughts3) and underperforms only beyond adapter capacity. RL: **matches even at rank 1.** Optimal LoRA LR ≈ **10× (9.8×)** full-FT LR. **All layers incl. MLP** are much better than attention-only. LoRA is **less tolerant of large batches**. ~2/3 of full-FT FLOPs per step. [Thinking Machines, 2025-09-29](https://thinkingmachines.ai/blog/lora/) | Llama-3 & Qwen3 incl. 1B | **Directly sets the recipe:** LoRA is fine for 5–25M-token SFT. Use all-linear targets, LR ≈ 10× full-FT, and moderate batch. The old r=16 is on the low side but defensible. r=64 costs almost nothing. |
| **On-policy distillation (OPD)** | Student samples, teacher scores per token (reverse KL). AIME24 **70%** in ~150 steps from a 60% SFT checkpoint. **9–30× cheaper** than continuing SFT. [Thinking Machines, 2025-10-27](https://thinkingmachines.ai/blog/on-policy-distillation/). 2026 follow-ups: **OPD→RLVR sequential beats joint** and beats SFT→RL as a cold start ([arXiv:2609.04108](https://arxiv.org/abs/2609.04108)). OPD before RL raises the final RL ceiling ([arXiv:2609.28145](https://arxiv.org/abs/2609.28145)) | 8B student; 2026 papers do not name sizes in their abstracts | State of the art for *building* the best small model. It needs online teacher log-probs (same-family teacher in practice), so it is **not a data-augmentation arm**. Future work. |
| SFT-then-RL for SLMs (2026) | Difficulty-aware SFT→RL: SFT for "not-yet-mastered" skills, RL to consolidate partially-accessible ones. Reports beating SFT, distillation and RL baselines (no numbers in the abstract). [arXiv:2606.04466](https://arxiv.org/abs/2606.04466) | "two SLMs" | Supports **SFT first; RL only as a second stage**. |
| Consumer-GPU precedent | 1.5B math model trained with RL + memory optimisations on **one RTX 3080 Ti (16 GB)** ([arXiv:2506.08935](https://arxiv.org/abs/2506.08935)). The abstract gives no benchmark numbers. | 1.5B | Shows ~16 GB is the practical floor for local RL at 1.5B. **4 GB is not enough.** |

### 2.2 What this means for 0.5–2B models

- **The biggest and cheapest gain is SFT on good teacher solutions.** RL adds a further +5–17 pts on
  *competition* math (AIME/AMC) when it starts from an *already distilled* 1.5B (Open-RS, DeepScaleR, Tina).
- **RL gains on Qwen bases are partly elicitation or artefact** (1-shot RLVR; spurious rewards). For a study
  whose dependent variable is *"which augmentation data is better"*, RL after SFT would (a) add a second
  source of variance, (b) partly erase differences between arms (RL re-elicits the same latent skill
  whatever the SFT data), and (c) need a random-reward control to be credible on Qwen. **So the
  Sept-2026 decision "SFT only for C2" still stands.** RL is at best a *secondary probe* (§6.3).
- **Short CoT for ≤3B** (learnability gap). This matches the plan's GSM8K/algebra scope and its
  `max_new_tokens` 512–640.
- **LoRA is not a quality handicap at this data scale.** About 10k–50k samples × ~450 tokens is
  4.5–22.5M tokens, which is "small-to-medium". The LoRA Without Regret conditions: all-linear targets,
  LR ≈ 10×, moderate batch.

---

## 3. Libraries (versions from the PyPI JSON API, fetched 2026-11-07; raw output in `scratchpad/infra/pypi_versions.txt`)

| Library | Latest | Released | Python | Wheels | GRPO | LoRA/QLoRA for RL | Windows |
|---|---|---|---|---|---|---|---|
| **unsloth** / unsloth_zoo | **2026.11.3** | 2026-11-06 | 3.9–3.14 | pure-py | ✅ `loss_type` `grpo`/`dr_grpo`/`gspo` ([RL guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)) | ✅ "made it work with QLoRA and LoRA". **"5GB VRAM … for any model with 1.5B parameters or less"** | ✅ Core "Works on Linux and Windows" ([requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)). vLLM `fast_inference` needs Linux/WSL. Note: Unsloth says Qwen3.5 must use `fast_inference=False` (no vLLM) |
| **trl** | **1.17.1** | 2026-10-30 | ≥3.11 | pure-py | ✅ default `loss_type='dapo'`, plus `dr_grpo`, `sapo`, `cispo`, `vespo`. GSPO via `importance_sampling_level="sequence"`. `beta=0.0` by default (no ref model). vLLM `colocate` (default) or `server` ([GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)) | ✅ `peft_config` (+`quantization_config` for QLoRA) | runs where torch runs; vLLM path Linux-only |
| **vllm** | **0.33.0** | 2026-11-02 | 3.11–3.14 | **manylinux x86_64/aarch64 only** | — | serves LoRA | ❌ "vLLM does not support Windows natively". Use **WSL**. **Min compute capability 7.5**; CUDA 12.9 default binaries ([install docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)) |
| torch | 2.15.0 | 2026-10-28 | ≥3.11 | incl. `win_amd64` | — | — | ✅ |
| transformers / peft | 5.22.2 / 0.21.2 | 2026-11-06 / 2026-10-01 | ≥3.11 / ≥3.10 | pure-py | — | — | ✅ |
| bitsandbytes | 0.50.2 | 2026-08-27 | ≥3.10 | incl. `win_amd64` | — | 4-bit QLoRA | ✅ |
| **verl** | 0.9.1 | 2026-09-20 | **3.10–3.12** | pure-py | ✅ | ✅ LoRA via PEFT + FSDP/FSDP2 with vLLM/SGLang rollout. Examples are **8×80 GB+**, no single-small-GPU recipe ([verl LoRA doc](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)) | ❌ Linux (Ray) **[UNVERIFIED]** |
| **OpenRLHF** | 0.11.2 | 2026-09-14 | ≥3.10 | **manylinux** | ✅ PPO, REINFORCE++(-baseline), RLOO, GRPO (`group_norm`), Dr.GRPO, DAPO tricks | ✅ `--ds.lora.rank`, `--ds.load_in_4bit`. Quick-start on 8×A100-80 ([README](https://github.com/OpenRLHF/OpenRLHF)) | ❌ Linux/Docker |
| **LLaMA-Factory** | 0.9.5 | 2026-05-30 | ≥3.11 | pure-py | ❌ not in its method table (PPO/DPO/KTO/ORPO/SimPO only). GRPO points to the separate EasyR1 project ([README](https://github.com/hiyouga/LLaMA-Factory)) | n/a | partial (manual torch/bnb; FA2 needs compiling) |
| **axolotl** | 0.21.1 | 2026-10-27 | **≥3.12** | pure-py | ✅ GRPO/GDPO via TRL, vLLM for rollouts ([docs](https://docs.axolotl.ai/docs/rlhf.html)) | ✅ (async GRPO LoRA sync) | not mentioned |
| **torchtune** | 0.6.1 | **2025-04-07** | ≥3.9 | pure-py | — | — | **"no longer actively maintained: … wound down in 2025"** ([README](https://github.com/pytorch/torchtune)) → **do not use** |
| math-verify | 0.9.0 | 2026-01-10 | ≥3.10 | pure-py | reward fn | — | ✅ |
| lm-eval / lighteval | 0.4.13 / 0.13.0 | 2026-08-31 / 2025-11-24 | ≥3.10 | pure-py | — | — | ✅ |

**GRPO + LoRA by GPU memory:**

| VRAM | Feasible | Evidence |
|---|---|---|
| **4 GB** (laptop) | SFT: 0.6B bf16 LoRA ✅ (short seq). 1.7B only as 4-bit QLoRA **[ESTIMATE]**: Unsloth lists 3B QLoRA at 3.5 GB, so 1.7B is ≈2–2.5 GB. **GRPO: ❌.** Unsloth's floor is "5GB VRAM" for ≤1.5B, and vLLM colocation needs more. | Unsloth RL guide + requirements |
| **15–16 GB** (T4 Kaggle/Colab) | SFT bf16→**fp16** LoRA for ≤4B. GRPO+LoRA for ≤1.7B with vLLM colocate at a small `vllm_gpu_memory_utilization`. "With 15GB VRAM … any model up to 17B" with QLoRA GRPO (Unsloth claim). **T4 has no BF16, and FlashAttention-2 needs Ampere+ [UNVERIFIED]**, so TRL's padding-free packing (FA2-only per TRL docs) is unavailable. | Unsloth RL guide; [T4 page](https://www.nvidia.com/en-us/data-center/tesla-t4/) lists FP16 only; [TRL SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer) |
| **24 GB** (L4/A10) | SFT bf16 LoRA up to 4B. GRPO+LoRA for 1.7B with vLLM colocate comfortable. 4B GRPO-LoRA possible. | arithmetic §5 |
| **40–48 GB** (A100-40/L40S) | Also **full FT of 1.7B** (≈16 B/param ≈ 27.5 GB + activations **[ESTIMATE]**) for the LoRA-vs-full sanity check. | §5 |

**Pick: Unsloth (SFT + GRPO) on top of TRL, on Linux (a Modal container), with vLLM for evaluation.**
It is the only stack that is maintained, single-GPU-first, has LoRA-GRPO with Dr.GRPO/GSPO switches, and
runs on Windows natively for laptop debugging. verl and OpenRLHF are multi-GPU/Ray systems: overkill here
and Linux-only. torchtune is dead. LLaMA-Factory has no native GRPO.

---

## 4. Free compute (official pages fetched 2026-11-07 unless noted)

| Option | What is free (verified) | Fit for this project | Evidence |
|---|---|---|---|
| **Modal Starter** | **$30/month** of compute ("GPUs, CPUs, Memory"), no monthly fee. Per-second billing: **T4 $0.164/h, L4 $0.222/h, A10 $0.306/h, L40S $0.542/h, A100-40 $0.583/h, A100-80 $0.694/h, H100 $1.097/h**. Shared-Endpoint tokens are billed from the first request (since 2026-09-01), so use Functions, not Shared Endpoints. **Academics: "up to $10k"** in credits (apply). | ★★★ Linux containers, reproducible as code (`modal run`), bf16 GPUs, vLLM works. → **135 L4-h or 55 L40S-h per month** | [modal.com/pricing](https://modal.com/pricing) (verbatim price lines re-fetched), [modal.com/academics](https://modal.com/academics) |
| **Kaggle Notebooks** | ⚠ *Previously excluded by the user.* GPU quota is **~30 h/week** ("30 hours or sometimes higher depending on demand"), resets weekly. **12 h** per CPU/GPU session (9 h TPU). **T4×2 or P100.** Phone verification required **[UNVERIFIED-official, widely reported]**. Kaggle's own doc pages did not render (JS-only). Figures come from search snippets of `kaggle.com/docs/efficient-gpu-usage` and a 2026-09-21 review | ★★ Good fallback and **the best free path for third-party reproduction** (anyone can fork a public notebook). Caveats: **T4 has no BF16** (fp16 only). **P100 (cc 6.0) < vLLM minimum 7.5**. Sessions are capped. ~130 T4-h/month | [gpuperhour.com review 2026-09-21](https://gpuperhour.com/blog/free-cloud-gpus-and-credits); search results for [kaggle.com/docs/efficient-gpu-usage](https://www.kaggle.com/docs/efficient-gpu-usage), [Kaggle floating-quota post](https://www.kaggle.com/product-feedback/173129) |
| **Google Colab (free)** | "notebooks can run for at most 12 hours, depending on availability". Limits "not publish[ed] … can vary". GPUs "heavily restricted" and "not guaranteed". GPU type is not listed (users report T4) | ★ Ad-hoc debugging only. Not plannable | [Colab FAQ](https://research.google.com/colaboratory/faq.html) |
| **Lightning AI (free)** | **One-off** credits: 5 on sign-up (+25 if you add a card, which this project avoids). "~50 GPU hours on interruptible machines" for 30 credits. Free Studio restarts every 4 h. Phone number required. Credits expire in 12 months. T4 $0.55/h, L4 $0.79/h | ✗ Not monthly. The card requirement for the bulk conflicts with the no-payment rule | [lightning.ai/pricing](https://lightning.ai/pricing) |
| **HF ZeroGPU** | Free account **5 min/day** of GPU (half RTX PRO 6000, 48 GB). **Gradio SDK only.** 60 s default per call. Can host 2 Spaces if the account is >30 days old | ✗ Demos only (≈2.5 h/month). Fine for a public **demo of the final model** in the hand-off | [HF ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu) |
| **HF Inference Providers** | Free users: **no** monthly credits (PRO $2) | ✗ | [HF pricing](https://huggingface.co/docs/inference-providers/pricing) |
| **SageMaker Studio Lab** | **Closed to new users** (existing users: 4 h GPU per 24 h) | ✗ | gpuperhour review 2026-09-21. Official FAQ returned HTTP 403 |
| **Paperspace Gradient free** | M4000 8 GB (2015, Maxwell), 6 h auto-shutdown (last verified 2026-07-13 by the review) | ✗ Maxwell is below Unsloth's minimum compute capability 7.0 and vLLM's 7.5 | gpuperhour review; Unsloth requirements; vLLM docs |
| **Google TPU Research Cloud** | Free Cloud TPUs for accepted researchers. Must publish and give feedback. No open/closed status on the page | ✗ for this stack (TPU ≠ CUDA, so no Unsloth/vLLM-CUDA). Possible only with a JAX rewrite | [TRC about](https://sites.research.google/trc/about/) |
| **NVIDIA Academic Grant** | Up to 30,000 H100-h, **but the applicant must be full-time faculty**, and the page says it "is currently not accepting new applications" | ✗ now. Possible via the advisor in a future cycle | [NVIDIA AGP](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program.md) |
| **VNU / HUS HPC** | No public information found on student GPU access at VNU-Hanoi. VNG sponsored GPU resources for **VNU-HCM** (not Hanoi) for 2026–2028 | ? **Ask the advisor.** This is the only route to free A100-class time without quotas | search ([VNG→ĐHQG-HCM](https://doanhnhan.baophapluat.vn/vng-tai-tro-12-ty-dong-tai-nguyen-gpu-cho-dhqg-hcm-nang-cao-nang-luc-tinh-toan-cho-nghien-cuu-ai.html)) |
| **Azure for Students** | $100 credit listed in the GitHub Student Pack (per the review). Student subscriptions usually have **0 GPU quota [UNVERIFIED]** | ✗ likely | gpuperhour review |
| **NVIDIA NIM** (inference) | Already in the plan: 40 RPM, no credit cap on the researcher's account (confirmed by the user in Sept 2026). Not re-verified here (the NIM docs page returned empty) | ✓ generation/verification. Not usable for evaluating *our* fine-tuned models | RESEARCH_PLAN §6.2 |

**GPU-hours per month at $0 (sustainable, without new applications):**

| Source | GPU-h / month | Notes |
|---|---|---|
| Modal $30 | **135 L4** *or* **55 L40S** *or* 51 A100-40 *or* 183 T4 | cost ÷ price. Mix freely |
| Kaggle (if re-allowed) | ~**130 T4×2 session-hours** (30 h/wk × 4.3) | fp16 only. Floating quota |
| Laptop | ~**unlimited**, but 4 GB / ~15 effective TFLOPS **[ESTIMATE]** | dev/smoke tests |
| Colab free | unknown / not guaranteed | — |
| **Total plannable** | **≈ 135 L4-equiv-h (Modal) + optional ~130 T4-h (Kaggle)** | Modal alone covers the plan (§5–6) |

---

## 5. Cost / time estimates (arithmetic: `scratchpad/infra/estimate.py` → `estimate_out.txt`)

**Inputs.**

- Parameters: Qwen3-0.6B-Base 0.596B, 1.7B-Base 1.721B, 4B-Base 4.022B (HF API `safetensors.total`).
- KV cache: 112 KiB/token for 0.6B/1.7B and 144 KiB/token for 4B (from `config.json`: 28/28/36 layers,
  8 KV heads, head_dim 128).
- Dense peak TFLOPS from vendor pages: [L4](https://www.nvidia.com/en-us/data-center/l4/) 121 (242 is the
  sparsity figure), 300 GB/s. [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) 65 FP16, 320 GB/s.
  [A10](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) 125, 600 GB/s.
  [L40S](https://www.nvidia.com/en-us/data-center/l40s/) 362, 864 GB/s.
  [A100](https://www.nvidia.com/en-us/data-center/a100/) 312.
- Laptop 3050 Ti ≈ 15 TFLOPS fp16-tensor sustained at 35–40 W **[ESTIMATE: 2560 cores [UNVERIFIED] × 2 ×
  ~1.5 GHz × 2]**, ~176 GB/s (11 Gbps × 128-bit [UNVERIFIED bus width]).

**Assumptions [ESTIMATE].**

- Training FLOPs/token: LoRA with gradient checkpointing = **6N**, full FT = **8N**, ×1.15 for attention at
  packed 4k context. This is consistent with LoRA ≈ 2/3 of full-FT FLOPs (Thinking Machines).
- MFU: 0.30 on datacenter GPUs, 0.25 on T4, 0.20 on the laptop.
- Tokens/sample: **~450 for short GSM8K-style Vietnamese CoT**, **~2,500 for long distilled CoT**. The
  `max_seq_len` 4096 is the *packing* length, not the average.
- **Treat every number as ±2×.**

### (a) SFT, bf16 LoRA, 1 epoch — hours (and Modal $)

| Data | Model | Laptop | T4 (Kaggle/Modal) | L4 | L40S | A100-40 |
|---|---|---|---|---|---|---|
| 10k × 450 tok (4.5M) | 0.6B | 1.7 h | 0.32 h | 0.14 h ($0.03) | 0.05 h | 0.05 h |
| | 1.7B | OOM (bf16) | 0.91 h | **0.41 h ($0.09)** | 0.14 h ($0.07) | 0.16 h |
| | 4B | OOM | 2.1 h | 0.96 h ($0.21) | 0.32 h ($0.17) | 0.37 h |
| 50k × 450 tok (22.5M) | 0.6B | 8.6 h | 1.6 h | 0.71 h ($0.16) | 0.24 h | 0.27 h |
| | 1.7B | OOM | 4.6 h | 2.0 h ($0.45) | 0.68 h ($0.37) | 0.79 h |
| | 4B | OOM | 10.7 h | 4.8 h ($1.06) | **1.6 h ($0.87)** | 1.9 h |
| 50k × 2,500 tok (125M, long CoT) | 1.7B | OOM | 25 h | 11.4 h ($2.52) | 3.8 h ($2.06) | 4.4 h |
| | 4B | OOM | 59 h | 26.6 h ($5.89) | 8.9 h ($4.81) | 10.3 h |

- **Check on the old plan:** "0.6B, ~20k examples on the laptop ≈ 30–50 min". The arithmetic gives 9M
  tokens × 6 × 0.596B × 1.15 = 3.7e16 FLOP, which is **3.4 h at MFU 0.20 (1.7 h even at 0.40)**. The old
  number was 3–7× too optimistic.
- Old "1.7B on L4 ≈ 15–25 min": this estimate gives about 50 min for 20k samples. That is the same order,
  and at $0.222/h it is irrelevant to the budget.
- **Full FT of 1.7B** on 20k × 450 tokens: L40S 0.36 h ($0.20), A100-40 0.42 h. Affordable as a one-off
  LoRA-vs-full sanity check.
- Add a fixed **~5–10 min per Modal run** for container start and model load **[ESTIMATE]**. Cache weights
  in a `modal.Volume`.

### (b) GRPO + LoRA, 1.7B-class, 300 steps

| Setting | Generated tokens | L4 | L40S | A100-40 | T4 |
|---|---|---|---|---|---|
| **Short CoT** (64 prompts × 8 samples × 500 tok/step) | 77M | 12.8 h, **$2.83** | 4.1 h, **$2.20** | 4.3 h, $2.50 | 26 h (≈1 Kaggle week) |
| Long CoT (× 3,000 tok) | 461M | 151 h, $34 | 33 h, $18 | 32 h, $18 | 276 h ✗ |

These are model estimates: generation at 35% of min(compute, bandwidth) bound, plus an 8N policy update.
**Empirical anchors:** Tina ≈ **$8 for 450 steps** of long-CoT LoRA-GRPO on 2×L40S at ~$1/GPU-h (≈8 GPU-h;
Modal L40S equivalent ≈ $4.3). Open-RS = 96 A40-h ($42). The thesis setting (GSM8K/algebra, short CoT)
is the **short-CoT row: ≈ $2–3 per GRPO run on Modal**.

### (c) Evaluation, ~2,000 problems, vLLM offline, greedy

| Output length | Model | Laptop (WSL) | T4 | L4 | L40S |
|---|---|---|---|---|---|
| ~400 tok (short) | 0.6B | ~13 min | ~1 min | <1 min | <1 min |
| | 1.7B | OOM | ~3.4 min | **~2.2 min** | ~0.6 min |
| | 4B | OOM | ~20 min | ~9 min | ~1.4 min |
| ~3,000 tok (long CoT) | 1.7B | OOM | ~2 h | ~1.3 h ($0.28) | ~12 min ($0.11) |

- The model is optimistic for tiny models on big GPUs, because they are CPU/scheduler-bound **[ESTIMATE]**.
  A measured anchor: DeepSeek-R1-Distill-Qwen-1.5B with vLLM `benchmark_throughput` gives **5,418 output
  tok/s** at 128-token outputs (GPU not stated) ([cnblogs](https://www.cnblogs.com/SunShine110/p/19086381)).
  **Plan with 3–5k tok/s for 1.7B on L4. A short-answer eval of 2k items is then ≈3–5 min, under $0.02.**
- Evaluation cost is negligible. The real cost is the per-container overhead, so evaluate several
  checkpoints per container.

---

## 6. Comparison and recommendation

### 6.1 Training-recipe options

| Option | Expected quality at 0.6–1.7B | Fits C2 (data comparison)? | Cost per run (Modal) | Risk |
|---|---|---|---|---|
| **A. SFT, bf16 LoRA r=64 all-linear (recommended)** | ≈ full FT at 5–25M tokens (LoRA Without Regret) | ✅ single, clean instrument | $0.03–0.45 | low |
| B. SFT, 4-bit QLoRA r=16 (old laptop recipe) | slightly below A. Attention+MLP r=16 can hit capacity limits **[ESTIMATE]** | ⚠ mixing with bf16 runs at 1.7B confounds size with precision | $0 (laptop, ~2–3 h) | medium |
| C. SFT full FT | ≈ A | ✅ | 1.5–2× A (needs L40S/A100 at 1.7B) | low. Use as a 1-run sanity check only |
| D. SFT → GRPO (RLVR) | + on top of A, but partly elicitation; spurious-reward risk on Qwen | ❌ as main instrument (second variance source, washes out arm differences) | +$2–3 per GRPO run | medium–high |
| E. RFT (self rejection-sampling) | + over SFT for weak models (RFT paper) | ❌ it is a *data* method, so it would be another arm | +vLLM sampling | low |
| F. On-policy distillation | best known per FLOP | ❌ not a data arm; teacher must be online | needs 8B+ teacher on GPU | medium |

### 6.2 Compute-plan options

| Plan | Monthly $0 capacity | Reproducible by third party? | Precision consistency | Verdict |
|---|---|---|---|---|
| **P1. Modal-only for all training & eval (laptop = dev)** | 135 L4-h / 55 L40S-h | ✅ code + pinned image. Needs their own free Modal account | ✅ bf16 everywhere | **Primary** |
| P2. Old plan: laptop 0.6B (QLoRA) + Modal 1.7B | laptop ∞ + Modal | ⚠ laptop env is hard to replicate | ❌ 4-bit vs bf16 | Retire |
| P3. Kaggle T4×2 (⚠ previously excluded by user) | ~130 T4-h | ✅✅ public notebook, no card, anyone can fork | ⚠ fp16, no FA2 | **Fallback**, and the reproduction path, if the user re-allows it |
| P4. Colab free | unknown | ✅ | fp16 | Debug only |
| P5. VNU lab GPU / Modal academic grant | potentially large | ⚠ | ✅ | Apply in parallel (free, no downside) |

### 6.3 Recommended training recipe (one recipe, frozen for all arms)

1. **Base models:** **Qwen3-0.6B-Base + Qwen3-1.7B-Base** (unchanged), plus **Qwen3-4B-Base as a third
   size**. It now costs <$1 per run, and a third size makes "arm ranking is size-stable" much stronger.
   Add **Gemma-3-1B-pt as the second family** (required for Q2; see §6.6).
   *Note:* Qwen3.5 small bases exist since 2026-02 (0.8B 0.87B params, 2B 2.27B, 4B 4.66B; Apache-2.0;
   hybrid Gated-DeltaNet + attention) ([HF API listing](https://huggingface.co/api/models?author=Qwen&search=Qwen3.5),
   [Qwen3.5-2B-Base card](https://huggingface.co/Qwen/Qwen3.5-2B-Base)). Unsloth notes Qwen3.5 needs
   `fast_inference=False` (no vLLM in its RL path), which signals tooling immaturity. **Keep Qwen3 for C2**
   and use Qwen3.5 only as leaderboard rows. Base-model choice belongs to the model-selection report; this
   is only flagged here.
2. **Method:** SFT, **completion-only loss** on a prompt–completion dataset (TRL default
   `completion_only_loss=None` → completion-only for prompt-completion data; [TRL SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer)).
   Use **packing** with `packing_strategy="bfd"` (TRL: BFD packing ⇒ padding-free, FA2/3 required; Unsloth
   has its own packing path).
3. **Adapter:** **bf16 LoRA, r = 64, α = 64–128, all linear layers (q,k,v,o,gate,up,down)**, dropout 0.
   No 4-bit. (LoRA Without Regret: MLP inclusion matters most, and rank barely shifts the optimal LR.)
4. **Optimiser:** AdamW, **LR ≈ 2e-4** (≈10× a full-FT LR of 2e-5, the TRL SFT default), cosine, ~3%
   warm-up, **effective batch 16–32 sequences** (LoRA is less tolerant of large batches).
   **Run one LR sweep {1e-4, 2e-4, 4e-4} × epochs {1, 2, 3} on arm S0 per size, then freeze.** Cost
   ≈ 18 small runs ≈ $2.
5. **Seq length:** `max_length` = p99 of the training-text length (expected ≤1,024 for short CoT). 4,096 only
   if an arm needs it. Keep the S2 generator on **short CoT** (learnability gap).
6. **Seeds/variance:** ≥3 seeds for primary arms, bootstrap CIs (unchanged).
7. **One sanity run:** full FT vs LoRA on S0 @1.7B (L40S, ~$0.2) to show LoRA is not the bottleneck.
8. **Eval:** vLLM ≥0.33 on Linux, LoRA merged to bf16, greedy, `math_verify` (unchanged).
9. **Stack (pin exact versions in a lock file and the Modal image):** unsloth 2026.11.3, trl 1.17.1,
   transformers 5.22.2, peft 0.21.2, torch 2.15.0, vllm 0.33.0, math-verify 0.9.0 (versions as of today).
   The mutual compatibility of these exact pins is **[UNVERIFIED]**. Freeze whatever resolves in the first
   working image (`pip freeze` → `requirements.lock`) and record the image digest.
10. **RLVR second stage (RQ-F), separate from the SFT comparison.**
    - Setup: GRPO-LoRA (Unsloth/TRL, `loss_type="dr_grpo"`, `beta=0`, 8 samples/prompt, ~300 steps,
      32–64 prompts/step, `math_verify` reward).
    - Starting checkpoints: **S0/S1/S2/S5 SFT checkpoints × 3 seeds × {Qwen3-1.7B, Gemma-3-1B}**.
    - Mandatory control: a **random-reward run** × 3 seeds per family (spurious-rewards paper).
    - Question it answers: "does the arm advantage survive RL?"
    - Cost: ≈ 30 runs, **$50–100** (see §6.6).
    - The minimal variant (3 runs, ≈ $8) remains the fallback if the Modal cycles run out.

### 6.4 Recommended compute plan

- **Primary — Modal Starter, $30/month, never add a payment method (unchanged rule).**
  - Use **L4 for 0.6B/1.7B SFT + all eval**. Use **L40S for 4B SFT, the full-FT check and GRPO** (better
    $/FLOP: $0.0015 vs $0.0018 per TFLOPS-h).
  - One Modal app with `train(arm, size, seed)` and `evaluate(ckpt_list)` functions. HF weights are cached
    in a `modal.Volume`, and adapters and predictions are pushed to a Volume, then to HF Hub (private) for
    the hand-off.
- **Laptop:** data pipeline, Z3/SymPy, unit tests, and 50-step **smoke tests** of the exact training script
  on 0.6B. Use Windows-native Unsloth in a fresh venv, or WSL2 with `.wslconfig` memory raised.
  - Under the **Q2 grid (§6.6)** it also trains the **entire Qwen3-0.6B block in bf16 LoRA** (same precision
    as Modal, whole base on one machine), about 100 laptop-hours.
  - Its evaluation still runs on Modal L4.
- **Fallback (needs the user's OK — Kaggle was excluded earlier):**
  - Kaggle T4×2 with the same script in fp16 if Modal credit or terms change. If used, rerun **an entire
    size** there (never mix hardware or precision within a size).
  - Publish a Kaggle notebook version anyway as the zero-signup-cost reproduction path in the hand-off.
- **Apply in parallel (free, no downside):** Modal academic credits ("up to $10k"), and ask the advisor
  about VNU lab GPUs. The NVIDIA grant needs a faculty PI and is currently closed.

### 6.5 Minimal grid at $0 (one Modal cycle) — *superseded by §6.6 under the Q2 / no-scope-cut constraint*

| Block | Runs | L4-equiv-h (×2 safety) | $ |
|---|---|---|---|
| LR/epoch sweep on S0 (2 sizes × 9) | 18 | ~6 | ~1.5 |
| Primary: S0/S1/S2/S5 × {0.6B, 1.7B} × 3 seeds | 24 | ~25 | ~5.5 |
| Third size: S0/S1/S2/S5 × 4B × 2 seeds (L40S) | 8 | ~10 (L40S ~4 h) | ~2.5 |
| Ablations S3/S4/S6 × 1.7B × 2 seeds | 6 | ~6 | ~1.3 |
| Data-efficiency S0/S1/S2 × {2k,5k,10k} × 1.7B × 2 seeds | 18 | ~10 | ~2.2 |
| English control S0–S3 × 1.7B × 2 seeds | 8 | ~7 | ~1.5 |
| Full-FT sanity (L40S) | 1 | ~1 | ~0.3 |
| Eval of all checkpoints + leaderboard rows ≤4B | — | ~10 | ~2.2 |
| *Optional* GRPO probe (3 runs, L40S) | 3 | ~13 (L40S) | ~7 |
| **Total** | **~86 runs** | | **≈ $17 (without GRPO) / ≈ $24 (with)** of $30 |

Even with all estimates 2× off on top of the 2× safety factor, the core grid (without GRPO) is ≈ $34. That
fits **two** monthly cycles, and it still costs $0.

### 6.6 Q2-sized grid (user constraint: no scope cuts) and how to fit it into $0

**What a Q2 reviewer will expect, and what the evidence adds:**

- **≥3 seeds on every reported arm**, not only the primary ones.
- **≥2 model families**, not just two Qwen sizes. The spurious-rewards paper
  ([arXiv:2506.10947](https://arxiv.org/abs/2506.10947)) shows Qwen-specific effects that "fail to produce
  gains" on Llama3/OLMo2. A Qwen-only result invites exactly that objection.
- **Proper baselines:** S0 (no augmentation), S5 (translate-train), S3 (unchecked LLM).
- **Learning-rate tuning that is visibly fair across arms.**
- **A data-efficiency curve with ≥4 points.**
- **RLVR as a second stage.** The 2026 evidence says SFT initialisation changes the post-RL ceiling:
  - OPD-then-RL beats SFT-then-RL ([arXiv:2609.04108](https://arxiv.org/abs/2609.04108));
  - "students initialized with OPD reach higher final performance after RL"
    ([arXiv:2609.28145](https://arxiv.org/abs/2609.28145)).

  So "*does the augmentation advantage survive RLVR?*" is a legitimate question a reviewer can ask. Add it
  as **RQ-F**, reported separately and **with a random-reward control**. Keep the SFT-only C2 comparison as
  the primary instrument.

**Second family.** Use **Gemma-3-1B-pt**: 1.00B params, Gemma license, gated with manual accept
(`family_candidates.txt`, HF API). Gemma 3 is multilingual, but **Vietnamese pretraining coverage is
[UNVERIFIED]**.

| Candidate | Pros | Cons |
|---|---|---|
| Llama-3.2-1B | — | Vietnamese is not among its officially supported languages **[UNVERIFIED]** |
| OLMo-2-1B (1.48B, Apache, ungated) | fully open data | weak multilingual **[UNVERIFIED]** |
| Gemma-4-E2B | — | **5.1B total params** (any-to-any, per-layer embeddings), too heavy for the grid |

**Grid and cost.** From `scratchpad/infra/q2_grid.py` → `q2_grid_out.txt`.

- Per run: 10k samples × 450 tokens × **2 epochs**, plus 0.15 h overhead and 0.10 h eval.
- **×2 safety** on SFT, ×1.5 on RLVR.
- GRPO per run = 300 steps × 64 prompts × 8 samples × 500 tokens.

| Block | Runs | Hardware | GPU-h | Modal-$ equiv |
|---|---|---|---|---|
| LR × epoch sweep (9 cfg on S0) for Qwen3-0.6B / 1.7B / Gemma-1B / Qwen3-4B | 36 | L4 / L40S | 58 | 18.0 |
| **Core: 7 arms (S0–S6) × 3 seeds × 4 bases** | 84 | L4 (0.6B, 1.7B, G-1B), L40S (4B) | 135 | 41.9 |
| **Data-efficiency: S1/S2/S3/S5 × {2k,5k,10k,20k} × 3 seeds × {Qwen3-1.7B, Gemma-1B}** | 96 | T4 | 304 | 49.9 |
| English control: S0–S3 × 3 seeds × Qwen3-1.7B | 12 | L4 | 26 | 5.7 |
| Full-FT vs LoRA sanity (S0 + best arm, 1.7B) | 2 | L40S | 2.5 | 1.3 |
| **SFT subtotal** | **230** | | | **≈ $117** |
| **RLVR (RQ-F): GRPO-LoRA from S0/S1/S2/S5 SFT ckpts × 3 seeds × {Qwen3-1.7B, Gemma-1B} + random-reward control × 3 seeds × 2 bases** | 30 | L40S | 184 | **≈ $100** (64 prompts/step) / **≈ $50** (32 prompts/step) |
| **Total if everything ran on Modal** | **260** | | | **≈ $167–217** ≈ **5.6–7.2 Modal cycles** |

**Fitting it into $0 by combining every free source.** Rule: **one base = one hardware/precision for its
whole SFT block**; each sub-experiment is internally homogeneous; **all evaluation runs on Modal L4**
(one inference platform).

| Source | Assigned work | Load | Free capacity | Fits? |
|---|---|---|---|---|
| **Laptop RTX 3050 Ti** (bf16 LoRA, same precision as Modal; chunked CE / seq ≤1k) | **Entire Qwen3-0.6B block**: sweep 9 + core 21 = 30 runs, plus smoke tests | 30 × ~3.4 h ≈ **100 laptop-h [ESTIMATE]** | ~8 h/night → **~2 weeks** | ✅ saves ≈ $7 of Modal |
| **Kaggle T4×2** (⚠ needs the user to lift the earlier exclusion) | **Entire data-efficiency block** (96 runs, fp16, own 10k point so it is internally consistent). Run **2 independent jobs per T4×2 session**, one per GPU. One session-hour reportedly costs 1 quota-hour for both GPUs **[UNVERIFIED, secondary source]** | 304 T4-h → ~152 session-h | 30 h/wk → **~5 weeks** | ✅ saves ≈ $50 of Modal |
| **Modal $30/month** | Qwen3-1.7B, Gemma-1B, Qwen3-4B sweeps + cores, English control, full-FT check, **all evaluation**, **RLVR** | SFT ≈ $60 + RLVR $50–100 | $30/cycle | **≈ $110–160 → 4–5.3 cycles** ⛔ **binding** |
| Colab free | overflow smoke tests and debugging only | — | unplannable | — |
| **Modal academic credits** (apply now; "up to $10k") | would absorb all of the above | — | — | removes the bottleneck if granted |
| VNU lab GPU (ask advisor) / NVIDIA AGP (faculty PI, currently closed) | RLVR block | — | unknown | contingent |

**Where the binding bottleneck is:**

1. **Modal's $30/month cap, i.e. calendar time, not dollars.**
   - At $0 the Modal-assigned work needs **4–5 consecutive monthly cycles**.
   - The RLVR stage alone is **$50–100 = 45–60%** of it. It can only run on bf16-capable ≥24 GB GPUs.
   - The same RLVR block on Kaggle T4 would be ~1,180 T4-h ≈ 20+ weeks even with 2 jobs per session.
     **Kaggle cannot absorb RLVR.**
2. **Kaggle's weekly quota** (~5 weeks for the data-efficiency block). It runs in parallel with Modal, so
   it is not on the critical path.
3. Laptop throughput (~2 weeks), also in parallel.
4. **Non-GPU [outside this report's scope]:** NIM generation of 7 arms × up to 20k items at ~35 s/call and
   40 RPM (RESEARCH_PLAN §6.2), and the human MOS ratings. These may bind before GPUs do. Run generation
   first.

**Schedule (if the GPU phase starts Dec 2026).**

| Modal cycle | Modal work | In parallel |
|---|---|---|
| **Dec** | sweeps + Qwen3-1.7B core + Gemma-1B core (≈ $28) | laptop runs the 0.6B block; Kaggle starts the data-efficiency block |
| **Jan** | Qwen3-4B sweep + core (≈ $29) | |
| **Feb** | English control + full-FT check + first half of RLVR (≈ $30) | |
| **Mar** | second half of RLVR (+ **Apr** if 64 prompts/step) | |

This lands all numbers by **Mar–Apr 2027**, which is compatible with the Q1–Q2 2027 ARR target in §10,
but with no slack. **Recommendation: apply for Modal academic credits immediately.** If granted, everything
collapses into ~1–2 weeks of wall-clock.

**Efficiency levers that do *not* cut scope.** Apply in this order.

1. GRPO at 32 prompts/step instead of 64. This halves RLVR. It is within the published range: Tina's best
   checkpoint is at 450 steps; 1-shot RLVR works with tiny data.
2. Early-stop RLVR on a dev-set plateau.
3. Pack 2–4 short eval jobs per container.
4. Run Qwen3-4B on L40S, not L4 (better $/FLOP).
5. Measure real tokens/s in the first smoke run and re-plan. All numbers above are ±2× estimates.

---

## 7. Changes vs the Sept-2026 plan (for §6.3 / §7 / §10)

| Old | New | Why (evidence) |
|---|---|---|
| Modal L4 $0.80/h, 1.7B grid ≈ $6–10 | **L4 $0.222/h, L40S $0.542/h**. Whole grid incl. 4B ≈ $17 | modal.com/pricing (2026-11-07) |
| 0.6B on laptop, 4-bit QLoRA, 30–50 min/run | 0.6B in **bf16** LoRA (not 4-bit). Minimal grid: on Modal. Q2 grid: the whole 0.6B block on the laptop (~3.4 h/run, ~100 h). Evaluation always on Modal | precision confound; laptop estimate 1.7–3.4 h/run; 7.7 GB RAM; vLLM Linux-only |
| one family (Qwen3) | + **Gemma-3-1B-pt** second family (Q2) | Qwen-specific effects (spurious rewards) |
| ~55 runs | **230 SFT + 30 RLVR runs** (Q2, §6.6), 4–5 Modal cycles + laptop + Kaggle | user constraint: Q2, no scope cuts |
| LoRA r=16 | r=64 all-linear, LR ≈ 10× full-FT, batch 16–32, one LR/epoch sweep on S0 | LoRA Without Regret |
| "1 epoch" fixed | epochs ∈ {1,2,3} chosen once on S0, then frozen | TRL default is 3 epochs. Small data **[ESTIMATE]** |
| No DPO/RL | **unchanged for the C2 SFT comparison**. Add **RQ-F: RLVR second stage** (GRPO-LoRA from S0/S1/S2/S5 SFT checkpoints × 3 seeds × 2 families, with a random-reward control), reported separately | spurious rewards; 1-shot RLVR; Tina cost; SFT/OPD-init → RL ceiling papers (2609.04108, 2609.28145) |
| Two sizes | two required + **optional Qwen3-4B** | cost now <$1/run |
| No Kaggle | still excluded *pending user decision*. Recommended as **fallback + public reproduction notebook** | §4 |
| — | Short-CoT generation for S2 (no long thinking traces) | learnability gap (≤3B) |

## 8. Open items / unverified

- The Kaggle quota and phone-verification requirement come from secondary sources (the official docs are
  JS-rendered). Check the account settings page before relying on it.
- Laptop TFLOPS (core count, bus width) and all MFU/throughput figures are **estimates**. Measure them in
  the first Modal smoke run (tokens/s logged by the trainer) and replace §5.
- The exact version pins in §6.3(9) were not install-tested here (no heavy installs were allowed).
- Modal Starter terms (whether a card is ever required) were not re-checked beyond the pricing page. The
  plan's existing rule (no payment method) stands.

## Sources

Compute and pricing:

- https://modal.com/pricing
- https://modal.com/academics
- https://research.google.com/colaboratory/faq.html
- https://www.kaggle.com/docs/efficient-gpu-usage
- https://www.kaggle.com/product-feedback/173129
- https://gpuperhour.com/blog/free-cloud-gpus-and-credits
- https://lightning.ai/pricing
- https://huggingface.co/docs/hub/spaces-zerogpu
- https://huggingface.co/docs/inference-providers/pricing
- https://sites.research.google/trc/about/
- https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program.md

GPU specifications:

- https://www.nvidia.com/en-us/data-center/l4/
- https://www.nvidia.com/en-us/data-center/tesla-t4/
- https://www.nvidia.com/en-us/data-center/products/a10-gpu/
- https://www.nvidia.com/en-us/data-center/l40s/
- https://www.nvidia.com/en-us/data-center/a100/

Libraries:

- https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
- https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- https://github.com/unslothai/unsloth
- https://huggingface.co/docs/trl/main/en/grpo_trainer
- https://huggingface.co/docs/trl/main/en/sft_trainer
- https://verl.readthedocs.io/en/latest/advance/ppo_lora.html
- https://github.com/OpenRLHF/OpenRLHF
- https://github.com/hiyouga/LLaMA-Factory
- https://docs.axolotl.ai/docs/rlhf.html
- https://github.com/pytorch/torchtune
- PyPI JSON API `https://pypi.org/pypi/<pkg>/json`

Methods and papers:

- https://thinkingmachines.ai/blog/lora/
- https://thinkingmachines.ai/blog/on-policy-distillation/
- arXiv 2504.15777, 2503.16219, 2504.20571, 2503.20783, 2503.14476, 2507.18071, 2502.12143, 2506.10947,
  2506.08935, 2308.01825, 2505.09388, 2606.04466, 2609.04108, 2609.28145
- https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview
- https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

Models and throughput anchor:

- https://huggingface.co/Qwen/Qwen3.5-2B-Base
- HF API `https://huggingface.co/api/models/<id>`
- https://www.cnblogs.com/SunShine110/p/19086381

Local evidence:

- `scratchpad/infra/local_machine.txt`
- `pypi_versions.txt`
- `hf_params.txt`
- `qwen3_configs.txt`
- `qwen35_models.txt`
- `family_candidates.txt`
- `estimate.py`
- `estimate_out.txt`
- `q2_grid.py`
- `q2_grid_out.txt`

The full scratch path is
`C:\Users\tranh\AppData\Local\Temp\claude\E--MATH-REASONING-THESIS-vi-math-smt\884a7b4b-700c-4c10-ad97-d56477e30ce1\scratchpad\infra\`.
Copy it into the repo if these files should become provenance.
