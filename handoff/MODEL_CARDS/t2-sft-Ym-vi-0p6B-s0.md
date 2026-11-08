---
# Hugging Face model-card YAML (06_packaging.md §2.4). Filled by handoff/tools/fill_model_cards.py.
license: apache-2.0
language: [vi]
library_name: peft
pipeline_tag: text-generation
base_model: Qwen/Qwen3-0.6B-Base
base_model_relation: adapter
datasets: [vi-math-smt/Y--vi]
tags: [math, vietnamese, lora, vi-math-smt, arm-Y-]
---

# Qwen/Qwen3-0.6B-Base — arm Y-, VI, seed 0 (t2-sft-Ym-vi-0p6B-s0)

> Internal hand-off card (not yet released). Generated from `jobs/outputs/t2-modal/t2-sft-Ym-vi-0p6B-s0/manifest.json` plus `jobs/queue/t2_state.json` (eval metrics).
> Status of the run: **COMPLETE**. Do not cite numbers from a card whose status is not COMPLETE.

## Model description
- **Run id:** `t2-sft-Ym-vi-0p6B-s0` (kind `sft`, config hash `b3ad2d08`)
- **Base model:** `Qwen/Qwen3-0.6B-Base` (0.6B, 616,235,008 parameters); Qwen3 base models are Apache-2.0.
- **Arm:** `Y-` — symbolic arm, unverified (wave-1 first item per seed, format/leak/decontam gates only)
- **Seed:** 0
- **Adapter:** LoRA r 32 / α 32 on all text-decoder linear layers (down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj); trainable parameters
  20,185,088.
- **Where the adapter is:** local copy `jobs/outputs/t2-modal/t2-sft-Ym-vi-0p6B-s0/adapters/epoch3/` (in the `runs` bundle; sha256 in `handoff/ADAPTERS.sha256`); Modal Volume `t1r-eval`, path `/train/t2-sft-Ym-vi-0p6B-s0/adapters/epoch3` (pull: `modal volume get t1r-eval /train/t2-sft-Ym-vi-0p6B-s0/adapters/epoch3 jobs/outputs/t2-modal/t2-sft-Ym-vi-0p6B-s0/adapters/epoch3`; see `handoff/BUNDLE.md` §Adapters).

## Intended use and limitations
- Research artifact of the vi-math-smt thesis (symbolic vs LLM-generated training data for small models,
  Vietnamese math). Intended for measuring arm effects (P1–P3, H1–H4), **not** for tutoring or grading students.
- Small base model (≤ 1.7B): makes arithmetic and reasoning errors; answers in the template it was trained on
  (VI `Bài toán / Lời giải / Đáp số: \boxed{}`, EN `Problem / Solution / Final answer: \boxed{}`); not
  instruction-tuned for chat; no safety tuning.
- Do not compare dev-set NLL across arms (KNOWN_ISSUES M-1: it measures solution style); compare accuracy.
- Trained on 1,808 GSM8K-seeded items per arm: expect gains on GSM8K-style word problems, not on MATH/exam
  algebra beyond what the data covers (S0 lowers MATH and Vi-ExamMath accuracy vs the 4-shot base; R1 §0).
- Evaluated only on the frozen sets listed below; no claim about other domains, grades or languages.

## Training data
- Pool / file: `data/train/final/Y/vi/train_minus.jsonl` — datasheet: `handoff/DATASHEETS/train_pools.md`
- Items used: 1808 (pool 1808, dropped as too long 0, max_length 1024);
  data fingerprint `517091c9a0b1fa9c`.
- **Provenance of LLM-generated text:** problems and solutions were written by Anthropic Claude models (`claude-haiku-5-5` generator/renderer/solution writer/screen; ids per stage in the datasheet; every call logged under `logs/api/<WS>/claude.jsonl`). Model ids, prompt hashes and API-log locations per stage: `handoff/DATASHEETS/README.md`.

## Training procedure
- Platform modal_l40s, precision bf16; engine unsloth (packed: True); micro-batch 32
  × accumulation 1; gradient checkpointing False. Code: git `f427fcad49a3` (dirty: False).
- Schedule: 168 steps (56/epoch × 3 epochs), warm-up 5, decay 17 (WSD); hedge branches at epochs [1, 2] (stored, not reported)
- Outcome: 185 steps executed, skipped 0 (0.0), non-finite losses 0;
  final dev loss 1.06434; throughput 37234.8 real tok/s; peak memory 20.75 GB.
- Environment: python 3.12.10, torch 2.14.0, transformers 5.20.2, peft 0.21.2, trl 1.17.1, vllm 0.33.0, gpu NVIDIA L40S, cuda 13.0
- Started 2026-11-08T00:22:40Z, last update 2026-11-08T00:24:24Z.

## Evaluation
Grader: `eval/math_verify.py` 3.0.2 (math-verify 0.9.0 backend). Decoding: primary sets: K = 4 samples (T 0.6, top-p 0.95, top-k 20, one seed per request) + greedy, cap 1,024 tokens; Vi-ExamMath Core VERIFIED: greedy only (deviation from T5, cost); vLLM 0.33 bf16 on Modal L40S, `VLLM_BATCH_INVARIANT=1`; 0-shot in the training template (base rows: 4-shot).

| Eval run | Dataset | n | pass@1 over K=4 [95 % CI] | greedy [95 % CI] | item-level preds |
|---|---|---|---|---|---|
| `ev-t2-sft-Ym-vi-0p6B-s0-exam-greedy` | vi_exam_core_verified | 1367 | n/a | 32.1 [29.5, 34.7] | `jobs/outputs/t2-eval/ev-t2-sft-Ym-vi-0p6B-s0-exam-greedy.jsonl` |
| `ev-t2-sft-Ym-vi-0p6B-s0-primary` | gsm8k_platinum | 600 | 43.9 [40.8, 47.2] | 46.8 [42.8, 50.8] | `jobs/outputs/t2-eval/ev-t2-sft-Ym-vi-0p6B-s0-primary.jsonl` |
| `ev-t2-sft-Ym-vi-0p6B-s0-primary` | math_algprealg | 400 | 37.2 [33.4, 40.9] | 41.2 [36.2, 46.2] | `jobs/outputs/t2-eval/ev-t2-sft-Ym-vi-0p6B-s0-primary.jsonl` |

Per-run CIs are item-bootstrap CIs of this single run. Arm comparisons across seeds (lme4 GLMM + cluster bootstrap,
Holm, TOST) are in `docs/research_2026-11/exec/R1_results.md`. Adapters are evaluated 0-shot in their training
template: compare them with the **0-shot base** card (`base-qwen3-0.6b-0shot`); a difference to the 4-shot base mixes
training with prompt format.

## Reproduce
```
# config: jobs/outputs/t2-modal/t2-sft-Ym-vi-0p6B-s0/config.yaml
python -m vi_math_verified_aug.train.launcher ...   # cell runner; see handoff/RUNBOOK.md
```

## License and notices
- Adapter weights: Apache-2.0 (inherits Qwen3 base license; modified by the vi-math-smt project).
- Training data terms: see the datasheet above (`handoff/DATASHEETS/`).
- Contact / maintainer: see `handoff/README.md`.
