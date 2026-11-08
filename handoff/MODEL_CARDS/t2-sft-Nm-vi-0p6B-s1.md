---
# Hugging Face model-card YAML (06_packaging.md §2.4). Filled by handoff/tools/fill_model_cards.py.
license: apache-2.0
language: [vi]
library_name: peft
pipeline_tag: text-generation
base_model: Qwen/Qwen3-0.6B-Base
base_model_relation: adapter
datasets: [vi-math-smt/N--vi]
tags: [math, vietnamese, lora, vi-math-smt, arm-N-]
---

# Qwen/Qwen3-0.6B-Base — arm N-, VI, seed 1 (t2-sft-Nm-vi-0p6B-s1)

> Internal hand-off card (not yet released). Generated from `jobs/outputs/t2-modal/t2-sft-Nm-vi-0p6B-s1/manifest.json` plus `jobs/queue/t2_state.json` (eval metrics).
> Status of the run: **COMPLETE**. Do not cite numbers from a card whose status is not COMPLETE.

## Model description
- **Run id:** `t2-sft-Nm-vi-0p6B-s1` (kind `sft`, config hash `8fc8e6f4`)
- **Base model:** `Qwen/Qwen3-0.6B-Base` (0.6B, 616,235,008 parameters); Qwen3 base models are Apache-2.0.
- **Arm:** `N-` — LLM-native arm, unverified (first slot per seed, parse/format/decontam gates only)
- **Seed:** 1
- **Adapter:** LoRA r 32 / α 32 on all text-decoder linear layers (down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj); trainable parameters
  20,185,088.
- **Where the adapter is:** local copy `jobs/outputs/t2-modal/t2-sft-Nm-vi-0p6B-s1/adapters/epoch3/` (in the `runs` bundle; sha256 in `handoff/ADAPTERS.sha256`); Modal Volume `t1r-eval`, path `/train/t2-sft-Nm-vi-0p6B-s1/adapters/epoch3` (pull: `modal volume get t1r-eval /train/t2-sft-Nm-vi-0p6B-s1/adapters/epoch3 jobs/outputs/t2-modal/t2-sft-Nm-vi-0p6B-s1/adapters/epoch3`; see `handoff/BUNDLE.md` §Adapters).

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
- Pool / file: `data/train/final/N/vi/train_minus.jsonl` — datasheet: `handoff/DATASHEETS/train_pools.md`
- Items used: 1808 (pool 1808, dropped as too long 0, max_length 1024);
  data fingerprint `a3bf2eeb9982eb4f`.
- **Provenance of LLM-generated text:** problems and solutions were written by Anthropic Claude models (`claude-haiku-5-5` generator/renderer/solution writer/screen; ids per stage in the datasheet; every call logged under `logs/api/<WS>/claude.jsonl`). Model ids, prompt hashes and API-log locations per stage: `handoff/DATASHEETS/README.md`.

## Training procedure
- Platform modal_l40s, precision bf16; engine unsloth (packed: True); micro-batch 32
  × accumulation 1; gradient checkpointing False. Code: git `e999a5653c3d` (dirty: False).
- Schedule: 168 steps (56/epoch × 3 epochs), warm-up 5, decay 17 (WSD); hedge branches at epochs [1, 2] (stored, not reported)
- Outcome: 185 steps executed, skipped 0 (0.0), non-finite losses 0;
  final dev loss 1.041528; throughput 38216.7 real tok/s; peak memory 20.75 GB.
- Environment: python 3.12.10, torch 2.14.0, transformers 5.20.2, peft 0.21.2, trl 1.17.1, vllm 0.33.0, gpu NVIDIA L40S, cuda 13.0
- Started 2026-11-08T00:44:15Z, last update 2026-11-08T00:45:57Z.

## Evaluation
Grader: `eval/math_verify.py` 3.0.2 (math-verify 0.9.0 backend). Decoding: primary sets: K = 4 samples (T 0.6, top-p 0.95, top-k 20, one seed per request) + greedy, cap 1,024 tokens; Vi-ExamMath Core VERIFIED: greedy only (deviation from T5, cost); vLLM 0.33 bf16 on Modal L40S, `VLLM_BATCH_INVARIANT=1`; 0-shot in the training template (base rows: 4-shot).

| Eval run | Dataset | n | pass@1 over K=4 [95 % CI] | greedy [95 % CI] | item-level preds |
|---|---|---|---|---|---|
| `ev-t2-sft-Nm-vi-0p6B-s1-exam-greedy` | vi_exam_core_verified | 1367 | n/a | 30.4 [27.9, 32.7] | `jobs/outputs/t2-eval/ev-t2-sft-Nm-vi-0p6B-s1-exam-greedy.jsonl` |
| `ev-t2-sft-Nm-vi-0p6B-s1-primary` | gsm8k_platinum | 600 | 45.9 [42.7, 49.2] | 49.7 [45.5, 53.7] | `jobs/outputs/t2-eval/ev-t2-sft-Nm-vi-0p6B-s1-primary.jsonl` |
| `ev-t2-sft-Nm-vi-0p6B-s1-primary` | math_algprealg | 400 | 37.1 [33.4, 40.8] | 39.2 [34.2, 44.2] | `jobs/outputs/t2-eval/ev-t2-sft-Nm-vi-0p6B-s1-primary.jsonl` |

Per-run CIs are item-bootstrap CIs of this single run. Arm comparisons across seeds (lme4 GLMM + cluster bootstrap,
Holm, TOST) are in `docs/research_2026-11/exec/R1_results.md`. Adapters are evaluated 0-shot in their training
template: compare them with the **0-shot base** card (`base-qwen3-0.6b-0shot`); a difference to the 4-shot base mixes
training with prompt format.

## Reproduce
```
# config: jobs/outputs/t2-modal/t2-sft-Nm-vi-0p6B-s1/config.yaml
python -m vi_math_verified_aug.train.launcher ...   # cell runner; see handoff/RUNBOOK.md
```

## License and notices
- Adapter weights: Apache-2.0 (inherits Qwen3 base license; modified by the vi-math-smt project).
- Training data terms: see the datasheet above (`handoff/DATASHEETS/`).
- Contact / maintainer: see `handoff/README.md`.
