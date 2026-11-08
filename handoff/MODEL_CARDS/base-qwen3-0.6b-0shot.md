---
# Hugging Face model-card YAML (06_packaging.md §2.4). Filled by handoff/tools/fill_model_cards.py.
license: apache-2.0
language: [vi, en]
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3-0.6B-Base
base_model_relation: n/a (base model)
datasets: []
tags: [math, vietnamese, lora, vi-math-smt, arm-base]
---

# Qwen/Qwen3-0.6B-Base — base model, 0-shot in the arms template (base-qwen3-0.6b-0shot)

> Internal hand-off card (not yet released). Generated from `jobs/t2/ev-base-qwen3-0.6b-0shot-vi.metrics.json` plus `jobs/queue/t2_state.json` (eval metrics).
> Status of the run: **COMPLETE**. Do not cite numbers from a card whose status is not COMPLETE.

## Model description
- **Run id:** `base-qwen3-0.6b-0shot` (kind `base`, config hash `n/a`)
- **Base model:** `Qwen/Qwen3-0.6B-Base` (0.6B, n/a parameters); Qwen3 base models are Apache-2.0.
- **Arm:** `base` — no fine-tuning (base model; 4-shot rows, and 0-shot rows in the arms' own template)
- **Seed:** 0
- **Adapter:** LoRA r 32 / α 32 on all text-decoder linear layers (n/a); trainable parameters
  n/a.
- **Where the adapter is:** n/a.

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
- Pool / file: `n/a` — datasheet: n/a (no training)
- Items used: n/a (pool n/a, dropped as too long n/a, max_length n/a);
  data fingerprint `n/a`.
- **Provenance of LLM-generated text:** n/a

## Training procedure
- Platform n/a, precision n/a; engine n/a (packed: n/a); micro-batch n/a
  × accumulation n/a; gradient checkpointing n/a. Code: n/a.
- Schedule: n/a
- Outcome: n/a steps executed, skipped n/a (n/a), non-finite losses n/a;
  final dev loss n/a; throughput n/a real tok/s; peak memory n/a GB.
- Environment: n/a
- Started n/a, last update n/a.

## Evaluation
Grader: `eval/math_verify.py` 3.0.2 (math-verify 0.9.0 backend). Decoding: primary sets: K = 4 samples (T 0.6, top-p 0.95, top-k 20, one seed per request) + greedy, cap 1,024 tokens; Vi-ExamMath Core VERIFIED: greedy only (deviation from T5, cost); vLLM 0.33 bf16 on Modal L40S, `VLLM_BATCH_INVARIANT=1`; 0-shot in the training template (base rows: 4-shot).

| Eval run | Dataset | n | pass@1 over K=4 [95 % CI] | greedy [95 % CI] | item-level preds |
|---|---|---|---|---|---|
| `ev-base-qwen3-0.6b-0shot-vi-exam-greedy` | vi_exam_core_verified | 1367 | n/a | 18.1 [16.0, 20.0] | `jobs/outputs/t2-eval/ev-base-qwen3-0.6b-0shot-vi-exam-greedy.jsonl` |
| `ev-base-qwen3-0.6b-0shot-vi-primary` | gsm8k_platinum | 600 | 30.3 [27.8, 32.6] | 34.5 [30.7, 38.0] | `jobs/outputs/t2-eval/ev-base-qwen3-0.6b-0shot-vi-primary.jsonl` |
| `ev-base-qwen3-0.6b-0shot-vi-primary` | math_algprealg | 400 | 26.8 [23.6, 30.0] | 34.8 [30.2, 39.2] | `jobs/outputs/t2-eval/ev-base-qwen3-0.6b-0shot-vi-primary.jsonl` |

Per-run CIs are item-bootstrap CIs of this single run. Arm comparisons across seeds (lme4 GLMM + cluster bootstrap,
Holm, TOST) are in `docs/research_2026-11/exec/R1_results.md`. Adapters are evaluated 0-shot in their training
template: compare them with the **0-shot base** card (`base-qwen3-0.6b-0shot`); a difference to the 4-shot base mixes
training with prompt format.

## Reproduce
```
python -m vi_math_verified_aug.train.launcher ...   # base rows; see handoff/RUNBOOK.md
```

## License and notices
- Adapter weights: Apache-2.0 (inherits Qwen3 base license; modified by the vi-math-smt project).
- Training data terms: see the datasheet above (`handoff/DATASHEETS/`).
- Contact / maintainer: see `handoff/README.md`.
