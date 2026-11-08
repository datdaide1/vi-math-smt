---
# Hugging Face model-card YAML (06_packaging.md §2.4). Filled by handoff/tools/fill_model_cards.py.
license: {{license}}
language: [{{language}}]
library_name: {{library_name}}
pipeline_tag: text-generation
base_model: {{base_model}}
base_model_relation: {{base_model_relation}}
datasets: [{{datasets}}]
tags: [math, vietnamese, lora, vi-math-smt, {{arm_tag}}]
---

# {{title}}

> Internal hand-off card (not yet released). Generated from `{{manifest_path}}`{{metrics_note}}.
> Status of the run: **{{status}}**. Do not cite numbers from a card whose status is not COMPLETE.

## Model description
- **Run id:** `{{run_id}}` (kind `{{kind}}`, config hash `{{config_hash}}`)
- **Base model:** `{{base_model}}` ({{size}}, {{n_params}} parameters); Qwen3 base models are Apache-2.0.
- **Arm:** `{{arm}}` — {{arm_description}}
- **Seed:** {{seed}}
- **Adapter:** LoRA r 32 / α 32 on all text-decoder linear layers ({{lora_modules}}); trainable parameters
  {{trainable_params}}.
- **Where the adapter is:** {{adapter_location}}.

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
- Pool / file: `{{train_path}}` — datasheet: {{datasheet}}
- Items used: {{n_items}} (pool {{n_pool}}, dropped as too long {{n_dropped}}, max_length {{max_length}});
  data fingerprint `{{fingerprint}}`.
- **Provenance of LLM-generated text:** {{provenance}}

## Training procedure
- Platform {{platform}}, precision {{precision}}; engine {{engine}} (packed: {{packed}}); micro-batch {{micro_batch}}
  × accumulation {{accum}}; gradient checkpointing {{grad_ckpt}}. Code: {{code}}.
- Schedule: {{plan}}
- Outcome: {{steps}} steps executed, skipped {{skipped}} ({{skipped_frac}}), non-finite losses {{nonfinite}};
  final dev loss {{dev_loss}}; throughput {{tok_s}} real tok/s; peak memory {{peak_mem}} GB.
- Environment: {{env}}
- Started {{started}}, last update {{updated}}.

## Evaluation
Grader: `eval/math_verify.py` {{grader}}. Decoding: {{decoding}}.

{{eval_table}}

Per-run CIs are item-bootstrap CIs of this single run. Arm comparisons across seeds (lme4 GLMM + cluster bootstrap,
Holm, TOST) are in `docs/research_2026-11/exec/R1_results.md`. Adapters are evaluated 0-shot in their training
template: compare them with the **0-shot base** card (`base-qwen3-0.6b-0shot`); a difference to the 4-shot base mixes
training with prompt format.

## Reproduce
```
{{reproduce}}
```

## License and notices
- Adapter weights: Apache-2.0 (inherits Qwen3 base license; modified by the vi-math-smt project).
- Training data terms: see the datasheet above (`handoff/DATASHEETS/`).
- Contact / maintainer: see `handoff/README.md`.
