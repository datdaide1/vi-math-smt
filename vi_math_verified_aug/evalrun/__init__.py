"""Evaluation runner (T5 §4.1, workstream Y1).

``python -m vi_math_verified_aug.evalrun.cli --config EVAL.yaml``

vLLM fp16 when importable on CUDA, else HF ``generate``.  T=0.6 / top-p 0.95 / top-k 20, K=4 samples plus one
greedy pass, max_new_tokens 2048, the SFT training template (zero-shot; 4-shot in the same template for base rows).
Writes item-level predictions JSONL (one row per item x sample) with every field the GLMM needs::

    run_id, arm, model, size, seed, lang, dataset, item_id, sample_k, correct, tokens

``sample_k`` is 0..K-1 for the sampled decodes and -1 for the greedy decode (``decoding`` says which).
"""

PRED_FIELDS = ("run_id", "arm", "model", "size", "seed", "lang", "dataset", "item_id", "sample_k", "correct",
               "tokens")
GREEDY_K = -1
