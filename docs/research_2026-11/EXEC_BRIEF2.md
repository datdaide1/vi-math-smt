# Execution brief 2 — parallel independent workstreams (2026-11-07, approved direction)

Approved design: `docs/research_2026-11/01_TECHNICAL_DESIGN.md` (symbolic data × Vietnamese math; P1 faithfulness,
P2 verification premium × language with English mirror `exec/E1_english_mirror.md`, P3 coverage). Tech details:
`tech/T1..T6`. Content reviews: `content_review/`. Running in parallel elsewhere (do NOT touch their files):
G1 generator pilot (`vi_math_verified_aug/pilot_gen/`, `vi_math_verified_aug/llm_aug/`, `data/pilot/`),
S2 student pilot (`vi_math_verified_aug/infra/`, `jobs/`).

## Ownership
| WS | Owns |
|---|---|
| X1 grader | `vi_math_verified_aug/eval/math_verify.py`, `tests/test_math_verify.py`, `tests/grader_cases/` |
| X2 benchmark collection | `vi_math_verified_aug/vi_exam/`, `data/vi_exam/` |
| X3 eval sets | `vi_math_verified_aug/eval_sets/` (new), `data/eval/` (new). May read `data/clean/` (partial earlier work) |
| X4 symbolic IR library | `vi_math_verified_aug/symbolic/`, `tests/test_symbolic*.py` |
| X5 decontamination | `vi_math_verified_aug/eval/decontam.py`, `tests/test_decontam.py`, `data/contam/` |
Reports: `docs/research_2026-11/exec/<WS>_*.md`. Partial uncommitted code from stopped agents may exist in your
area: review it, keep what is sound, rewrite what is not. Do not git commit (the lead commits).

## Laptop vs cloud (rule added 2026-11-07 after RAM hit 0.02 GB)
- Laptop (7.7 GB RAM, much of it used by the desktop app): code, unit tests, light CPU scripts only. Each process
  ≤ ~300 MB, one Python process per workstream at a time, never import torch/transformers/TRL locally.
- Anything heavy goes to the cloud: GPU work (embeddings, training, eval, GRPO) → private Kaggle kernel; heavy
  CPU (full-size GLMM fits, big scans) → Kaggle CPU kernel (no GPU quota cost). Kaggle use must be coordinated
  with the lead (quota), and follow S2's reliability rules (incremental outputs, ENVCHECK, per-task time budgets,
  pinned versions, private kernels/datasets).

## Modal budget (user, 2026-11-07): pick the GPU by performance/quality/speed, not T4
Real rates (`modal billing rates`): T4 0.59 · L4 0.80 · A10G 1.10 · L40S 1.95 · A100-40 2.10 · A100-80 2.50 ·
H100 3.95 $/h; CPU 0.047 $/core-h. Choose by cost per finished job (throughput × price + ~2–5 min cold start),
not by hourly price. Default picks: inference/embeddings → **L4** (bf16, FA2, 24 GB, cheapest modern GPU);
training ≤2B or GRPO → **L40S** (best $/TFLOP per T3); ≥4B or long-context → **A100-80/H100** only when it
shortens wall-clock enough to pay for itself. One hardware/precision per model size and block (T3/T6 rule).
| Allocation of the $30/month | Cap |
|---|---|
| X5 embeddings (L4) | $0.50 |
| Y2 statistical fits (L4; GPU-vectorised bootstrap/simulation/NumPyro — user: GPU, not CPU-only) | $2 |
| S2/Y1 parity & throughput checks (L4/L40S) | $5 |
| Training blocks that don't fit T4 well (e.g. Qwen3-4B on L40S, full-FT vs LoRA check) | $17 |
| Reserve | $5.50 |
Log spend before/after each job with `modal billing summary`; never exceed your cap.

## Shared rules
- **No NIM calls** (reserved for G1). **No Kaggle** (reserved for S2). Gemini: not needed by any X workstream.
- Claude API only where stated in your task; key `ANTHROPIC_API_KEY` + header `anthropic-workspace-id` from
  `ANTHROPIC_WORKSPACE_ID` (`.env`); SDK in `E:\venv_tools`. Log every call under `logs/api/<WS>/`. Never print keys.
- Test items (Vi-GSM8K / Vi-MATH test, Vi-ExamMath) may go ONLY to Claude, never to NIM/Gemini.
- Python: project `.venv` (3.11) for running package code/tests; extra packages into `E:\venv_tools` or a documented
  env. Windows long-path issue: keep venvs at short paths.
- Evidence for every result (command + output). Final message ≤200 words: built, measured, blocked.
