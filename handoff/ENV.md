# ENV — environments (laptop, Kaggle, Modal)

Checked 2026-11-07 22:30–23:00 (UTC+7). Three environments, on purpose: a **light CPU env** on the laptop (pipeline,
verification, grading, API clients, tests), and **pinned GPU images** on Kaggle and Modal (training, vLLM eval,
embeddings, GLMM fits). torch/transformers/vLLM are never installed in the laptop env.

## 1. Laptop (Windows 11, 7.7 GB RAM, often < 2 GB free) — one locked env

| Item | Value |
|---|---|
| Lock | `pyproject.toml` + **`uv.lock`** (universal: Windows + Linux, CPU-only, 50 packages); pip mirror `requirements.lock.txt` |
| Python | 3.11 (`requires-python >= 3.11`); `.venv` is 3.11.7 |
| Core deps | sympy 1.14.0, z3-solver 5.1.0.0, numpy 2.4.6, lxml 6.1.3, beautifulsoup4 4.15.0 (pinned via `[tool.uv] constraint-dependencies`), requests, python-dotenv |
| Extras | `api` (anthropic 2.6.0, httpx2), `data` (pyarrow 26.0.1, huggingface-hub), `grader` (math-verify 0.9.0, latex2sympy2_extended 1.11.0), `dev` (pytest, pytest-cov) |
| Install | `uv sync --locked --all-extras` (pip: `pip install -r requirements.lock.txt && pip install -e . --no-deps`) |

**Verification done (2026-11-07):**
- `uv lock --check` → exit 0 ("Resolved 50 packages"): `uv.lock` is consistent with `pyproject.toml` (uv 0.10.5).
- `uv export --format requirements.txt --all-extras --no-hashes --no-emit-project` is identical to
  `requirements.lock.txt` (diff empty).
- `uv sync --locked --all-extras --dry-run` → the shared `.venv` has **drifted** from the lock: it would uninstall 25
  and install 33 packages (e.g. adds anthropic, math-verify, pyarrow, huggingface-hub, **pyyaml**; removes pandas,
  openai, python-dateutil). Not applied tonight because other workstreams run from `.venv`; see KNOWN_ISSUES K-ENV-1.
  Consequence measured: `pyyaml` is missing from `.venv`, so the seven yaml-dependent `tests/test_train_*.py` files skip themselves
  there (importorskip); `handoff/test_report.md` runs them with `E:\venv_y1` instead.
- Memory: with `OPENBLAS_NUM_THREADS=1` every light test file stays ≤ 190 MB committed (`test_report.md`). Without it,
  `import numpy` alone commits ~250 MB of OpenBLAS thread buffers on this 8-thread laptop and two test files hit a
  300 MB cap. **Set `OPENBLAS_NUM_THREADS=1` (and `OMP_NUM_THREADS=1`) for every laptop process.**

**Interpreters in use tonight** (to be consolidated onto the lock; see KNOWN_ISSUES K-ENV-2):

| Env | Python | Packages | Used for | torch? |
|---|---|---|---|---|
| `.venv` (repo) | 3.11.7 | 40 | package code, most tests | no |
| `E:\venv_tools` | 3.11.7 | 77 | anthropic SDK, Kaggle CLI, Modal CLI, pyarrow, sympy | no |
| `E:\venv_y1` | 3.11.7 | 60 | Y1 train/eval tests (has pyyaml) | **yes (2.14.0+cpu)** — never import it on the laptop |
| `E:\venv_dc` | 3.11.7 | 44 | X5 decontam encoders (lazy) | **yes (2.14.1+cu126)** |
| `E:\venv_mv` | 3.11.7 | 14 | math-verify backend experiments (X1) | no |
| `E:\venv_stats` | 3.11.7 | 21 | Y2 stats experiments | no |

Target state: one `.venv` from `uv sync --locked --all-extras`; the Kaggle and Modal CLIs as uv tools
(`uv tool install kaggle`, `uv tool install modal`) or in a `tools` dependency group; the five `E:\venv_*` deleted.

## 2. Kaggle (T4×2, private kernels) — pins verified by S2 (v1, v2) and used by the T1r **pilot**

> **Platform decision (lead, 2026-11-08, R-GPU):** every comparison block (S0 + arms, VI + EN, 0.6B and up) trains and
> evaluates on **Modal L40S bf16** (§3, `t1r-eval`); Kaggle T4 fp16 is pilot-only (the same seed on another GPU flips
> 11–15 % of dev items). Kaggle remains for the LFM2.5-2.6B block (P10, T4 HF loop), CPU kernels (lme4 GLMM) and the
> indicative base 0-shot fallback. Each run's `manifest.json` records `platform`, `precision`, `env` and the git
> `code` stamp.

```
image: python 3.13.16 · driver 580.178.04 · CUDA 13.0 · torch 2.14.0+cu130 · triton 3.8.0
pip install 'vllm==0.33.0'            # ≈7.6 min per session; pulls transformers 5.20.2 · peft 0.21.2 · accelerate 1.15.0 · datasets 4.8.5
pip uninstall -y torchao              # REQUIRED: image torchao 0.10.0 makes peft >= 0.18 raise at LoRA dispatch
# Unsloth isolated so torch/vLLM stay untouched:
pip install --no-deps --target /tmp/uns_site unsloth==2026.11.3 unsloth_zoo==2026.11.3 trl==1.17.1 \
    cut_cross_entropy==25.1.1 bitsandbytes==0.50.2       # PYTHONPATH=/tmp/uns_site for the Unsloth process only
```
- Source: [`exec/S2_student_pilot.md`](../docs/research_2026-11/exec/S2_student_pilot.md) §5 and the T1r lock file (transformers 5.20.2 chosen over O1's 5.17.0 because vLLM 0.33.0 resolves it); kernel template
  `vi_math_verified_aug/train/kaggle/job_template.py`, pins file `vi_math_verified_aug/train/requirements-kaggle.txt`.
- Every kernel prints an `ENVCHECK` block into `results/meta.json` and **exits 5 before any GPU work** if a pin does
  not match (fail-fast). The Kaggle image drifts weekly: re-check the pins at the start of each week.
- Precision on T4: fp16 (no bf16 on sm_75); fp16 safety measured (no overflow; tightest 1.7B, 4.6× headroom).
- Open item: Unsloth exactness was gated on transformers 5.17.0 (O1) and re-gated per run on 5.20.2 by the
  start-of-job logits gate (T2 §0) — the per-run gate result is in each run's `pack_gate.json`.

## 3. Modal images (pinned in code; each run writes its own env into the run manifest)

| App (`modal.App`) | File | Image | Pins | GPU |
|---|---|---|---|---|
| **`t1r-eval`** (tranche training cells, eval, GRPO smoke; Volumes `t1r-eval` = data/adapters/preds, `t1r-hf-cache`) | `vi_math_verified_aug/evalrun/modal_eval.py` | debian_slim py3.12 | vllm 0.33.0, transformers 5.20.2, peft 0.21.2, accelerate 1.15.0, math-verify 0.9.0, datasets 4.8.5, pyyaml; `pip uninstall torchao`; Unsloth 2026.11.3 set in `/uns_site`; env `VLLM_USE_FLASHINFER_SAMPLER=0`, eval `VLLM_BATCH_INVARIANT=1` | **L40S pinned, no fallback** (train + eval); GRPO smoke on T4 fp16 |
| `g5-vllm-cycle`, `g5-vllm-cycle2` (Qwen3-32B-AWQ blind cycle check, chunked) | `vi_math_verified_aug/arm_y/vllm_cycle_modal*.py` | debian_slim | vllm 0.33.0, transformers 5.20.2 | L40S (A100-80 tested) |
| `g5-decontam`, `d1-n-decontam` (bge-m3) | `arm_y/decontam_modal.py`, `scale_gen/embed_modal.py` | debian_slim py3.11 | torch 2.14.1, transformers 5.22.2, pyarrow 26.0.1, numpy 2.4.6 | `["L4","A10G","L40S"]` |
| `r-gpu-bench` (FP8/bf16/compile benchmarks) | `bench/r_gpu/modal_rgpu.py` | debian_slim | as `t1r-eval` + torchao (variant) | L40S, H100, L4 |
| `qa-t-tokcheck`, `qa-t-nllprobe` | `vi_math_verified_aug/qa/` | debian_slim | as `t1r-eval` | CPU / T4 |
| `s2-student-pilot` | `vi_math_verified_aug/infra/modal_s2.py` | debian_slim py3.12 | vllm 0.33.0, transformers 5.20.2, peft 0.21.2, accelerate 1.15.0, datasets | L4/A10G/L40S/H100 (benchmarks) |
| `x5-embed` | `vi_math_verified_aug/eval/decontam_modal.py` | debian_slim py3.11 | torch 2.14.1, transformers 5.22.2, pyarrow 26.0.1, numpy 2.4.6 | L4 |
| `s3-seed-embed`, `g3-embed`, `g4-embed` | `seeds/embed_modal.py`, `pilot_symbolic/embed_modal.py`, `pilot_symbolic_v2/embed_modal.py` | debian_slim py3.11 | torch 2.14.1, transformers 5.22.2, numpy 2.4.6, huggingface_hub 2.4.0 | L4/A10G |
| `y2-analysis` | `vi_math_verified_aug/analysis/modal_jobs.py` | debian_slim py3.11 | apt r-base-core + r-cran-lme4; numpy 2.2.4, pandas 2.2.3, scipy 1.15.2, statsmodels 0.14.4, pytest 8.3.5, jax[cuda12] 0.5.3, numpyro 0.18.0 | L4 (fallback A10G/L40S) |
| `w2-gpu-check` | `vi_math_verified_aug/infra/modal_gpu_check.py` | debian_slim py3.11 | — | probe |

Rules (S2 §7, EXEC_BRIEF2): GPU fallback lists in the decorator (`with_options(gpu=list)` raises); `retries=0`;
resolve local paths only under `modal.is_local()` (an import-time path bug caused a crash loop); log
`modal billing report` before/after each job. Note the two transformers pins (5.20.2 for training/eval, 5.22.2 for
embedding apps): they never share a process, but a successor unifying images should re-run the eval parity check.

Measured on the final platform (T2, R-GPU): L40S bf16 Unsloth packed mb 32 ≈ 35.4k tok/s; a 3-epoch run on 1,808 items
≈ 190 s incl. setup; eval batch-invariant → 5,000/5,000 identical responses across two containers; vLLM-LoRA vs PEFT
greedy parity: same verdict on 94 % of 100 items (kernel-level divergence, no adapter-load error).

## 4. Reproducibility record per run
Each training/eval run writes `manifest.json` with `env` (python, torch, transformers, peft, trl, accelerate, vllm,
GPU, CUDA) and `info`; `handoff/tools/fill_model_cards.py` copies it into the model card. Kaggle kernels additionally
write `results/env_check.json` / `meta.json`.
