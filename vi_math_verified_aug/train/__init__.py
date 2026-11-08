"""Student training for the Nov-2026 design (workstream Y1).

Written fresh from ``docs/research_2026-11/tech/T3_training.md`` §3.1 (SFT) and §4.1 (GRPO).  Nothing here imports
the legacy ``notebooks/``, ``src/`` or the September ``scripts/sft.py`` stubs.

Modules
-------
config     one YAML file per run (kind / arm / model / size / seed / budget) -> validated ``RunConfig``
template   the plain-text base-model template shared by SFT, GRPO and evaluation
data       tokenise -> drop (never truncate) over-long items -> seeded subsample -> padded, length-grouped steps
schedule   WSD learning-rate schedule and the epoch-branch plan
sft        the SFT trainer (custom loop: fp16 autocast + GradScaler, fp32 LoRA, resumable checkpoints)
grpo       the GRPO stage (TRL ``GRPOTrainer``, Dr.GRPO loss, binary ``math_verify`` reward, random-reward control)
reward     reward functions (grader adapter + seeded Bernoulli(0.5) control)
modeling   model / tokenizer / LoRA loading, including the offline ``tiny-random`` smoke model
cli        ``python -m vi_math_verified_aug.train.cli {sft,grpo,merge,check} --config RUN.yaml``
kaggle     Kaggle kernel template + laptop-side renderer (uses S2's ``infra.kaggle_runner`` read-only)
"""

EXIT_OK = 0
EXIT_ERROR = 3
EXIT_ABORT_OVERFLOW = 4      # >max_skip_frac of optimiser steps skipped by the GradScaler
EXIT_GATE_FAILED = 6         # engine=unsloth start-of-job logits gate failed (no training step was taken)
EXIT_GUARD_ABORT = 7         # a fail-fast guard fired (pre-flight / in-loop / mid-run); manifest ABORTED_<reason>
EXIT_DATA_ERROR = 8          # data / setup error (missing file, n_train shortfall > 1 %): no engine fallback
EXIT_CELL_STOPPED = 9        # not run: an earlier arm of its (seed-0) cell failed a guard -> the cell stopped
EXIT_NEEDS_RESUME = 75       # wall-clock budget reached; a checkpoint was written (EX_TEMPFAIL)
