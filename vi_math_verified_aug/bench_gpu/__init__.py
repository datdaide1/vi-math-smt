"""R-GPU platform benchmark (2026-11-07): FP8 vs bf16, GPU choice, concurrency.  Runs ``train.sft`` and
``evalrun.runner`` unchanged; only adds optional, recorded in-process patches (FP8, torch.compile)."""
