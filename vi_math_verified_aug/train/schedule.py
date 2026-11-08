"""WSD (warm-up / stable / decay) learning-rate schedule and the epoch-branch plan (T3 §3.1).

Main run of ``total`` optimiser steps: linear warm-up over ``warmup`` steps, constant, then a linear decay to 0
over the last ``decay_frac`` of the steps.

Epoch branches (LR/epoch sweep): for each e in ``branch_epochs`` the trainer forks from the stable phase at
``fork = e*spe - decay(e*spe)`` and runs the decay over the remaining steps of epoch e, so the branch model has
seen exactly e epochs of data with the same decay length a fresh e-epoch WSD run would use.  The main run then
restores the fork state and continues in the stable phase.  Extra compute per branch = its decay length
(≈ 0.1·e epochs), i.e. {1,2,3}-epoch models for 3.3 epochs instead of 6.  Gate G6 checks branch-vs-fresh fidelity.
The one difference from a fresh e-epoch run is the warm-up length (3% of the main run, not of e epochs).
"""
from __future__ import annotations


def decay_len(total: int, decay_frac: float) -> int:
    return max(1, round(decay_frac * total))


def warmup_len(total: int, warmup_frac: float) -> int:
    return round(warmup_frac * total)


def lr_mult(step: int, total: int, warmup: int, decay: int) -> float:
    """Multiplier for 0-based optimiser step ``step`` of a WSD run with ``total`` steps."""
    if step < warmup:
        return (step + 1) / warmup
    start = total - decay
    if step >= start:  # D decay steps get D/(D+1), ..., 1/(D+1): strictly decreasing towards 0
        return max(0.0, (total - step) / (decay + 1))
    return 1.0


def plan(steps_per_epoch: int, epochs: int, branch_epochs: list[int], warmup_frac: float,
         decay_frac: float) -> dict:
    total = steps_per_epoch * epochs
    W, D = warmup_len(total, warmup_frac), decay_len(total, decay_frac)
    branches = []
    for e in sorted(set(branch_epochs)):
        end = e * steps_per_epoch
        d = decay_len(end, decay_frac)
        fork = end - d
        if fork < W:
            raise ValueError(f"branch epoch {e}: fork step {fork} falls inside the warm-up ({W} steps)")
        if end > total - D:
            raise ValueError(f"branch epoch {e} ends inside the main decay")
        branches.append({"epoch": e, "fork_step": fork, "end_step": end, "decay": d})
    return {"total": total, "warmup": W, "decay": D, "steps_per_epoch": steps_per_epoch, "epochs": epochs,
            "branches": branches}


def branch_mult(step: int, branch: dict, warmup: int) -> float:
    return lr_mult(step, branch["end_step"], warmup, branch["decay"])
