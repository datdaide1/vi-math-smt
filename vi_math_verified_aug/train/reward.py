"""Reward functions for GRPO and the single scoring entry point used by the eval runner.

Both call the project grader ``vi_math_verified_aug.eval.math_verify`` through :func:`score` (X1 owns and is
rewriting the grader; only its public ``grade`` / ``is_correct`` API is used here).

* :class:`MathVerifyReward` -- 1.0 if the response's final answer is equivalent to the gold, else 0.0.
  No format reward, no length penalty (T3 §4.1).
* :class:`RandomReward`     -- the spurious-reward control: Bernoulli(p) independent of correctness, keyed by
  (seed, global step, row index) so a resumed run draws the same rewards.  It still computes the true
  correctness and logs it as a metric (``true_accuracy``) so the control's real accuracy is visible.
"""
from __future__ import annotations

import hashlib
import json


def _grader():
    from ..eval import math_verify as mv
    return mv


def grader_version() -> str:
    try:
        mv = _grader()
        return f"{getattr(mv, 'GRADER_VERSION', '?')}/{getattr(mv, 'BACKEND', '?')}"
    except Exception as e:  # noqa: BLE001
        return f"unavailable:{type(e).__name__}"


def score(response: str, gold, locale: str | None = None) -> dict:
    """Grade one raw response.  Returns {correct, span, flag}.  Never raises (grader errors -> incorrect + flag)."""
    if isinstance(gold, str) and gold.startswith("{"):
        try:
            gold = json.loads(gold)
        except ValueError:
            pass
    try:
        mv = _grader()
        if hasattr(mv, "grade"):
            v = mv.grade(response, gold, locale=locale)
            return {"correct": bool(v.get("correct")), "span": v.get("span"), "flag": v.get("flag")}
        return {"correct": bool(mv.is_correct(response, gold)), "span": None, "flag": None}
    except Exception as e:  # noqa: BLE001
        return {"correct": False, "span": None, "flag": f"grader-error:{type(e).__name__}:{str(e)[:120]}"}


def _text(c) -> str:
    """TRL passes strings for plain-text prompts and [{'role','content'}] lists for conversational ones."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return c[-1].get("content", "")
    return str(c)


class MathVerifyReward:
    __name__ = "math_verify"

    def __init__(self, locale: str | None = None):
        self.locale = locale

    def __call__(self, prompts, completions, gold, **kw) -> list[float]:
        from .template import cut_at_stop  # QA-T G-reward: same view of the completion as the eval runner
        res = [score(cut_at_stop(_text(c)), g, self.locale) for c, g in zip(completions, gold)]
        log = kw.get("log_metric")
        if log and res:
            log("graded_no_answer_frac", sum(1 for r in res if r["flag"] == "no-answer") / len(res))
        return [1.0 if r["correct"] else 0.0 for r in res]


class RandomReward:
    __name__ = "random_reward"

    def __init__(self, seed: int, p: float = 0.5, locale: str | None = None):
        self.seed, self.p, self.locale = seed, p, locale
        self._calls = 0

    def draw(self, step: int, idx: int) -> float:
        h = hashlib.sha256(f"rr:{self.seed}:{step}:{idx}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2 ** 64
        return 1.0 if u < self.p else 0.0

    def __call__(self, prompts, completions, gold, **kw) -> list[float]:
        st = kw.get("trainer_state")
        step = getattr(st, "global_step", None)
        if step is None:
            step = self._calls
        self._calls += 1
        out = [self.draw(step, i) for i in range(len(completions))]
        log = kw.get("log_metric")
        if log and completions:
            from .template import cut_at_stop
            true = [score(cut_at_stop(_text(c)), g, self.locale)["correct"] for c, g in zip(completions, gold)]
            log("true_accuracy", sum(true) / len(true))
        return out
