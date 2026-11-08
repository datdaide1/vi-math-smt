"""SUPERSEDED (kept for the calibration record): the production common screen is D1's frozen
``scale_gen.screen`` (screen / haiku_v3_request / haiku_v3_parse / common_pass), which arm_y.pipeline.stage_vgate
calls, so N+ and Y+ run literally identical code. This module documents the rule as calibrated by G5.

The COMMON validity screen of the "+" arms (lead decision 2026-11-07 ~23:05): used verbatim by arm Y (G5) and
arm N (D1), so Y+ vs N+ differ only in their arm-specific verification (program certificate vs solve agreement).

    screen_pass(haiku_labels, lite_row) = haiku_ok(haiku_labels) AND lite_row["lite_ok"]

* Haiku leg: ``haiku_request`` = prompts/validity_judge.v3 (G2 rubric, cached system block), claude-haiku-5-5,
  effort low, max_tokens 4000; labels = ``pilot_symbolic.judge.lab_v3``; haiku_ok = not absurd AND not unanswerable
  AND teacher_would_write.
* flash-lite leg: D1's ``scale_gen.lite`` (prompts/validity_lite.v1, packs of 10, gemini-3.5/3.1-flash-lite), its
  ``lite_ok`` label.
* Calibration (G5, 358 Opus-labelled items; data/pilot/symbolic_v3/calib_common_screen.json): pass 180/358,
  precision vs Opus-accept 71% [64-77], recall 71%, absurd among accepts 6% [3-11] (flash-lite alone 10% [7-15],
  Haiku alone 8% [5-12]).
"""
from __future__ import annotations

from ..common.llm import LLMRequest
from ..common.prompts import PromptRegistry
from ..pilot_symbolic.judge import lab_v3

MODEL = "claude-haiku-5-5"
_REG = PromptRegistry()


def haiku_request(seed: str, op: str, op_instruction: str, problem: str, item: str, split: str = "train") -> LLMRequest:
    rp = _REG.get("validity_judge", 3).render(seed=seed, op=op, op_instruction=op_instruction, problem=problem)
    return LLMRequest(model=MODEL, system=None, messages=[{"role": "user", "content": rp.user}], max_tokens=4000,
                      temperature=None, item_id=item, split=split, prompt_id=rp.prompt_id, prompt_version=rp.version,
                      prompt_sha256=rp.sha256, tag="common_screen_haiku",
                      extra={"output_config": {"effort": "low"},
                             "system": [{"type": "text", "text": rp.system, "cache_control": {"type": "ephemeral"}}]})


def haiku_labels(text: str | None) -> dict | None:
    from ..common.llm import parse_json
    v = parse_json(text) if text else None
    return lab_v3(v) if isinstance(v, dict) else None


def haiku_ok(lab: dict | None) -> bool:
    return bool(lab) and not lab["absurd"] and not lab["unansw"] and bool(lab["teacher"])


def screen_pass(lab: dict | None, lite_row: dict | None) -> bool:
    return haiku_ok(lab) and bool((lite_row or {}).get("lite_ok"))


__all__ = ["haiku_request", "haiku_labels", "haiku_ok", "screen_pass", "MODEL"]
