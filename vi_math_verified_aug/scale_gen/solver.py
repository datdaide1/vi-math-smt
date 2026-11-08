"""The ONE shared solution writer for training rows of arms N and Y (lead / QA-T, 2026-11-07). Stable API for G4.

    from vi_math_verified_aug.scale_gen.solver import SOLVER_SPEC, solve_request, parse_solve, training_row

* model ``claude-haiku-5-5``, effort ``low`` (adaptive thinking), max_tokens 3000, no system prompt;
* prompt ``solution_writer.v2`` (VI) / ``solution_writer_en.v2`` (EN, a 1:1 translation, E1 R4);
* run it through the Message Batches API (``batchrun.BatchManager`` or ``LLMClient.batches``); requests are
  cache-keyed, so an identical problem text is never paid twice.
* training row = {question, solution = clean_trace(trace), answer = the trace's own last ``\\boxed{}``}.
  "+" keeps a row only if ``answer`` equals the arm's verified answer; "−" takes the trace unchecked.
"""
from __future__ import annotations

from ..common.llm import LLMRequest
from .checks import _num_value, agree, clean_trace, last_boxed
from .common import PROMPT_ID, PROMPT_VER, SOLVER
from .common import solve_request as _solve_request

SOLVER_SPEC = {"model": SOLVER, "effort": "low", "thinking": "adaptive (default)", "max_tokens": 3000,
               "prompt": {lang: f"{PROMPT_ID[lang]['solve']}.v{PROMPT_VER['solve']}" for lang in ("vi", "en")},
               "postprocess": "scale_gen.checks.clean_trace", "answer": "last \\boxed{} of the trace"}


def solve_request(problem: str, lang: str, item_key: str, sample_idx: int = 0) -> LLMRequest:
    """``item_key`` is metadata only (logged), not part of the cache key."""
    return _solve_request(problem, lang, item_key, sample_idx)


def parse_solve(text: str | None) -> dict:
    b = last_boxed(text)
    return {"boxed": b, "solution": clean_trace(text) if text else None, "ok": bool(text and b)}


def training_row(question: str, solve_text: str, *, lang: str, verified_answer: str | None = None, **meta) -> dict | None:
    """{question, solution, answer} (+ ``trace_matches_verified`` when a verified answer is given); None if the
    trace has no boxed answer (format failure)."""
    p = parse_solve(solve_text)
    if not p["ok"] or not p["solution"]:
        return None
    row = {"question": question, "solution": p["solution"], "answer": p["boxed"], **meta}
    if verified_answer is not None:
        row["trace_matches_verified"] = agree(solve_text, str(verified_answer), lang)
    return row


__all__ = ["SOLVER_SPEC", "solve_request", "parse_solve", "training_row", "clean_trace", "agree", "last_boxed",
           "_num_value"]
