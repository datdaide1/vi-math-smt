"""The one plain-text template (T3 §3.1, T5 §4.1).  Base models, no chat template::

    Bài toán:
    {problem}

    Lời giải:
    {solution ... }
    Đáp số: \\boxed{answer}<eos>

SFT trains on the completion (everything after ``Lời giải:\\n``) including EOS; the prompt is masked.  GRPO and
evaluation use exactly :func:`prompt` as the prompt.  Few-shot rows (base-model baselines) prepend k solved
examples in the same template, separated by a blank line.
"""
from __future__ import annotations

import re

TEMPLATE_NAME = "vi_plain_v1"
PROMPT = "Bài toán:\n{q}\n\nLời giải:\n"
ANSWER_LINE = "Đáp số: \\boxed{{{a}}}"
# English mirror runs (E1): the same template, word for word in English ("en_plain_v1").
PROMPTS = {"vi": PROMPT, "en": "Problem:\n{q}\n\nSolution:\n"}
ANSWER_LINES = {"vi": ANSWER_LINE, "en": "Final answer: \\boxed{{{a}}}"}
# Generation stops (few-shot base rows continue with the next "Bài toán:"/"Problem:"; SFT models emit EOS).
STOP_STRINGS = ["\n\nBài toán:", "\nBài toán:", "\n\nProblem:", "\nProblem:"]

_CUE_RE = re.compile(r"^\W*(?:####|đáp số|đáp án|final answer|the answer is|kết luận)", re.IGNORECASE)
_TAIL_RE = re.compile(r"(\n\s*(?:####|Đáp số\s*:|Đáp án\s*:|Final answer\s*:|The answer is)[^\n]*)+\s*$",
                      re.IGNORECASE)


def prompt(question: str, lang: str = "vi") -> str:
    return PROMPTS[lang].format(q=question.strip())


def _last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    k = j
    while k < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        k += 1
    return text[j:k - 1] if depth == 0 else None


def completion(solution: str, answer: str | None = None, lang: str = "vi") -> str:
    """Solution body + the final ``Đáp số: \\boxed{answer}`` line (EOS is appended at tokenisation).

    Any trailing ``#### x`` / ``Đáp số: ...`` / ``Đáp án: ...`` lines are replaced, so every arm ends the same way.
    ``answer`` is the item's canonical answer; if None, the last ``\\boxed{}`` of the solution is used.
    """
    body = solution.rstrip()
    if answer is None:
        answer = _last_boxed(body)
        if answer is None:
            raise ValueError("no answer given and no \\boxed{} in the solution")
    body = _TAIL_RE.sub("", body).rstrip()
    # QA-T R-fmt: generator outputs end in "**Đáp số:** \\boxed{x}", "Vậy đáp số là $\\boxed{x}$." or a bare boxed
    # line; drop every trailing line that carries a \\boxed{} or an answer cue, so exactly one answer line remains.
    lines = body.split("\n")
    while lines and ("\\boxed{" in lines[-1] or _CUE_RE.search(lines[-1]) or not lines[-1].strip()):
        lines.pop()
    body = "\n".join(lines).rstrip()
    line = ANSWER_LINES[lang].format(a=str(answer).strip())
    return f"{body}\n{line}" if body else line


def fewshot_prefix(shots: list[dict], lang: str = "vi") -> str:
    """k solved examples in the training template, each followed by a blank line."""
    parts = [prompt(s["question"], lang) + completion(s["solution"], s.get("answer") or s.get("answer_canonical"),
                                                      lang) for s in shots]
    return "".join(p + "\n\n" for p in parts)


def cut_at_stop(text: str) -> str:
    cut = len(text)
    for s in STOP_STRINGS:
        i = text.find(s)
        if 0 <= i < cut:
            cut = i
    return text[:cut]
