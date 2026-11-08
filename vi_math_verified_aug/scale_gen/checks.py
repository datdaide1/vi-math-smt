"""Deterministic parse / format checks (the whole N- gate) and answer agreement for the N+ screen.

N- = parse/format checks only (+ the decontamination gate that applies to every arm). Each check is a named flag so
rejection rates per check and per language can be reported (E1 R8 manipulation check).
"""
from __future__ import annotations

import re
import unicodedata

from ..common.llm import parse_json
from ..eval import math_verify as MV

VI_LEAK = re.compile(r"\\boxed|đáp (?:án|số)\s*(?:là|=|:)|gợi ý|lời giải|\banswer\b|\bhint\b", re.I)
EN_LEAK = re.compile(r"\\boxed|the answer is|answer\s*[:=]|\bhint\b|\bsolution\b|đáp|lời giải", re.I)
VI_CHARS = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.I)
NUM = re.compile(r"(?<![\w.,])-?\d+(?:[.,]\d+)*(?![\w])")
BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def _num_value(s: str, lang: str):
    """A single number in the generator's ``answer`` field -> canonical string, else None."""
    if s is None:
        return None
    t = str(s).strip().replace("\u00a0", " ").replace("−", "-")
    t = re.sub(r"\\boxed\{(.*)\}", r"\1", t).strip().strip("$").strip()
    t = re.sub(r"\s*(?:đồng|dollars?|usd|\$|%|kg|km|m|cm|giờ|phút|hours?|minutes?)\.?$", "", t, flags=re.I).strip()
    if lang == "vi":
        if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+(?:,\d+)?", t):      # 12.500 / 1.250,5
            t = t.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"-?\d+,\d+", t):
            t = t.replace(",", ".")
    else:
        if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", t):
            t = t.replace(",", "")
    t = t.replace(" ", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", t) or re.fullmatch(r"-?\d+/\d+", t):
        return t
    return None


def vi_ratio(text: str) -> float:
    letters = [c for c in unicodedata.normalize("NFC", text) if c.isalpha()]
    if not letters:
        return 0.0
    words = re.findall(r"\w+", text)
    vi_words = sum(bool(VI_CHARS.search(w)) for w in words)
    return vi_words / max(len(words), 1)


def answer_in_text(problem: str, ans: str, lang: str) -> bool:
    """The (multi-digit) answer value appears as a standalone number in the problem text (T1 arm-B 'leak')."""
    try:
        a = float(ans) if "/" not in ans else None
    except ValueError:
        return False
    if a is None or abs(a) < 10:
        return False
    for m in NUM.finditer(problem):
        v = _num_value(m.group(0), lang)
        try:
            if v is not None and "/" not in v and abs(float(v) - a) < 1e-9:
                return True
        except ValueError:
            continue
    return False


def parse_gen(text: str | None, lang: str, op: str) -> dict:
    j = parse_json(text) if text else None
    j = j if isinstance(j, dict) else {}
    prob = j.get("problem") if isinstance(j.get("problem"), str) else None
    ans_raw = j.get("answer")
    ans = _num_value(ans_raw, lang) if ans_raw is not None else None
    sol = j.get("solution") if isinstance(j.get("solution"), str) else None
    flags = {
        "parse_fail": not j,
        "no_problem": not prob,
        "answer_not_number": ans is None,
        "op_mismatch": str(j.get("op_applied", "")).strip() != op,
        "length": bool(prob) and not (40 <= len(prob) <= 1600),
    }
    if prob:
        leak_re = VI_LEAK if lang == "vi" else EN_LEAK
        flags["leak_regex"] = bool(leak_re.search(prob))
        r = vi_ratio(prob)
        flags["wrong_language"] = (r < 0.25) if lang == "vi" else (r > 0.05)
        flags["answer_in_text"] = bool(ans) and answer_in_text(prob, ans, lang)
    else:
        flags.update(leak_regex=False, wrong_language=False, answer_in_text=False)
    return {"problem": prob, "answer": ans, "answer_raw": None if ans_raw is None else str(ans_raw),
            "gen_solution": sol, "plan": j.get("plan"), "self_check": j.get("self_check"),
            "op_applied": j.get("op_applied"), "change_summary": j.get("change_summary"),
            "fmt_flags": flags, "fmt_ok": not any(flags.values())}


def last_boxed(text: str | None) -> str | None:
    if not text:
        return None
    m = list(BOX.finditer(text))
    return m[-1].group(1).strip() if m else None


def agree(solve_text: str | None, gen_answer: str | None, lang: str) -> bool:
    """Does the blind solve's final answer equal the generator's answer? (X1 grader, typed gold from the answer.)"""
    if not solve_text or gen_answer is None:
        return False
    try:
        gold = MV.infer_gold(gen_answer, locale=lang)
        return bool(MV.is_correct(solve_text, gold, locale=lang))
    except Exception:
        b = last_boxed(solve_text)
        return b is not None and _num_value(b, lang) == gen_answer


_ANS_LINE = re.compile(
    r"^[\s*_#>-]*(?:(?:đáp\s*(?:số|án)|kết\s*quả|vậy|final\s+answer|the\s+answer\s+is|answer|so|therefore)"
    r"[^\n]{0,40}?)?[:：]?[\s*_]*\$*\s*\\boxed\{(?:[^{}]|\{[^{}]*\})*\}\s*\$*\s*[.。]?[\s*_]*$", re.I)


def clean_trace(text: str) -> str:
    """Solver trace -> training solution body (shared by arms N and Y; lead / QA-T, 2026-11-07).

    1. drop trailing lines whose content is only the final boxed answer (optionally after "Đáp số:", "Answer:",
       "Vậy", ...), because the training template appends ``Đáp số: \\boxed{answer}`` itself;
    2. unwrap any remaining ``\\boxed{x}`` -> ``x`` so the training completion contains exactly one boxed answer.
    """
    lines = (text or "").rstrip().splitlines()
    while lines and (not lines[-1].strip() or _ANS_LINE.match(lines[-1])):
        lines.pop()
    return BOX.sub(lambda m: m.group(1), "\n".join(lines)).strip()
