"""Provisional answer extraction from the (private) third-party worked solution.

Replaces the prototype's free regex (~37 % precise, R6) with ordered, typed extractors. Every result carries
``method`` and ``confidence`` so the QA queue can prioritise. The answer is *provisional* key #1 only; it is
never trusted without the SymPy key or the blind-solve key agreeing (T5 §3.1 stage 2).

Methods (first hit wins):
  key_table            MCQ letter from the solution's key table               (high)
  mcq_choice           "Chọn B" / "Đáp án: B" in the question's solution       (high)
  tf_word              Đúng / Sai for a true/false statement                   (high)
  short_answer_line    "Đáp án: 12" / "Trả lời: 12" / "Đáp số: 12"             (high)
  vay_math             last formula of the last "Vậy …" sentence, after dropping formulas copied from the
                       problem text and pure domain conditions                 (medium)
  vay_text             a number in the "Vậy …" sentence with no formula        (low)
  last_math            last formula of the solution                            (low)
"""
from __future__ import annotations

import re

RE_MATH = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]|\$\$(.+?)\$\$|\$(.+?)\$", re.S)
RE_CHOICE = re.compile(r"(?:Chọn|Đáp án|ĐA)\s*(?:đáp án|phương án|câu)?\s*:?\s*\(?\*?\*?([A-D])\b(?![a-zà-ỹ])")
RE_SHORT = re.compile(r"^\s*(?:Đáp án|Đáp số|Trả lời|Kết quả)\s*:?\s*(.+)$", re.I)
RE_DOMAIN = re.compile(r"^\s*[a-z]\s*(\\ge|\\le|\\ne|\\geqslant|\\leqslant|>|<|≥|≤|≠)", re.I)


def math_spans(s: str) -> list[str]:
    return [next(g for g in m.groups() if g is not None).strip() for m in RE_MATH.finditer(s)]


def final_value(raw: str | None) -> str | None:
    """Readable final member: drop array wrappers, keep the last line, keep the last '=' member of a chain
    ('(8 + 25).2 = 66 m' -> '66 m'; 'B = … \\\\ B = 2' -> '2'). Set/tuple answers ('x = 2; x = 3') are kept whole."""
    if not raw:
        return raw
    s = re.sub(r"\\begin\{array\}(\{[^}]*\})?|\\end\{array\}", "", raw)
    lines = [x for x in re.split(r"\\\\", s) if x.strip()]
    s = lines[-1] if lines else s
    several = re.search(r"[;,]\s*(\\left)?\(?\s*\{?\s*[a-zA-Z](_\{?\d\}?)?\s*\}?\s*[;,]?\s*[a-zA-Z]?\s*\)?\s*=", s)
    if not several and "=" in s and not re.search(r"^\s*(\\left)?\(", s):
        s = s.split("=")[-1]  # 'A = … = 3' or 'A = 3' -> '3'; 'x = 2; x = 3' and '(x;y) = (2;1)' kept whole
    return re.sub(r"(\\,|\\;|\s)+", " ", s).strip().rstrip(".;, ")


def _canon(s: str) -> str:
    s = re.sub(r"\\left|\\right|\\,|\\;|\\!|\\ |\\mkern\s*\d+mu|\s|\{|\}", "", s)
    return s.rstrip(".,;:")


def extract(sol_lines: list[str], ans_type: str, problem_text: str, key_letter: str | None = None) -> dict:
    res = {"answer_raw": None, "method": None, "confidence": None}
    if ans_type == "mcq":
        if key_letter:
            return {"answer_raw": key_letter, "method": "key_table", "confidence": "high"}
        for ln in reversed(sol_lines):
            m = RE_CHOICE.search(ln)
            if m:
                return {"answer_raw": m.group(1), "method": "mcq_choice", "confidence": "high"}
        return res
    if not sol_lines:
        return res
    if ans_type == "true_false":
        txt = " ".join(sol_lines[-4:])
        hits = re.findall(r"\b(Đúng|Sai|đúng|sai)\b", txt)
        if hits:
            return {"answer_raw": "Đúng" if hits[-1].lower() == "đúng" else "Sai", "method": "tf_word",
                    "confidence": "high" if len(set(h.lower() for h in hits)) == 1 else "medium"}
        return res
    for ln in reversed(sol_lines):
        m = RE_SHORT.match(ln)
        if m and len(m.group(1)) < 80:
            return {"answer_raw": m.group(1).strip(" ."), "method": "short_answer_line", "confidence": "high"}
    given = {_canon(x) for x in math_spans(problem_text)}
    vay = [ln for ln in sol_lines if re.search(r"(^|\W)Vậy\b", ln)]
    if vay:
        last = vay[-1]
        tail = last[re.search(r"(^|\W)Vậy\b", last).end():]
        spans = [s for s in math_spans(tail) if _canon(s) not in given and not (RE_DOMAIN.match(s) and ans_type != "parameter_condition")
                 and not re.fullmatch(r"\(?[A-Za-z]\)?|\\left\(\s*[A-Za-z]\s*\\right\)", s.strip())]
        if spans:
            # sets of roots are often written as several formulas ('x = 2; x = 3'): keep all of them;
            # otherwise ('A = 1', 'S = {…}', '(x; y) = (…)') the last informative formula
            ans = "; ".join(spans) if ans_type in ("solution_set", "parameter_condition") else spans[-1]
            return {"answer_raw": ans, "method": "vay_math", "confidence": "medium", "conclusion": last}
        nums = re.findall(r"-?\d+(?:[,\.]\d+)?", tail)
        if nums:
            return {"answer_raw": nums[-1], "method": "vay_text", "confidence": "low", "conclusion": last}
    for ln in reversed(sol_lines):
        sp = [s for s in math_spans(ln) if _canon(s) not in given]
        if sp:
            return {"answer_raw": sp[-1], "method": "last_math", "confidence": "low"}
    return res
