"""The project's single answer grader: typed gold, Vietnamese-aware prediction parsing (X1, 2026-11-07).

Written fresh from ``docs/research_2026-11/tech/T2_verification_selection.md`` §4 and
``docs/research_2026-11/04_data_eval.md`` §4.2 (bugs B1-B6).  Report: ``docs/research_2026-11/exec/X1_grader.md``.
Tests: ``tests/test_math_verify.py`` + ``tests/grader_cases/*.jsonl``.

Design
------
1. **Gold is structured and never extracted.**  A gold is a dict
   ``{"answer_canonical": str, "answer_type": one of ANSWER_TYPES, "locale": "vi"|"en"|"mixed",
   "vars": [...], "target_var": str, "round_dp": int, "percent": bool}`` (T2's ``type``/``value`` keys are
   accepted too).  ``answer_canonical`` uses ``.`` as the decimal mark, no digit grouping, ``;`` between
   container items, and ``[a,b)`` for intervals.  Only the *prediction* is extracted and locale-parsed, so a
   partial prediction (``8``) can never match a multi-part gold (``x=7; y=8``), and ``125`` never equals 0.125.
2. **Extraction (prediction only):** drop ``<think>…</think>``; last ``\\boxed{}`` (brace-balanced; several boxes
   in the final paragraph are joined for a container gold and are a *hedge* (wrong) for a scalar gold); else
   ``####``; else the last cue (Đáp số / Đáp án / Kết luận / Vậy / The answer is); else a short pure-math
   response as a whole; else the last number — unless the response ends in a degenerate repetition loop.
3. **Structure before numbers:** ``(1,2)`` is a pair, ``x = 2 hoặc x = 3`` / ``x_1=2; x_2=3`` / ``S=\\{2;3\\}`` are
   sets, ``[1;3)`` / ``]1;3[`` / ``x \\ge 2`` are intervals — all decided before any ``,`` or ``.`` is read as a
   decimal or grouping mark.  Arity must match; sets are multisets; tuples are ordered (matched by name when
   both sides name their variables).
4. **Locale-aware numbers** (T2 §4.3): ``{,}`` is a decimal comma; ``\\,``/space between 3-digit groups and
   repeated separators with 3-digit groups are grouping; both ``.`` and ``,`` → the last is the decimal mark;
   one separator followed by ≠3 digits (or after a leading ``0``) is a decimal mark; one separator followed by
   exactly 3 digits is *ambiguous* and follows the response's own convention (its unambiguous tokens vote),
   then ``gold.locale``; for ``mixed`` golds the tie goes to the gold type (int → grouping, else decimal) and
   the verdict carries ``flag="ambiguous-locale"``.
5. **Exact numbers.**  Integers, decimals and fractions compare as exact ``Fraction`` (``0.333 ≠ 1/3``); a
   decimal literal never equals an irrational gold.  Tolerance only with ``round_dp`` (|p−g| ≤ ½·10^−dp).
6. **Expression backend** (radicals, π, polynomials, matrices, equations): HF ``math-verify==0.9.0`` when it is
   importable, else a built-in SymPy path.  ``BACKEND`` names the active one; ``set_backend()`` switches.
   When this file is imported as a top-level module named ``math_verify`` (the frozen snapshot shipped to
   Kaggle), the HF package cannot be imported under the same name and the SymPy path is used.

Public API
----------
grade(pred, gold, *, locale=None) -> dict   correct, span, method, flag, conv, gold_type, backend, grader_version, matched (gold | accept_also[i])
is_correct(pred, gold, **legacy)  -> bool   ``gold`` = typed dict (preferred) or a canonical gold string
infer_gold(raw, *, problem=None, locale="mixed") -> dict   type a legacy gold string ONCE (migration)
Compatibility (old callers): extract_answer(text), normalize_answer(s), _to_float(s), _sympy_equal(a, b).
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Union

GRADER_VERSION = "3.0.2"
ANSWER_TYPES = ("int", "rational", "decimal", "radical", "tuple", "set", "interval", "expr", "text", "choice")
_TYPE_ALIASES = {"ineq": "interval", "integer": "int", "fraction": "rational", "mcq": "choice"}
NUMERIC_TYPES = ("int", "rational", "decimal")
CONTAINER_TYPES = ("tuple", "set")
LOCALES = ("vi", "en", "mixed")

# ============================================================================================================ #
# expression backend
# ============================================================================================================ #
_MV_TIMEOUT = None if os.name == "nt" else 5  # math-verify's own timeouts use signals / processes: off on Windows


def _load_math_verify():
    if __name__ == "math_verify":  # this file *is* the top-level module "math_verify" (Kaggle snapshot)
        return None
    try:
        import math_verify as _hf  # noqa: F401  (the HF package, absolute import)
        if getattr(_hf, "__file__", None) == __file__ or not hasattr(_hf, "verify"):
            return None
        from math_verify import parse as _p, verify as _v
        from importlib.metadata import version as _ver
        for name in ("math_verify", "math_verify.parser", "math_verify.grader", "math_verify.utils",
                     "latex2sympy2_extended"):
            logging.getLogger(name).setLevel(logging.CRITICAL)
        return _p, _v, _ver("math-verify")
    except Exception:
        return None


_MV = _load_math_verify()
_use_mv = _MV is not None and os.environ.get("VI_GRADER_BACKEND", "").lower() != "sympy"
BACKEND = f"math-verify-{_MV[2]}" if _use_mv else "sympy-fallback"


def set_backend(name: str) -> str:
    """Select ``"math-verify"`` (if installed) or ``"sympy"`` for non-numeric atoms; returns ``BACKEND``."""
    global _use_mv, BACKEND
    if name == "math-verify":
        if _MV is None:
            raise RuntimeError("math-verify is not importable here")
        _use_mv = True
    elif name == "sympy":
        _use_mv = False
    else:
        raise ValueError(name)
    BACKEND = f"math-verify-{_MV[2]}" if _use_mv else "sympy-fallback"
    return BACKEND


# ============================================================================================================ #
# gold
# ============================================================================================================ #
@dataclass(frozen=True)
class Gold:
    type: str
    value: str
    locale: str = "mixed"
    vars: Optional[tuple] = None
    target_var: Optional[str] = None
    round_dp: Optional[int] = None
    percent: bool = False

    def items(self) -> list:
        if self.type not in CONTAINER_TYPES:
            return [self.value]
        v = self.value.strip()
        if v in ("", "\\emptyset", "\\varnothing", "\\{\\}"):
            return []
        return [x.strip() for x in _split_top(v, (";",)) if x.strip()]


def as_gold(gold, *, locale: Optional[str] = None) -> Gold:
    """Typed gold dict / :class:`Gold` / legacy string (typed once by :func:`infer_gold`) -> :class:`Gold`."""
    if isinstance(gold, Gold):
        return gold
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        gold = repr(gold) if isinstance(gold, float) else str(gold)
    if isinstance(gold, str):
        gold = infer_gold(gold, locale=locale or "mixed")
    if not isinstance(gold, dict):
        raise TypeError(f"gold must be a dict or a string, not {type(gold).__name__}")
    gtype = gold.get("answer_type", gold.get("type"))
    gtype = _TYPE_ALIASES.get(gtype, gtype)
    value = gold.get("answer_canonical", gold.get("value"))
    if gtype not in ANSWER_TYPES:
        raise ValueError(f"answer_type {gtype!r} not in {ANSWER_TYPES}")
    if value is None:
        raise ValueError("gold has no answer_canonical")
    loc = locale or gold.get("locale") or "mixed"  # no locale = unknown provenance: type-guided tie-break, flagged
    if loc not in LOCALES:
        raise ValueError(f"locale {loc!r} not in {LOCALES}")
    vars_ = gold.get("vars")
    return Gold(type=gtype, value=_nfc(str(value)).strip(), locale=loc,
                vars=tuple(v.replace("{", "").replace("}", "") for v in vars_) if vars_ else None,
                target_var=gold.get("target_var"), round_dp=gold.get("round_dp"),
                percent=bool(gold.get("percent", False)))


# ============================================================================================================ #
# text utilities
# ============================================================================================================ #
_DEC = "\u2396"  # stands for LaTeX "{,}" (a decimal comma that is never a list separator)
_UNI = {
    "\u2212": "-", "\u2013": "-", "\u2014": "-", "\u00d7": "\\times ", "\u00f7": "\\div ", "\u221a": "\\sqrt",
    "\u03c0": "\\pi ", "\u221e": "\\infty ", "\u2264": "\\le ", "\u2265": "\\ge ", "\u2260": "\\ne ",
    "\u222a": "\\cup ", "\u2229": "\\cap ", "\u2208": "\\in ", "\u2248": "\\approx ", "\u2205": "\\emptyset ",
    "\u00b0": "^\\circ ", "\u00b2": "^2", "\u00b3": "^3", "\u00a0": " ", "\u2009": " ", "\u202f": " ",
    "\uff1a": ":", "\u00b1": "\\pm ", "\u2032": "'",
}


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _uni(s: str) -> str:
    s = _nfc(s)
    for a, b in _UNI.items():
        s = s.replace(a, b)
    return s


def _close_brace(s: str, i: int) -> Optional[int]:
    """Index of the ``}`` matching the ``{`` at ``s[i]`` (escaped ``\\{``/``\\}`` are not counted)."""
    depth = 0
    for k in range(i, len(s)):
        c = s[k]
        if c in "{}" and k > 0 and s[k - 1] == "\\" and not (k > 1 and s[k - 2] == "\\"):
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return k
    return None


def _split_top(s: str, seps) -> list:
    """Split ``s`` on any of ``seps`` at bracket depth 0 (``()[]{}`` and ``\\{ \\}``)."""
    out, cur, depth, i = [], [], 0, 0
    while i < len(s):
        c = s[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        if depth == 0:
            hit = next((x for x in seps if s.startswith(x, i)), None)
            if hit is not None:
                out.append("".join(cur))
                cur, i = [], i + len(hit)
                continue
        cur.append(c)
        i += 1
    out.append("".join(cur))
    return out


def _unwrap(s: str, o: str, c: str) -> Optional[str]:
    """Inner text if ``s`` is exactly one ``o … c`` group enclosing everything, else None."""
    s = s.strip()
    if not (s.startswith(o) and s.endswith(c)) or len(s) < len(o) + len(c):
        return None
    inner = s[len(o):len(s) - len(c)]
    depth = 0
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return None
    return inner if depth == 0 else None


def _unwrap_cmd(s: str, cmd: str, keep: bool = True) -> str:
    """``\\cmd{X}`` -> `` X `` (balanced; ``keep=False`` drops X)."""
    for _ in range(50):
        m = re.search(r"\\%s\s*\{" % cmd, s)
        if not m:
            break
        j = m.end() - 1
        k = _close_brace(s, j)
        if k is None:
            return s[:m.start()] + s[m.end():]
        s = s[:m.start()] + (" " + s[j + 1:k] + " " if keep else " ") + s[k + 1:]
    return s


# ============================================================================================================ #
# 1. cleaning and extraction (prediction only)
# ============================================================================================================ #
def _clean(text: str) -> tuple:
    flag = None
    t = _uni(text or "")
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    if re.search(r"</think>", t, re.I):  # template opened <think> in the prompt
        t = re.split(r"</think>", t, flags=re.I)[-1]
    if re.search(r"<think>", t, re.I):
        flag = "unclosed-think"
        t = re.sub(r"<think>", " ", t, flags=re.I)
    t = re.sub(r"<<[^<>\n]*>>", "", t)  # GSM8K calculator annotations
    t = re.sub(r"\[asy\].*?\[/asy\]", " ", t, flags=re.S)
    return t, flag


def _boxes(body: str) -> list:
    """All ``\\boxed{…}``/``\\fbox{…}`` as (content, start, end), brace-balanced (nested braces ok)."""
    out, i = [], 0
    pat = re.compile(r"\\(?:boxed|fbox)\s*")
    while True:
        m = pat.search(body, i)
        if not m:
            return out
        j = m.end()
        if j < len(body) and body[j] == "{":
            k = _close_brace(body, j)
            if k is None:
                return out
            out.append((body[j + 1:k].strip(), m.start(), k + 1))
            i = k + 1
        else:  # \boxed 5
            m2 = re.match(r"[^\s$]+", body[j:])
            if not m2:
                return out
            out.append((m2.group(0).rstrip(".,"), m.start(), j + m2.end()))
            i = j + m2.end()


_CUE = re.compile(
    r"(?<![^\W\d_])(?:đáp\s*số|đáp\s*án(?:\s*(?:cuối\s*cùng|đúng))?|kết\s*luận|kết\s*quả(?:\s*cuối\s*cùng)?|"
    r"trả\s*lời|vậy|the\s+(?:final\s+)?answer\s+is|final\s+answer|answer\s*:)"
    r"(?![^\W\d_])\s*(?:(?:là|is|bằng|=|:)\s*)*", re.I)
_LASTNUM = re.compile(r"-?\\[dt]?frac\s*\{-?\d+\}\s*\{\d+\}|(?<!\^)(?<!\^\{)(?<![\w.])-?\d+(?:(?:[.,]|\{,\})\d+)*"
                      r"(?:\s*/\s*\d+)?(?:\s*(?:nghìn|ngàn|triệu|tỷ|tỉ)(?![^\W\d_]))?")


def _degenerate(body: str) -> bool:
    """The response ends inside a repetition loop (max-token truncation of a looping base model)."""
    t = re.sub(r"\s+", " ", body).strip()
    if len(t) < 200:
        return False
    if re.search(r"\d{60,}\s*$", t):  # 6.6666666… running into the token limit
        return True
    probe = t[-70:-25]
    return len(probe.strip()) >= 20 and t[-1500:].count(probe) >= 3


def _same_surface(a: str, b: str) -> bool:
    x, y = re.sub(r"\s+", "", a), re.sub(r"\s+", "", b)
    if x == y:
        return True
    pa, pb = _prenorm(a, "expr"), _prenorm(b, "expr")
    fa, fb = _plain_fraction(pa), _plain_fraction(pb)
    if fa is not None or fb is not None:
        return fa is not None and fa == fb
    if _has_prose(pa) or _has_prose(pb) or re.search(r"[;,]|hoặc", pa + pb):
        return False
    return _sym_eq(pa, pb)  # "x+11" and "11+x" are one answer written twice, not a hedge


def _extract(body: str, arity: int) -> tuple:
    """(span, method).  ``arity`` > 1 lets several boxes of the final paragraph form one container answer."""
    boxes = [b for b in _boxes(body)
             if re.search(r"\w", re.sub(r"\\[hv]?phantom\s*\{[^{}]*\}", "", b[0]))]
    if boxes:
        a = body.rfind("\n\n", 0, boxes[-1][1])
        a = 0 if a < 0 else a
        final = [b for b in boxes if b[1] >= a]
        distinct = []
        for c, _, _ in final:
            if not any(_same_surface(c, d) for d in distinct):
                distinct.append(c)
        if len(distinct) >= 2:
            atomic = all(not re.search(r"[;,]|hoặc|\\\{|\\cup", c) for c, _, _ in final)
            if arity > 1 and atomic:
                parts = []
                for c, s0, _ in final:
                    m = re.search(r"([A-Za-z](?:_\{?\w+\}?)?)\s*\$?\s*=\s*\$?\s*$", body[max(a, s0 - 15):s0])
                    parts.append(f"{m.group(1)} = {c}" if m and "=" not in c else c)
                return " ; ".join(parts), "boxed-multi"
            return " ; ".join(distinct), "boxed-hedge"
        return boxes[-1][0], "boxed"
    if "####" in body:
        tail = body.rsplit("####", 1)[1].strip()
        if tail:
            return tail.splitlines()[0].strip(), "####"
    cues = list(_CUE.finditer(body))
    if cues:
        rest = body[cues[-1].end():]
        lines = rest.split("\n")
        line = lines[0].strip() or next((ln.strip() for ln in lines[1:] if ln.strip()), "")
        line = re.split(r"(?<=[^\d\s])\.(?=\s|$)|(?<=\d)\.(?!\d)", line.replace("**", ""))[0].strip()
        if line and re.search(r"[\w\\]", line):
            return line, "cue"
    t = body.strip()
    if t and len(t) <= 200 and "\n" not in t and not _has_prose(_prenorm(t, "int")):
        return t, "whole"
    nums = list(_LASTNUM.finditer(body))
    if nums and arity > 1 and not _degenerate(body):  # container gold, no cue: "Các giá trị là 1, 2 và 3."
        last = [ln for ln in t.split("\n") if ln.strip()][-1]
        tail = re.split(r"(?<![^\W\d_])(?:là|is|are)(?![^\W\d_])|:", last)[-1].strip().rstrip(".")
        if len(_LASTNUM.findall(tail)) >= 2:
            return tail, "last-line"
    if nums:
        if _degenerate(body):
            return None, "degenerate"
        return nums[-1].group(0).strip(), "last-number"
    return None, "none"


# ============================================================================================================ #
# 2. pre-normaliser (prediction spans; gold strings in infer_gold)
# ============================================================================================================ #
_MULT = {"nghìn": 1000, "ngàn": 1000, "triệu": 10 ** 6, "tỷ": 10 ** 9, "tỉ": 10 ** 9,
         "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}
_UNITS = (
    r"đồng|vnđ|vnd|đô\s*la|đô|usd|dollars?|cents?|xu|"
    r"km/h|km/giờ|m/s|km|cm|dm|mm|mét|met|kg|mg|gam|gram|tấn|tạ|yến|lít|ml|"
    r"giờ|tiếng|phút|giây|ngày|tuần|tháng|năm|tuổi|độ|dặm|pound|ounce|"
    r"đơn\s*vị(?:\s*vuông)?|units?|feet|foot|ft|inch(?:es)?|miles?|pounds?|degrees?|hours?|minutes?|days?|"
    r"weeks?|years?|học\s*sinh|người|cái|chiếc|quả|cây|cuốn|quyển|viên|con|bông|trang|lần|cách|vé|chai|hộp|gói"
)
_UNIT_RE = re.compile(r"(?<![^\W\d_])(?:%s)(?:\s+(?:vuông|khối|square|cubic))?(?:\s*\^\s*\{?[23]\}?)?(?![^\W\d_])"
                      % _UNITS, re.I)
_UNIT1_RE = re.compile(r"(?<=[\d}])\s*(?:m|g|l|s|h|đ)(?:\s*\^\s*\{?[23]\}?)?(?![^\W\d_])")
_MATH_WORDS = {"sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "lg", "exp", "lim", "max", "min", "gcd",
               "lcm", "mod", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "det", "sqrt", "frac", "pi",
               "infty", "hoặc", "và", "or", "and", "cup", "cap", "in", "pm", "le", "ge", "leq", "geq", "lt", "gt",
               "nghìn", "ngàn", "triệu", "tỷ", "tỉ", "thousand", "million", "billion"}
_EN_STOP = {"is", "of", "to", "so", "we", "it", "be", "an", "at", "by", "on", "the", "are", "if", "as"}


def _prenorm(span: str, gtype: str) -> str:
    s = _uni(span)
    s = s.replace("\\$", " ").replace("$", " ")
    s = re.sub(r"\\[()\[\]]", " ", s)
    s = s.replace("**", " ")
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|big|Big|bigg|Bigg)(?![A-Za-z])\s*\.?", "", s)
    s = s.replace("\\displaystyle", " ")
    s = re.sub(r"(\d)\s*\{,\}\s*(\d)", r"\1" + _DEC + r"\2", s)        # 1{,}5
    s = re.sub(r"(\d)\\,(?=\d{3}(?!\d))", r"\1", s)                    # 1\,000\,000
    s = re.sub(r"(\d),\s*\\!\s*(?=\d{3}(?!\d))", r"\1", s)             # 10,\!000
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\u2397")
    s = re.sub(r"\\[,;:> ]|~|\\quad|\\qquad", " ", s)
    s = s.replace("\u2397", "\\\\")
    for cmd in ("mathrm", "textbf", "textit", "mbox", "textrm", "operatorname", "mathit"):
        s = re.sub(r"\\%s\s*\{" % cmd, r"\\text{", s)
    if gtype in ("text", "choice"):
        s = _unwrap_cmd(s, "text")
        return re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\\text\s*\{\s*(hoặc|và|or|and)\s*\}", r" \1 ", s, flags=re.I)
    s = re.sub(r"\\mathbf\s*\{", r"\\text{", s) if gtype != "expr" else s
    no_text = re.sub(r"\\text\s*\{[^{}]*\}(?:\s*\^\s*\{?[23]\}?)?", " ", s)
    s = no_text if re.search(r"[\d\\]", no_text) else _unwrap_cmd(s, "text")
    s = re.sub(r"\(?\s*(?:với\s*)?k\s*\\in\s*\\mathbb\s*\{?\s*Z\s*\}?\s*\)?", " ", s)  # (k ∈ Z)
    s = re.sub(r"(?<![A-Za-z\\])([a-z])\s*(\d+)\s*\\pi", r"\2\1\\pi", s)  # VN "k2\pi" = 2k\pi
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", " ", s)                       # degrees
    s = s.replace("\\%", "%")
    s = _UNIT_RE.sub(" ", s)
    if gtype not in ("expr", "radical"):
        s = _UNIT1_RE.sub(" ", s)
    s = re.sub(r"(?<![\d.,])(\d{1,3})((?: \d{3})+)(?![\d.,]?\d)", lambda m: m.group(1) + m.group(2).replace(" ", ""),
               s)                                                         # 1 000 000
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[.,:;]+$", "", s).strip() if not s.endswith("...") else s
    return s


def _trim_prose(span: str) -> str:
    """Drop a prose lead-in: "tập nghiệm của phương trình là $S = \\{1; 2\\}$" -> "S = \\{1; 2\\}".
    Cuts after the last prose text chunk that is followed by a ``$…$`` math chunk; without ``$``, after the last
    "là / bằng / is / :" when what follows is prose-free.  Connectors (hoặc, và) are not prose."""
    if "$" in span:
        parts = span.split("$")
        cut = None
        for i in range(0, len(parts) - 1, 2):  # even = text, odd = math
            if _has_prose(parts[i]) and parts[i + 1].strip():
                cut = i
        return span if cut is None else "$".join(parts[cut + 1:])
    m = list(re.finditer(r"(?<![^\W\d_])(?:là|bằng|is|are)(?![^\W\d_])|:", span))
    if m and _has_prose(span[:m[-1].start()]):
        rest = span[m[-1].end():].strip()
        if rest and not _has_prose(_prenorm(rest, "int")):
            return rest
    return span


def _has_prose(s: str) -> bool:
    t = re.sub(r"\\(?:begin|end)\s*\{[A-Za-z*]*\}", " ", s)
    t = re.sub(r"\\[A-Za-z]+", " ", t)
    for w in re.findall(r"[^\W\d_]{2,}", t):
        lw = w.lower()
        if lw in _MATH_WORDS:
            continue
        if not w.isascii() or len(w) >= 3 or lw in _EN_STOP:
            return True
    return False


# ============================================================================================================ #
# 3. locale-aware numbers
# ============================================================================================================ #
@dataclass
class _Ctx:
    conv: str          # "vi" | "en" | "mixed"
    decided: bool      # convention decided by the response's own unambiguous tokens
    amb: str           # reading of an ambiguous d[.,]ddd token when not decided: "dec" | "grp"
    list_comma: bool   # an unbracketed d,d (≠3 digits) is a list separator, not a decimal comma
    ambiguous: bool = False


def _votes(body: str) -> tuple:
    vi = en = 0
    vi += 2 * len(re.findall(r"\d\s*\{,\}\s*\d", body))
    for m in re.finditer(r"(?<![\w.,])\d+(?:[.,]\d+)+(?![\d])", body):
        tok = m.group(0)
        seps = re.findall(r"[.,]", tok)
        groups = re.split(r"[.,]", tok)
        before = body[m.start() - 1] if m.start() else " "
        after = body[m.end()] if m.end() < len(body) else " "
        if len(set(seps)) == 2:
            if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d+", tok):
                vi += 1
            elif re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d+", tok):
                en += 1
        elif len(seps) >= 2:
            if len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:]):
                vi += seps[0] == "."
                en += seps[0] == ","
        else:
            head, tail = groups
            if len(tail) == 3 and head != "0":
                continue  # ambiguous: no vote
            if seps[0] == ".":
                en += 1
            elif before not in "([{;," and after not in ")]};,":
                vi += 1  # "x = 2,5 kg" (not "(1,2)" or "{1,2}", which may be lists)
    return vi, en


def _context(body: str, g: Gold) -> _Ctx:
    vi, en = _votes(body)
    container = g.type in CONTAINER_TYPES or g.type == "interval"
    if vi != en:
        conv = "vi" if vi > en else "en"
        return _Ctx(conv, True, "dec", conv == "en")
    if g.locale in ("vi", "en"):
        return _Ctx(g.locale, False, "dec", g.locale == "en")
    return _Ctx("mixed", False, "grp" if g.type == "int" else "dec", container)


def _read_run(run: str, ctx: Optional[_Ctx]) -> Optional[str]:
    """One digit run with separators -> canonical '.'-decimal string, or None if it is not one number.
    ``ctx=None`` reads a canonical gold string ('.' decimal only)."""
    if _DEC in run:
        head, _, tail = run.partition(_DEC)
        if "," in head or "," in tail or "." in tail or _DEC in tail:
            return None
        if "." in head and not re.fullmatch(r"\d{1,3}(?:\.\d{3})+", head):
            return None
        return head.replace(".", "") + "." + tail
    if ctx is None:
        return run if re.fullmatch(r"\d+(?:\.\d+)?", run) else None
    seps = re.findall(r"[.,]", run)
    if not seps:
        return run
    if len(set(seps)) == 2:
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d+", run):
            return run.replace(".", "").replace(",", ".")
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d+", run):
            return run.replace(",", "")
        return None
    sep = seps[0]
    parts = run.split(sep)
    if len(parts) > 2:
        if len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:]):
            return "".join(parts)
        return None
    head, tail = parts
    if len(tail) != 3 or head == "0" or len(head) > 3:
        return head + "." + tail
    if not ctx.decided:
        ctx.ambiguous = True
    if ctx.conv == "vi":
        return head + "." + tail if sep == "," else head + tail
    if ctx.conv == "en":
        return head + tail if sep == "," else head + "." + tail
    return head + tail if ctx.amb == "grp" else head + "." + tail


_RUN = re.compile(r"(?<![\w\\])\d+(?:(?:[.,]|%s)\d+)*" % _DEC)


def _localize(s: str, ctx: Optional[_Ctx]) -> str:
    """Rewrite every digit run of ``s`` in canonical form (runs that are not one number are left alone)."""
    def rep(m):
        r = _read_run(m.group(0), ctx)
        return r if r is not None else m.group(0)
    return _RUN.sub(rep, s).replace(_DEC, ".")


def _plain_fraction(t: str) -> Optional[Fraction]:
    """Exact value of a plain number already in canonical form: 12, -0.5, 3/4, \\frac{3}{4}, 2\\frac14, 1 1/2."""
    t = t.strip()
    m = re.fullmatch(r"(-?)(\d+)\s+(\d+)\s*/\s*(\d+)", t)  # mixed number "19 1/3"
    if m and int(m.group(4)):
        v = int(m.group(2)) + Fraction(int(m.group(3)), int(m.group(4)))
        return -v if m.group(1) else v
    t = t.replace(" ", "")
    t = re.sub(r"^\+", "", t)
    while re.fullmatch(r"\((.*)\)", t) and _unwrap(t, "(", ")") is not None:
        t = t[1:-1]
    num = r"-?\d+(?:\.\d+)?"
    if re.fullmatch(num, t) or re.fullmatch(r"-?\.\d+", t):
        return Fraction(t)
    m = re.fullmatch(rf"({num})/({num})", t)
    if m:
        d = Fraction(m.group(2))
        return Fraction(m.group(1)) / d if d else None
    m = re.fullmatch(rf"(-?)\\frac(?:\{{({num})\}}|(\d))(?:\{{({num})\}}|(\d))", t)
    if m:
        d = Fraction(m.group(4) or m.group(5))
        if not d:
            return None
        v = Fraction(m.group(2) or m.group(3)) / d
        return -v if m.group(1) else v
    m = re.fullmatch(r"(-?)(\d+)\\frac(?:\{(\d+)\}|(\d))(?:\{(\d+)\}|(\d))", t)
    if m:
        d = int(m.group(5) or m.group(6))
        if not d:
            return None
        v = int(m.group(2)) + Fraction(int(m.group(3) or m.group(4)), d)
        return -v if m.group(1) else v
    return None


def _is_decimal_literal(t: str) -> bool:
    return bool(re.fullmatch(r"-?\d*\.\d+", t.replace(" ", "")))


# ============================================================================================================ #
# 4. symbolic equality (backend)
# ============================================================================================================ #
_MAX_SYM = 400


def _tex_to_sympy_str(s: str) -> str:
    t = s
    t = re.sub(r"\\(?:left|right)(?![A-Za-z])\s*\.?", "", t)
    t = re.sub(r"\\[dt]frac", r"\\frac", t)
    t = re.sub(r"\\(?:mathrm|text|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", t)

    def arg(u: str, j: int):
        while j < len(u) and u[j] == " ":
            j += 1
        if j >= len(u):
            return None
        if u[j] == "{":
            k = _close_brace(u, j)
            return None if k is None else (u[j + 1:k], k + 1)
        if u[j] == "\\":
            m = re.match(r"\\[A-Za-z]+", u[j:])
            return (m.group(0), j + m.end()) if m else None
        return u[j], j + 1

    for _ in range(40):
        m = re.search(r"\\frac", t)
        if not m:
            break
        a1 = arg(t, m.end())
        a2 = arg(t, a1[1]) if a1 else None
        if not a2:
            break
        t = t[:m.start()] + "((" + a1[0] + ")/(" + a2[0] + "))" + t[a2[1]:]
    for _ in range(40):
        m = re.search(r"\\sqrt\s*(?:\[([^\]]+)\])?", t)
        if not m:
            break
        a1 = arg(t, m.end())
        if not a1:
            break
        rep = f"(({a1[0]})**(1/({m.group(1)})))" if m.group(1) else f"sqrt({a1[0]})"
        t = t[:m.start()] + rep + t[a1[1]:]
    t = re.sub(r"\\binom\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"binomial(\1,\2)", t)
    t = re.sub(r"(\d+|\))!", r"factorial(\1)", t)
    for a, b in (("\\cdot", "*"), ("\\times", "*"), ("\\div", "/"), ("\\pi", " pi "), ("\\infty", " oo "),
                 ("\\ln", " log "), ("\\log", " log "), ("\\exp", " exp ")):
        t = t.replace(a, b)
    t = re.sub(r"\\(arcsin|arccos|arctan|sin|cos|tan|cot|sec|csc)", r" \1", t)
    t = t.replace("{", "(").replace("}", ")").replace("\\", " ")
    t = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", " I ", t)
    return t


def _to_sympy(s: str):
    import sympy as sp
    from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application, parse_expr,
                                            rationalize, standard_transformations)
    m = re.fullmatch(r"\s*\\begin\s*\{[pbvB]?matrix\}(.*)\\end\s*\{[pbvB]?matrix\}\s*", s, re.S)
    if m:
        rows = [r for r in re.split(r"\\\\", m.group(1)) if r.strip()]
        return sp.Matrix([[_to_sympy(c) for c in r.split("&")] for r in rows])
    tr = standard_transformations + (implicit_multiplication_application, convert_xor, rationalize)
    return parse_expr(_tex_to_sympy_str(s), transformations=tr,
                      local_dict={"I": sp.I, "pi": sp.pi, "oo": sp.oo, "E": sp.E}, evaluate=True)


def _sympy_eq(p: str, g: str) -> bool:
    import sympy as sp
    try:
        if ("=" in p) != ("=" in g):
            return False
        if "=" in p:
            pl, pr = p.split("=", 1)
            gl, gr = g.split("=", 1)
            dp = _to_sympy(pl) - _to_sympy(pr)
            dg = _to_sympy(gl) - _to_sympy(gr)
            if sp.simplify(dp - dg) == 0 or sp.simplify(dp + dg) == 0:
                return True
            q = sp.simplify(dp / dg)
            return bool(q.is_number and q != 0)
        a, b = _to_sympy(p), _to_sympy(g)
        if isinstance(a, sp.MatrixBase) or isinstance(b, sp.MatrixBase):
            return (isinstance(a, sp.MatrixBase) and isinstance(b, sp.MatrixBase) and a.shape == b.shape
                    and bool(sp.simplify(a - b).is_zero_matrix))
        if a == b:
            return True
        d = sp.simplify(a - b)
        if d == 0:
            return True
        if not a.free_symbols and not b.free_symbols:  # closed forms simplify() cannot match: compare at 50 digits
            rel = sp.N(sp.Abs(a - b) / sp.Max(1, sp.Abs(b)), 50)
            return bool(rel.is_number and rel.is_finite and rel < sp.Float("1e-30"))
        return False
    except Exception:
        return False


def _mv_eq(p: str, g: str) -> bool:
    parse, verify, _ = _MV
    try:
        gp = parse("$" + g + "$", parsing_timeout=_MV_TIMEOUT)
        pp = parse("$" + p + "$", parsing_timeout=_MV_TIMEOUT)
        if not gp or not pp:
            return False
        return bool(verify(gp, pp, float_rounding=10, timeout_seconds=_MV_TIMEOUT))
    except Exception:
        return False


def _sym_eq(p: str, g: str) -> bool:
    if not p.strip() or not g.strip() or len(p) > _MAX_SYM or len(g) > _MAX_SYM:
        return False
    if re.sub(r"\s+", "", p) == re.sub(r"\s+", "", g):
        return True
    if _use_mv:
        # math-verify 0.9.0 leaves some sums unsimplified when the gold is a bare symbol (x vs x/2 + x/2 -> False);
        # an exact SymPy zero-difference check is the second opinion on its negatives.
        return _mv_eq(p, g) or _sympy_eq(p, g)
    return _sympy_eq(p, g)


def _numeric(x: str) -> Optional[complex]:
    f = _plain_fraction(x)
    if f is not None:
        return complex(float(f))
    try:
        import sympy as sp
        v = _to_sympy(x)
        if not getattr(v, "free_symbols", {1}):
            return complex(sp.N(v, 30))
    except Exception:
        pass
    return None


# ============================================================================================================ #
# 5. typed comparison
# ============================================================================================================ #
def _atom_eq(pred: str, gval: str, g: Gold, ctx: _Ctx) -> bool:
    """One predicted atom (already free of names/prose) vs one canonical gold atom."""
    t = pred.strip()
    pct = "%" in t
    t = t.replace("%", " ")
    mult = None
    for w, k in _MULT.items():
        if re.search(r"(?<![^\W\d_])%s(?![^\W\d_])" % w, t, re.I):
            mult = k
            t = re.sub(r"(?<![^\W\d_])%s(?![^\W\d_])" % w, " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return False
    t = _localize(t, ctx)
    gv = _localize(gval.replace("\\%", "").replace("%", ""), None)
    fg = _plain_fraction(gv)
    fp = _plain_fraction(t)
    cands = [fp] if fp is not None else []
    if fp is not None:
        if pct and not g.percent:
            cands.append(fp / 100)
        if mult:
            cands.append(fp * mult)
    if g.round_dp is not None:
        vg = complex(float(fg)) if fg is not None else _numeric(gv)
        vps = [complex(float(c)) for c in cands] or ([_numeric(t)] if _numeric(t) is not None else [])
        tol = 0.5 * 10 ** (-g.round_dp) + 1e-12
        return vg is not None and any(abs(v - vg) <= tol for v in vps)
    if fg is not None and cands:
        return any(c == fg for c in cands)
    if fp is not None and fg is None:
        if _is_decimal_literal(t):  # 1.4142 is not \sqrt{2}: decimals are exact
            return False
        return _sym_eq(t, gv)
    if fg is not None:
        if re.fullmatch(r"[-\d.\s/]*", t):
            return False
        ex = _exact_rational(t)  # rational gold: the prediction must be exactly that rational (√2 ≠ 1.41421…)
        if ex is not None:
            return ex == fg
        if _numeric(t) is not None:  # a closed-form constant that is not rational
            return False
        return _sym_eq(t, gv)
    return _sym_eq(t, gv)


def _exact_rational(x: str) -> Optional[Fraction]:
    """Exact rational value of a closed-form LaTeX constant (``2^{10}``, ``\\sqrt{9}``), else None."""
    if len(x) > _MAX_SYM:
        return None
    try:
        import sympy as sp
        v = _to_sympy(x)
        if getattr(v, "free_symbols", {1}):
            return None
        v = sp.simplify(v)
        if v.is_Rational:
            return Fraction(int(v.p), int(v.q))
    except Exception:
        return None
    return None


_NAME = r"(?:[A-Za-z](?:_\{?\w+\}?)?'?|[A-Z]{2,3})"
_CONNECT = r"\s+(?:hoặc|và|or|and)\s+|\\vee|\\lor|\\text\s*\{\s*(?:hoặc|và|or|and)\s*\}"


def _split_name(item: str) -> tuple:
    m = re.match(r"^\s*(%s)\s*=\s*(.+)$" % _NAME, item, re.S)
    if m and "=" not in m.group(2):
        return m.group(1).replace("{", "").replace("}", ""), m.group(2).strip()
    return None, item.strip()


def _last_rhs(item: str) -> str:
    parts = _split_top(item, ("=", "\\approx"))
    return parts[-1].strip() if len(parts) > 1 and parts[-1].strip() else item.strip()


def _deprose(item: str) -> tuple:
    """(atom, had_prose): an atom that still holds words falls back to its last number/fraction."""
    if not _has_prose(item):
        return item, False
    nums = list(_LASTNUM.finditer(item.replace(_DEC, "{,}")))
    if not nums:
        return item, True
    return nums[-1].group(0).replace("{,}", _DEC).strip(), True


def _comma_split(s: str, ctx: _Ctx, bracketed: bool) -> list:
    """Depth-0 commas: list separators, except inside one digit run that reads as one number."""
    out, cur, depth = [], [], 0
    for i, c in enumerate(s):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        if c == "," and depth == 0:
            prev = s[i - 1] if i else " "
            nxt = s[i + 1] if i + 1 < len(s) else " "
            if not bracketed and prev.isdigit() and nxt.isdigit():
                a = i
                while a and (s[a - 1].isdigit() or s[a - 1] in ".,"):
                    a -= 1
                b = i + 1
                while b < len(s) and (s[b].isdigit() or s[b] in ".,"):
                    b += 1
                run = s[a:b]
                tail = run.split(",")[-1]
                reads_as_number = _read_run(run, ctx) is not None
                is_num = reads_as_number and not (ctx.list_comma and run.count(",") == 1
                                                  and len(tail) != 3 and "." not in run)
                if reads_as_number and not is_num and not ctx.decided:
                    ctx.ambiguous = True  # "2,5" taken as a list only by the tie-break: flag it
                if is_num:
                    cur.append(c)
                    continue
            out.append("".join(cur))
            cur = []
            continue
        cur.append(c)
    out.append("".join(cur))
    return out


def _strip_lhs(s: str) -> str:
    m = re.match(r"^\s*(?:%s|\([A-Za-z0-9_{},;\s]+\))\s*(?:=|\\in)\s*(?=\\\{|[({\[\]]|\\emptyset|\\varnothing)"
                 % _NAME, s)
    return s[m.end():] if m else s


def _container_items(s: str, ctx: _Ctx) -> list:
    s = _strip_lhs(s.strip())
    s = re.sub(r"^[A-Z]'?\s*(?=\()", "", s)  # point M(1; -3)
    bracketed = False
    for o, c in (("\\{", "\\}"), ("{", "}"), ("(", ")"), ("[", "]")):
        inner = _unwrap(s, o, c)
        if inner is not None and (o in ("\\{", "{") or _split_top(inner, (";", ","))[1:]):
            s, bracketed = inner, True
            break
    s = re.sub(_CONNECT, " ⎘ ", s, flags=re.I)  # hoặc / và / or / and / \vee
    if len(_split_top(s, (";",))) > 1:  # ';' present: it is the separator and commas are decimal marks
        parts = _split_top(s, (";", "⎘"))
    else:                               # "7, -2, and -5"
        parts = [p for piece in _split_top(s, ("⎘",)) for p in _comma_split(piece, ctx, bracketed)]
    items = []
    for it in (x.strip() for x in parts):
        if not it:
            continue
        if it.count("\\pm") == 1:
            items += [it.replace("\\pm", "+"), it.replace("\\pm", "-")]
        else:
            items.append(it)
    return items


def _item_eq(item: str, gval: str, g: Gold, ctx: _Ctx) -> bool:
    if "=" not in gval:
        item = _last_rhs(_split_name(item)[1])
    item, _ = _deprose(item)
    return _atom_eq(item, gval, g, ctx)


def _target_value(span: str, var: str) -> Optional[str]:
    ms = list(re.finditer(r"(?<![A-Za-z\\])(%s)\s*=\s*" % _NAME, span))
    hits = []
    for i, m in enumerate(ms):
        if m.group(1).replace("{", "").replace("}", "") != var.replace("{", "").replace("}", ""):
            continue
        end = ms[i + 1].start() if i + 1 < len(ms) else len(span)
        val = re.split(r"\s+(?:thì|khi|và|nên|and|then|so)\s+|[,;]\s", span[m.end():end] + " ")[0]
        hits.append(val.strip())
    return hits[-1] if hits else None


# ---- intervals ---------------------------------------------------------------------------------------------- #
_REL = {"<": ("<", False), ">": (">", False), "\\lt": ("<", False), "\\gt": (">", False),
        "\\le": ("<", True), "\\leq": ("<", True), "\\leqslant": ("<", True), "<=": ("<", True),
        "\\ge": (">", True), "\\geq": (">", True), "\\geqslant": (">", True), ">=": (">", True)}
_REL_RE = re.compile(r"\\leqslant|\\geqslant|\\leq|\\geq|\\le|\\ge|\\lt|\\gt|<=|>=|<|>")
_INF = {"+\\infty": 1, "\\infty": 1, "+oo": 1, "oo": 1, "infty": 1, "+infty": 1, "-\\infty": -1, "-oo": -1,
        "-infty": -1}


def _endpoint(x: str, ctx: Optional[_Ctx]):
    t = x.strip().replace(" ", "")
    if t in _INF:
        return float("inf") * _INF[t]
    return _localize(x.strip(), ctx)


def _bracket_interval(p: str, ctx):
    p = p.strip()
    pt = _unwrap(p, "\\{", "\\}")
    if pt is not None and len(_split_top(pt, (",", ";"))) == 1:
        a = _endpoint(pt, ctx)
        return (True, a, a, True)
    if len(p) < 5 or p[0] not in "([]" or p[-1] not in ")[]":
        return None
    inner = p[1:-1]
    parts = _split_top(inner, (";",))
    if len(parts) != 2:
        parts = _split_top(inner, (",",))
    if len(parts) != 2:
        return None
    return (p[0] == "[", _endpoint(parts[0], ctx), _endpoint(parts[1], ctx), p[-1] == "]")


def _inequality(piece: str, ctx):
    lo, hi, lo_c, hi_c = float("-inf"), float("inf"), False, False
    conj = [c for c in re.split(r"\s+(?:và|and)\s+|\\wedge|\\land|,|;", piece) if c.strip()]
    for c in conj:
        ops = _REL_RE.findall(c)
        toks = [t.strip() for t in _REL_RE.split(c)]
        if not ops:
            return None
        var = [i for i, t in enumerate(toks) if re.fullmatch(_NAME, t)]
        if len(var) != 1:
            return None
        v = var[0]
        for i, op in enumerate(ops):
            sym, closed = _REL[op]
            if i + 1 == v:  # bound OP var
                b = _endpoint(toks[i], ctx)
                if sym == "<":
                    lo, lo_c = b, closed
                else:
                    hi, hi_c = b, closed
            elif i == v:    # var OP bound
                b = _endpoint(toks[i + 1], ctx)
                if sym == "<":
                    hi, hi_c = b, closed
                else:
                    lo, lo_c = b, closed
            else:
                return None
    return (lo_c, lo, hi, hi_c)


def _intervals(s: str, ctx) -> Optional[list]:
    t = s.strip()
    t = re.sub(r"^\s*(?:%s)\s*(?:=|\\in)\s*" % _NAME, "", t)
    t = re.sub(r"\\mathbb\s*\{?\s*R\s*\}?|^\s*R\s*$", "(-\\\\infty;+\\\\infty)", t)
    pieces = [x for x in _split_top(re.sub(r"\\cup|\s+U\s+", "|", t), ("|",)) if x.strip()]
    if len(pieces) == 1 and _REL_RE.search(pieces[0]):
        pieces = [x for x in re.split(r"\s+(?:hoặc|or)\s+|\\vee|\\lor", pieces[0], flags=re.I) if x.strip()]
    out = []
    for pc in pieces:
        iv = _inequality(pc, ctx) if _REL_RE.search(pc) else _bracket_interval(pc, ctx)
        if iv is None:
            return None
        out.append(iv)
    return out


def _ep_eq(a, b, g: Gold) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, float) and isinstance(b, float) and a == b
    return _atom_eq(a, b, g, _Ctx("en", True, "dec", True))  # both already localised


def _sort_key(iv):
    if isinstance(iv[1], float):
        return iv[1]
    v = _numeric(iv[1])
    return v.real if v is not None else 0.0


def _intervals_eq(p, q, g: Gold) -> bool:
    if p is None or q is None or len(p) != len(q):
        return False
    for (plc, plo, phi, phc), (glc, glo, ghi, ghc) in zip(sorted(p, key=_sort_key), sorted(q, key=_sort_key)):
        if not (_ep_eq(plo, glo, g) and _ep_eq(phi, ghi, g)):
            return False
        if (not isinstance(glo, float) and plc != glc) or (not isinstance(ghi, float) and phc != ghc):
            return False
    return True


def _norm_text(s: str) -> str:
    t = _nfc(s).lower()
    t = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"[\\{}$()\[\].,;:!?\"'`*]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # leading filler that never changes the answer: "Ngày 20 tháng 6" = "20 tháng 6", "the parabola" = "parabola"
    return re.sub(r"^(?:(?:ngày|vào|vào ngày|là|the|on|an?)\s+)+", "", t)


_EMPTY_PRED = re.compile(r"^\s*(?:\\emptyset|\\varnothing|\\\{\s*\\\}|\{\s*\}|S\s*=\s*\\emptyset|vô\s*nghiệm|"
                         r"không\s*có\s*nghiệm(?:\s*nào)?|no\s+(?:real\s+)?solutions?)\s*$", re.I)
_EMPTY_ANY = re.compile(r"vô\s*nghiệm|không\s*có\s*nghiệm|\\emptyset|\\varnothing|no\s+(?:real\s+)?solutions?",
                        re.I)


# ============================================================================================================ #
# 6. top level
# ============================================================================================================ #
def grade(pred, gold, *, locale: Optional[str] = None) -> dict:
    """Grade one raw model response against one typed gold (plus its ``accept_also`` alternatives).

    A gold dict may carry ``accept_also``: a list of alternative gold dicts (same schema; a missing ``locale``
    inherits the main gold's).  The response is correct if it matches the main gold or any alternative;
    ``matched`` records which one (``"gold"`` | ``"accept_also[i]"`` | None).  See :func:`_grade_one`.
    """
    res = _grade_one(pred, gold, locale=locale)
    res["matched"] = "gold" if res["correct"] else None
    alts = gold.get("accept_also") if isinstance(gold, dict) else None
    if res["correct"] or not alts:
        return res
    for i, alt in enumerate(alts):
        if isinstance(alt, dict) and "locale" not in alt and gold.get("locale"):
            alt = {**alt, "locale": gold["locale"]}
        r = _grade_one(pred, alt, locale=locale)
        if r["correct"]:
            r["matched"] = f"accept_also[{i}]"
            return r
    return res


def _grade_one(pred, gold, *, locale: Optional[str] = None) -> dict:
    """Grade one raw model response against one typed gold.

    Returns ``{correct, span, method, flag, conv, gold_type, backend, grader_version}``.  ``method`` is the
    extraction route (boxed | boxed-multi | boxed-hedge | #### | cue | whole | last-number | degenerate | none).
    ``flag`` marks items worth adjudicating: no-answer | degenerate | hedged | arity p!=g | ambiguous-locale |
    prose-span | unparsed-interval | unclosed-think | text-gold.
    """
    g = as_gold(gold, locale=locale)
    res = dict(correct=False, span=None, method="none", flag=None, conv=None, gold_type=g.type,
               backend=BACKEND, grader_version=GRADER_VERSION)
    if pred is None or not str(pred).strip():
        res["flag"] = "no-answer"
        return res
    body, flag = _clean(str(pred))
    res["flag"] = flag
    arity = len(g.items()) if g.type in CONTAINER_TYPES else 1
    span, how = _extract(body, arity)
    res["span"], res["method"] = span, how
    if g.type in CONTAINER_TYPES and not g.items() and how in ("cue", "whole", "last-number", "none",
                                                                "degenerate"):
        m = _EMPTY_ANY.search(body[-300:])
        if m:
            res.update(correct=True, span=m.group(0), method="empty-set")
            return res
    if span is None or not span.strip():
        res["flag"] = "degenerate" if how == "degenerate" else "no-answer"
        return res
    if how == "boxed-hedge":
        res["flag"] = "hedged"
        return res
    ctx = _context(body, g)
    res["conv"] = ctx.conv + ("" if ctx.decided else "?")
    s = _prenorm(span if g.type in ("text", "choice") else _trim_prose(span), g.type)
    ok = False

    if g.type == "choice":
        letters = re.findall(r"(?<![A-Za-z\\])([A-E])(?![A-Za-z])", s)
        ok = bool(letters) and letters[-1] == g.value.strip().strip("()").upper()

    elif g.type == "text":
        ns, ng = _norm_text(s), _norm_text(g.value)
        tail = re.split(r"\s(?:là|is|bằng)\s|:", " " + ns + " ")[-1].strip()
        ok = bool(ng) and ng in (ns, tail)
        res["flag"] = res["flag"] or (None if ok else "text-gold")

    elif g.type == "interval":
        gi = _intervals(_prenorm(g.value, "interval"), None)
        pi = _intervals(s, ctx)
        if pi is None:
            res["flag"] = res["flag"] or "unparsed-interval"
        ok = _intervals_eq(pi, gi, g)

    elif g.type in CONTAINER_TYPES:
        gitems = g.items()
        if not gitems:
            ok = bool(_EMPTY_PRED.match(s))
        else:
            pitems = _container_items(s, ctx)
            if len(pitems) != len(gitems):
                res["flag"] = f"arity {len(pitems)}!={len(gitems)}"
            elif g.type == "tuple":
                named = [_split_name(x) for x in pitems]
                names = [n for n, _ in named]
                if g.vars and all(names) and len(set(names)) == len(names) and set(names) == set(g.vars):
                    order = {n: i for i, n in enumerate(g.vars)}
                    named.sort(key=lambda t: order[t[0]])
                elif g.vars and any(names) and not {n for n in names if n} <= set(g.vars):
                    named = []
                ok = bool(named) and all(_item_eq(v, gv, g, ctx) for (_, v), gv in zip(named, gitems))
            else:
                left = list(gitems)
                ok = True
                for it in pitems:
                    v = _split_name(it)[1]
                    hit = next((gv for gv in left if _item_eq(v, gv, g, ctx)), None)
                    if hit is None:
                        ok = False
                        break
                    left.remove(hit)
                ok = ok and not left

    else:  # scalar: int | rational | decimal | radical | expr
        work = s
        if g.target_var:
            tv = _target_value(work, g.target_var)
            if tv is not None:
                work = tv
        if "=" in g.value:
            items = [work]
        elif not _has_prose(work):
            items = _container_items(work, ctx)
            bracketed_pair = len(items) >= 2 and re.match(r"^\s*[A-Z]?'?\s*[(\[]", work)
            if bracketed_pair and not (g.value.startswith("(") or g.value.startswith("[")):
                res["flag"] = f"arity {len(items)}!=1"
                items = []
        else:
            pieces = [x.strip() for x in re.split(_CONNECT + r"|;", work, flags=re.I) if x.strip()]
            items = pieces if len(pieces) >= 2 and not any(_has_prose(x) for x in pieces) else [work]
        if items:
            distinct = []
            for it in items:
                if not any(_item_eq(it, d_, g, ctx) or _same_surface(_last_rhs(it), _last_rhs(d_))
                           for d_ in distinct):
                    distinct.append(it)
            if len(distinct) >= 2:
                res["flag"] = "hedged"
            else:
                atom, prose = _deprose(_last_rhs(_split_name(items[0])[1]) if "=" not in g.value else items[0])
                if prose:
                    res["flag"] = res["flag"] or "prose-span"
                ok = _atom_eq(atom, g.value, g, ctx)

    if ctx.ambiguous and not res["flag"]:
        res["flag"] = "ambiguous-locale"
    res["correct"] = bool(ok)
    return res


def is_correct(pred, gold, *, locale: Optional[str] = None, **_legacy) -> bool:
    """True iff the response ``pred`` answers ``gold`` (typed dict, or a legacy canonical gold string).

    Legacy keywords (``rel_tol``, ``abs_tol``) are accepted and ignored: numbers are exact; put ``round_dp``
    in the gold for rounded answers.
    """
    if gold is None:
        return False
    return grade(pred, gold, locale=locale)["correct"]


# ============================================================================================================ #
# 7. gold typing (migration of legacy gold strings; NOT extraction)
# ============================================================================================================ #
_INTERVAL_CUES = re.compile(r"khoảng|interval|tập\s*xác\s*định|miền|tập\s*giá\s*trị|domain|range|"
                            r"tập\s*nghiệm\s*của\s*bất|chạy\s*trên|interval\s*notation", re.I)
_CHOICE_RE = re.compile(r"\(?\s*([A-E])\s*\)?")


def _canon_scalar(t: str, loc: str) -> tuple:
    """(answer_type, canonical value, target_var) of one scalar gold string."""
    t = t.strip()
    target = None
    m = re.match(r"^\s*(%s)\s*=\s*(.+)$" % _NAME, t)
    if m and "=" not in m.group(2) and not re.search(r"[A-Za-z]", re.sub(r"\\[A-Za-z]+", "", m.group(2))):
        target, t = m.group(1).replace("{", "").replace("}", ""), m.group(2).strip()
    ctx = _Ctx("vi" if loc == "vi" else "en", True, "dec", loc != "vi")
    t = re.sub(r"^\+", "", _localize(t, ctx).strip())
    if re.fullmatch(r"-?\d+", t):
        return "int", str(int(t)), target
    if re.fullmatch(r"-?\d*\.\d+", t):
        f = Fraction(t)
        if f.denominator == 1:
            return "int", str(f.numerator), target
        t = re.sub(r"^(-?)\.", r"\g<1>0.", t)
        return "decimal", t.rstrip("0").rstrip("."), target
    f = _plain_fraction(t)
    if f is not None:
        if f.denominator == 1:
            return "int", str(f.numerator), target
        return "rational", t, target
    if "\\sqrt" in t and not re.search(r"[A-Za-z]", re.sub(r"\\[A-Za-z]+", "", t)):
        return "radical", t, target
    return "expr", t, target


def infer_gold(raw, *, problem: Optional[str] = None, locale: str = "mixed") -> dict:
    """Type a legacy gold string once: e.g. the boxed answer of a MATH solution or a GSM8K ``####`` value.

    The whole string is the answer — nothing is searched for inside it, so a multi-part gold stays multi-part.
    ``problem`` (optional) breaks the ``(a,b)`` tuple-vs-open-interval tie.  Returns a typed gold dict.
    """
    s = _uni(str(raw)).strip()
    bx = _boxes(s)
    if len(bx) == 1 and bx[0][1] == 0 and bx[0][2] == len(s):
        s = bx[0][0]
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"\\(?:left|right)(?![A-Za-z])\s*\.?", "", s)
    s = re.sub(r"(\d),\s*\\!\s*(?=\d{3}(?!\d))", r"\1", s)
    s = re.sub(r"\\!|\\displaystyle", "", s)
    s = re.sub(r"(\d)\s*\{,\}\s*(?=\d)", r"\1" + _DEC, s)
    s = re.sub(r"(\d)\\,(?=\d{3}(?!\d))", r"\1", s)
    s = s.replace("\\\\", "\u2397")
    s = re.sub(r"\\[,;: ]|~|\\quad|\\qquad", " ", s)
    s = s.replace("\u2397", "\\\\").strip().rstrip(".").strip()
    out: dict = {"locale": locale}
    if re.search(r"\\?%\s*$", s):
        s = re.sub(r"\\?%\s*$", "", s).strip()
        out["percent"] = True
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "", s).strip()

    mc = re.fullmatch(r"\\text\s*\{\s*\(?([A-E])\)?\s*\}|\(?([A-E])\)", s) or \
        (re.fullmatch(r"[A-E]", s) and re.fullmatch(r"([A-E])", s))
    if mc:
        return {**out, "answer_type": "choice", "answer_canonical": mc.group(1) or mc.group(2)}
    mt = re.fullmatch(r"\\(?:text|textbf|mathrm|mbox|textrm)\s*\{([^{}]*)\}", s)
    if mt:
        inner = mt.group(1).strip()
        if not re.fullmatch(r"[-+\d.,\s/]+", inner):
            return {**out, "answer_type": "text", "answer_canonical": inner}
        s = inner
    if re.fullmatch(r"[^\W\d_][^\W\d_\s]*(?:\s+[^\W\d_]+)*", s) and _has_prose(s):
        return {**out, "answer_type": "text", "answer_canonical": s}
    s = re.sub(r"\\text\s*\{\s*(?:và|hoặc|or|and)\s*\}", " and ", s, flags=re.I)
    wo = re.sub(r"\\(?:text|mbox|mathrm|textrm)\s*\{[^{}]*\}(?:\s*\^\s*\{?[23]\}?)?", " ", s).strip()
    if wo and re.search(r"[\d\\]", wo):
        s = wo
    if re.search(r"\d", s):
        s = _UNIT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()

    body = re.sub(r"^\s*[A-Za-z]\s*\\in\s*", "", s)
    pair = bool(re.fullmatch(r"[\[\(\]].*[\]\)\[]", body)) and len(_split_top(body[1:-1], (",", ";"))) == 2
    if ("\\cup" in body and _intervals(body, None)) or \
            (pair and ("\\infty" in body or body[0] in "[]" or body[-1] in "[]")) or \
            (pair and problem and _INTERVAL_CUES.search(problem)) or re.fullmatch(r"\\mathbb\s*\{?R\}?", body):
        return {**out, "answer_type": "interval", "answer_canonical": body}
    if _REL_RE.search(body) and _intervals(body, None) is not None:
        return {**out, "answer_type": "interval", "answer_canonical": body}

    loc = "vi" if locale == "vi" else "en"
    ctx = _Ctx(loc, True, "dec", loc == "en")
    if _unwrap(s, "\\{", "\\}") is not None:
        items = _container_items(s, ctx)
        return {**out, "answer_type": "set",
                "answer_canonical": "; ".join(_canon_scalar(_split_name(i)[1], loc)[1] for i in items)}
    if _unwrap(s, "(", ")") is not None and len(_split_top(s[1:-1], (",", ";"))) >= 2:
        items = _container_items(s, ctx)
        named = [_split_name(i) for i in items]
        res = {**out, "answer_type": "tuple",
               "answer_canonical": "; ".join(_canon_scalar(v, loc)[1] for _, v in named)}
        if all(n for n, _ in named):
            res["vars"] = [n for n, _ in named]
        return res
    items = _container_items(s, ctx) if not _unwrap(s, "\\begin", "}") else [s]
    if len(items) >= 2 and not re.search(r"\\begin\s*\{", s):
        named = [_split_name(i) for i in items]
        names = [n for n, _ in named]
        is_tuple = all(names) and len(set(names)) == len(names) and \
            not all(re.fullmatch(r"[A-Za-z]_\{?\d+\}?", n) for n in names)
        res = {**out, "answer_type": "tuple" if is_tuple else "set",
               "answer_canonical": "; ".join(_canon_scalar(v, loc)[1] for _, v in named)}
        if is_tuple:
            res["vars"] = names
        return res
    gtype, val, target = _canon_scalar(s, loc)
    res = {**out, "answer_type": gtype, "answer_canonical": val}
    if target:
        res["target_var"] = target
    return res


# ============================================================================================================ #
# 8. thin compatibility layer for pre-X1 callers (jobs/*/run.py, data_fix/mta_vi.py, vi_exam/build.py)
# ============================================================================================================ #
def extract_answer(text) -> Optional[str]:
    """Prediction-side answer span (last box > #### > cue > whole > last number).  Never use it on golds.
    ``is_correct(extract_answer(r), gold)`` works, but prefer ``grade(r, gold)``: container golds need the full
    response to join several final boxes."""
    if not isinstance(text, str) or not text.strip():
        return None
    body, _ = _clean(text)
    return _extract(body, 1)[0]


def normalize_answer(s, locale: str = "vi") -> str:
    """Canonical string of an answer span (debugging / old callers only; grading uses :func:`grade`).
    ``locale="vi"`` reads ``3,5`` as 3.5 (the old callers handle Vietnamese exam answers)."""
    if s is None or not str(s).strip():
        return ""
    t = str(s)
    bx = _boxes(t)
    t = bx[-1][0] if bx else t
    d = infer_gold(t, locale=locale)
    return d["answer_canonical"].replace(" ", "")


def _to_float(s) -> Optional[float]:
    """Deprecated helper kept for vi_exam/build.py: float value of a plain numeric string, else None."""
    if s is None:
        return None
    f = _plain_fraction(_localize(_prenorm(str(s), "int"), _Ctx("mixed", False, "dec", False)))
    return float(f) if f is not None else None


def _sympy_equal(a, b) -> bool:
    """Deprecated helper kept for vi_exam/build.py: symbolic equality of two LaTeX atoms (active backend)."""
    return _sym_eq(str(a), str(b))
