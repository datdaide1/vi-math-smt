"""The single accuracy definition for the whole project (T0.2).

Every arm and every leaderboard model is scored through :func:`is_correct`.
Consolidates the three divergent scorers in ``src/evaluate/`` into one; the old
regexes are NOT copied (several were broken — see the review notes).

Public API
----------
extract_answer(text)            -> str | None   pull the final answer out of a model output
normalize_answer(s)             -> str          canonical string form
is_correct(pred, gold, ...)     -> bool         normalize + SymPy-equivalence + numeric tolerance

Vietnamese notes
----------------
* Decimal comma: ``3,14`` == ``3.14``.  Thousands dot: ``1.000.000`` == ``1000000``.
  These conflict, so we only collapse ``.``/``,`` as separators when the grouping
  is unambiguous (``\\d{1,3}(?:[.,]\\d{3})+`` -> thousands; a single trailing
  ``,\\d+`` / ``.\\d+`` -> decimal).
* Unit words (``nghìn đồng``, ``km``, ``cm``, ``đô la`` …) and ``\\text{…}`` are stripped.
"""
from __future__ import annotations

import math
import re
from typing import Optional

try:  # sympy is a hard dep, but keep import failure survivable for extract-only use
    from sympy import simplify, nsimplify, Rational
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )

    _TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
    _HAVE_SYMPY = True
except Exception:  # pragma: no cover
    _HAVE_SYMPY = False


# --------------------------------------------------------------------------- #
# answer extraction
# --------------------------------------------------------------------------- #

_VI_CUES = [
    r"đáp\s*án\s*(?:cuối\s*cùng)?\s*(?:là|:)?",
    r"đáp\s*số\s*(?:là|:)?",
    r"kết\s*quả\s*(?:là|:)?",
    r"vậy[^\n.=]*?(?:=|là)\s*",
    r"the\s+(?:final\s+)?answer\s+is\s*:?",
]
_CUE_RE = re.compile("(?:%s)" % "|".join(_VI_CUES), re.IGNORECASE)


def _last_boxed(text: str) -> Optional[str]:
    """Return the content of the last ``\\boxed{...}`` / ``\\fbox{...}`` (brace-balanced)."""
    for key in ("\\boxed", "\\fbox"):
        idx = text.rfind(key)
        if idx < 0:
            continue
        i = text.find("{", idx)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1 : j].strip()
    return None


def _last_number(text: str) -> Optional[str]:
    # allow fractions a/b, decimals with . or , , leading sign, LaTeX \frac handled elsewhere
    nums = re.findall(r"-?\d+(?:[.,]\d+)*(?:\s*/\s*-?\d+(?:[.,]\d+)*)?", text)
    return nums[-1].strip() if nums else None


def extract_answer(text: str) -> Optional[str]:
    """Priority: ``\\boxed{}`` > ``####`` (GSM8K) > Vietnamese/English cue > last number."""
    if not isinstance(text, str) or not text.strip():
        return None

    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()

    if "####" in text:
        tail = text.rsplit("####", 1)[-1].strip()
        cand = tail.splitlines()[0].strip() if tail else ""
        if cand:
            return cand

    last_cue = None
    for m in _CUE_RE.finditer(text):
        last_cue = m
    if last_cue is not None:
        tail = text[last_cue.end() :].strip()
        # take the first line / sentence chunk after the cue
        chunk = re.split(r"[\n.]|(?<=\d)\s+(?=[A-ZĐ])", tail, maxsplit=1)[0].strip()
        chunk = chunk.rstrip(".").strip()
        if chunk:
            return chunk

    return _last_number(text)


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #

_UNIT_WORDS = (
    r"nghìn\s*đồng|triệu\s*đồng|tỉ\s*đồng|tỷ\s*đồng|đồng|"
    r"đô\s*la|dollars?|"
    r"km/h|m/s|km²|m²|cm²|km2|m2|cm2|km|cm|dm|mm|m|kg|g|"
    r"giờ|phút|giây|ngày|tuần|tháng|năm|"
    r"sản\s*phẩm|học\s*sinh|người|cây|quả|viên|chiếc|cái|đơn\s*vị|"
    r"phần\s*trăm|độ"
)
_UNIT_RE = re.compile(r"\s*(?:%s)\s*$" % _UNIT_WORDS, re.IGNORECASE)


def _strip_latex(s: str) -> str:
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\cfrac", "\\frac")
    s = re.sub(r"\\text\s*\{[^{}]*\}", "", s)
    s = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\operatorname\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\^\s*\{?\\?circ\}?", "", s)          # degrees
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("\\pi", "pi").replace("\\infty", "oo")
    s = s.replace("\\approx", "").replace("\\pm", "")
    # \frac{a}{b} -> ((a)/(b))   (repeat for nesting)
    for _ in range(6):
        new = re.sub(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s)
        if new == s:
            break
        s = new
    # \sqrt{x} -> sqrt(x) ; \sqrt[n]{x} -> (x)**(1/n)
    s = re.sub(r"\\sqrt\s*\[([^\]]+)\]\s*\{([^{}]+)\}", r"((\2)**(1/(\1)))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", s)
    s = s.replace("^", "**")
    s = s.replace("\\", "")
    return s


def _collapse_number_seps(s: str) -> str:
    # unambiguous thousands grouping:  1.000.000 / 1,000,000 -> 1000000
    s = re.sub(r"(?<=\d)([.,])(?=\d{3}(?:\D|$))(?=(?:\d{3}([.,])?)+(?:\D|$))", "", s)
    s = re.sub(r"(?<=\d)[.,](?=\d{3}(?:\D|$))", "", s)
    # a single decimal comma -> dot   (3,14 -> 3.14) ; leave a lone dot alone
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    return s


def normalize_answer(s) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    unwrapped = _last_boxed(s)
    if unwrapped is not None:
        s = unwrapped.strip()
    s = s.replace("$", "").replace("\\$", "")
    # drop an "x =" / "S =" prefix, keep the RHS
    if "=" in s:
        s = s.split("=")[-1].strip()
    s = _strip_latex(s)
    s = _UNIT_RE.sub("", s).strip()
    s = _UNIT_RE.sub("", s).strip()  # a second unit word (e.g. "365 nghìn đồng")
    s = _collapse_number_seps(s)
    s = s.replace(" ", "")
    # tuple / coordinate separators -> comma
    s = s.replace(";", ",")
    # strip one outer paren pair around a tuple:  (1,2) stays, ((3)/(4)) stays
    return s.strip()


# --------------------------------------------------------------------------- #
# equivalence
# --------------------------------------------------------------------------- #

def _to_float(s: str) -> Optional[float]:
    try:
        if "/" in s and s.count("/") == 1:
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)
    except Exception:
        return None


def _sympy_equal(a: str, b: str) -> bool:
    if not _HAVE_SYMPY:
        return False
    try:
        ea = parse_expr(a, transformations=_TRANSFORMS, evaluate=True)
        eb = parse_expr(b, transformations=_TRANSFORMS, evaluate=True)
    except Exception:
        return False
    try:
        if simplify(ea - eb) == 0:
            return True
    except Exception:
        pass
    try:
        d = complex(ea.evalf() - eb.evalf())
        if abs(d) < 1e-9:
            return True
        mb = abs(complex(eb.evalf()))
        if mb > 0 and abs(d) / mb < 1e-9:
            return True
    except Exception:
        pass
    return False


def _split_tuple(s: str) -> list[str]:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts = [p for p in s.split(",") if p != ""]
    return parts if len(parts) > 1 else [s]


_INT_RE = re.compile(r"^-?\d+$")


def _atoms_equal(p: str, g: str, rel_tol: float, abs_tol: float) -> bool:
    if p == "" or g == "":
        return False
    if p == g:
        return True
    pt, gt = _split_tuple(p), _split_tuple(g)
    if len(pt) != len(gt):
        return False
    if len(pt) > 1:
        return all(_atoms_equal(x, y, rel_tol, abs_tol) for x, y in zip(pt, gt))
    if _INT_RE.match(p) and _INT_RE.match(g):        # exact for integers (rel_tol lies on big ints)
        return int(p) == int(g)
    fp, fg = _to_float(p), _to_float(g)
    if fp is not None and fg is not None:
        return math.isclose(fp, fg, rel_tol=rel_tol, abs_tol=abs_tol)
    return _sympy_equal(p, g)


def is_correct(pred, gold, *, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    """True iff ``pred`` matches ``gold`` as a math answer.

    Each side is reduced to a small candidate set — the raw string and the
    :func:`extract_answer` result — and a match on any (pred, gold) pair counts.
    """
    if pred is None or gold is None:
        return False

    def candidates(x: str) -> list[str]:
        out, seen = [], set()
        for c in (x, extract_answer(x)):
            if c is None:
                continue
            n = normalize_answer(c)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    ps, gs = candidates(str(pred)), candidates(str(gold))
    return any(_atoms_equal(p, g, rel_tol, abs_tol) for p in ps for g in gs)
