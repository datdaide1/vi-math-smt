# Vendored (frozen) copy of vi_math_verified_aug/symbolic/numerals.py (uncommitted W5 code), 2026-11-07, for the G1 pilot.
"""Vietnamese numeral normalizer / formatter used by arm A (span checks, tag checks, rendering)
and by the shared leak gate.

parse_literal(s) -> set[Fraction]   every plausible value of a literal as written in a VN/translated
                                    text ("48", "1.000", "2,5", "0.2", "3/4", "một nửa", "hai mươi", "tá")
fmt_like(v, original_literal)       format a new value in the same surface style as the literal it replaces
value_strings(v)                    all surface forms of v (for "answer appears in text" leak checks)
"""
from __future__ import annotations

import re
from fractions import Fraction

_UNITS = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5,
          "sáu": 6, "bảy": 7, "tám": 8, "chín": 9}
_SPECIAL = {
    "nửa": Fraction(1, 2), "một nửa": Fraction(1, 2), "rưỡi": Fraction(1, 2), "phân nửa": Fraction(1, 2),
    "đôi": Fraction(2), "gấp đôi": Fraction(2), "gấp ba": Fraction(3), "gấp bốn": Fraction(4),
    "gấp rưỡi": Fraction(3, 2), "tá": Fraction(12), "một tá": Fraction(12), "nửa tá": Fraction(6),
    "chục": Fraction(10), "một chục": Fraction(10), "trăm": Fraction(100), "nghìn": Fraction(1000),
    "ngàn": Fraction(1000), "triệu": Fraction(10 ** 6), "một phần tư": Fraction(1, 4),
    "một phần ba": Fraction(1, 3), "hai phần ba": Fraction(2, 3), "ba phần tư": Fraction(3, 4),
    "một phần năm": Fraction(1, 5), "một phần mười": Fraction(1, 10), "một phần hai": Fraction(1, 2),
    "một phần sáu": Fraction(1, 6), "một phần tám": Fraction(1, 8), "hai phần năm": Fraction(2, 5),
    "ba phần năm": Fraction(3, 5), "bốn phần năm": Fraction(4, 5), "một phần bảy": Fraction(1, 7),
    "một phần chín": Fraction(1, 9), "một phần mười hai": Fraction(1, 12),
}
_ORD_PREFIX = ("gấp ", "bằng ")


def _words_to_int(s: str):
    """Vietnamese cardinal words (up to the thousands) -> int, or None."""
    toks = s.split()
    if not toks:
        return None
    total, cur = 0, 0
    i = 0
    seen = False
    while i < len(toks):
        t = toks[i]
        if t in _UNITS:
            cur += _UNITS[t]
            seen = True
        elif t == "mười":
            cur += 10
            seen = True
        elif t in ("mươi",):
            cur = (cur if cur else 1) * 10
        elif t == "trăm":
            cur = (cur if cur else 1) * 100
            seen = True
        elif t in ("nghìn", "ngàn"):
            total += (cur if cur else 1) * 1000
            cur = 0
            seen = True
        elif t == "triệu":
            total += (cur if cur else 1) * 10 ** 6
            cur = 0
            seen = True
        elif t in ("linh", "lẻ"):
            pass
        else:
            return None
        i += 1
    return total + cur if seen else None


_DIG = re.compile(r"^[+-]?\d+(?:[.,\s]\d+)*$")


def _digit_values(s: str) -> set:
    s = s.strip().replace(" ", " ").replace("\\,", "")
    s = s.replace("{,}", ",")
    out = set()
    if re.fullmatch(r"[+-]?\d+\s*/\s*\d+", s):
        a, b = re.split(r"\s*/\s*", s)
        if int(b) != 0:
            out.add(Fraction(int(a), int(b)))
        return out
    if not _DIG.match(s):
        return out
    s = s.replace(" ", "")
    seps = re.findall(r"[.,]", s)
    if not seps:
        out.add(Fraction(int(s)))
        return out
    groups = re.split(r"[.,]", s)
    # interpretation 1: all separators are grouping (needs 3-digit groups after the first)
    if all(len(g) == 3 for g in groups[1:]) and len(set(seps)) == 1:
        out.add(Fraction(int("".join(groups))))
    # interpretation 2: last separator is the decimal mark, the others grouping
    if len(set(seps[:-1])) <= 1 and (not seps[:-1] or seps[:-1][0] != seps[-1]):
        intpart = "".join(groups[:-1])
        if all(len(g) == 3 for g in groups[1:-1]):
            out.add(Fraction(intpart + "." + groups[-1]) if intpart else Fraction("0." + groups[-1]))
    return out


def parse_literal(s: str) -> set:
    """All plausible values of a literal span as written."""
    if s is None:
        return set()
    raw = s.strip()
    t = raw.lower().strip()
    t = re.sub(r"^\$|\$$|%$|\s*phần trăm$", "", t).strip()
    t = t.replace("\\$", "").strip()
    vals = _digit_values(t)
    if vals:
        return vals
    # "2 rưỡi" / "3 phần 4"
    m = re.fullmatch(r"(\d+)\s*rưỡi", t)
    if m:
        return {Fraction(int(m.group(1))) + Fraction(1, 2)}
    m = re.fullmatch(r"(\d+)\s*phần\s*(\d+)", t)
    if m and int(m.group(2)):
        return {Fraction(int(m.group(1)), int(m.group(2)))}
    for p in _ORD_PREFIX:
        if t.startswith(p) and t not in _SPECIAL:
            t2 = t[len(p):]
            r = parse_literal(t2)
            if r:
                return r
    t = re.sub(r"\s+lần$", "", t)
    if t in _SPECIAL:
        return {_SPECIAL[t]}
    m = re.fullmatch(r"(.+?)\s+rưỡi", t)
    if m:
        w = _words_to_int(m.group(1))
        if w is not None:
            return {Fraction(w) + Fraction(1, 2)}
    m = re.fullmatch(r"(.+?)\s+phần\s+(.+)", t)
    if m:
        a, b = _words_to_int(m.group(1)), _words_to_int(m.group(2))
        if a is not None and b:
            return {Fraction(a, b)}
    w = _words_to_int(t)
    if w is not None:
        return {Fraction(w)}
    return set()


def is_digit_literal(s: str) -> bool:
    return bool(re.search(r"\d", s or ""))


def literal_style(lit: str) -> dict:
    """Surface style of a digit literal: decimal mark and grouping mark."""
    lit = (lit or "").strip()
    st = {"dec": ",", "group": None}
    seps = re.findall(r"\d([.,])\d", lit)
    if not seps:
        return st
    vals = _digit_values(re.sub(r"[^\d.,/]", "", lit))
    groups = re.split(r"[.,]", re.sub(r"[^\d.,]", "", lit))
    if len(seps) >= 2 and len(set(seps)) == 1:
        st["group"] = seps[0]
        st["dec"] = "," if seps[0] == "." else "."
    elif len(seps) == 1 and len(groups[-1]) == 3 and len(groups[0]) <= 3 and groups[0] != "0":
        # ambiguous single 3-digit group: treat as grouping (translated GSM8K mostly writes 1,200 / 1.200)
        st["group"] = seps[0]
        st["dec"] = "," if seps[0] == "." else "."
    else:
        st["dec"] = seps[-1]
        if len(set(seps)) == 2:
            st["group"] = seps[0]
    return st


def fmt_number(v: Fraction, dec: str = ",", group: str | None = None, max_dp: int = 2) -> str:
    v = Fraction(v)
    neg = v < 0
    v = abs(v)
    if v.denominator == 1:
        s = str(v.numerator)
        frac = ""
    else:
        # finite decimal with <= max_dp places, else a/b
        for dp in range(1, max_dp + 1):
            if (v * 10 ** dp).denominator == 1:
                n = v * 10 ** dp
                s_all = str(n.numerator).rjust(dp + 1, "0")
                s, frac = s_all[:-dp], s_all[-dp:]
                break
        else:
            return ("-" if neg else "") + f"{v.numerator}/{v.denominator}"
    if group and len(s) > 3:
        parts = []
        while len(s) > 3:
            parts.insert(0, s[-3:])
            s = s[:-3]
        parts.insert(0, s)
        s = group.join(parts)
    return ("-" if neg else "") + s + ((dec + frac) if frac else "")


def fmt_like(v: Fraction, original: str) -> str:
    """Format v in the surface style of the literal it replaces (digits only; word literals get digits)."""
    if original and "/" in original and Fraction(v).denominator != 1:
        return f"{Fraction(v).numerator}/{Fraction(v).denominator}"
    st = literal_style(original)
    return fmt_number(v, st["dec"], st["group"])


def value_strings(v: Fraction) -> set:
    """Surface forms of v that would reveal it in a text."""
    v = Fraction(v)
    out = set()
    for dec in (",", "."):
        out.add(fmt_number(v, dec, None, max_dp=6))
        for grp in (".", ",", " "):
            if grp != dec:
                out.add(fmt_number(v, dec, grp, max_dp=6))
    if v.denominator != 1:
        out.add(f"{v.numerator}/{v.denominator}")
    return {s for s in out if s}


_NUM_IN_TEXT = re.compile(r"(?<![\w.,])\d+(?:[.,\s]\d{3})*(?:[.,]\d+)?(?:\s*/\s*\d+)?(?![\w])")


def numbers_in_text(text: str) -> list:
    """[(literal, set(values), start, end)] for digit literals in text."""
    out = []
    for m in re.finditer(r"\d+(?:[.,]\d+)*(?:\s*/\s*\d+)?", text):
        lit = m.group(0)
        out.append((lit, parse_literal(lit), m.start(), m.end()))
    return out


def text_contains_value(text: str, v: Fraction) -> bool:
    v = Fraction(v)
    for lit, vals, s, e in numbers_in_text(text):
        if v in vals:
            return True
    return False
