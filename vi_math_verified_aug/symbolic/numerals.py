"""Vietnamese numeral normalizer and formatter (T1 §2.1 acceptance rule (ii), §2.2 renderer).

Conventions handled:
- digits with the Vietnamese "." thousands separator and "," decimal mark ("1.000", "2,5", "1.234,5"),
  and the English convention kept by translated GSM8K ("1,200", "0.10", "$1.50"); an ambiguous literal
  such as "1.500" returns BOTH readings, and callers check membership;
- fractions "2/5", "3 phần 4", percent "20%", "20 phần trăm";
- cardinal number words: "hai", "mười lăm", "hai mươi mốt", "một trăm linh năm", "ba nghìn", "1,5 triệu";
- fraction and multiplier words: "nửa", "một nửa", "rưỡi" ("hai giờ rưỡi" 2.5, "trăm rưỡi" 150),
  "một phần ba", "ba phần tư", "gấp đôi", "gấp 3 lần", "tá" (12), "chục" (10), "nửa tá".

Public API
    parse_literal(s)        -> set[Fraction]   every plausible value of a literal as written
    parse_number(s)         -> Fraction | None preferred (Vietnamese-convention) value
    int_to_words(n)         -> str             Vietnamese cardinal words (0 <= n < 10**12)
    fmt_vi(v, ...)          -> str             Vietnamese-style digits ("1.250", "2,5")
    fmt_like(v, original)   -> str             same surface style as the literal it replaces
    numbers_in_text(text)   -> list[NumMatch]  digit literals and number-word phrases with spans
    contains_value(text, v) -> bool            does any numeral in the text denote v (leak check)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction

__all__ = ["parse_literal", "parse_number", "int_to_words", "fraction_to_words", "fmt_vi", "fmt_like",
           "literal_style", "numbers_in_text", "contains_value", "value_strings", "NumMatch",
           "is_word_literal", "normalize_text"]

# --------------------------------------------------------------------------------------------- words
_DIGIT_WORDS = {"không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6, "bảy": 7,
                "tám": 8, "chín": 9}
# variants that only occur after "mươi"/"mười" (or as units in compounds)
_UNIT_VARIANTS = {"mốt": 1, "tư": 4, "lăm": 5, "nhăm": 5, "bảy": 7}
_SCALES = {"nghìn": 1000, "ngàn": 1000, "triệu": 10 ** 6, "tỷ": 10 ** 9, "tỉ": 10 ** 9}
_GROUP_WORDS = {"tá": 12, "chục": 10}
_HALF_WORDS = {"nửa", "phân nửa"}
_NUMBER_TOKENS = (set(_DIGIT_WORDS) | set(_UNIT_VARIANTS) | set(_SCALES) | {"mười", "mươi", "trăm", "linh", "lẻ",
                                                                           "rưỡi", "rưởi"})


def normalize_text(s: str) -> str:
    """NFC, lower-case, collapse whitespace, unify the narrow no-break space."""
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def _below_thousand(toks: list[str]) -> list[int]:
    """Parse tokens for a number < 1000. Returns all readings (colloquial "ba trăm năm" = 350 or 305)."""
    if not toks:
        return [0]
    n = len(toks)
    i = 0
    hundreds = None
    # hundreds
    if n >= 2 and toks[1] == "trăm" and toks[0] in _DIGIT_WORDS:
        hundreds = _DIGIT_WORDS[toks[0]]
        i = 2
    elif toks[0] == "trăm":
        hundreds = 1
        i = 1
    rest = toks[i:]
    base = (hundreds or 0) * 100
    if not rest:
        return [base] if hundreds is not None else []
    if rest[0] in ("linh", "lẻ"):
        if hundreds is None or len(rest) != 2 or rest[1] not in _DIGIT_WORDS and rest[1] not in _UNIT_VARIANTS:
            return []
        v = _DIGIT_WORDS.get(rest[1], _UNIT_VARIANTS.get(rest[1]))
        return [base + v]
    # tens
    if rest[0] == "mười":
        tens = 10
        rest = rest[1:]
        if not rest:
            return [base + 10]
        if len(rest) == 1 and rest[0] in ("lăm", "nhăm", "năm"):
            return [base + 15]
        if len(rest) == 1 and rest[0] in _DIGIT_WORDS and rest[0] != "không":
            return [base + tens + _DIGIT_WORDS[rest[0]]]
        if len(rest) == 1 and rest[0] in ("bảy", "tư"):
            return [base + tens + _UNIT_VARIANTS[rest[0]]]
        return []
    if len(rest) >= 2 and rest[1] == "mươi" and rest[0] in _DIGIT_WORDS and rest[0] not in ("không", "một"):
        tens = _DIGIT_WORDS[rest[0]] * 10
        rest = rest[2:]
        if not rest:
            return [base + tens]
        if len(rest) == 1:
            t = rest[0]
            if t in ("mốt",):
                return [base + tens + 1]
            if t in ("lăm", "nhăm", "năm"):
                return [base + tens + 5]
            if t in ("tư", "bốn"):
                return [base + tens + 4]
            if t in _DIGIT_WORDS and t != "không":
                return [base + tens + _DIGIT_WORDS[t]]
        return []
    if len(rest) == 1 and rest[0] in _DIGIT_WORDS:
        d = _DIGIT_WORDS[rest[0]]
        if hundreds is not None:
            # colloquial "ba trăm năm" = 350; strict reading 305 needs "linh", keep both (ambiguous)
            return sorted({base + d * 10, base + d}) if d else [base]
        return [d]
    return []


def _words_to_ints(s: str) -> list[int]:
    """Cardinal number words -> all readings (usually one). [] if not a number phrase."""
    toks = s.split()
    if not toks or any(t not in _NUMBER_TOKENS for t in toks):
        return []
    # "rưỡi" after a scale word: "trăm rưỡi" 150, "nghìn rưỡi" 1500, "triệu rưỡi" 1.5e6
    half_bonus = 0
    if toks[-1] in ("rưỡi", "rưởi"):
        if len(toks) < 2:
            return []
        prev = toks[-2]
        if prev == "trăm":
            half_bonus = 50
        elif prev in _SCALES:
            half_bonus = _SCALES[prev] // 2
        else:
            return []
        toks = toks[:-1]
    # split on scale words (largest first by position)
    readings = [0]
    cur: list[str] = []
    for t in toks:
        if t in _SCALES:
            mult = _SCALES[t]
            part = _below_thousand(cur) if cur else [1]
            if not part:
                return []
            readings = [r + p * mult for r in readings for p in part]
            cur = []
        else:
            cur.append(t)
    if cur:
        part = _below_thousand(cur)
        if not part:
            return []
        readings = [r + p for r in readings for p in part]
    return sorted({r + half_bonus for r in readings})


def int_to_words(n: int) -> str:
    """Vietnamese cardinal words for 0 <= n < 10**12 (standard written forms: mốt, tư, lăm, linh)."""
    n = int(n)
    if n < 0:
        return "âm " + int_to_words(-n)
    if n == 0:
        return "không"
    units = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

    def under_1000(x: int, full: bool) -> list[str]:
        h, r = divmod(x, 100)
        t, u = divmod(r, 10)
        out: list[str] = []
        if h or full:
            out += [units[h], "trăm"]
        if t == 0:
            if u:
                if h or full:
                    out.append("linh")
                out.append(units[u])
        elif t == 1:
            out.append("mười")
            if u == 5:
                out.append("lăm")
            elif u:
                out.append(units[u])
        else:
            out += [units[t], "mươi"]
            if u == 1:
                out.append("mốt")
            elif u == 4:
                out.append("tư")
            elif u == 5:
                out.append("lăm")
            elif u:
                out.append(units[u])
        return out

    parts: list[str] = []
    scales = [(10 ** 9, "tỷ"), (10 ** 6, "triệu"), (1000, "nghìn"), (1, "")]
    started = False
    for size, name in scales:
        chunk, n = divmod(n, size)
        if chunk:
            parts += under_1000(chunk, full=started)
            if name:
                parts.append(name)
            started = True
    return " ".join(parts)


def fraction_to_words(v: Fraction) -> str:
    """1/2 -> 'một nửa', 3/4 -> 'ba phần tư', 2 -> 'hai'."""
    v = Fraction(v)
    if v.denominator == 1:
        return int_to_words(v.numerator)
    if v == Fraction(1, 2):
        return "một nửa"
    den = "tư" if v.denominator == 4 else int_to_words(v.denominator)
    return f"{int_to_words(v.numerator)} phần {den}"


# --------------------------------------------------------------------------------------------- digits
_DIGITS_RE = re.compile(r"^[+-]?\d+(?:[.,\s]\d+)*$")


def _digit_values(s: str) -> set[Fraction]:
    s = s.strip().replace("\\,", "").replace("{,}", ",").replace("{.}", ".")
    out: set[Fraction] = set()
    m = re.fullmatch(r"([+-]?\d+)\s*/\s*(\d+)", s)
    if m:
        if int(m.group(2)) != 0:
            out.add(Fraction(int(m.group(1)), int(m.group(2))))
        return out
    if not _DIGITS_RE.match(s):
        return out
    neg = s.startswith("-")
    s = s.lstrip("+-")
    if " " in s:  # "1 000 000" grouping by spaces
        groups = s.split(" ")
        if all(len(g) == 3 for g in groups[1:]) and all(g.isdigit() for g in groups):
            s = "".join(groups)
        else:
            return out
    seps = re.findall(r"[.,]", s)
    groups = re.split(r"[.,]", s)
    if not seps:
        out.add(Fraction(int(s)))
    else:
        # reading 1: every separator groups thousands (one kind of separator, 3-digit groups)
        if (len(set(seps)) == 1 and all(len(g) == 3 for g in groups[1:]) and groups[0] not in ("", "0")
                and not groups[0].startswith("0")):
            out.add(Fraction(int("".join(groups))))
        # reading 2: last separator is the decimal mark, earlier ones (all the other kind) group
        last = seps[-1]
        earlier = seps[:-1]
        if all(e != last for e in earlier) and len(set(earlier)) <= 1 and all(len(g) == 3 for g in groups[1:-1]):
            ip = "".join(groups[:-1]) or "0"
            out.add(Fraction(f"{ip}.{groups[-1]}"))
    return {-v for v in out} if neg else out


_MAGNITUDE = {"nghìn": 1000, "ngàn": 1000, "triệu": 10 ** 6, "tỷ": 10 ** 9, "tỉ": 10 ** 9, "trăm": 100,
              "chục": 10, "tá": 12}


def parse_literal(s: str | None) -> set[Fraction]:
    """All plausible values of a literal (span text) as written. Empty set = not a numeral."""
    if s is None:
        return set()
    t = normalize_text(s)
    # LaTeX: \frac{1}{2}, \dfrac12, -\frac{3}{4}
    m = re.fullmatch(r"\$?\s*(-)?\s*\\[dt]?frac\s*(?:\{\s*(-?\d+)\s*\}|(\d))\s*(?:\{\s*(\d+)\s*\}|(\d))\s*\$?", t)
    if m:
        num, den = int(m.group(2) or m.group(3)), int(m.group(4) or m.group(5))
        if den == 0:
            return set()
        v = Fraction(num, den)
        return {-v if m.group(1) else v}
    t = re.sub(r"^\$(?=\s*-?\d)|(?<=\d)\$$", "", t)
    t = re.sub(r"^(?:\$|\\\$|us\$)\s*", "", t)
    t = re.sub(r"\s*(?:\$|đ|vnđ|vnd|usd|đô la|đồng)$", "", t)
    pct = False
    if t.endswith("%") or t.endswith("\\%"):
        t = t.rstrip("%").rstrip("\\").strip()
        pct = True
    elif t.endswith(" phần trăm"):
        t = t[: -len(" phần trăm")].strip()
        pct = True
    if not t:
        return set()
    # multiplier phrases: "gấp đôi", "gấp 3 lần", "gấp rưỡi", "3 lần"
    m = re.fullmatch(r"(?:gấp\s+)?(.+?)\s+lần", t) or re.fullmatch(r"gấp\s+(.+)", t)
    if m:
        inner = m.group(1).strip()
        if inner == "đôi":
            return {Fraction(2)}
        if inner in ("rưỡi", "rưởi"):
            return {Fraction(3, 2)}
        return parse_literal(inner)
    if t == "đôi":
        return {Fraction(2)}
    vals = _digit_values(t)
    if vals:
        return vals
    # digits + magnitude word: "15 nghìn", "1,5 triệu", "2 tá"
    m = re.fullmatch(r"([\d.,]+)\s+(nghìn|ngàn|triệu|tỷ|tỉ|trăm|chục|tá)(?:\s+(rưỡi|rưởi))?", t)
    if m:
        base = _digit_values(m.group(1))
        mult = _MAGNITUDE[m.group(2)]
        extra = Fraction(mult, 2) if m.group(3) else 0
        return {b * mult + extra for b in base}
    # "2 rưỡi" (2.5)
    m = re.fullmatch(r"(\d+)\s+(?:rưỡi|rưởi)", t)
    if m:
        return {Fraction(int(m.group(1))) + Fraction(1, 2)}
    # fractions "3 phần 4", "một phần ba", "ba phần tư"
    m = re.fullmatch(r"(.+?)\s+phần\s+(.+)", t)
    if m:
        nums = _digit_values(m.group(1)) or {Fraction(x) for x in _words_to_ints(m.group(1))}
        den_s = m.group(2)
        dens = _digit_values(den_s) or {Fraction(x) for x in _words_to_ints(den_s)}
        if den_s == "tư":
            dens = {Fraction(4)}
        return {a / b for a in nums for b in dens if b}
    if t in _HALF_WORDS or t == "một nửa":
        return {Fraction(1, 2)}
    m = re.fullmatch(r"(?:(.+?)\s+)?(tá|chục)", t)  # "một tá", "nửa tá", "hai chục"
    if m:
        g = _GROUP_WORDS[m.group(2)]
        if m.group(1) is None:
            return {Fraction(g)}
        if m.group(1) in _HALF_WORDS:
            return {Fraction(g, 2)}
        return {Fraction(x * g) for x in _words_to_ints(m.group(1))}
    # "hai giờ rưỡi"-style is handled by callers; plain words (with optional trailing rưỡi on units)
    m = re.fullmatch(r"(.+?)\s+(?:rưỡi|rưởi)", t)
    if m and m.group(1).split()[-1] not in _SCALES and m.group(1).split()[-1] != "trăm":
        ints = _words_to_ints(m.group(1))
        if ints:
            return {Fraction(x) + Fraction(1, 2) for x in ints}
    ints = _words_to_ints(t)
    return {Fraction(x) for x in ints}


def parse_number(s: str | None) -> Fraction | None:
    """Single preferred value (Vietnamese convention first: '.' groups thousands, ',' is decimal)."""
    vals = parse_literal(s)
    if not vals:
        return None
    if len(vals) == 1:
        return next(iter(vals))
    # ambiguous only for a single 3-digit group ("1.500", "1,200"): both conventions read it as grouping
    ints = [v for v in vals if v.denominator == 1]
    return max(ints) if ints else sorted(vals)[0]


def is_word_literal(s: str) -> bool:
    return bool(s) and not re.search(r"\d", s)


# ---------------------------------------------------------------------------------------- formatting
def fmt_vi(v, dec: str = ",", group: str | None = ".", max_dp: int = 2, group_min: int = 5) -> str:
    """Format v with Vietnamese conventions. Non-terminating / long decimals fall back to 'p/q'.
    Grouping is applied only when the integer part has >= group_min digits (school texts write 1500)."""
    v = Fraction(v)
    neg = v < 0
    v = abs(v)
    frac = ""
    if v.denominator == 1:
        ip = str(v.numerator)
    else:
        for dp in range(1, max_dp + 1):
            if (v * 10 ** dp).denominator == 1:
                n = (v * 10 ** dp).numerator
                s_all = str(n).rjust(dp + 1, "0")
                ip, frac = s_all[:-dp], s_all[-dp:]
                break
        else:
            return ("-" if neg else "") + f"{v.numerator}/{v.denominator}"
    if group and len(ip) >= group_min:
        parts = []
        while len(ip) > 3:
            parts.insert(0, ip[-3:])
            ip = ip[:-3]
        parts.insert(0, ip)
        ip = group.join(parts)
    return ("-" if neg else "") + ip + ((dec + frac) if frac else "")


def literal_style(lit: str) -> dict:
    """Surface style of a literal: kind ('digits'|'words'|'fraction'|'multiplier'), dec/group marks, dp, pct."""
    t = normalize_text(lit)
    st = {"kind": "digits", "dec": ",", "group": None, "dp": 0, "pct": t.endswith("%"),
          "prefix": "", "suffix": "", "group_min": 4}
    if t.startswith("gấp"):
        st["kind"] = "multiplier"
        st["suffix"] = " lần" if t.endswith(" lần") else ""
        st["digits_inside"] = bool(re.search(r"\d", t))
        return st
    if not re.search(r"\d", t):
        st["kind"] = "words"
        st["fraction"] = (" phần " in f" {t} ") or t in _HALF_WORDS or t == "một nửa"
        return st
    if re.fullmatch(r"\$?\s*\d+\s*/\s*\d+", t):
        st["kind"] = "fraction"
        return st
    m = re.match(r"^(\$\s*)", lit.strip())
    if m:
        st["prefix"] = m.group(1)
    seps = re.findall(r"\d([.,])\d", t)
    body = re.sub(r"[^\d.,]", "", t)
    groups = re.split(r"[.,]", body)
    if not seps:
        return st
    if len(seps) >= 2 and len(set(seps)) == 1:
        st["group"] = seps[0]
        st["dec"] = "," if seps[0] == "." else "."
    elif len(seps) == 1 and len(groups[-1]) == 3 and groups[0] not in ("0", ""):
        st["group"] = seps[0]
        st["dec"] = "," if seps[0] == "." else "."
    else:
        st["dec"] = seps[-1]
        st["dp"] = len(groups[-1])
        if len(set(seps)) == 2:
            st["group"] = seps[0]
    return st


def fmt_like(v, original: str) -> str:
    """Format v in the surface style of the literal it replaces.

    digits stay digits with the same decimal/grouping marks (and the same number of decimals for money
    like '0.10'); words stay words ('hai' -> 'ba', 'một nửa' -> 'một phần ba'); 'gấp đôi' -> 'gấp ba'."""
    v = Fraction(v)
    st = literal_style(original)
    if st["kind"] == "multiplier":
        if v == 2 and not st.get("digits_inside"):
            return "gấp đôi" + st["suffix"]
        inner = fmt_vi(v) if st.get("digits_inside") else (fraction_to_words(v) if v.denominator == 1 else fmt_vi(v))
        return f"gấp {inner}" + (st["suffix"] or (" lần" if st.get("digits_inside") else ""))
    if st["kind"] == "words":
        if v.denominator == 1 and v >= 0:
            w = int_to_words(v.numerator)
        elif v > 0 and v.denominator <= 12 and v < 1:
            w = fraction_to_words(v)
        else:
            return fmt_vi(v)
        if original[:1].isupper():
            w = w[:1].upper() + w[1:]
        return w
    if st["kind"] == "fraction" or (v.denominator != 1 and (v * 100).denominator != 1):
        return f"{v.numerator}/{v.denominator}"
    s = fmt_vi(v, dec=st["dec"], group=st["group"], max_dp=max(2, st["dp"]), group_min=st["group_min"])
    if st["dp"] and v.denominator != 1 or (st["dp"] and v.denominator == 1 and st["dp"] == 2):
        # keep money-style fixed decimals: '1.50' -> '2.25', '0.10' -> '0.15'
        body = fmt_vi(v, dec=st["dec"], group=st["group"], max_dp=st["dp"], group_min=st["group_min"])
        ip, _, fp = body.partition(st["dec"])
        s = ip + st["dec"] + fp.ljust(st["dp"], "0")
    return st["prefix"] + s + ("%" if st["pct"] else "")


def value_strings(v) -> set[str]:
    """Digit surface forms of v that would reveal it in a text."""
    v = Fraction(v)
    out = set()
    for dec in (",", "."):
        for grp in (None, ".", ",", " "):
            if grp == dec:
                continue
            out.add(fmt_vi(v, dec=dec, group=grp, max_dp=6, group_min=4))
    if v.denominator != 1:
        out.add(f"{v.numerator}/{v.denominator}")
    return out


# -------------------------------------------------------------------------------------- text scanning
@dataclass(frozen=True)
class NumMatch:
    start: int
    end: int
    literal: str
    values: frozenset
    kind: str  # "digits" | "words"


_DIGIT_IN_TEXT = re.compile(r"(?<![\w])\d+(?:[.,]\d+)*(?:\s*/\s*\d+)?(?:\s*%)?")
_WORD_TOKEN = r"(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|mốt|tư|lăm|nhăm|trăm|nghìn|ngàn|triệu|tỷ|tỉ|linh|lẻ|rưỡi)"
_WORD_PHRASE = re.compile(rf"(?<!\w)(?:{_WORD_TOKEN})(?:\s+{_WORD_TOKEN})*(?!\w)", re.IGNORECASE)
_FRACTION_WORDS = re.compile(rf"(?<!\w)(?:một nửa|nửa|(?:{_WORD_TOKEN}\s+)+phần\s+(?:{_WORD_TOKEN})(?:\s+{_WORD_TOKEN})*)(?!\w)",
                             re.IGNORECASE)


def numbers_in_text(text: str, words: bool = False) -> list[NumMatch]:
    """Digit literals in text (always) and, with words=True, number-word phrases.

    Word phrases are conservative: a lone "một" (article) and a lone "năm" (also 'year') are skipped."""
    out: list[NumMatch] = []
    for m in _DIGIT_IN_TEXT.finditer(text):
        lit = m.group(0).rstrip()
        # do not swallow a sentence-final period: "là 5." -> "5"
        while lit and lit[-1] in ".,":
            lit = lit[:-1]
        vals = parse_literal(lit)
        if vals:
            out.append(NumMatch(m.start(), m.start() + len(lit), lit, frozenset(vals), "digits"))
    if words:
        taken = [(x.start, x.end) for x in out]
        for rx in (_FRACTION_WORDS, _WORD_PHRASE):
            for m in rx.finditer(text):
                lit = m.group(0)
                low = normalize_text(lit)
                if low in ("một", "năm", "tư", "lẻ", "linh", "mươi", "trăm", "nghìn", "ngàn", "triệu", "ba", "bảy"):
                    # ambiguous alone ("một" article, "năm" year, "ba" father, "bảy" ...). Callers that need them
                    # anchor them through given spans instead.
                    if low not in ("ba", "bảy"):
                        continue
                if any(s < m.end() and m.start() < e for s, e in taken):
                    continue
                vals = parse_literal(lit)
                if vals:
                    out.append(NumMatch(m.start(), m.end(), lit, frozenset(vals), "words"))
                    taken.append((m.start(), m.end()))
    out.sort(key=lambda x: x.start)
    return out


def contains_value(text: str, v, words: bool = True) -> bool:
    """True if any numeral in text (digits, and number words if words=True) can denote v."""
    v = Fraction(v)
    return any(v in m.values for m in numbers_in_text(text, words=words))
