"""Shared helpers for the X3 eval-set builders: I/O, hashing, deterministic text repairs, parity checks."""
from __future__ import annotations

import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "eval" / "_sources"
WORK = ROOT / "data" / "eval" / "_work"
OUT = ROOT / "data" / "eval"
SEED = 20261107
BUILD_DATE = "2026-11-07"

VI_GSM8K_TEST = ROOT / "data" / "test" / "gsm8k" / "gsm8k_test_vietnamese.jsonl"
VI_MATH_TEST = {s: ROOT / "data" / "test" / "math" / f"{s}_vi.jsonl" for s in ("algebra", "prealgebra")}
ANNOT = ROOT / "docs" / "research_2026-11" / "content_review" / "annotations"


# ----------------------------------------------------------------------------------------------------------- I/O
def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(p: Path, rows: Iterable[dict]) -> str:
    """Write rows (UTF-8, LF, one JSON object per line) and return the file's sha256."""
    p.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    p.write_text(text, encoding="utf-8", newline="\n")
    return sha256_bytes(text.encode("utf-8"))


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def ws(s: str) -> str:
    """Whitespace-normalised form used only for text matching."""
    return re.sub(r"\s+", " ", nfc(s)).strip()


# ------------------------------------------------------------------------------------------ boxed answer extraction
def all_boxed(text: str) -> list[str]:
    """Contents of every ``\\boxed{..}`` / ``\\fbox{..}`` (brace-balanced), in order."""
    out = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth, j = 1, m.end()
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            out.append(text[m.end():j - 1])
    return out


def last_boxed(text: str) -> Optional[str]:
    b = all_boxed(text)
    return b[-1] if b else None


# ----------------------------------------------------------------------------- deterministic transport repairs (MATH)
# JSON-escape damage: "\r" inside "\right" became a carriage return, "\t" inside "\times"/"\text" a TAB, etc.
_CTRL2LETTER = {"\r": "r", "\t": "t", "\x0c": "f", "\x08": "b", "\x0b": "v", "\x07": "a"}
# literal two-character "\n" that is not the start of a real LaTeX command
_REAL_N_CMDS = {
    "neq", "ne", "ni", "nu", "not", "nabla", "newline", "noindent", "nmid", "nleq", "ngeq", "nless", "ngtr",
    "nsubseteq", "nsupseteq", "nsubset", "natural", "neg", "nolimits", "normalsize", "notin", "nonumber",
    "nexists", "ncong", "nsim", "nparallel", "nearrow", "nwarrow", "nleftarrow", "nrightarrow", "nRightarrow",
    "nLeftarrow", "nLeftrightarrow", "nleftrightarrow", "ntriangleleft", "ntriangleright", "nprec", "nsucc",
    "newcommand", "nobreak", "nonstopmode", "numberwithin",
}


def repair_ctrl_escapes(vi: str, en_ref: str) -> tuple[str, list[str]]:
    """Restore ``\\right`` etc. where a control character replaced ``\\`` + letter.

    A control char ``c`` followed by a word ``w`` is restored to ``\\`` + letter(c) + ``w`` only if that LaTeX
    command occurs in the English reference text (so legitimate TABs, e.g. asy indentation, are kept).
    """
    log = []

    def sub(m: re.Match) -> str:
        c, word = m.group(1), m.group(2)
        cmd = "\\" + _CTRL2LETTER[c] + word
        if word and re.search(re.escape(cmd) + r"(?![A-Za-z])", en_ref):
            log.append(f"ctrl->{cmd}")
            return cmd
        return m.group(0)

    out = re.sub(r"([\r\t\x0c\x08\x0b\x07])([A-Za-z]*)", sub, vi)
    return out, log


def repair_literal_newlines(vi: str, en_ref: str = "") -> tuple[str, list[str]]:
    """Replace a literal two-character ``\\n`` (not a LaTeX command such as ``\\neq``) by a real newline.

    A ``\\n<letters>`` token is kept if it is a known LaTeX command or occurs verbatim in the English reference.
    """
    log = []

    def sub(m: re.Match) -> str:
        if m.group(0).startswith("\\\\"):  # "\\n" = LaTeX line break followed by n...: leave
            return m.group(0)
        word = m.group(1)
        if word and (("n" + word) in _REAL_N_CMDS
                     or re.search(re.escape("\\n" + word) + r"(?![A-Za-z])", en_ref)):
            return m.group(0)
        # "\nk" etc.: the letters after "\n" are ordinary text that followed the broken newline
        log.append("literal\\n->newline")
        return "\n" + word

    out = re.sub(r"\\\\n|\\n([A-Za-z]*)", sub, vi)
    return out, log


def deterministic_repair(vi: str, en_ref: str) -> tuple[str, list[str]]:
    a, l1 = repair_ctrl_escapes(vi, en_ref)
    b, l2 = repair_literal_newlines(a, en_ref)
    return b, l1 + l2


# ---------------------------------------------------------------------------------------------- parity checks
_TEXTY = {"text", "textbf", "textit", "mbox", "mathrm", "rm", "emph", "textrm", "it", "bf", "operatorname",
          "hbox", "textnormal", "textsf", "texttt", "mathbf", "mathit"}
_ALIAS = {"dfrac": "frac", "tfrac": "frac", "le": "leq", "ge": "geq", "ne": "neq", "lt": "<", "gt": ">"}


def latex_cmds(t: str) -> collections.Counter:
    """Multiset of LaTeX commands, ignoring text-mode wrappers and spacing commands."""
    c = collections.Counter()
    for m in re.finditer(r"\\([A-Za-z]+)", t):
        w = m.group(1)
        if w in _TEXTY:
            continue
        c[_ALIAS.get(w, w)] += 1
    return c


def _norm_numbers(t: str) -> str:
    t = nfc(t)
    t = t.replace("{,}", ",").replace("{.}", ".")
    t = re.sub(r"\\!", "", t)
    t = re.sub(r"(\d)\s*\\[,;: ]\s*(?=\d{3}(?!\d))", r"\1", t)  # 54\,321 thin-space grouping
    t = re.sub(r"(\d)[\u00a0\u202f](?=\d{3}(?!\d))", r"\1", t)  # non-breaking-space grouping
    return t


def digit_tokens(t: str) -> collections.Counter:
    """Locale-agnostic multiset of numerals: separators ``.`` and ``,`` between digits are removed.

    "2,125" (EN) and "2.125" (VI) -> "2125"; "2.5" (EN) and "2,5" (VI) -> "25".  Leading zeros are dropped.
    """
    t = _norm_numbers(t)
    c = collections.Counter()
    for m in re.finditer(r"\d+(?:[.,]\d+)*", t):
        d = re.sub(r"[.,]", "", m.group(0)).lstrip("0") or "0"
        c[d] += 1
    return c


def counter_diff(a: collections.Counter, b: collections.Counter) -> dict:
    """{'missing': a-b, 'extra': b-a} as plain dicts (empty dicts when equal)."""
    return {"missing": dict(a - b), "extra": dict(b - a)}


_EN_NUMBER_WORDS = {
    "zero": "0", "nonzero": "0", "non-zero": "0", "one": "1", "single": "1", "two": "2", "twice": "2", "double": "2",
    "both": "2", "pair": "2", "three": "3", "thrice": "3", "triple": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "dozen": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000", "million": "1000000", "half": "2",
    "halves": "2", "third": "3", "quarter": "4 25", "quarters": "4 25", "fourth": "4", "fifth": "5", "tenth": "10",
    "percent": "100", "zeroes": "0", "zeros": "0", "dozens": "12",
    # coin names and month names encode a number that a translation may write out
    "penny": "1", "pennies": "1", "nickel": "5", "nickels": "5", "dime": "10", "dimes": "10",
    "january": "1", "february": "2", "march": "3", "april": "4", "june": "6", "july": "7",
    "august": "8", "september": "9", "october": "10", "november": "11", "december": "12",
}


def en_number_words(en: str) -> collections.Counter:
    """Numerals that an English number/coin/month *word* may legitimately become in a translation."""
    c = collections.Counter()
    for w in re.findall(r"[A-Za-z]+(?:-zero)?", en.lower()):
        for v in _EN_NUMBER_WORDS.get(w, "").split():
            c[v] += 1
    return c


def parity(en: str, vi: str, allow_en_number_words: bool = False) -> dict:
    """Numeral- and LaTeX-multiset parity of a Vietnamese text against its English source.

    With ``allow_en_number_words`` a Vietnamese numeral may stand for an English number *word* ("nonzero" ->
    "khác 0", "twice" -> "2 lần"): extra Vietnamese numerals are tolerated up to the count of such English words.
    Missing English numerals are never tolerated.
    """
    nd = counter_diff(digit_tokens(en), digit_tokens(vi))
    if allow_en_number_words and nd["extra"]:
        words = en_number_words(en)
        nd["extra_explained_by_en_words"] = {k: v for k, v in nd["extra"].items() if v <= words.get(k, 0)}
        nd["extra"] = {k: v for k, v in nd["extra"].items() if v > words.get(k, 0)}
    ld = counter_diff(latex_cmds(en), latex_cmds(vi))
    return {
        "numbers_ok": not nd["missing"] and not nd["extra"],
        "latex_ok": not ld["missing"] and not ld["extra"],
        "numbers": nd,
        "latex": ld,
    }


# --------------------------------------------------------------------------------------------- misc text probes
_VI_DIACRITIC = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)


def has_vi(t: str) -> bool:
    return bool(_VI_DIACRITIC.search(t))


def braces_balanced(t: str) -> bool:
    t = t.replace("\\{", "").replace("\\}", "")
    d = 0
    for ch in t:
        d += {"{": 1, "}": -1}.get(ch, 0)
        if d < 0:
            return False
    return d == 0


def dollars_even(t: str) -> bool:
    return t.replace("\\$", "").count("$") % 2 == 0


def asy_blocks(t: str) -> list[str]:
    return re.findall(r"\[asy\].*?\[/asy\]", t, flags=re.S)


def strip_asy_strings(block: str) -> str:
    """asy code with string literals blanked (labels may be legitimately translated)."""
    return re.sub(r'"[^"\n]*"', '""', block)
