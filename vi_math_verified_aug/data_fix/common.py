"""Shared helpers for the W3 data-fix builders (IO, datacards, LaTeX and number utilities)."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CLEAN = DATA / "clean"
SOURCES = CLEAN / "_sources"
ANN = ROOT / "docs" / "research_2026-11" / "content_review" / "annotations"

# Inputs that must never be written to (guarded by write_jsonl / write_json).
READ_ONLY_DIRS = [DATA / "test", DATA / "translation", DATA / "finetune", DATA / "generate"]

# Control characters that are never legitimate in a translated row
# (\t and \n are allowed; \t only where the English source has it, see bad_ctrl()).
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0d\x0e-\x1f\x7f]")


# --------------------------------------------------------------------------- IO
def read_jsonl(p: Path | str) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def read_json(p: Path | str) -> Any:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _guard(p: Path) -> Path:
    p = Path(p).resolve()
    for d in READ_ONLY_DIRS:
        if d.resolve() in p.parents:
            raise PermissionError(f"refusing to write inside read-only input dir: {p}")
    return p


def write_jsonl(p: Path | str, rows: Iterable[dict]) -> int:
    p = _guard(Path(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            s = json.dumps(r, ensure_ascii=False)
            assert json.loads(s) == r, "JSON round-trip failed"
            f.write(s + "\n")
            n += 1
    return n


def write_json(p: Path | str, obj: Any) -> None:
    p = _guard(Path(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def sha256_file(p: Path | str) -> str:
    """sha256 of a file; for a directory, sha256 over 'name:sha256' of its files (sorted, non-recursive)."""
    h = hashlib.sha256()
    if Path(p).is_dir():
        for q in sorted(Path(p).iterdir()):
            if q.is_file() and not q.name.endswith(".bak"):
                h.update(f"{q.name}:{sha256_file(q)}\n".encode())
        return h.hexdigest()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path | str) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def datacard(out_path: Path | str, *, name: str, version: str, description: str,
             sources: list[dict], transformations: list[str], counts: dict,
             changed: list[dict], license: str, script: str, extra: dict | None = None) -> dict:
    """Write ``<out>.datacard.json`` next to an output file and return it.

    ``sources`` items: {"path"|"url", "sha256"?, "rows"?, "role"}.
    ``changed`` items: {"id", "fields", "reason"} — every row that differs from its input.
    """
    out_path = Path(out_path)
    card = {
        "name": name,
        "version": version,
        "file": rel(out_path),
        "sha256": sha256_file(out_path) if out_path.exists() else None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": git_head(),
        "script": script,
        "description": description,
        "license": license,
        "sources": [{**s, **({"sha256": sha256_file(ROOT / s["path"])} if "path" in s and "sha256" not in s
                              and (ROOT / s["path"]).exists() else {})} for s in sources],
        "transformations": transformations,
        "counts": counts,
        "n_changed": len(changed),
        "changed_ids": changed,
    }
    if extra:
        card.update(extra)
    stem = out_path.name[: -len(out_path.suffix)] if out_path.suffix else out_path.name
    write_json(out_path.with_name(stem + ".datacard.json"), card)
    return card


# ------------------------------------------------------------------- LaTeX utils
def all_boxed(text: str) -> list[str]:
    """Contents of every ``\\boxed{..}`` / ``\\fbox{..}`` (brace-balanced). Unclosed ones are skipped."""
    out = []
    if not text:
        return out
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            c = text[i]
            if c == "\\":
                i += 2
                continue
            depth += (c == "{") - (c == "}")
            i += 1
        if depth == 0:
            out.append(text[m.end():i - 1])
    return out


def last_boxed(text: str) -> str | None:
    b = all_boxed(text)
    return b[-1] if b else None


def has_unclosed_boxed(text: str) -> bool:
    return len(re.findall(r"\\(?:boxed|fbox)\s*\{", text or "")) > len(all_boxed(text or ""))


def brace_balance(s: str) -> int:
    s = re.sub(r"\\[{}]", "", s or "")
    return s.count("{") - s.count("}")


def unescaped_dollars(s: str) -> int:
    return len(re.findall(r"(?<!\\)\$", s or ""))


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "")).strip()


def norm_boxed(s: str | None) -> str:
    """Light normalisation for comparing boxed answers across EN/VI (not a grader)."""
    if s is None:
        return ""
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\(?:text|mbox|textbf|mathrm)\{[^{}]*\}", "", s)
    s = re.sub(r"\^\\circ|°|\\!|\\,|\\;|\\ |\s+", "", s)
    s = s.replace("{,}", ",")
    return s


# ------------------------------------------------------------- escape damage repair
# LaTeX commands starting with t / r / f / b / v (the letters JSON turns into control chars).
_KNOWN_T = {"times", "text", "theta", "tan", "to", "tfrac", "triangle", "top", "tau", "tilde", "textbf", "textit",
            "textrm", "textup", "textsf", "texttt", "textstyle", "tbinom", "tag", "therefore", "thinspace", "tt",
            "tanh", "triangleq", "triangleleft", "triangleright", "textnormal"}
_KNOWN_R = {"right", "rightarrow", "rho", "rangle", "rbrace", "rbrack", "rceil", "rfloor", "rvert", "rVert", "rm",
            "raise", "rule", "rightharpoonup", "rightleftharpoons", "root", "rq"}
_KNOWN_F = {"frac", "forall", "floor", "fbox", "flat", "frown", "fill", "footnotesize", "fontsize"}
_KNOWN_B = {"begin", "binom", "bar", "beta", "boxed", "bf", "big", "bigg", "bmod", "bot", "backslash",
            "bullet", "bigcup", "bigcap", "bmatrix", "bigl", "bigr", "biggl", "biggr", "blacksquare", "breve",
            "boldsymbol", "bigstar", "because", "bowtie"}
_KNOWN_V = {"vec", "vee", "vert", "varphi", "varepsilon", "vartheta", "vdots", "vspace", "vdash", "vline",
            "varpi", "varrho", "varsigma"}
_CTRL = {"\r": ("r", _KNOWN_R), "\x0c": ("f", _KNOWN_F), "\x08": ("b", _KNOWN_B), "\x0b": ("v", _KNOWN_V),
         "\t": ("t", _KNOWN_T)}


def _is_cmd(word: str, known: set[str], en_text: str | None) -> bool:
    if word in known or (en_text is not None and re.search(re.escape("\\" + word) + r"(?![A-Za-z])", en_text)):
        return True
    return False


def repair_control_chars(text: str, en_text: str | None = None) -> tuple[str, list[str]]:
    """Map JSON-decoded control chars back to LaTeX escapes (``\\right`` became CR + ``ight``).

    A control char c followed by letters w is repaired to backslash + letter(c) + w. CR/FF/BS/VT are never
    legitimate in this corpus, so they are always repaired (except a CRLF pair -> LF). A TAB is only repaired
    when letter+w is a known LaTeX command (``\\times``, ``\\text``...) or a command present in the English
    source; asy indentation TABs (TAB + ``draw``) are kept.
    Returns (fixed_text, list of repairs applied).
    """
    if not text or not any(c in text for c in _CTRL):
        return text, []
    notes: list[str] = []
    out = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in _CTRL:
            letter, known = _CTRL[c]
            if c == "\r" and text[i + 1:i + 2] == "\n":
                out.append("\n")
                i += 2
                notes.append("CRLF->LF")
                continue
            word = letter + re.match(r"[A-Za-z]*", text[i + 1:]).group(0)
            if c != "\t" or _is_cmd(word, known, en_text):
                out.append("\\" + letter)
                notes.append({"\r": "CR", "\x0c": "FF", "\x08": "BS", "\x0b": "VT", "\t": "TAB"}[c] + "->\\" + letter)
                i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out), notes


_LIT_NL_RE = re.compile(r"\\n(?![a-zA-Z])")


def repair_literal_newlines(text: str, en_text: str | None = None) -> tuple[str, int]:
    """Replace a literal two-char ``\\n`` (not a TeX command such as ``\\neq``, ``\\nu``) by a newline.

    A ``\\n`` followed by a letter is ambiguous (``\\nThe`` vs ``\\neq``); those are only replaced when
    the letters do not form a known command stem and the English source has no such command.
    """
    if not text or "\\n" not in text:
        return text, 0
    n = 0

    def sub_safe(m):
        nonlocal n
        n += 1
        return "\n"

    t = _LIT_NL_RE.sub(sub_safe, text)

    # \n followed by a letter: replace when the following word is not a TeX command stem.
    def sub_word(m):
        nonlocal n
        word = m.group(1)
        cmd = "\\n" + word
        known = re.match(r"(eq|eqq|u|ot|otin|abla|eg|ewline|ormalsize|i|ewcommand|ewpage|exists|mid|parallel|"
                         r"leq|geq|subseteq|rightarrow|Rightarrow|leftarrow|Leftarrow|cong|sim|vdash|prec|succ)$",
                         word)
        if known or (en_text and cmd in en_text):
            return m.group(0)
        n += 1
        return "\n" + word

    t = re.sub(r"(?<!\\)\\n([A-Za-zÀ-ỹ]+)", sub_word, t)
    return t, n


def bad_ctrl(text: str, en_text: str | None = None) -> bool:
    if CTRL_RE.search(text or ""):
        return True
    if "\t" in (text or "") and en_text is not None and "\t" not in en_text:
        return bool(re.search(r"\t[A-Za-z]", text))
    return False


# ------------------------------------------------------------------ number utils
EN_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000, "million": 1000000, "dozen": 12, "twice": 2, "double": 2, "triple": 3, "half": 0.5,
    "quarter": 0.25, "once": 1, "thrice": 3, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "pair": 2, "couple": 2, "score": 20,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12, "noon": 12, "midnight": 12,
    "weekly": 7, "week": 7, "daily": 1, "hourly": 1, "monthly": 1, "annually": 1, "yearly": 1,
}
VI_WORDS = {
    "không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7,
    "tám": 8, "chín": 9, "mười": 10, "mươi": 10, "trăm": 100, "nghìn": 1000, "ngàn": 1000, "triệu": 1000000,
    "tá": 12, "chục": 10, "đôi": 2, "nửa": 0.5, "rưỡi": 0.5, "gấp đôi": 2, "gấp ba": 3, "phần tư": 0.25,
    "tuần": 7, "cặp": 2, "trưa": 12, "nhất": 1, "thứ hai": 2, "thứ ba": 3, "thứ tư": 4, "thứ năm": 5,
}
_EN_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w])|(?<![\w\d])\.(\d+)")
_VI_NUM = re.compile(r"(?<![\w])(\d+(?:[.,\s]\d{3})*(?:[.,]\d+)?|\d+)(?![\w])")


def _f(x: float) -> float:
    return round(float(x), 6)


def en_numbers(text: str) -> list[float]:
    """Digit numbers in English prose: ``1,234`` thousands, ``3.5`` decimal, ``$.35``."""
    out = []
    t = re.sub(r"<<[^<>]*>>", " ", text or "")
    for m in re.finditer(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\d])|(?<![\d])\.(\d+)", t):
        if m.group(3):
            out.append(_f("0." + m.group(3)))
        else:
            out.append(_f(m.group(1).replace(",", "") + (m.group(2) or "")))
    return out


def vi_number_candidates(text: str) -> list[set[float]]:
    """Digit numbers in Vietnamese prose, each with every plausible reading.

    ``1.000`` -> {1000, 1.0}; ``3,5`` -> {3.5}; ``1,000`` -> {1000, 1.0}; ``2.5`` -> {2.5}.
    """
    out = []
    t = re.sub(r"<<[^<>]*>>", " ", text or "")
    for m in re.finditer(r"(?<![\d.,])\d+(?:[.,]\d+)*(?![\d])|(?<![\d])\.(\d+)", t):
        s = m.group(0)
        c: set[float] = set()
        if s.startswith("."):
            out.append({_f("0" + s)})
            continue
        parts = re.split(r"[.,]", s)
        if len(parts) == 1:
            c.add(_f(s))
        else:
            if all(len(p) == 3 for p in parts[1:]):
                c.add(_f("".join(parts)))  # thousands grouping either separator
            if len(parts) == 2:
                c.add(_f(parts[0] + "." + parts[1]))  # decimal either separator
            if len(parts) > 2 and not all(len(p) == 3 for p in parts[1:]):
                # e.g. 1.234,5 / 1,234.5
                c.add(_f("".join(parts[:-1]) + "." + parts[-1]))
        out.append(c)
    return out


def word_values(text: str, table: dict) -> list[float]:
    t = (text or "").lower()
    vals = []
    for w, v in table.items():
        vals += [v] * len(re.findall(r"(?<!\w)" + re.escape(w) + r"(?!\w)", t))
    return [_f(v) for v in vals]


def number_multiset_diff(en: str, vi: str) -> dict:
    """Directional digit-number check EN<->VI (R1 §5 style).

    missing: EN digit numbers with no VI reading (digits or VI number words) left to match;
    added:   VI digit numbers with no EN value (digits or EN number words).
    Only digit numbers are required to match; spelled numbers serve as cover.
    """
    en_d = Counter(en_numbers(en))
    vi_c = vi_number_candidates(vi)
    vi_w = Counter(word_values(vi, VI_WORDS))
    en_w = Counter(word_values(en, EN_WORDS))
    # match EN digits against VI digit candidates first
    unmatched_vi = list(range(len(vi_c)))
    missing = []
    for val, k in sorted(en_d.items()):
        for _ in range(k):
            hit = next((j for j in unmatched_vi if val in vi_c[j]), None)
            if hit is not None:
                unmatched_vi.remove(hit)
            elif vi_w.get(val, 0) > 0:
                vi_w[val] -= 1
            else:
                missing.append(val)
    added = []
    en_w_left = Counter(en_w)
    for j in unmatched_vi:
        cand = vi_c[j]
        w = next((v for v in cand if en_w_left.get(v, 0) > 0), None)
        if w is not None:
            en_w_left[w] -= 1
        else:
            added.append(sorted(cand)[0])
    # set-level view: a value that never appears on the other side (ignores repetition differences)
    vi_all = set().union(*vi_c) if vi_c else set()
    vi_all |= set(word_values(vi, VI_WORDS))
    en_all = set(en_d) | set(en_w)
    missing_set = sorted({v for v in missing if v not in vi_all})
    added_set = sorted({v for v in added if v not in en_all})
    return {"missing_in_vi": missing, "added_in_vi": added, "missing_set": missing_set, "added_set": added_set}


def gsm8k_final(ans: str) -> str | None:
    m = re.search(r"####\s*(.+?)\s*$", ans or "")
    return m.group(1).strip() if m else None


def canon_num(s: str) -> str:
    """GSM8K gold canonical string: strip thousands commas / spaces, drop trailing .0"""
    s = (s or "").strip().replace(",", "").replace(" ", "")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return repr(f)
    except ValueError:
        return s


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def load_annotations(tag: str) -> dict[int, dict]:
    p = ANN / f"{tag}.jsonl"
    out = {}
    if p.exists():
        for r in read_jsonl(p):
            out[(r.get("subject"), r["index"]) if "subject" in r else r["index"]] = r
    return out


def bootstrap_path() -> None:
    """Allow ``python -I path/to/script.py`` (isolated mode drops the script dir and cwd)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
