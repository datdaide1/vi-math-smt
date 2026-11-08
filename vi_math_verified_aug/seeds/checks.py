"""Deterministic checks shared by ``detect`` (on the existing translations) and ``mta`` (on re-translations).

G0 structure (T4 §3.5): control characters, brace balance, ``$`` parity (MATH), no new/translated LaTeX macros,
closed ``\\boxed{}``, English residue, length ratio. G1 content: prose number multiset (problem), the final
``#### N`` / last ``\\boxed{}`` vs the English gold, triggered glossary forbid-list (problem + solution), construction
rules CR-1/2/3/5 (problem), GSM8K calculator annotations present.

The number/LaTeX/IO utilities are the W3 ones (``data_fix/common.py``: reviewed, kept as sound); the glossary is
``glossary_vi_math_s3.json`` (W3 v0.1 + S3 G54b).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from ..data_fix.common import (
    all_boxed, brace_balance, canon_num, gsm8k_final, has_unclosed_boxed, norm_boxed, number_multiset_diff,
    repair_control_chars, repair_literal_newlines,
)

GLOSSARY = json.loads((Path(__file__).with_name("glossary_vi_math_s3.json")).read_text(encoding="utf-8"))
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0d\x0e-\x1f\x7f]")
EN_STOP = re.compile(r"\b(the|and|of|is|are|that|this|how|many|much|what|each|with|has|have|does|she|he|they)\b")
ASY_RE = re.compile(r"\[asy\].*?\[/asy\]", re.S)
MATH_RE = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$(?:\\\$|[^$])*?(?<!\\)\$|\[asy\].*?\[/asy\]", re.S)
ANNOT_RE = re.compile(r"<<[^<>]*>>")
# commands a faithful translation may add (\frac -> \dfrac, unit text, spacing); seen in detect, all benign
BENIGN_NEW_CMDS = {"dfrac", "tfrac", "frac", "left", "right", "cdot", "times", "text", "mbox", "quad", "qquad"}

__all__ = ["prose", "g0", "g1", "fired", "deterministic_repair", "asy_blocks", "GLOSSARY", "gold_matches"]


def prose(s: str, dataset: str) -> str:
    """Text outside math / code / calculator annotations (GSM8K ``$`` are dollar amounts, not delimiters)."""
    if dataset == "gsm8k":
        return ANNOT_RE.sub(" ", s or "")
    return MATH_RE.sub(" ", ANNOT_RE.sub(" ", s or ""))


def asy_blocks(s: str) -> list[str]:
    return ASY_RE.findall(s or "")


def latex_cmds(s: str) -> Counter:
    return Counter(re.findall(r"\\([A-Za-zÀ-ỹđĐ]+)", s or ""))


def unescaped_dollars(s: str) -> int:
    return len(re.findall(r"(?<!\\)\$", s or ""))


_INLINE = re.compile(r"(?<!\\)\$(?!\$)((?:\\\$|[^$])*?)(?<!\\)\$")
_TEXTCMD = re.compile(r"\\(?:text|mbox|textbf|textit|mathrm|operatorname|textrm|emph|textsf|mathit)\s*\{[^{}]*\}")


def _swallowed(s: str) -> str | None:
    """An inline math span holding Vietnamese prose (outside \\text{}) — the R2 '\\$18.09 -> $18.09' breakage."""
    s = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]|\[asy\].*?\[/asy\]", " ", s or "", flags=re.S)
    for m in _INLINE.finditer(s):
        body = _TEXTCMD.sub(" ", m.group(1))
        if re.search(r"[À-ỹđĐ]", body) or re.search(r"\b(?:đô|la|và|của|là|với)\b", body):
            return m.group(0)
    return None


def fired(en: str) -> tuple[list[dict], list[dict]]:
    g = [e for e in GLOSSARY["entries"] if re.search(e["en"], en, re.I)]
    r = [c for c in GLOSSARY["rules"] if re.search(c["en"], en, re.I)]
    return g, r


# --------------------------------------------------------------------------------------------- repairs
def deterministic_repair(r: dict) -> tuple[dict, list[dict]]:
    """T4 §3.8 step 1 on one aligned row. Returns ({problem_vi, solution_vi, gold_raw_vi}, [change records])."""
    out = {k: r[k] for k in ("problem_vi", "solution_vi", "gold_raw_vi")}
    changes: list[dict] = []
    en_all = r["problem_en"] + "\n" + r["solution_en"]
    for fld, en in (("problem_vi", r["problem_en"]), ("solution_vi", r["solution_en"]), ("gold_raw_vi", en_all)):
        t, notes = repair_control_chars(out[fld], en)
        if notes:
            out[fld] = t
            changes.append({"field": fld, "kind": "escape_damage",
                            "reason": "JSON-decoded control char restored to a LaTeX escape (R2 §4: \\right->CR+ight "
                                      "etc.): " + ", ".join(f"{k}x{v}" for k, v in Counter(notes).items())})
        if "\\n" in out[fld] and "\\n" not in re.sub(r"\\n(?:eq|u|ot|abla|eg|ewline|i|mid|exists|leq|geq|cong|sim|"
                                                         r"subseteq|rightarrow|Rightarrow|leftarrow|Leftarrow|parallel|"
                                                         r"prec|succ|otin|vdash)", "", en):
            t, n = repair_literal_newlines(out[fld], en)
            if n:
                out[fld] = t
                changes.append({"field": fld, "kind": "literal_newline",
                                "reason": f"literal '\\n' -> newline x{n} (R2 §4)"})
    if r["dataset"] == "gsm8k":
        gold = canon_num(r["gold_raw_en"])
        fin = gsm8k_final(out["solution_vi"])
        if fin is not None and canon_num(fin) != gold and re.fullmatch(r"-?\d{1,3}(\.\d{3})+", fin) \
                and fin.replace(".", "") == gold:
            out["solution_vi"] = re.sub(r"####\s*.+?\s*$", "#### " + gold, out["solution_vi"])
            changes.append({"field": "solution_vi", "kind": "final_line_format",
                            "reason": f"final line '#### {fin}' (VN thousands dot) -> '#### {gold}' (R1 §5)"})
    else:
        for fld, en in (("problem_vi", r["problem_en"]), ("solution_vi", r["solution_en"])):
            ve, ee = asy_blocks(out[fld]), asy_blocks(en)
            if ve != ee and len(ve) == len(ee) and ee:
                t = out[fld]
                for a, b in zip(ve, ee):
                    t = t.replace(a, b, 1)
                out[fld] = t
                changes.append({"field": fld, "kind": "asy_restored",
                                "reason": f"[asy] block(s) edited by the translator -> English block(s) copied back "
                                          f"verbatim (R2 §5 item 4; {sum(a != b for a, b in zip(ve, ee))} block(s))"})
    return out, changes


# --------------------------------------------------------------------------------------------- gates
def _len_ratio(vi: str, en: str) -> float:
    return len(vi) / max(1, len(en))


def g0(vi_p: str, vi_s: str, en_p: str, en_s: str, dataset: str) -> list[str]:
    fails = []
    for fld, vi, en in (("problem", vi_p, en_p), ("solution", vi_s, en_s)):
        if CTRL_RE.search(vi or ""):
            fails.append(f"G0 control characters in {fld}")
        if "\t" in (vi or "") and "\t" not in en and re.search(r"\t[A-Za-z]", vi):
            fails.append(f"G0 TAB+letters in {fld} (escape damage)")
        if brace_balance(vi) != brace_balance(en):
            fails.append(f"G0 brace balance differs in {fld} (VI {brace_balance(vi)} vs EN {brace_balance(en)})")
        if dataset == "math":
            # parity, not equality: the translator legitimately turns `$4$` into `4` (detect: 306 rows differ in
            # count, almost all benign). Broken delimiters show as an odd count or as prose swallowed into math.
            if unescaped_dollars(vi) % 2 != unescaped_dollars(en) % 2:
                fails.append(f"G0 unescaped '$' parity broken in {fld} (VI {unescaped_dollars(vi)} vs EN "
                             f"{unescaped_dollars(en)})")
            else:
                sw = _swallowed(vi)
                if sw:
                    fails.append(f"G0 unescaped '$' delimits prose in {fld}: {sw[:80]!r} (a '\\$' amount broken)")
        if asy_blocks(vi) != asy_blocks(en):
            fails.append(f"G0 [asy] blocks differ from English in {fld} ({len(asy_blocks(vi))} vs {len(asy_blocks(en))})")
        t = prose(vi, dataset)
        words = re.findall(r"[A-Za-zÀ-ỹđĐ]+", t)
        if words and len(EN_STOP.findall(t)) / len(words) > 0.06:
            fails.append(f"G0 English left in {fld} ({len(EN_STOP.findall(t))} EN stopwords / {len(words)} words)")
        lr = _len_ratio(vi or "", en)
        if not (0.6 <= lr <= 2.5):
            fails.append(f"G0 length ratio {lr:.2f} in {fld}")
    new = set(latex_cmds(vi_p + vi_s)) - set(latex_cmds(en_p + en_s))
    new -= {"n"} | BENIGN_NEW_CMDS                 # \n: handled by the literal-\n repair
    if new:
        fails.append(f"G0 new or translated LaTeX commands {sorted(new)}")
    if has_unclosed_boxed(vi_s):
        fails.append("G0 \\boxed{ without its closing brace in solution")
    return fails


def gold_matches(vi_s: str, gold_raw_en: str, dataset: str, en_s: str | None = None) -> tuple[bool, str | None]:
    """Does the VI solution end in the English gold? GSM8K: the '####' line; MATH: the last \\boxed{} (typed grader,
    with a light-normalisation fallback for VI unit text)."""
    if dataset == "gsm8k":
        fin = gsm8k_final(vi_s)
        return (fin is not None and canon_num(fin) == canon_num(gold_raw_en)), fin
    b = all_boxed(vi_s)
    if not b:
        return False, None
    last = b[-1]
    if norm_boxed(last) == norm_boxed(gold_raw_en):
        return True, last
    try:
        from ..eval.math_verify import grade, infer_gold
        g = infer_gold(gold_raw_en, locale="en")
        if grade("\\boxed{" + last + "}", g, locale="vi")["correct"] or \
                grade("\\boxed{" + last + "}", g, locale="en")["correct"]:
            return True, last
        # several final boxes (a multi-answer gold such as "18, 12" from a solution boxing each)
        if len(b) > 1 and grade(" ".join("\\boxed{" + x + "}" for x in b), g)["correct"]:
            return True, last
    except Exception:
        pass
    if en_s is not None:      # the EN solution's own last box may differ from ground_truth formatting
        eb = all_boxed(en_s)
        if eb and norm_boxed(eb[-1]) == norm_boxed(last):
            return True, last
    return False, last


def g1(vi_p: str, vi_s: str, en_p: str, en_s: str, gold_raw_en: str, dataset: str) -> list[str]:
    fails = []
    miss, add = number_diff(en_p, vi_p, dataset)
    if miss or add:
        fails.append(f"G1 numbers in problem: EN values missing in VI {miss}, VI values not in EN {add}")
    ok, got = gold_matches(vi_s, gold_raw_en, dataset, en_s)
    if not ok:
        want = f"'#### {canon_num(gold_raw_en)}'" if dataset == "gsm8k" else f"\\boxed{{{gold_raw_en}}}"
        fails.append(f"G1 final answer must be {want} (got {got!r})")
    if dataset == "gsm8k":
        miss = Counter(ANNOT_RE.findall(en_s)) - Counter(ANNOT_RE.findall(vi_s))
        if miss:
            fails.append(f"G1 calculator annotations missing in VI solution: {sorted(miss)[:4]}")
    gl, rules = fired(en_p + "\n" + en_s)
    for e in gl:
        if not e.get("forbid"):
            continue
        if e.get("skip_if_en") and re.search(e["skip_if_en"], en_p + "\n" + en_s, re.I):
            continue                     # the English itself uses the metric unit (e.g. "a gallon is 4 liters")
        for fld, en, vi in (("problem", en_p, vi_p), ("solution", en_s, vi_s)):
            if re.search(e["en"], en, re.I):
                m = re.search(e["forbid"], prose(vi, dataset), re.I)
                if m:
                    fails.append(f"G1 glossary {e['id']} in {fld}: forbidden '{m.group(0)}' — {e['line']}")
    for c in rules:
        men = re.search(c["en"], en_p, re.I)
        if not men:
            continue
        forbid = c.get("vi_forbid")
        if c["id"] == "CR-3":
            # S3 fix of the W3 rule: the forbidden rendering is "nhiều hơn N lần" with N = the ADDITIVE number of
            # "N more than k times"; the correct template "nhiều hơn k lần số ... là N" also contains "hơn k lần".
            n_add = men.group(1)
            forbid = rf"(nhiều|ít) hơn {re.escape(n_add)} lần" if re.fullmatch(r"\d+", n_add) else None
        if forbid:
            m = re.search(forbid, vi_p, re.I)
            if m:
                fails.append(f"G1 {c['id']} in problem: '{m.group(0)}' — {c['line']}")
        if c.get("vi_require") and c["vi_require"] not in vi_p:
            fails.append(f"G1 {c['id']} in problem: required '{c['vi_require']}' missing — {c['line']}")
    return fails


def _num_text(s: str, dataset: str, variant: int = 1) -> str:
    r"""Text whose digit numbers are compared. GSM8K: prose without calculator annotations. MATH: everything except
    [asy] code and LaTeX command names; ``{,}`` and ``,\!`` dropped. ``variant`` 1: a comma between digits inside
    math is a list separator (coordinates ``$(1,3)$``); 2: also bare ``(a,b)`` pairs in prose; 0: neither (a VI
    decimal comma inside math, ``$86,8\%$``)."""
    if dataset == "gsm8k":
        return ANNOT_RE.sub(" ", s or "")
    t = ASY_RE.sub(" ", s or "")
    t = re.sub(r"\{,\}|,\\!", "", t)

    def mathfix(m):
        b = m.group(0)
        if variant >= 1:
            b = re.sub(r"(\d)\s*,\s*(?=-?\d)", r"\1 ; ", b)
        return re.sub(r"\\[a-zA-Z]+|[{}$^_]", " ", b)
    t = MATH_RE.sub(mathfix, t)
    if variant >= 2:
        t = re.sub(r"\((\s*-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?\s*)\)", r"(\1 ; \2)", t)
    return re.sub(r"\\[a-zA-Z]+|[{}$^_]", " ", t)


_COIN_VALUES = [(r"\bquarters\b|\ba quarter\b(?! of)", 25.0), (r"\bdimes?\b", 10.0), (r"\bnickels?\b", 5.0),
                (r"\bpenn(?:y|ies)\b", 1.0), (r"\bhalf[- ]dollars?\b", 50.0)]


def number_diff(en_p: str, vi_p: str, dataset: str) -> tuple[list, list]:
    """Set-level digit-number diff EN->VI of the problem (W3 ``number_multiset_diff``) under the most lenient comma
    reading, with benign renderings removed: coin names rendered with their value (quarter -> '25 xu'),
    'a/an' -> '1', 'N and a half' -> 'N,5'."""
    ok_add = {1.0} | {v for pat, v in _COIN_VALUES if re.search(pat, en_p, re.I)}
    best = None
    for variant in ((0,) if dataset == "gsm8k" else (1, 2, 0)):
        en_t = re.sub(r"(\d+) and a half", lambda m: f"{m.group(1)}.5", _num_text(en_p, dataset, variant))
        en_t = re.sub(r"(?<=[A-Za-z]),(?=\d)", ", ", en_t)          # "8 roses,12 daisies" (missing space)
        nd = number_multiset_diff(en_t, _num_text(vi_p, dataset, variant))
        miss, add = list(nd["missing_set"]), [a for a in nd["added_set"] if a not in ok_add]
        if best is None or len(miss) + len(add) < len(best[0]) + len(best[1]):
            best = (miss, add)
        if not miss and not add:
            break
    return best


def fail_kind(f: str) -> str:
    """Short category of a gate failure (for counting)."""
    m = re.match(r"(G[01]) (glossary \w+|CR-\d|numbers|final answer|calculator|control|TAB|brace|unescaped|\[asy\]|"
                 r"English|length|new or translated|\\boxed)", f)
    if not m:
        return f[:30]
    k = m.group(2)
    return f"{m.group(1)} {k}"
