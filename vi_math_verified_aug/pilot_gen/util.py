"""G1 helpers: prompt loading, answer extraction/comparison, leak regex, chrF, Wilson CI."""
from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from .numerals import parse_literal, text_contains_value

PROMPTS = Path(__file__).with_name("prompts")


def prompt(name: str, **kw) -> tuple[str, str]:
    txt = (PROMPTS / name).read_text(encoding="utf-8")
    s, u = txt.split("### USER", 1)
    s = s.replace("### SYSTEM", "", 1).strip()
    u = u.strip()
    for k, v in kw.items():
        s = s.replace(f"<<{k}>>", str(v))
        u = u.replace(f"<<{k}>>", str(v))
    return s, u


OP_INSTR = {
    "M1": "Đổi các số liệu, giữ nguyên cấu trúc lời giải.",
    "M2": "Đảo bài toán: cho biết kết quả của bài gốc, hỏi một dữ kiện ban đầu.",
    "M3": "Thêm đúng một bước tính hợp lý vào giữa câu chuyện (không thêm vào sau đáp số).",
    "M4": "Thay đổi một quan hệ (thêm↔bớt, gấp k lần, phần trăm…).",
    "M5": "Hỏi một đại lượng khác cần ít nhất hai bước tính.",
    "M6": "Ghép với bài sau thành một bài nhiều bước: ",
}


def card_text(c: dict) -> str:
    items = "; ".join(f"{a} ({b})" for a, b in c["vat_pham"])
    return (f"bối cảnh: {c['boi_canh']}; tên người: {', '.join(c['ten'])}; vật phẩm và giá: {items}; "
            f"đơn vị: {', '.join(c['don_vi'])}")


def last_boxed(s: str) -> str | None:
    if not s:
        return None
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    d = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0:
                return s[j + 1:k]
    return None


def answer_values(a) -> set:
    """All plausible Fraction values of an answer string (boxed content or generator 'answer')."""
    if a is None:
        return set()
    s = str(a)
    s = re.sub(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\text\{[^{}]*\}|\\mathrm\{[^{}]*\}", " ", s)
    s = s.replace("{,}", ",").replace("\\,", "").replace("\\!", "").replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "").replace("−", "-").replace("\u00a0", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    m = re.findall(r"-?\d+(?:[.,\s]\d{3})*(?:[.,]\d+)?(?:\s*/\s*\d+)?", s)
    if not m:
        return set()
    tok = m[-1] if len(m) > 1 and "=" in s else m[0]
    tok = tok.strip()
    neg = tok.startswith("-")
    vals = parse_literal(tok.lstrip("-")) or set()
    if " " in tok and not vals:
        vals = parse_literal(tok.replace(" ", ""))
    return {(-v if neg else v) for v in vals}


def same_answer(a, b) -> bool:
    va, vb = (a if isinstance(a, set) else answer_values(a)), (b if isinstance(b, set) else answer_values(b))
    return bool(va & vb)


LEAK_RE = re.compile(r"\b[hd]_\d+|\banswer\b|\bSMT\b|ràng buộc|\bZ3\b|<BÀI|\\boxed|đáp (?:án|số)\s*(?:là|=|:)|"
                     r"gợi ý|kết quả (?:mong muốn|cuối cùng phải)|lời giải", re.I)


def leak_regex(text: str) -> list:
    return [m.group(0) for m in LEAK_RE.finditer(text or "")]


def answer_in_text(text: str, ans_vals: set) -> bool:
    return any(text_contains_value(text or "", v) for v in ans_vals)


def _ngrams(s: str, n: int) -> Counter:
    s = re.sub(r"\s+", "", s)
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """chrF (character n-gram F-beta, n=1..6, whitespace removed), 0..1."""
    ps, rs = [], []
    for k in range(1, n + 1):
        h, r = _ngrams(hyp, k), _ngrams(ref, k)
        ov = sum((h & r).values())
        if sum(h.values()):
            ps.append(ov / sum(h.values()))
        if sum(r.values()):
            rs.append(ov / sum(r.values()))
    if not ps or not rs:
        return 0.0
    p, r = sum(ps) / len(ps), sum(rs) / len(rs)
    if p + r == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def fmt_ci(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    if n == 0:
        return "–"
    return f"{100 * p:.0f}% [{100 * lo:.0f}–{100 * hi:.0f}] ({k}/{n})"


def kappa(pairs: list) -> float:
    """Cohen's kappa for a list of (a, b) labels."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
