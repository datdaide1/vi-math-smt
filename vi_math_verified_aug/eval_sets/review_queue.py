"""Step 3: translation-defect review queue for the test items (question text only; golds come from English).

Run from the repo root with the project venv::

    .venv/Scripts/python.exe -m vi_math_verified_aug.eval_sets.review_queue

Triggers (each queued item lists every trigger that fired):
- ``platinum_revised``      GSM8K-Platinum changed the gold -> forced re-translation from English
- ``r1_annotation`` / ``r2_annotation``  question-side defects recorded by content reviews R1/R2
- ``number_multiset``       EN<->VI numeral multisets differ (locale-agnostic, see ``common.digit_tokens``)
- ``latex_multiset``        EN<->VI LaTeX command multisets differ (MATH)
- ``pat:*``                 R1/R2 mistranslation patterns ("N of the remaining", "more than N ... so với", ...)
- ``gloss:*``               glossary hits: an English term whose standard Vietnamese rendering is absent, or a
                            known wrong rendering is present ("congruent" -> "đồng dạng")
- ``fmt:*``                 English thousands comma left in Vietnamese text, unbalanced braces/``$``, asy edits
- ``leak:english``          an untranslated English sentence in the Vietnamese question

Output: ``data/eval/_work/review_queue.jsonl`` and ``review_queue_summary.json``.
"""
from __future__ import annotations

import collections
import json
import re

from vi_math_verified_aug.eval_sets.common import (
    WORK, asy_blocks, braces_balanced, dollars_even, has_vi, read_jsonl, strip_asy_strings, write_json, write_jsonl,
)

# Question-side defects from the content reviews (solution-only notes are excluded: the eval scores questions only).
R1_TEST_QUESTION_DEFECTS = {  # data/test/gsm8k row index -> R1 code
    23: "NL-LEAK (EN clock format)", 130: "NL-MEANING critical ('10 of the remaining')",
    217: "NL-FLUENCY calque 'nhiều hơn 7 ... so với'", 382: "NL-MEANING major ('40 more gold pesos')",
    475: "NL-FLUENCY ('break in' -> 'đi thử')", 583: "NL-FLUENCY calque", 865: "NL-FLUENCY calque",
    1179: "NL-FLUENCY ('thêm 7 chiếc')",
}
R2_TEST_QUESTION_DEFECTS = {
    "math-test-algebra-0645": "NL-FLUENCY ('slope' -> 'Độ dốc')",
    "math-test-prealgebra-0765": "X-DECSEP major ('54,321' reads as a decimal)",
}

# (name, English regex, Vietnamese renderings that must be present (any), Vietnamese wrong renderings (any))
GLOSSARY = [
    ("congruent", r"\bcongruent\b", [], [r"đồng dạng"]),
    ("common_ratio", r"\bcommon ratio\b", [r"công bội"], [r"t[iỉỷ] số chung"]),
    ("relatively_prime", r"\brelatively prime\b", [r"nguyên tố cùng nhau"], [r"tương đối nguyên tố"]),
    ("monic", r"\bmonic\b", [], [r"đơn thức"]),
    ("kite", r"\bkite\b", [], [r"hình thoi"]),
    ("counterclockwise", r"counter-?clockwise", [r"ngược chiều kim"], []),
    ("clockwise", r"(?<!counter)(?<!counter-)\bclockwise", [], [r"ngược chiều kim"]),
    ("improper_fraction", r"\bimproper fraction", [], [r"không đúng quy cách"]),
    ("integer_vs_prime", r"\bintegers?\b", [], [r"nguyên tố"]),  # only when English has no "prime" (below)
    ("prime", r"(?<!\\)\bprimes?\b", [r"nguyên tố"], []),  # not LaTeX \prime
    ("product", r"\bproduct\b", [r"\btích\b"], [r"sản phẩm"]),
    ("reciprocal", r"\breciprocal\b", [r"nghịch đảo"], []),
    ("remainder", r"\bremainder\b", [r"\bdư\b", r"còn lại"], []),
    ("median", r"\bmedian\b", [r"trung vị"], []),
    ("nonnegative", r"\bnon-?negative\b", [r"không âm"], []),
    ("positive", r"(?<!non)(?<!non-)\bpositive\b", [r"dương"], []),
    ("negative", r"(?<!non)(?<!non-)\bnegative\b", [r"\bâm\b"], []),
    ("even", r"\beven (?:number|integer|digit)", [r"chẵn"], []),
    ("odd", r"\bodd (?:number|integer|digit)", [r"\blẻ\b"], []),
    ("hundredth", r"\bhundredths?\b", [r"phần trăm", r"hai chữ số thập phân", r"chữ số thập phân thứ hai", r"0[,.]01"], []),
    ("tenth", r"nearest tenth", [r"phần mười", r"một chữ số thập phân", r"chữ số thập phân thứ nhất", r"0[,.]1"], []),
    ("nearest_hundred", r"nearest hundred\b(?!th)", [r"hàng trăm"], [r"phần trăm"]),
    ("at_least", r"\bat least\b", [r"ít nhất", r"tối thiểu", r"không ít hơn", r"không nhỏ hơn", r"trở lên"], []),
    ("at_most", r"\bat most\b", [r"nhiều nhất", r"tối đa", r"không quá", r"không vượt quá", r"không lớn hơn"], []),
    ("strictly", r"\bstrictly\b", [r"thực sự", r"hẳn", r"nghiêm ngặt", r"chặt"], [r"đáng kể"]),
    ("imperial_to_metric", r"\b(?:yards?|feet|foot|inch(?:es)?|miles?|gallons?|pounds?|ounces?|quarts?|pints?)\b",
     [], [r"\b(?:mét|km|ki-lô-mét|lít|kg|ki-lô-gam|gam|cm|xăng-ti-mét)\b"]),
]
# Metric units that are legitimately present in English stop the imperial->metric check from firing.
_EN_METRIC = re.compile(r"\b(?:meters?|metres?|km|kilometers?|liters?|litres?|kg|kilograms?|grams?|cm|centimeters?)\b",
                        re.I)


def _strip_math(t: str) -> str:
    t = re.sub(r"\[asy\].*?\[/asy\]", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{(\w+\*?)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\)", " ", t, flags=re.S)
    return t


def pattern_hits(en: str, vi: str) -> list[str]:
    hits = []
    if re.search(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|half|some|all|most) of the remaining\b",
                 en, re.I):
        hits.append("pat:of_the_remaining")
    # calque "nhiều/ít hơn N X so với Y" (reads "more than N")
    if re.search(r"(?:nhiều|ít) hơn \d[\d.,]*(?:\s+\S+){0,6}?\s+so với", vi):
        hits.append("pat:more_than_N_so_voi")
    # additive relation rendered multiplicatively: VI "hơn N lần" without EN "N times"
    for m in re.finditer(r"(?:nhiều|ít|lớn|nhỏ) hơn (\d+) lần", vi):
        if not re.search(rf"\b{m.group(1)} times\b", en):
            hits.append("pat:N_lan_additive")
            break
    if re.search(r"\b\d+ (?:more|less|fewer) than (?:twice|thrice|half|\w+ times|\d+ times)", en) and \
            re.search(r"hơn \d+ lần", vi):
        hits.append("pat:N_more_than_k_times")
    # "thước" alone reads as "metre"/"ruler"; "kích thước" (size), "thước đo" (protractor), "thước Anh" (yard) are fine
    if re.search(r"(?<!kích )\bthước\b(?! (?:Anh|đo|kẻ))", vi) and not re.search(r"\bruler", en, re.I):
        hits.append("pat:thuoc")
    return hits


def glossary_hits(en: str, vi: str) -> list[str]:
    hits = []
    en_l, vi_l = en.lower(), vi.lower()
    for name, en_pat, need, bad in GLOSSARY:
        if not re.search(en_pat, en_l):
            continue
        if name == "imperial_to_metric" and _EN_METRIC.search(en):
            continue
        if name == "integer_vs_prime" and re.search(r"\bprimes?\b", en_l):
            continue
        if need and not any(re.search(p, vi_l) for p in need):
            hits.append(f"gloss:{name}:missing")
        if bad and any(re.search(p, vi_l) for p in bad):
            hits.append(f"gloss:{name}:wrong")
    return hits


def format_hits(en: str, vi: str) -> list[str]:
    hits = []
    # English thousands comma kept in Vietnamese (reads as a decimal comma), only where English has the same token
    for m in re.finditer(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d,])", vi):
        if m.group(0) in en:
            hits.append("fmt:en_thousands_comma")
            break
    if braces_balanced(en) and not braces_balanced(vi):
        hits.append("fmt:unbalanced_braces")
    if dollars_even(en) and not dollars_even(vi):
        hits.append("fmt:odd_dollar")
    ea, va = asy_blocks(en), asy_blocks(vi)
    if len(ea) != len(va):
        hits.append("fmt:asy_count")
    elif any(strip_asy_strings(a).split() != strip_asy_strings(b).split() for a, b in zip(ea, va)):
        hits.append("fmt:asy_code_edit")
    if re.search(r"[\r\x0c\x08\x0b\x07]", vi):
        hits.append("fmt:control_char")
    return hits


def english_leak(vi: str) -> bool:
    for sent in re.split(r"(?<=[.?!:])\s+|\n", _strip_math(vi)):
        words = re.findall(r"[A-Za-z]+", sent)
        if len(words) >= 6 and not has_vi(sent):
            return True
    return False


def build() -> list[dict]:
    queue = []
    gsm = read_jsonl(WORK / "gsm8k_platinum_base.jsonl")
    math = read_jsonl(WORK / "math_algprealg_base.jsonl")
    for r in gsm + math:
        en, vi = r["question_en"], r["question_vi"]
        trig = []
        if r.get("platinum_status") == "revised":
            trig.append("platinum_revised")
        if r["dataset"] == "gsm8k_platinum" and r["gsm8k_index"] in R1_TEST_QUESTION_DEFECTS:
            trig.append("r1_annotation:" + R1_TEST_QUESTION_DEFECTS[r["gsm8k_index"]])
        if r["id"] in R2_TEST_QUESTION_DEFECTS:
            trig.append("r2_annotation:" + R2_TEST_QUESTION_DEFECTS[r["id"]])
        if not r["number_parity_ok"]:
            trig.append("number_multiset")
        if r.get("latex_parity_ok") is False:
            trig.append("latex_multiset")
        trig += pattern_hits(en, vi)
        trig += glossary_hits(en, vi)
        trig += format_hits(en, vi)
        if english_leak(vi):
            trig.append("leak:english")
        if trig:
            queue.append({
                "id": r["id"], "dataset": r["dataset"], "triggers": trig,
                "forced": "platinum_revised" in trig,
                "in_math500": r.get("in_math500", False),
                "number_parity_diff": r["number_parity_diff"],
            })
    return queue


def main() -> None:
    q = build()
    write_jsonl(WORK / "review_queue.jsonl", q)
    summ: dict = {}
    for ds in ("gsm8k_platinum", "math_algprealg"):
        sub = [x for x in q if x["dataset"] == ds]
        cnt = collections.Counter(t.split(":")[0] if t.startswith(("r1_", "r2_")) else t
                                  for x in sub for t in x["triggers"])
        summ[ds] = {"queued_items": len(sub), "in_math500": sum(x["in_math500"] for x in sub),
                    "trigger_counts": dict(cnt.most_common())}
    write_json(WORK / "review_queue_summary.json", summ)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
