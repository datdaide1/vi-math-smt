"""Task W3-2: Vi-MATH test -> v1.1 (structured EN gold, deterministic repairs, review queue, slices).

* gold: attached from the English MATH test by position within subject (alignment verified: row counts,
  level/type equality, last-\\boxed agreement after normalisation).
* deterministic fixes (problem + solution): JSON-escape damage (CR/FF/BS/VT, TAB before a command stem),
  literal ``\\n`` tokens, ``$`` amounts that lost their backslash (``\\$18.09`` -> ``$18.09``), English
  thousands commas in prose (``54,321`` -> ``54.321``, only when the same token is in the English prose).
* review queue (NOT auto-fixed): congruent->đồng dạng, [asy] edited/dropped/moved, counterclockwise, monic,
  common ratio, relatively prime, unit swaps, untranslated solutions, number/LaTeX parity breaks,
  multi-answer golds.
* slices: algebra+prealgebra in-scope (2,058) and Vi-MATH-500 alg+prealg (matched by exact EN problem text).

No LLM is used on any test item.
Run: python -X utf8 -I vi_math_verified_aug/data_fix/build_math_test.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vi_math_verified_aug.data_fix.common import (  # noqa: E402
    CLEAN, DATA, SOURCES, all_boxed, bad_ctrl, brace_balance, datacard, last_boxed, load_annotations, norm_boxed,
    norm_ws, read_json, read_jsonl, rel, repair_control_chars, repair_literal_newlines, unescaped_dollars, write_jsonl,
)

VERSION = "1.1"
OUT = CLEAN / "vi_math_test"
SCRIPT = "vi_math_verified_aug/data_fix/build_math_test.py"
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
            "prealgebra", "precalculus"]
VI_DIR = DATA / "test" / "math"
EN_DIR = DATA / "generate" / "MATH" / "test"
M500 = SOURCES / "math500_test.jsonl"

ASY = re.compile(r"\[asy\].*?\[/asy\]", re.S)
MATH_SPAN = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$(?:\\\$|[^$])*?(?<!\\)\$|\[asy\].*?\[/asy\]",
                       re.S)
TERMS = [  # (reason, EN regex, VI forbidden regex)  — glossary forbid-list (T4 §4)
    ("congruent->dong_dang", r"(?<!non-)(?<!non)congruent", r"(?<!không )đồng dạng"),
    ("non_congruent->khong_dong_dang", r"non-?congruent", r"không đồng dạng"),
    ("counterclockwise->clockwise", r"counter-?clockwise", r"(?<!ngược )chiều kim đồng hồ"),
    ("monic->don_thuc", r"\bmonic\b", r"đơn thức"),
    ("common_ratio->ti_so_chung", r"common ratio", r"tỉ số chung|tỷ số chung"),
    ("relatively_prime->tuong_doi_nguyen_to", r"relatively prime", r"tương đối nguyên tố"),
    ("kite->hinh_thoi", r"\bkite\b", r"hình thoi"),
    ("divides_direction", r"\b\S+ divides \S+", r"chia hết cho"),
    ("median->trung_binh", r"\bmedian\b", r"\btrung bình\b"),
    ("dozen->chuc", r"\bdozen\b", r"\bchục\b"),
]
UNIT_SWAP = [(r"\byards?\b", r"\bmét\b|\bthước\b"), (r"\bgallons?\b", r"\blít\b"), (r"\bfeet|foot\b", r"\bmét\b"),
             (r"\bmiles?\b", r"\bki-?lô-?mét\b|\bkm\b"), (r"\bpounds?\b", r"\bkg\b|\bki-?lô-?gam\b"),
             (r"\binch(es)?\b", r"\bcm\b|xăng-?ti-?mét")]
EN_STOP = re.compile(r"\b(the|and|of|is|are|that|this|we|have|which|so|then|since|therefore)\b")
VI_DIAC = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)
MULTI_ANS = re.compile(r"separated by commas|Enter all|enter all|all possible|all the (?:solutions|values)|"
                       r"two smallest|Enter the (?:ordered|letters)|in the form", re.I)
LATEX_CMD = re.compile(r"\\[A-Za-z]+")


def prose(s: str) -> str:
    return MATH_SPAN.sub(" ", s or "")


def untranslated(s: str) -> bool:
    t = prose(s)
    words = re.findall(r"[A-Za-zÀ-ỹ]+", t)
    return len(words) >= 8 and len(EN_STOP.findall(t)) >= 4 and len(VI_DIAC.findall(t)) < 3


def fix_dollars(vi: str, en: str) -> tuple[str, int]:
    """Re-escape money amounts: EN ``\\$18.09`` rendered in VI as bare ``$18.09`` (breaks math mode)."""
    if unescaped_dollars(vi) % 2 == 0:
        return vi, 0
    n = 0
    for amt in set(re.findall(r"\\\$(\d[\d,.]*)", en)):
        pat = re.compile(r"(?<!\\)\$(" + re.escape(amt) + r")(?![\d])")
        if pat.search(vi) and unescaped_dollars(vi) % 2 == 1:
            vi, k = pat.subn(r"\\$\1", vi, count=1)
            n += k
    return vi, n


def fix_thousands(vi: str, en: str) -> tuple[str, list[str]]:
    """Prose-only: ``54,321`` -> ``54.321`` when the same comma-grouped token is in the English prose (T4 §3.6)."""
    en_tok = set(re.findall(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d,])", prose(en)))
    if not en_tok:
        return vi, []
    out, notes, last = [], [], 0
    for m in MATH_SPAN.finditer(vi):
        out.append(_sub_th(vi[last:m.start()], en_tok, notes))
        out.append(m.group(0))
        last = m.end()
    out.append(_sub_th(vi[last:], en_tok, notes))
    return "".join(out), notes


def _sub_th(seg: str, en_tok: set[str], notes: list[str]) -> str:
    def r(m):
        if m.group(0) in en_tok:
            notes.append(f"{m.group(0)}->{m.group(0).replace(',', '.')}")
            return m.group(0).replace(",", ".")
        return m.group(0)
    return re.sub(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d,])", r, seg)


def repair(vi: str, en: str) -> tuple[str, list[str]]:
    notes = []
    t, cc = repair_control_chars(vi, en)
    if cc:
        notes.append("escape damage: " + ", ".join(f"{k}x{v}" for k, v in Counter(cc).items()))
    t, nl = repair_literal_newlines(t, en)
    if nl:
        notes.append(f"literal \\n -> newline x{nl}")
    t, nd = fix_dollars(t, en)
    if nd:
        notes.append(f"re-escaped $ amount x{nd}")
    return t, notes


def main() -> None:
    ann = load_annotations("vi_math_test")
    rows, queue, changed = [], [], []
    align = {}
    for subj in SUBJECTS:
        vi = read_jsonl(VI_DIR / f"{subj}_vi.jsonl")
        en = read_json(EN_DIR / f"{subj}.json")
        assert len(vi) == len(en), (subj, len(vi), len(en))
        lvl = sum(v["level"] == e["level"] and v["type"] == e["type"] for v, e in zip(vi, en))
        box = sum(norm_boxed(last_boxed(v["solution"])) == norm_boxed(last_boxed(e["solution"])) for v, e in zip(vi, en))
        en_gold = sum(last_boxed(e["solution"]) is not None for e in en)
        align[subj] = {"rows": len(vi), "level_type_equal": lvl, "last_boxed_equal_after_norm": box,
                       "en_has_boxed": en_gold}
        assert lvl == len(vi), f"{subj}: level/type misalignment"
        for i, (v, e) in enumerate(zip(vi, en)):
            rid = f"math/test/{subj}/{i}"
            p_vi, s_vi = v["problem"], v["solution"]
            p_en, s_en = e["problem"], e["solution"]
            reasons, notes = [], []
            # -------- deterministic repairs
            p_fix, pn = repair(p_vi, p_en)
            s_fix, sn = repair(s_vi, s_en)
            p_fix, th = fix_thousands(p_fix, p_en)
            if th:
                pn.append("EN thousands comma localised: " + ", ".join(th))
            if pn:
                notes.append("problem: " + "; ".join(pn))
            if sn:
                notes.append("solution: " + "; ".join(sn))
            # -------- gold
            boxes = all_boxed(s_en)
            gold = boxes[-1] if boxes else None
            vi_box = last_boxed(s_fix)
            if gold is None:
                reasons.append("no_en_boxed_gold")
            if len(boxes) > 1 and MULTI_ANS.search(p_en) and len({norm_boxed(b) for b in boxes}) > 1:
                reasons.append(f"multi_answer_gold: EN boxes {boxes} (gold=last only)")
            if norm_boxed(vi_box) != norm_boxed(gold):
                reasons.append(f"vi_boxed_ne_en_gold: VI {vi_box!r} vs EN {gold!r}")
            # -------- residual structural checks on the repaired problem
            if bad_ctrl(p_fix, p_en):
                reasons.append("problem_control_chars_remaining")
            if brace_balance(p_fix) != brace_balance(p_en):
                reasons.append(f"problem_brace_balance VI {brace_balance(p_fix)} vs EN {brace_balance(p_en)}")
            if unescaped_dollars(p_fix) % 2 != unescaped_dollars(p_en) % 2:
                reasons.append("problem_dollar_parity")
            new_cmds = set(LATEX_CMD.findall(p_fix)) - set(LATEX_CMD.findall(p_en))
            if new_cmds:
                reasons.append(f"problem_new_latex_cmds {sorted(new_cmds)}")
            # -------- asy
            a_vi_p, a_en_p = ASY.findall(p_fix), ASY.findall(p_en)
            a_vi_s, a_en_s = ASY.findall(s_fix), ASY.findall(s_en)
            if len(a_vi_p) < len(a_en_p):
                reasons.append("asy_dropped_from_problem")
            elif len(a_vi_p) > len(a_en_p):
                reasons.append("asy_moved_into_problem (possible answer leak)")
            elif a_vi_p != a_en_p:
                reasons.append("asy_edited_in_problem")
            if len(a_vi_s) == len(a_en_s) and a_vi_s != a_en_s:
                reasons.append("asy_edited_in_solution")
            # -------- terminology / meaning patterns
            for name, en_rx, vi_rx in TERMS:
                for fld, en_t, vi_t in (("problem", p_en, p_fix), ("solution", s_en, s_fix)):
                    if re.search(en_rx, en_t, re.I) and re.search(vi_rx, vi_t, re.I):
                        reasons.append(f"term:{name} in {fld}")
            for en_rx, vi_rx in UNIT_SWAP:
                if re.search(en_rx, prose(p_en), re.I) and re.search(vi_rx, prose(p_fix), re.I) \
                        and not re.search(vi_rx, prose(p_en), re.I):
                    reasons.append(f"unit_swap problem: EN /{en_rx}/ VI /{vi_rx}/")
            if untranslated(s_fix):
                reasons.append("solution_untranslated")
            if untranslated(p_fix):
                reasons.append("problem_untranslated")
            lr = len(p_fix) / max(1, len(p_en))
            if lr < 0.6 or lr > 2.5:
                reasons.append(f"problem_length_ratio {lr:.2f}")
            if re.search(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d,])", prose(p_fix)):
                reasons.append("problem_thousands_comma_remaining")
            a = ann.get((subj, i))
            if a and a.get("severity") in ("major", "critical"):
                reasons.append(f"R2_annotation:{a['severity']}:{','.join(a['codes'])}: {a['note']}")

            row = {
                "id": rid, "subject": subj, "index": i, "level": v["level"], "type": v["type"],
                "problem": p_fix, "solution": s_fix,
                "gold": gold, "gold_all_boxed": boxes, "gold_source": "EN MATH test, last \\boxed of solution",
                "en_problem": p_en, "en_solution": s_en,
                "flags": reasons, "review_status": "pending" if reasons else "auto_pass",
                "math500_unique_id": None,
            }
            rows.append(row)
            fields = [f for f, a_, b_ in (("problem", p_vi, p_fix), ("solution", s_vi, s_fix)) if a_ != b_]
            if fields:
                changed.append({"id": rid, "fields": fields, "reason": " | ".join(notes)})
            if reasons:
                queue.append({"id": rid, "subject": subj, "index": i, "level": v["level"],
                              "en_problem": p_en, "vi_problem": p_fix, "en_solution": s_en, "vi_solution": s_fix,
                              "gold": gold, "reasons": reasons, "priority": priority(reasons),
                              "suggested_fix": suggest(reasons, a_en_p, a_vi_p)})

    # -------- MATH-500 match
    m500 = read_jsonl(M500)
    key = {(r["subject"], norm_ws(r["en_problem"])): r for r in rows}
    sub_map = {"Algebra": "algebra", "Counting & Probability": "counting_and_probability", "Geometry": "geometry",
               "Intermediate Algebra": "intermediate_algebra", "Number Theory": "number_theory",
               "Prealgebra": "prealgebra", "Precalculus": "precalculus"}
    m500_hit, m500_gold_eq = 0, 0
    for m in m500:
        r = key.get((sub_map[m["subject"]], norm_ws(m["problem"])))
        if r is None:
            continue
        m500_hit += 1
        r["math500_unique_id"] = m["unique_id"]
        r["math500_answer"] = m["answer"]
        m500_gold_eq += norm_boxed(m["answer"]) == norm_boxed(r["gold"])
    align["math500"] = {"rows": len(m500), "matched_by_exact_problem_text": m500_hit,
                        "math500_answer_eq_our_gold_after_norm": m500_gold_eq}
    print(align)

    OUT.mkdir(parents=True, exist_ok=True)
    full_p = OUT / f"vi_math_test_full_v{VERSION}.jsonl"
    write_jsonl(full_p, rows)
    inscope = [r for r in rows if r["subject"] in ("algebra", "prealgebra")]
    ins_p = OUT / f"vi_math_test_algprealg_v{VERSION}.jsonl"
    write_jsonl(ins_p, inscope)
    m5 = [r for r in inscope if r["math500_unique_id"]]
    m5_p = OUT / f"vi_math500_algprealg_v{VERSION}.jsonl"
    write_jsonl(m5_p, m5)
    queue.sort(key=lambda q: (q["priority"], q["subject"], q["index"]))
    q_p = OUT / "review_queue.jsonl"
    write_jsonl(q_p, queue)

    rc = Counter(re.split(r"[ :]", r)[0] + (":" + r.split(":")[1].split(" ")[0] if r.startswith("term:") else "")
                 for q in queue for r in q["reasons"])
    srcs = [{"path": rel(VI_DIR), "role": "Vietnamese MATH test, 7 subject files (input, unmodified)"},
            {"path": rel(EN_DIR), "role": "English MATH test (gold), aligned by position within subject"},
            {"path": rel(M500), "url": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500",
             "role": "MATH-500 membership (exact problem-text match)"}]
    tf = ["align VI row i <-> EN row i within subject; verified level/type equality and last-\\boxed agreement",
          "gold := last \\boxed{} of the EN solution (brace-balanced); gold_all_boxed keeps every EN box",
          "fix: JSON-escape damage CR/FF/BS/VT -> \\r \\f \\b \\v; TAB -> \\t only before a letter",
          "fix: literal two-char \\n -> newline (not before TeX command stems such as \\neq, \\nu, \\not)",
          "fix: bare $ amount re-escaped when EN has \\$<same amount> and VI $-parity is odd",
          "fix: prose EN thousands comma (54,321 -> 54.321) only for tokens also in EN prose; math spans untouched",
          "flag (no change): terminology forbid-list, [asy] edits/drops/moves, unit swaps, untranslated text, "
          "brace/$/LaTeX-command parity, multi-answer golds, VI boxed != EN gold, R2 hand annotations",
          "NO LLM used on any test item"]
    qc = {"queue_items": len(queue), "by_priority": dict(Counter(q["priority"] for q in queue)),
          "reason_counts": dict(rc),
          "algprealg_items_in_queue": sum(q["subject"] in ("algebra", "prealgebra") for q in queue),
          "problem_side_items": sum(q["priority"] <= 2 for q in queue)}
    common = dict(sources=srcs, license="MIT (MATH, Hendrycks et al.); translation: project", script=SCRIPT)
    datacard(full_p, name="Vi-MATH-test (full)", version=VERSION,
             description="5,000 Vietnamese MATH test items with structured EN gold, deterministic LaTeX repairs and "
                         "review flags. Grade against `gold` (EN), not the VI \\boxed.",
             transformations=tf, counts={"rows_in": 5000, "rows_out": len(rows), "flagged": len(queue),
                                         "repaired_rows": len(changed), "alignment_checks": align,
                                         "review_queue": qc}, changed=changed, **common)
    datacard(ins_p, name="Vi-MATH-test in-scope (algebra + prealgebra)", version=VERSION,
             description="C2 secondary eval: algebra + prealgebra subset of Vi-MATH-test v1.1.",
             transformations=tf + ["filter subject in {algebra, prealgebra}"],
             counts={"rows_out": len(inscope), "by_subject": dict(Counter(r["subject"] for r in inscope)),
                     "flagged": sum(r["review_status"] == "pending" for r in inscope)},
             changed=[c for c in changed if c["id"].split("/")[2] in ("algebra", "prealgebra")], **common)
    datacard(m5_p, name="Vi-MATH-500 alg+prealg slice", version=VERSION,
             description="Items of the in-scope slice that belong to MATH-500 (HuggingFaceH4/MATH-500), for "
                         "comparability with MATH-500 numbers.",
             transformations=tf + ["filter subject in {algebra, prealgebra}",
                                   "keep rows whose EN problem equals a MATH-500 problem (whitespace-normalised)"],
             counts={"rows_out": len(m5), "by_subject": dict(Counter(r["subject"] for r in m5)),
                     "flagged": sum(r["review_status"] == "pending" for r in m5),
                     "math500_alignment": align["math500"]},
             changed=[c for c in changed if any(c["id"] == r["id"] for r in m5)], **common)
    datacard(q_p, name="Vi-MATH-test review queue", version=VERSION,
             description="Meaning/structure defects for the Claude/human pass. Priority 1 = problem-side meaning or "
                         "gold risk, 2 = problem-side structure/terminology, 3 = solution-only.",
             transformations=["deterministic flags only"], counts=qc, changed=[], **common)
    print("rows", len(rows), "inscope", len(inscope), "m500 slice", len(m5), "changed", len(changed), qc)


def priority(reasons: list[str]) -> int:
    r = " | ".join(reasons)
    p1 = ("congruent->dong_dang in problem", "non_congruent", "asy_dropped", "asy_moved", "unit_swap",
          "problem_untranslated", "multi_answer_gold", "no_en_boxed_gold", "R2_annotation",
          "monic->don_thuc in problem", "median->trung_binh in problem", "dozen->chuc in problem",
          "counterclockwise->clockwise in problem", "problem_control_chars", "problem_length_ratio")
    if any(k in r for k in p1):
        return 1
    p2 = ("in problem", "asy_edited_in_problem", "problem_brace", "problem_dollar", "problem_new_latex",
          "problem_thousands")
    if any(k in r for k in p2):
        return 2
    return 3


def suggest(reasons: list[str], a_en: list[str], a_vi: list[str]) -> str | None:
    if "asy_edited_in_problem" in reasons and len(a_en) == len(a_vi):
        return "copy the English [asy] block(s) back verbatim into the VI problem"
    if any("congruent->dong_dang" in r for r in reasons):
        return "replace 'đồng dạng' (similar) by 'bằng nhau' (congruent) where EN says congruent"
    if any("non_congruent" in r for r in reasons):
        return "'không đồng dạng' -> 'không bằng nhau'"
    return None


if __name__ == "__main__":
    main()
