"""Task W3-1: Vi-GSM8K test -> v1.1 full (1,319) + Vi-GSM8K-Platinum (1,209) + review queue.

No LLM is called on test items (EXEC_BRIEF). Fixes applied here are deterministic only:
  * ``#### 2,125`` final lines (English thousands comma kept in VI) -> ``#### 2125``.
  * calculator annotation ``<<a*b=c>>`` restored from English when the VI annotation is
    arithmetically wrong and the English one at the same position is right (R1: test 154).
Everything meaning-related goes to the review queue (id, EN, VI, reasons).

Run: python -X utf8 -I vi_math_verified_aug/data_fix/build_gsm8k_test.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vi_math_verified_aug.data_fix.common import (  # noqa: E402
    CLEAN, DATA, SOURCES, canon_num, datacard, gsm8k_final, load_annotations, number_multiset_diff,
    read_json, read_jsonl, rel, write_jsonl,
)

VERSION_FULL = "1.1"
VERSION_PLAT = "1.0"
OUT = CLEAN / "vi_gsm8k_test"
SCRIPT = "vi_math_verified_aug/data_fix/build_gsm8k_test.py"

VI_PATH = DATA / "test" / "gsm8k" / "gsm8k_test_vietnamese.jsonl"
EN_PATH = DATA / "generate" / "gsm8k" / "test.json"
PLAT_PATH = SOURCES / "gsm8k_platinum_test.jsonl"

# ---------------------------------------------------------------- R1 pattern list (§5, §6)
EN_PATTERNS = {
    # CR-2: "N of the remaining" -> R1 found 4/4 occurrences mistranslated as "N X còn lại"
    "of_the_remaining": re.compile(r"(?<![/\d])\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|half|some|all|most)"
                                   r" of the remaining\b", re.I),
    # CR-3: additive-of-multiple "N more/less than twice / k times"
    "n_more_than_k_times": re.compile(r"\b(\d+|\w+) (more|less|fewer) than (twice|double|triple|half|thrice|"
                                      r"\w+ times|\d+ times)\b", re.I),
    # CR-5 strictness
    "strictly": re.compile(r"\bstrictly\b|\bat least\b|\bat most\b|\bno more than\b|\bno less than\b", re.I),
    # CR-6 fraction multiplier (ambiguous even in EN)
    "fraction_times": re.compile(r"\b\d+/\d+ times (more|as|less)\b", re.I),
    # CR-4 direction
    "subtracted_from": re.compile(r"\bsubtracted from\b|\bless than (?:half|twice|\d)", re.I),
}
# CR-1 anchor: EN "N more X than Y" while VI puts the number right after "hơn" (anchor may be dropped)
EN_CR1 = re.compile(r"\b(\d+(?:\.\d+)?|\w+) (more|fewer|less) (?:[\w$-]+ ){0,3}?than\b", re.I)
VI_CR1 = re.compile(r"(nhiều|ít|lớn|nhỏ|cao|thấp|dài|ngắn|nặng|nhẹ|đắt|rẻ|già|trẻ|nhanh|chậm|xa|muộn|sớm) hơn"
                    r" (\$?\d[\d.,]*|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\b", re.I)
VI_CALQUE = re.compile(r"(nhiều|ít) hơn \$?\d[\d.,]* [^,.;:?]{0,40}? so với", re.I)
VI_K_LAN = re.compile(r"(nhiều|ít) hơn \d+ lần", re.I)
VI_REMAINING = re.compile(r"(?<![/\w])(?<!phần )(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+[^\s,.]+(?:\s+[^\s,.]+)?\s+còn lại",
                          re.I)
# units: imperial in EN, metric/thước in VI -> conversion or mistranslation risk (R1 6922)
EN_IMPERIAL = re.compile(r"\b(yards?|feet|foot|inch(?:es)?|miles?|gallons?|quarts?|pints?|ounces?|pounds?|lbs?)\b",
                         re.I)
EN_METRIC = re.compile(r"\b(meters?|metres?|centimet\w*|kilomet\w*|liters?|litres?|grams?|kilograms?|kg|km|cm|ml)\b",
                       re.I)
VI_METRIC = re.compile(r"(?<!kích )\b(mét|thước(?! Anh)|lít|gam|ki-lô-gam|kilogam|kg|km|cm|ml)\b", re.I)
EN_LEAK = re.compile(r"\b(the|and|of|how many|what is|is|are|was|were|each|every|per|with|from|has|have)\b")
# (feet/inch/yard/pound/gallon are kept as transliterated units by policy (T4 D9) and are not leaks)
EN_UNITS_LEFT = re.compile(r"\b(miles|bails?|hours?|minutes|seconds|dollars|cents|degrees|apples|cookies|"
                           r"each|per day|per hour|total)\b",
                           re.I)
ANNOT_RE = re.compile(r"<<([^<>]*)>>")


def annot_ok(a: str) -> bool:
    try:
        lhs, rhs = a.split("=", 1)
        lhs = lhs.replace("x", "*").replace("^", "**")
        return abs(eval(lhs, {"__builtins__": {}}) - float(rhs)) < 1e-6  # noqa: S307 - arithmetic only
    except Exception:
        return True  # unparsable -> do not touch


def en_leak_score(vi_q: str) -> int:
    t = re.sub(r"\$[^$]*\$", " ", vi_q)
    return len(EN_LEAK.findall(t))


def main() -> None:
    vi = read_jsonl(VI_PATH)
    en = read_json(EN_PATH)
    plat = read_jsonl(PLAT_PATH)
    assert len(vi) == len(en) == 1319, (len(vi), len(en))
    ann = load_annotations("vi_gsm8k_test")

    # ---------------- alignment checks (VI<->EN by position; Platinum<->EN by exact question text)
    gt_eq = sum(canon_num(v["ground_truth"]) == canon_num(e["ground_truth"]) for v, e in zip(vi, en))
    en_final_eq = sum(canon_num(gsm8k_final(e["answer"])) == canon_num(e["ground_truth"]) for e in en)
    q2i = {e["question"].strip(): i for i, e in enumerate(en)}
    assert len(q2i) == len(en), "duplicate EN questions"
    plat_idx = [q2i.get(p["question"].strip()) for p in plat]
    assert None not in plat_idx, "platinum row without exact EN match"
    assert all(a < b for a, b in zip(plat_idx, plat_idx[1:])), "platinum order not monotone"
    p_by_i = {i: p for i, p in zip(plat_idx, plat)}
    checks = {
        "vi_rows": len(vi), "en_rows": len(en), "platinum_rows": len(plat),
        "vi_ground_truth_eq_en_ground_truth": gt_eq,
        "en_####_eq_en_ground_truth": en_final_eq,
        "platinum_exact_question_match": len(plat) - plat_idx.count(None),
        "platinum_order_monotone": True,
        "platinum_status": dict(Counter(p["cleaning_status"] for p in plat)),
    }
    print("checks", checks)

    rows, queue, changed_full = [], [], []
    for i, (v, e) in enumerate(zip(vi, en)):
        rid = f"gsm8k/test/{i}"
        p = p_by_i.get(i)
        status = p["cleaning_status"] if p else "removed"
        gold_orig = canon_num(e["ground_truth"])
        gold_plat = canon_num(gsm8k_final(p["answer"])) if p else None
        q_vi, a_vi = v["question"], v["answer"]
        reasons: list[str] = []
        change: list[str] = []

        # ---- deterministic fix 1: "#### 2,125" -> "#### 2125"
        fin = gsm8k_final(a_vi)
        if fin is not None and fin != canon_num(fin) and canon_num(fin) == gold_orig:
            a_vi = re.sub(r"####\s*.+?\s*$", "#### " + canon_num(fin), a_vi)
            change.append(f"final line '#### {fin}' -> '#### {canon_num(fin)}' (thousands separator)")

        # ---- deterministic fix 2: calculator annotations
        va, ea = ANNOT_RE.findall(a_vi), ANNOT_RE.findall(e["answer"])
        if va != ea:
            if len(va) == len(ea):
                for x, y in zip(va, ea):
                    if x != y and not annot_ok(x) and annot_ok(y):
                        a_vi = a_vi.replace(f"<<{x}>>", f"<<{y}>>", 1)
                        change.append(f"calculator annotation <<{x}>> -> <<{y}>> (VI arithmetic wrong; EN restored)")
                    elif x != y:
                        reasons.append(f"sol_annotation_diff: VI <<{x}>> vs EN <<{y}>>")
            else:
                reasons.append(f"sol_annotation_count: VI {len(va)} vs EN {len(ea)}")

        # ---- deterministic flags (question side)
        nd = number_multiset_diff(e["question"], q_vi)
        if nd["missing_set"] or nd["added_set"]:
            reasons.append(f"number_set: value absent on other side: missing_in_vi={nd['missing_set']} "
                           f"added_in_vi={nd['added_set']}")
        elif nd["missing_in_vi"] or nd["added_in_vi"]:
            reasons.append(f"number_count: repetition differs: missing_in_vi={nd['missing_in_vi']} "
                           f"added_in_vi={nd['added_in_vi']}")
        for name, rx in EN_PATTERNS.items():
            m = rx.search(e["question"])
            if m:
                reasons.append(f"en_pattern:{name} '{m.group(0)}'")
        if EN_CR1.search(e["question"]) and VI_CR1.search(q_vi):
            reasons.append(f"cr1_anchor_risk: EN '{EN_CR1.search(e['question']).group(0)}' / VI "
                           f"'{VI_CR1.search(q_vi).group(0)}'")
        if VI_CALQUE.search(q_vi):
            reasons.append(f"vi_calque: '{VI_CALQUE.search(q_vi).group(0)}'")
        if VI_K_LAN.search(q_vi):
            reasons.append(f"vi_more_than_k_times: '{VI_K_LAN.search(q_vi).group(0)}'")
        if EN_PATTERNS["of_the_remaining"].search(e["question"]):
            m = VI_REMAINING.search(q_vi)
            if m or "trong số" not in q_vi:
                reasons.append(f"vi_N_X_con_lai: EN 'N of the remaining' but VI lacks 'trong số'"
                               f"{' / ' + repr(m.group(0)) if m else ''}")
        if EN_IMPERIAL.search(e["question"]) and VI_METRIC.search(q_vi) and not EN_METRIC.search(e["question"]):
            reasons.append(f"unit_swap_risk: EN '{EN_IMPERIAL.search(e['question']).group(0)}' / VI "
                           f"'{VI_METRIC.search(q_vi).group(0)}'")
        if en_leak_score(q_vi) >= 3:
            reasons.append(f"english_leak_question: {en_leak_score(q_vi)} EN stopwords")
        if EN_UNITS_LEFT.search(a_vi):
            reasons.append(f"english_leak_solution: '{EN_UNITS_LEFT.search(a_vi).group(0)}'")
        lr = len(q_vi) / max(1, len(e["question"]))
        if lr < 0.6 or lr > 2.0:
            reasons.append(f"length_ratio {lr:.2f}")
        a = ann.get(i)
        if a and a.get("severity") in ("major", "critical"):
            reasons.append(f"R1_annotation:{a['severity']}:{','.join(a['codes'])}: {a['note']}")
        if status == "revised":
            reasons.append(f"platinum_revised: gold {gold_orig} -> {gold_plat}; VI question unchanged, VI solution "
                           f"is a translation of the original (now wrong) EN solution")

        row = {
            "id": rid, "index": i,
            "question": q_vi, "answer": a_vi,
            "ground_truth": gold_orig,
            "en_question": e["question"], "en_answer": e["answer"],
            "platinum_status": status,
            "platinum_ground_truth": gold_plat,
            "flags": reasons,
            "review_status": "pending" if reasons else "auto_pass",
        }
        rows.append(row)
        if change:
            changed_full.append({"id": rid, "fields": ["answer"], "reason": "; ".join(change)})
        if reasons:
            queue.append({"id": rid, "index": i, "platinum_status": status,
                          "en_question": e["question"], "vi_question": q_vi,
                          "en_answer": p["answer"] if p else e["answer"], "vi_answer": a_vi,
                          "gold": gold_plat if gold_plat is not None else gold_orig,
                          "reasons": reasons,
                          "priority": priority(reasons, status)})

    OUT.mkdir(parents=True, exist_ok=True)
    full_p = OUT / f"vi_gsm8k_test_full_v{VERSION_FULL}.jsonl"
    write_jsonl(full_p, rows)

    plat_rows, changed_plat = [], []
    for r in rows:
        if r["platinum_status"] == "removed":
            continue
        pr = dict(r)
        pr["ground_truth"] = r["platinum_ground_truth"]
        pr["en_answer"] = p_by_i[r["index"]]["answer"]
        pr["original_ground_truth"] = r["ground_truth"]
        if r["platinum_status"] == "revised":
            pr["vi_answer_stale"] = True  # VI solution translates the original EN solution
            changed_plat.append({"id": r["id"], "fields": ["ground_truth", "en_answer"],
                                 "reason": f"platinum revised gold {r['ground_truth']} -> {pr['ground_truth']}"})
        plat_rows.append(pr)
    plat_p = OUT / f"vi_gsm8k_platinum_v{VERSION_PLAT}.jsonl"
    write_jsonl(plat_p, plat_rows)
    assert len(plat_rows) == 1209

    queue.sort(key=lambda q: (q["priority"], q["index"]))
    q_p = OUT / "review_queue.jsonl"
    write_jsonl(q_p, queue)

    reason_counts = Counter(r.split(":")[0].split(" ")[0] for q in queue for r in q["reasons"])
    srcs = [{"path": rel(VI_PATH), "role": "Vietnamese translation (input, unmodified)"},
            {"path": rel(EN_PATH), "role": "English GSM8K test (gold, alignment by position)"},
            {"path": rel(PLAT_PATH), "url": "https://huggingface.co/datasets/madrylab/gsm8k-platinum",
             "role": "GSM8K-Platinum test (status + revised gold), aligned by exact question text"}]
    common_tf = [
        "align VI row i <-> EN GSM8K test row i (position; VI ground_truth == EN ground_truth checked)",
        "align Platinum rows to EN by exact question text (1209/1209 exact, order monotone)",
        "ground_truth canonicalised (thousands commas stripped)",
        "fix: VI '#### 2,125'-style final lines -> '#### 2125'",
        "fix: VI calculator annotation restored from EN when VI arithmetic is wrong and EN is right",
        "flag (no change): R1 pattern list (of the remaining, N more than k times, strictly, fraction-times,"
        " CR-1 anchor risk, VI calque, 'nhiều hơn N lần', 'N X còn lại', unit swap, English leak, length ratio),"
        " EN<->VI digit-number multiset mismatch, R1 hand annotations (major/critical), Platinum revised",
        "NO LLM used on any test item",
    ]
    q_counts = {"queue_items": len(queue), "by_priority": dict(Counter(q["priority"] for q in queue)),
                "reason_counts": dict(reason_counts),
                "in_platinum": sum(q["platinum_status"] != "removed" for q in queue)}
    datacard(full_p, name="Vi-GSM8K-test (full)", version=VERSION_FULL,
             description="Full 1,319-item Vietnamese GSM8K test with EN source, original GSM8K gold, Platinum status "
                         "and deterministic review flags. Use `ground_truth` (never parse ####).",
             sources=srcs, transformations=common_tf,
             counts={"rows_in": len(vi), "rows_out": len(rows), "flagged": len(queue),
                     "auto_pass": len(rows) - len(queue), "alignment_checks": checks, "review_queue": q_counts},
             changed=changed_full, license="MIT (GSM8K); translation: project", script=SCRIPT)
    datacard(plat_p, name="Vi-GSM8K-Platinum", version=VERSION_PLAT,
             description="1,209 Vi-GSM8K test items kept by GSM8K-Platinum (110 removed as flawed); ground_truth is the "
                         "Platinum gold (10 revised). `vi_answer_stale` marks revised rows whose VI solution "
                         "translates the old solution. Rows with review_status=pending await the Claude/human pass.",
             sources=srcs, transformations=common_tf + [
                 "drop 110 rows not in Platinum (status 'removed')",
                 "ground_truth := Platinum '####' gold; original kept as original_ground_truth"],
             counts={"rows_in": len(rows), "rows_out": len(plat_rows),
                     "removed": sum(r["platinum_status"] == "removed" for r in rows),
                     "status": dict(Counter(r["platinum_status"] for r in plat_rows)),
                     "flagged": sum(r["review_status"] == "pending" for r in plat_rows)},
             changed=changed_plat + [c for c in changed_full if p_by_i.get(int(c["id"].split("/")[-1]))],
             license="MIT (GSM8K / GSM8K-Platinum); translation: project", script=SCRIPT)
    datacard(q_p, name="Vi-GSM8K-test review queue", version=VERSION_FULL,
             description="Items needing a Claude/human pass (id, EN, VI, reasons). Priority 1 = critical evidence "
                         "(R1 annotation, of-the-remaining, Platinum revised, number mismatch, k-times), "
                         "2 = relational-pattern risk, 3 = style/solution-only.",
             sources=srcs, transformations=["deterministic flags only"], counts=q_counts, changed=[],
             license="MIT (GSM8K)", script=SCRIPT)
    print("full", len(rows), "platinum", len(plat_rows), "queue", len(queue), q_counts)
    print("changed (full)", changed_full)


def priority(reasons: list[str], status: str) -> int:
    r = " ".join(reasons)
    if any(k in r for k in ("R1_annotation", "of_the_remaining", "platinum_revised", "number_set",
                            "vi_N_X_con_lai", "vi_more_than_k_times", "unit_swap_risk")):
        return 1
    if any(k in r for k in ("n_more_than_k_times", "cr1_anchor_risk", "vi_calque", "strictly", "fraction_times",
                            "subtracted_from", "length_ratio", "english_leak_question", "number_count")):
        return 2
    return 3


if __name__ == "__main__":
    main()
