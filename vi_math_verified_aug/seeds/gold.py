"""Stage ``gold``: a typed gold for every seed (grader 3.0.2 ``infer_gold``) + EN/VI gold agreement.

* ``gold`` = ``infer_gold(EN ground truth)`` (GSM8K: the canonical ``#### N`` value; MATH: the English
  ``ground_truth``), ``locale="mixed"`` (the X3 eval-set convention), typed once here, never re-extracted.
* Agreement checks, recorded per seed:
  - ``agree_vi_field``: the Vietnamese ``ground_truth`` field (after the escape repair) graded against ``gold``;
  - ``agree_vi_solution``: the Vietnamese solution (its ``####`` line / last ``\\boxed{}``) graded against ``gold``;
  - ``agree_en_solution``: the English solution graded against ``gold`` (catches the MATH last-box truncation of
    multi-answer golds, R2 §4.4).
"""
from __future__ import annotations

import json
import re
from collections import Counter

from ..data_fix.common import all_boxed, canon_num, gsm8k_final
from ..eval.math_verify import GRADER_VERSION, grade, infer_gold
from .assemble import current_rows
from .common import WORK, write_jsonl


def _sol_pred(sol: str, dataset: str) -> str | None:
    if dataset == "gsm8k":
        f = gsm8k_final(sol)
        return None if f is None else "\\boxed{" + f + "}"
    return sol


def typed(r: dict) -> dict:
    raw = canon_num(r["gold_raw_en"]) if r["dataset"] == "gsm8k" else r["gold_raw_en"]
    return infer_gold(raw, problem=r["problem_en"], locale="mixed")


def _text_gold(en: str, vi: str) -> dict:
    return {"locale": "mixed", "answer_type": "text", "answer_canonical": en,
            "accept_also": [{"answer_type": "text", "answer_canonical": vi}]}


# Manual resolutions after reading every disagreement of the first gold pass (S3 report §3). "DROP" = the gold is
# not representable by grader 3.0.2 (a set of points), so the seed is dropped (freeze rule D3).
GOLD_FIXES: dict[tuple[str, int], tuple[object, str]] = {
    ("vi_math_train_algebra", 334): ("DROP", "two intersection points (-2,18),(8,38); ground_truth keeps only the last "
                                             "box and grader 3.0.2 cannot type a set of tuples"),
    ("vi_math_train_algebra", 504): ("DROP", "two intersection points (-2,23),(0,1); same as algebra 334"),
    ("vi_math_train_algebra", 409): ({"locale": "mixed", "answer_type": "set", "answer_canonical": "6; 8; 10"},
                                     "infer_gold read '6, 8\\text{, and }10' as {6, 810}; the answer is {6, 8, 10}"),
    ("vi_math_train_algebra", 1425): (None, "the first \\boxed{2} is an intermediate step; gold 4 confirmed"),
    ("vi_math_train_algebra", 1556): ({"locale": "mixed", "answer_type": "decimal", "answer_canonical": "8.24"},
                                      "'\\approx 8.24 \\text{ mph}' typed as expr; the answer is the decimal 8.24"),
    ("vi_math_train_algebra", 412): (_text_gold("Friday", "Thứ Sáu"), "text gold: Vietnamese day name accepted"),
    ("vi_math_train_algebra", 856): (_text_gold("Tuesday", "Thứ Ba"), "text gold: Vietnamese day name accepted"),
    ("vi_math_train_algebra", 867): (_text_gold("Sunday", "Chủ Nhật"), "text gold: Vietnamese day name accepted"),
    ("vi_math_train_algebra", 1486): (_text_gold("Sunday", "Chủ nhật"), "text gold: Vietnamese day name accepted"),
    ("vi_math_train_prealgebra", 92): (_text_gold("Wednesday", "Thứ Tư"), "text gold: Vietnamese day name accepted"),
    ("vi_math_train_prealgebra", 174): (_text_gold("August", "Tháng 8"), "text gold: Vietnamese month accepted"),
    ("vi_math_train_prealgebra", 838): (_text_gold("Softball, Kickball, Picnic", "Bóng mềm, Đá bóng, Dã ngoại"),
                                        "text gold: the Vietnamese ordering accepted"),
    ("vi_math_train_prealgebra", 1078): ({"locale": "mixed", "answer_type": "int", "answer_canonical": "4",
                                          "accept_also": [{"answer_type": "text", "answer_canonical": "four"},
                                                          {"answer_type": "text", "answer_canonical": "bốn"}]},
                                         "'\\mbox{four}' is the count 4"),
}


def run() -> dict:
    rows = current_rows()
    out, c = [], Counter()
    for r in rows:
        g = typed(r)
        fix = GOLD_FIXES.get((r["source"], r["index"]))
        gold_fix = None
        if fix is not None:
            gold_fix = {"action": "drop" if fix[0] == "DROP" else ("confirm" if fix[0] is None else "replace"),
                        "reason": fix[1], "inferred": g}
            if isinstance(fix[0], dict):
                g = fix[0]
        vf = r["gold_raw_vi"]
        if r["dataset"] == "gsm8k":
            vf = canon_num(re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", vf or ""))
        a_field = grade("\\boxed{" + str(vf) + "}", g)["correct"] if vf not in (None, "", "None") else None
        sp = _sol_pred(r["solution_vi"], r["dataset"])
        a_vi = grade(sp, g)["correct"] if sp else False
        sp_en = _sol_pred(r["solution_en"], r["dataset"])
        a_en = grade(sp_en, g)["correct"] if sp_en else False
        en_boxes = all_boxed(r["solution_en"]) if r["dataset"] == "math" else []
        multi = len({b.strip() for b in en_boxes}) > 1
        rec = {"seed_id": r["seed_id"], "source": r["source"], "index": r["index"], "gold": g,
               "agree_vi_field": a_field, "agree_vi_solution": a_vi, "agree_en_solution": a_en,
               "en_multi_box": multi, "en_boxes": en_boxes if multi else None, "gold_fix": gold_fix}
        out.append(rec)
        c["n"] += 1
        c[f"type:{g['answer_type']}"] += 1
        c["vi_field_disagree"] += a_field is False
        c["vi_solution_disagree"] += not a_vi
        c["en_solution_disagree"] += not a_en
        c["en_multi_box"] += multi
    write_jsonl(WORK / "gold.jsonl", out)
    summ = {"grader_version": GRADER_VERSION, **dict(c),
            "disagree_ids": [(o["source"], o["index"]) for o in out
                             if o["agree_vi_field"] is False or not o["agree_vi_solution"] or
                             not o["agree_en_solution"]][:200]}
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return summ
