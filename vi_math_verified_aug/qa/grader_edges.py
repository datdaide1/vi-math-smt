"""QA-T: grader edge cases on the exact response format the SFT template produces (VI and EN).

    python -m vi_math_verified_aug.qa.grader_edges            # active backend (sympy-fallback on the laptop)
    VI_GRADER_BACKEND=sympy python -m ...                     # force the Kaggle-snapshot backend
"""
from __future__ import annotations

import sys

from vi_math_verified_aug.eval import math_verify as mv
from vi_math_verified_aug.train import reward as R
from vi_math_verified_aug.train import template as T


def G(v, t, loc="mixed"):
    return {"answer_canonical": v, "answer_type": t, "locale": loc}


B = "\\boxed"
CASES = [
    # (response, gold, expected)  -- responses end the way SFT targets end: "Đáp số: \boxed{..}"
    (r"... " + "\nĐáp số: " + B + "{12}", G("12", "int"), True),
    ("...\nĐáp số: " + B + "{12.}", G("12", "int"), True),
    ("...\nFinal answer: " + B + "{12}", G("12", "int", "en"), True),
    ("Vậy x = 2,5\nĐáp số: " + B + "{2,5}", G("2.5", "decimal", "vi"), True),
    ("Đáp số: " + B + "{2{,}5}", G("2.5", "decimal", "vi"), True),
    ("Đáp số: " + B + "{1.500}", G("1500", "int", "vi"), True),
    ("Đáp số: " + B + "{1,500}", G("1500", "int", "en"), True),
    ("Đáp số: " + B + "{1,500}", G("1500", "int"), True),
    ("Đáp số: " + B + "{1.5}", G("1500", "int"), False),
    ("Đáp số: " + B + r"{\$18}", G("18", "int"), True),
    ("Đáp số: " + B + "{18 đồng}", G("18", "int"), True),
    ("Đáp số: " + B + r"{18 \text{ km}}", G("18", "int"), True),
    ("Đáp số: " + B + r"{20\%}", G("20", "int"), True),
    ("Đáp số: " + B + r"{\frac{3}{4}}", G("3/4", "rational"), True),
    ("Đáp số: " + B + "{0.75}", G("3/4", "rational"), True),
    ("Đáp số: " + B + "{3/4}", G("0.75", "decimal"), True),
    ("Đáp số: " + B + r"{\dfrac{3}{4}}", G("3/4", "rational"), True),
    ("Đáp số: " + B + "{x = 3}", G("3", "int"), True),
    ("Đáp số: " + B + r"{2\sqrt{3}}", G(r"2\sqrt{3}", "radical"), True),
    ("Đáp số: " + B + r"{\sqrt{12}}", G(r"2\sqrt{3}", "radical"), True),
    ("Đáp số: " + B + "{(1; 2)}", G("1; 2", "tuple", "vi"), True),
    ("Đáp số: " + B + r"{x \ge 2}", G(r"[2,\infty)", "interval", "vi"), True),
    ("Đáp số: " + B + "{-5}", G("-5", "int"), True),
    ("Đáp số: " + B + "{\u22125}", G("-5", "int"), True),
    ("Đáp số: " + B + "{12}", "12", True),          # legacy string gold (SFT quick-eval path)
    ("Đáp số: " + B + "{1.250}", "1250", True),
    ("Đáp số: " + B + "{0,5}", "0.5", True),
    ("Đáp số: " + B + "{5}\n\nBài toán:\nAn có 3 quả táo.\n\nLời giải:\n...\nĐáp số: " + B + "{7}", G("5", "int"),
     "uncut"),  # GRPO reward sees the raw completion (no cut_at_stop)
]


def main():
    bad = 0
    print("grader", mv.GRADER_VERSION, mv.BACKEND)
    for resp, gold, want in CASES:
        if want == "uncut":
            raw = R.score(resp, gold, None)["correct"]
            cut = R.score(T.cut_at_stop(resp), gold, None)["correct"]
            print(f"{'BAD' if raw != cut else 'OK '} continuation: raw-graded={raw} cut-graded={cut}")
            bad += raw != cut
            continue
        v = R.score(resp, gold, None)
        ok = v["correct"] == want
        bad += not ok
        g = gold if isinstance(gold, str) else gold["answer_canonical"]
        print(f"{'OK ' if ok else 'BAD'} want={want!s:5} got={v['correct']!s:5} span={v['span']!r:22} "
              f"flag={v['flag']} | gold={g}")
    print(f"{bad} unexpected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
