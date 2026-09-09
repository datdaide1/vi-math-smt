"""T0.2 acceptance: hand-labelled cases for eval/math_verify.py.

Covers integers, decimals, fractions, \\frac, radicals, tuples/coords, \\boxed vs
bare, LaTeX noise, unit suffixes, and the Vietnamese decimal-comma / thousands-dot
traps.
"""
import pytest

from vi_math_verified_aug.eval.math_verify import (
    extract_answer,
    is_correct,
    normalize_answer,
)

# --------------------------------------------------------------------------- #
# (pred, gold, expected)
# --------------------------------------------------------------------------- #
CASES = [
    # --- integers ---
    ("42", "42", True),
    ("42", "43", False),
    ("The answer is 120.", "120", True),
    ("-7", "-7", True),
    ("007", "7", True),
    # --- \boxed extraction ---
    (r"... nên \(x=365\). Vậy \(\boxed{365}\).", "365", True),
    (r"\boxed{\dfrac{11}{3}}", "11/3", True),
    (r"blah \boxed{365 \text{ nghìn đồng}}", "365", True),
    (r"\boxed{365 \text{ nghìn đồng}; 520 \text{ nghìn đồng}}", "365; 520", True),
    # --- #### GSM8K ---
    ("Chi tiết...\n#### 18", "18", True),
    ("... #### 18", "19", False),
    # --- decimals & Vietnamese decimal comma ---
    ("3,14", "3.14", True),
    ("3.14", "3,14", True),
    (r"\boxed{2,5}", "2.5", True),
    ("0,5", "1/2", True),
    # --- Vietnamese thousands dot ---
    ("1.000.000", "1000000", True),
    ("1.000.000", "1000001", False),
    ("giá là 885.000 đồng", "885000", True),
    # --- simple fractions ---
    ("1/2", "0.5", True),
    ("7/2", "3.5", True),
    ("2/4", "1/2", True),
    (r"\frac{7}{2}", "3.5", True),
    (r"\dfrac{-3}{4}", "-0.75", True),
    (r"\frac{1}{3}", "0.333", False),  # 3-dp gold is not exactly 1/3
    # --- radicals ---
    (r"\sqrt{2}", "1.4142135623730951", True),
    (r"2\sqrt{3}", r"\sqrt{12}", True),
    (r"\boxed{\sqrt{9}}", "3", True),
    # --- tuples / coordinates ---
    (r"\left( \frac{7}{2}; 1 \right)", "(3.5, 1)", True),
    ("(1, 2)", "(1, 3)", False),
    ("x = -1", "-1", True),
    ("(x;y) = (2;3)", "(2, 3)", True),
    # --- units / text noise ---
    ("51 đô la", "51", True),
    ("120 km", "120", True),
    (r"\boxed{682\ \text{nghìn đồng}}", "682", True),
    ("Vậy quãng đường là 120 km.", "120", True),
    ("Diện tích là 15,5 m²", "15.5", True),
    # --- percent / degree ---
    (r"\boxed{20\%}", "20", True),
    (r"45^\circ", "45", True),
    # --- expression equivalence ---
    ("2*x + 3*x", "5*x", True),
    (r"\frac{x}{2} + \frac{x}{2}", "x", True),
    # --- clearly wrong / empty ---
    ("", "5", False),
    ("không biết", "5", False),
]


@pytest.mark.parametrize("pred,gold,expected", CASES)
def test_is_correct(pred, gold, expected):
    assert is_correct(pred, gold) is expected


def test_extract_answer_priority():
    assert extract_answer(r"Vậy x = 5. \boxed{7}") == "7"
    assert extract_answer("bước cuối\n#### 42\n") == "42"
    assert extract_answer("Đáp án: 365 nghìn đồng") == "365 nghìn đồng"
    assert extract_answer("tổng cộng có 12 quả táo và 3 quả cam") == "3"
    assert extract_answer("") is None


def test_normalize_answer():
    assert normalize_answer(r"\boxed{}") == ""
    assert normalize_answer("1.000.000") == "1000000"
    assert normalize_answer("3,14") == "3.14"
    assert normalize_answer(r"365\ \text{nghìn đồng}") == "365"
    assert normalize_answer("x = 42") == "42"
