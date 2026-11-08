"""Q1 (Vi-ExamMath answer-key QA): key re-typing, gold validation, status rule, audit bound. No API calls."""
from vi_math_verified_aug.vi_exam_qa import finalize as F
from vi_math_verified_aug.vi_exam_qa import keys as K


def test_clean_key_mathtype_braces():
    assert K.grade_ok(K.boxed(r"\left( {x;y} \right) = \left( { - 4;5} \right)"),
                      {"answer_type": "tuple", "answer_canonical": "-4; 5", "vars": ["x", "y"], "locale": "vi"})
    assert K.grade_ok(K.boxed(r"\left\{ { - 6;1} \right\}"), {"answer_type": "set", "answer_canonical": "-6; 1"})
    # a partial page key must not match a two-root gold
    assert not K.grade_ok(K.boxed("7"), {"answer_type": "set", "answer_canonical": "2; 7"})


def test_sympy_gold_types():
    g = K.sympy_gold({"key_type": "solve_system", "variables": ["x", "y"], "values": [["2", "1"]]})
    assert g["answer_type"] == "tuple" and g["vars"] == ["x", "y"]
    assert K.grade_ok(r"\boxed{(x;y)=(2;1)}", g)
    g = K.sympy_gold({"key_type": "solve_equation", "values": ["-sqrt(73)/6 - 7/6", "-7/6 + sqrt(73)/6"]})
    assert g["answer_type"] == "set"
    assert K.grade_ok(r"\boxed{x = \frac{-7 \pm \sqrt{73}}{6}}", g) in (True, False)  # parsed without crashing
    assert K.grade_ok(r"\boxed{\frac{-7+\sqrt{73}}{6}; \frac{-7-\sqrt{73}}{6}}", g)
    g = K.sympy_gold({"key_type": "simplify_equivalence", "values": ["(x - 1)/(sqrt(x) - 1)"]})
    assert g["role"] == "equivalence"
    assert F.sympy_match(K.sympy_gold({"key_type": "simplify_equivalence", "values": ["(x - 1)/(sqrt(x) - 1)"]}),
                         {"answer_type": "expr", "answer_canonical": r"\sqrt{x}+1", "locale": "vi"})


def test_make_gold_validates_schema():
    assert F.make_gold({"answer_type": "set", "answer_canonical": "-1; 3"})["answer_type"] == "set"
    assert F.make_gold({"answer_type": "bogus", "answer_canonical": "1"}) is None
    g = F.make_gold({"answer_type": "int", "answer_canonical": 24, "percent": False,
                     "accept_also": [{"answer_type": "decimal", "answer_canonical": "24.0"}]})
    assert g["answer_canonical"] == "24" and "percent" not in g and len(g["accept_also"]) == 1


def test_parse_json_takes_last_block():
    t = "analysis {\"a\": 1}\n```json\n{\"valid\": true, \"confidence\": \"high\"}\n```"
    assert F.parse_json(t) == {"valid": True, "confidence": "high"}


def test_clopper_pearson_matches_t5():
    assert abs(F.cp_upper(0, 150) - 0.0198) < 5e-4      # rule of three ~ 2.0 %
    assert abs(F.cp_upper(1, 150) - 0.0312) < 5e-4
    assert abs(F.cp_upper(2, 150) - 0.0414) < 5e-4
