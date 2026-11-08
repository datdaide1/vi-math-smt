"""Rejection rules: answer tail (formal C3 claim), identity, clean answer, plausibility, uniqueness."""
import random
from fractions import Fraction as F

import pytest

from vi_math_verified_aug.symbolic.executor import execute
from vi_math_verified_aug.symbolic.fixtures import ALGEBRA_IDS, GSM8K_IDS, all_fixtures, load_fixture
from vi_math_verified_aug.symbolic.ir import ProgramBuilder
from vi_math_verified_aug.symbolic.legacy_ops import LEGACY_OPS, apply_legacy, legacy_wrap
from vi_math_verified_aug.symbolic.rules import (answer_issues, clean_answer, gate, is_answer_tail, is_identity,
                                                 is_semantic_tail, plausibility_issues, unique_solution)

FIX = all_fixtures()


# ------------------------------------------------------------------------- C3: old operators are tails
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("name", LEGACY_OPS)
def test_legacy_operators_are_answer_tails_on_every_fixture(name, in_place):
    """Formal C3 claim: wrap(answer), answer*(answer+1), answer+2h (src/mutation_informalize/mutation_engine.py),
    expressed in the IR, are ALL rejected by is_answer_tail — also when the old answer node is renamed and the
    query qid redefined in place."""
    for sid, P in FIX.items():
        for s in range(3):
            P2 = apply_legacy(name, P, random.Random(s), in_place=in_place)
            assert is_answer_tail(P, P2), (name, sid, in_place)
            assert gate(P, P2) is not None


def test_legacy_identity_wrap_is_also_an_identity():
    for sid, P in FIX.items():
        P2 = legacy_wrap(P, "(a+k)-k", F(7, 4))
        assert execute(P2)[P2.query] == execute(P)[P.query]
        assert is_identity(P, P2), sid


def test_legacy_ops_are_semantic_tails():
    """Independent value-level confirmation on forward programs."""
    for sid in GSM8K_IDS:
        P = FIX[sid]
        if P.unknowns():
            continue
        for name in LEGACY_OPS:
            if name == "wrap:(a+k)-k":
                continue  # constant in the old answer: no gradient to compare
            assert is_semantic_tail(P, apply_legacy(name, P, random.Random(0))), (sid, name)


def test_tail_rule_does_not_fire_on_story_combinations():
    """A new query = difference of two story nodes (no invented constant) is not a tail."""
    P = FIX["gsm8k-60"]
    P2 = P.copy()
    play = next(q.qid for q in P2.derived() if q.label_vi.startswith("số giờ xem một con gái thi đấu"))
    prac = next(q.qid for q in P2.derived() if q.label_vi.startswith("số giờ xem một con gái tập"))
    n = P2.add_quantity(None, "measure", "giờ", "chênh lệch", "derived")
    P2.add_op("sub", [prac, play], n)
    P2.query = n
    assert not is_answer_tail(P, P2)


def test_intermediate_tail_detected():
    """x + k on an old intermediate x (invented k) is a tail too."""
    P = FIX["gsm8k-3"]
    P2 = P.copy()
    rem = next(q.qid for q in P2.derived() if q.label_vi.startswith("số trang còn lại"))
    k = P2.add_quantity(5, "count", "trang", "k", "given", origin="inserted")
    n = P2.add_quantity(None, "count", "trang", "", "derived")
    P2.add_op("add", [rem, k], n)
    P2.query = n
    assert is_answer_tail(P, P2)


def test_inverse_problem_is_not_a_tail():
    P = FIX["gsm8k-70"]
    from vi_math_verified_aug.symbolic.operators import m2_inverse
    P2, _ = m2_inverse(P, random.Random(1))
    assert not is_answer_tail(P, P2)


# ------------------------------------------------------------------------- identity
def test_identity_same_program():
    P = FIX["gsm8k-20"]
    assert is_identity(P, P.copy())


def test_not_identity_when_values_change():
    P = FIX["gsm8k-20"]
    P2 = P.with_values({1: F(12)})
    assert not is_identity(P, P2)


# ------------------------------------------------------------------------- clean / plausible
def _single(text, lit, qtype, unit, value=None, kind="sub", other=None):
    b = ProgramBuilder("t", text)
    a = b.given(lit, qtype, unit, "a", value=value)
    c = b.given(other[0], other[1], other[2], "c") if other else None
    out = b.op(kind, [a, c], qtype, unit, "kết quả")
    return b.build(out)


def test_clean_answer_rules():
    P = _single("Giá 10,25 đô la, giảm 3 đô la. Hỏi?", "10,25", "money", "đô la",
                other=("3", "money", "đô la"))
    assert clean_answer(P)                              # 7.25 USD is on the 0.01 grid
    P = _single("Có 10 kg, chia 3 phần. Hỏi?", "10", "measure", "kg", kind="div",
                other=("3", "ratio", ""))
    assert any("2 decimals" in i for i in answer_issues(P))
    P = _single("Có 20 triệu đồng, mất 3 đồng. Hỏi?", "20 triệu", "money", "đồng",
                other=("3", "money", "đồng"))
    assert any(">= 1e7" in i or "grid" in i for i in answer_issues(P))


def test_clean_answer_requires_new_answer():
    P = FIX["gsm8k-0"]
    assert "answer equals the seed answer" in answer_issues(P.copy(), P)


def test_plausibility_ranges():
    P = FIX["gsm8k-40"]
    assert plausibility_issues(P) == []
    P2 = P.with_values({1: F(10 ** 9)})                  # a 1e9 USD concert ticket
    assert plausibility_issues(P2)


# ------------------------------------------------------------------------- uniqueness
def test_unique_forward_and_linear_system():
    assert unique_solution(FIX["gsm8k-0"])[0]
    assert unique_solution(FIX["gsm8k-150"]) == (True, "sympy")
    assert unique_solution(FIX["algebra-52"])[0]


def test_non_unique_quadratic_in_abstract_domain():
    t = "Tìm $x$ biết $x^2 = 9$."
    b = ProgramBuilder("quad", t, kind="symbolic")
    e = b.given("2", "abstract", "", "số mũ")
    nine = b.given("9", "abstract", "", "vế phải")
    x = b.unknown("abstract", "", "x")
    b.equation(b.op("pow", [x, e], "abstract", "", "x^2"), nine)
    P = b.build(x)
    ok, why = unique_solution(P)
    assert not ok and "2 solutions" in why


def test_count_domain_selects_positive_root():
    t = "Số học sinh bình phương bằng 9. Hỏi có 2 lớp?"
    b = ProgramBuilder("quad2", t)
    nine = b.given("9", "abstract", "", "vế phải")
    e = b.given("2", "abstract", "", "số mũ")
    x = b.unknown("count", "học sinh", "số học sinh")
    b.equation(b.op("pow", [x, e], "abstract", "", "x^2"), nine)
    P = b.build(x)
    assert unique_solution(P)[0]
    assert execute(P)[x] == 3


def test_underdetermined_integer_system_z3():
    t = "Có tổng cộng 27 bữa ăn và gói hàng. Hỏi?"
    b = ProgramBuilder("under", t)
    tot = b.given("27", "count", "đơn hàng", "tổng")
    m = b.unknown("count", "bữa ăn", "số bữa ăn")
    p = b.unknown("count", "gói hàng", "số gói hàng")
    b.equation(b.op("add", [m, p], "count", "đơn hàng", "tổng"), tot)
    P = b.build(m)
    ok, why = unique_solution(P)
    assert not ok and "z3" in why


def test_underdetermined_integer_system_with_unique_integer_point():
    """m + p = 2 with m >= 2 (seed constraint): exactly one non-negative integer solution, found by Z3."""
    t = "Có tổng cộng 2 món. Hỏi?"
    b = ProgramBuilder("under1", t)
    tot = b.given("2", "count", "món", "tổng")
    m = b.unknown("count", "món", "m")
    p = b.unknown("count", "món", "p")
    b.equation(b.op("add", [m, p], "count", "món", "tổng"), tot)
    b.constraint(f"q{m} >= 2")
    P = b.build(m)
    assert unique_solution(P) == (True, "z3")
