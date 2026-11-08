"""Operators M1-M6 and renderer support. Property tests use seeded loops over all fixtures (no hypothesis dep)."""
import random
from fractions import Fraction as F

import pytest

from vi_math_verified_aug.symbolic.executor import check, execute, type_violations
from vi_math_verified_aug.symbolic.fixtures import ALGEBRA_IDS, GSM8K_IDS, all_fixtures
from vi_math_verified_aug.symbolic.ir import Program, ProgramBuilder
from vi_math_verified_aug.symbolic.operators import (NotApplicable, OPERATORS, m1_resample, m2_inverse,
                                                     m3_insert_step, m4_relation_change, m5_query_shift, m6_compose,
                                                     mutate)
from vi_math_verified_aug.symbolic.render import (TextEdit, apply_substitutions, parse_tags, render_substitutions,
                                                  strip_tags, tag_issues, tag_multiset_ok, tagged_text)
from vi_math_verified_aug.symbolic.rules import gate, is_answer_tail, is_identity
from vi_math_verified_aug.symbolic.validate import validate_program

FIX = all_fixtures()
PARTNERS = [FIX[k] for k in GSM8K_IDS]
N_SEEDS = 3


def _run(name, P, s):
    kw = {"partners": PARTNERS} if name == "M6" else {}
    return mutate(name, P, random.Random(1000 * s + 7), **kw)


# ------------------------------------------------------------------------- properties over all operators
@pytest.mark.parametrize("name", list(OPERATORS))
def test_property_no_operator_ever_produces_an_answer_tail(name):
    produced = 0
    for sid, P in FIX.items():
        for s in range(N_SEEDS):
            r = _run(name, P, s)
            if not r.ok:
                continue
            produced += 1
            P2 = r.program
            assert not is_answer_tail(P, P2), (name, sid, s)
            assert not is_identity(P, P2), (name, sid, s)
            assert check(P2, execute(P2)) == [], (name, sid, s)     # types hold at every node
            assert gate(P, P2, name) is None, (name, sid, s)
            for e in r.edits:                                     # the editor never sees the new answer
                d = e.to_editor_dict()
                assert "answer" not in d
    assert produced > 0, f"{name} never applied on the fixtures"


@pytest.mark.parametrize("name", ["M1", "M2", "M3", "M4", "M5"])
def test_property_new_answer_differs_from_seed(name):
    for sid, P in FIX.items():
        r = _run(name, P, 0)
        if r.ok:
            assert execute(r.program)[r.program.query] != execute(P)[P.query], (name, sid)


def test_property_operators_never_make_half_candies():
    """'0,5 cái kẹo' is impossible: every node with an indivisible unit stays integral under every operator."""
    for sid in GSM8K_IDS:
        P = FIX[sid]
        for name in OPERATORS:
            for s in range(N_SEEDS):
                r = _run(name, P, s)
                if r.ok:
                    assert type_violations(r.program, execute(r.program)) == [], (sid, name, s)


# ------------------------------------------------------------------------- M1
@pytest.mark.parametrize("sid", GSM8K_IDS)
def test_m1_rendered_text_revalidates(sid):
    """Deterministic M1 rendering: the substituted text re-anchors and passes formalizer acceptance."""
    P = FIX[sid]
    P2, edits = m1_resample(P, random.Random(3))
    assert all(e.kind == "SUBST" for e in edits) and edits
    R = render_substitutions(P2, edits)
    assert R.text == apply_substitutions(P.text, edits)
    assert validate_program(R, execute(P2)[P2.query]) == []


def test_m1_values_in_range_and_on_grid():
    P = FIX["gsm8k-40"]
    for s in range(20):
        P2, edits = m1_resample(P, random.Random(s))
        for e in edits:
            v0, v1 = P.quantities[e.qid].value, P2.quantities[e.qid].value
            assert v0 / 3 <= v1 <= 3 * v0 and v1 != v0


@pytest.mark.parametrize("sid", ["gsm8k-100", "gsm8k-80", "gsm8k-25"])
def test_m1_z3_handles_coupled_integrality(sid):
    """tries=0 forces the Z3 Int CSP path: divisibility chains (36 = 12k) and the seed equality constraint
    (100 + 75 == 175) are satisfied exactly."""
    P = FIX[sid]
    ints = [q.qid for q in P.givens() if q.qtype.value != "ratio" and not q.distractor]
    P2, edits = m1_resample(P, random.Random(5), tries=0, only=ints)
    assert P2.meta.get("m1_solver") == "z3"
    assert all(P2.quantities[q].value != P.quantities[q].value for q in ints)
    assert check(P2, execute(P2)) == []


def test_m1_never_breaks_seed_constraint():
    P = FIX["gsm8k-25"]
    for s in range(10):
        P2, _ = m1_resample(P, random.Random(s))
        v = {q.label_vi: q.value for q in P2.givens()}
        assert v["số bóng trong đợt đầu tiên"] + v["số bóng trong đợt tiếp theo"] == \
            v["số bóng tennis nạp vào máy lúc đầu"]


# ------------------------------------------------------------------------- M2
def test_m2_inverse_structure():
    P = FIX["gsm8k-70"]
    P2, edits = m2_inverse(P, random.Random(2))
    assert P2.quantities[P2.query].role == "unknown"
    assert P2.quantities[P.query].role == "given" and P2.quantities[P.query].origin == "stated_fact"
    assert [e.kind for e in edits] == ["UNKNOWN", "STATE_FACT", "NEW_QUESTION"]
    # the stated fact is the old query's value under the resampled given, the answer is that given
    env = execute(P2)
    assert env[P2.query] != P.quantities[P2.query].value


def test_m2_rejects_non_unique_inverse():
    """(2x+5)^2 = v asked for x has two real roots in the abstract domain: M2 must not return that inverse."""
    P = FIX["algebra-69"]
    x = next(q.qid for q in P.givens() if q.label_vi == "giá trị của x")
    P1 = P.with_values({x: F(4)})
    from vi_math_verified_aug.symbolic.operators import _make_inverse
    from vi_math_verified_aug.symbolic.rules import unique_solution
    assert not unique_solution(_make_inverse(P1, x, execute(P1)[P1.query]))[0]
    for s in range(5):
        r = mutate("M2", P, random.Random(s))
        if r.ok:
            assert r.program.query != x


# ------------------------------------------------------------------------- M3 / M4 / M5 / M6
def test_m3_inserts_strictly_upstream():
    P = FIX["gsm8k-3"]
    P2, edits = m3_insert_step(P, random.Random(4))
    assert P2.query == P.query
    new = [q for q in P2.quantities.values() if q.qid not in P.quantities]
    assert {q.origin for q in new} == {"inserted", "operator"}
    xp = next(q for q in new if q.role == "derived")
    assert xp.qid in P2.ancestors(P2.query)
    assert edits[0].kind == "INSERT_STEP" and edits[0].value


def test_m3_at_the_query_would_be_a_tail():
    """Inserting on the answer itself (forced) is exactly the old failure; the gate refuses it."""
    P = FIX["gsm8k-20"]
    with pytest.raises(NotApplicable):
        m3_insert_step(P, random.Random(0), x=P.query)


def test_m4_changes_one_relation():
    P = FIX["gsm8k-20"]
    P2, edits = m4_relation_change(P, random.Random(0))
    diff = [(a.kind, b.kind) for a, b in zip(P.ops, P2.ops) if a.kind != b.kind]
    assert len(diff) <= 1 and edits[0].kind == "REPLACE_RELATION"


def test_m5_targets_depth_two():
    P = FIX["gsm8k-40"]
    P2, edits = m5_query_shift(P, random.Random(0))
    assert P2.query != P.query and P2.depth(P2.query) >= 2
    assert edits[0].kind == "NEW_QUESTION" and edits[0].label


def test_m6_compose_uses_partner_material():
    P = FIX["gsm8k-40"]
    P2, edits = m6_compose(P, random.Random(0), PARTNERS)
    assert P.query in P2.ancestors(P2.query)
    assert any(q.origin == "partner" for q in P2.quantities.values() if q.qid in P2.ancestors(P2.query))
    assert [e.kind for e in edits] == ["JOIN", "REFERENCE"]
    assert not is_answer_tail(P, P2)


def test_operators_not_applicable_are_reported():
    t = "Lan có 3 cái kẹo và 4 cái kẹo. Hỏi có tất cả bao nhiêu?"
    b = ProgramBuilder("tiny", t)
    a = b.given("3", "count", "cái kẹo", "a")
    c = b.given("4", "count", "cái kẹo", "c")
    P = b.build(b.op("add", [a, c], "count", "cái kẹo", "tổng"))
    r = mutate("M5", P, random.Random(0))
    assert not r.ok and r.reason


# ------------------------------------------------------------------------- renderer support
def test_tagged_text_round_trip():
    P = FIX["gsm8k-0"]
    t = tagged_text(P)
    assert strip_tags(t) == P.text
    assert sorted(parse_tags(t)) == sorted((q.qid, P.literal(q.qid)) for q in P.givens())
    assert tag_multiset_ok(t, P, seed=P)


def test_tag_check_rejects_stray_numerals_and_missing_tags():
    P = FIX["gsm8k-0"]
    t = tagged_text(P)
    assert not tag_multiset_ok(t + " Thêm 7 cái nữa.", P, seed=P)          # untagged new number
    # known limit: untagged numerals are allowed BY VALUE, so a stray 5 passes because 'tháng 5' is in the seed
    assert tag_multiset_ok(t + " Thêm 5 cái nữa.", P, seed=P)
    assert not tag_multiset_ok(t.replace('<q n="1">48</q>', "48"), P, seed=P)   # missing tag
    bad = t.replace('<q n="1">48</q>', '<q n="1">49</q>')
    assert any("literal" in i for i in tag_issues(bad, P, seed=P))       # wrong value in a tag


def test_tag_check_allows_seed_dates():
    """'tháng 4', 'tháng 5' are untagged in the seed; keeping them is fine."""
    P = FIX["gsm8k-0"]
    assert "tháng 4" in strip_tags(tagged_text(P))
    assert tag_multiset_ok(tagged_text(P), P, seed=P)


def test_tag_check_on_m2_editor_output():
    P = FIX["gsm8k-70"]
    P2, edits = m2_inverse(P, random.Random(2))
    fact = next(e for e in edits if e.kind == "STATE_FACT")
    tg = {q.qid: q for q in P2.givens()}
    parts = [f'<q n="{k}">{fact.value if k == P.query else P.literal(k)}</q>' for k in tg]
    # simulated editor output: every given of P2 tagged with its value
    text = "Bài toán: " + ", ".join(parts) + ". Hỏi?"
    vals = {k: v.value for k, v in tg.items()}
    ok_text = text
    for k, v in vals.items():
        if k != P.query and P.literal(k) is not None:
            from vi_math_verified_aug.symbolic.numerals import fmt_like
            ok_text = ok_text.replace(f'<q n="{k}">{P.literal(k)}</q>', f'<q n="{k}">{fmt_like(v, P.literal(k))}</q>')
    assert tag_multiset_ok(ok_text, P2, seed=P), tag_issues(ok_text, P2, seed=P)


def test_apply_substitutions_rejects_overlap():
    with pytest.raises(ValueError):
        apply_substitutions("abc 12 def", [TextEdit("SUBST", span=(4, 6), text="3"),
                                           TextEdit("SUBST", span=(5, 6), text="4")])
