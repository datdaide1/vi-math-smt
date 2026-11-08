"""Regression tests for the minimal X4 fixes made during the G3 symbolic pilot (exec/G3_symbolic_pilot.md §6).

Each bug was found on real Haiku-formalized Vi-GSM8K seeds (data/pilot/symbolic/run_a_x4_as_delivered): programs
built by ``from_expr_json`` carry no ``verb_vi`` and may contain anonymous intermediate nodes, which the hand-written
fixtures never had.
"""
import random

import pytest

from vi_math_verified_aug.symbolic.fixtures import all_fixtures
from vi_math_verified_aug.symbolic.ir import from_expr_json
from vi_math_verified_aug.symbolic.operators import NotApplicable, _OP_VI, m3_insert_step, m4_relation_change

TEXT = ("Lan có 6 hộp, mỗi hộp có 5 quả táo. Mai có 4 hộp, mỗi hộp có 3 quả táo. "
        "Hỏi cả hai bạn có tất cả bao nhiêu quả táo?")


def _apples(expr_total: str = "d1 + d2"):
    j = {"quantities": [
        {"id": "a", "role": "given", "kind": "count", "unit": "hộp", "label": "số hộp của Lan", "value": "6",
         "span": "6", "context": "Lan có 6 hộp"},
        {"id": "b", "role": "given", "kind": "count", "unit": "quả táo", "label": "số táo mỗi hộp của Lan",
         "value": "5", "span": "5", "context": "mỗi hộp có 5 quả"},
        {"id": "c", "role": "given", "kind": "count", "unit": "hộp", "label": "số hộp của Mai", "value": "4",
         "span": "4", "context": "Mai có 4 hộp"},
        {"id": "d", "role": "given", "kind": "count", "unit": "quả táo", "label": "số táo mỗi hộp của Mai",
         "value": "3", "span": "3", "context": "mỗi hộp có 3 quả"},
        {"id": "d1", "role": "derived", "kind": "count", "unit": "quả táo", "label": "số táo của Lan", "expr": "a * b"},
        {"id": "d2", "role": "derived", "kind": "count", "unit": "quả táo", "label": "số táo của Mai", "expr": "c * d"},
        {"id": "t", "role": "derived", "kind": "count", "unit": "quả táo", "label": "tổng số táo của hai bạn",
         "expr": expr_total}],
        "query": "t"}
    return from_expr_json("g3-apples", TEXT, j)


def test_op_over_derived_inputs_gets_a_sentence():
    """Bug: the final op (inputs all derived) had sentence_id None -> M3 INSERT_STEP without after_sentence."""
    P = _apples()
    assert all(o.sentence_id is not None for o in P.ops)
    top = P.producer(P.query)
    assert top.sentence_id == 1          # latest sentence among its inputs' producers (Mai's sentence)


def test_m4_never_invents_a_verb_that_is_not_in_the_text():
    """Bug: with no verb_vi, M4 emitted verb_old 'thêm'/'bớt' even when the sentence has no such word."""
    P = _apples()
    seen = 0
    for s in range(20):
        try:
            _, edits = m4_relation_change(P, random.Random(s))
        except NotApplicable:
            continue
        e = edits[0]
        seen += 1
        assert e.verb_old in P.text or e.verb_old in _OP_VI.values(), e
        assert e.sentence_id is not None
    assert seen > 0


def test_m4_does_not_swap_a_world_constant_factor():
    """Bug (G3 run A, Vi-GSM8K train idx of seed 1ec27e664964): '/ 100' (percent base, a world constant declared
    with kind ratio) was flipped to '* 100' -> 'gói mới' costs 450.150 đô la; the gate let it through."""
    text = ("Mark nhận một gói điện thoại mới đắt hơn 30% so với gói cũ của anh ấy. Nếu gói cũ của anh ấy là "
            "150 đô la một tháng thì gói mới của anh ấy có giá bao nhiêu?")
    j = {"quantities": [
        {"id": "p", "role": "given", "kind": "percent", "unit": "%", "label": "phần trăm gói mới đắt hơn gói cũ",
         "value": "30", "span": "30", "context": "đắt hơn 30%"},
        {"id": "b", "role": "given", "kind": "money", "unit": "đô la", "label": "giá gói cũ mỗi tháng",
         "value": "150", "span": "150", "context": "là 150 đô la"},
        {"id": "w", "role": "world_const", "kind": "ratio", "unit": "", "label": "cơ số phần trăm", "value": "100"},
        {"id": "x", "role": "derived", "kind": "money", "unit": "đô la", "label": "số tiền gói mới đắt hơn gói cũ",
         "expr": "b * p / w"},
        {"id": "n", "role": "derived", "kind": "money", "unit": "đô la", "label": "giá gói mới mỗi tháng",
         "expr": "b + x"}], "query": "n"}
    P = from_expr_json("g3-mark", text, j)
    wc = {q.qid for q in P.quantities.values() if q.role == "world_const"}
    for s in range(30):
        try:
            P2, _ = m4_relation_change(P, random.Random(s))
        except NotApplicable:
            continue
        for o in P.ops:
            if wc & set(o.inputs):
                assert next(x for x in P2.ops if x.oid == o.oid).kind == o.kind, s


def test_m3_never_inserts_on_an_anonymous_node():
    """Bug: '(a + b) * c' compiles to an unlabelled node; M3 inserted on it -> operands [''] in the edit."""
    P = _apples("(d1 + d2) * 1")
    assert any(q.role == "derived" and not q.label_vi for q in P.quantities.values())
    for s in range(30):
        try:
            _, edits = m3_insert_step(P, random.Random(s))
        except NotApplicable:
            continue
        assert all(edits[0].operands), edits[0]
        assert edits[0].after_sentence is not None


@pytest.mark.parametrize("sid", sorted(all_fixtures()))
def test_m4_factor_change_names_the_factor_tag(sid):
    """Bug: a 'gấp k lần' -> 'gấp k′ lần' edit carried the derived output's qid (no tag) instead of the factor's."""
    P = all_fixtures()[sid]
    for s in range(12):
        try:
            P2, edits = m4_relation_change(P, random.Random(s))
        except NotApplicable:
            return
        e = edits[0]
        if e.value is not None:
            q = P2.quantities[e.qid]
            assert q.role == "given" and q.span is not None and q.value != P.quantities[e.qid].value
