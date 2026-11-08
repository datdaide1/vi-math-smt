"""IR data structures, JSON schema, executor, type propagation, validators, fixtures."""
import json
import os
from fractions import Fraction as F
from pathlib import Path

import pytest

from vi_math_verified_aug.symbolic.executor import (ExecError, auto_constraints, check, eval_constraint, execute,
                                                   infer_types, type_violations)
from vi_math_verified_aug.symbolic.fixtures import ALGEBRA_IDS, GSM8K_IDS, SEEDS, all_fixtures, load_fixture
from vi_math_verified_aug.symbolic.ir import (SCHEMA_VERSION, IRError, Program, ProgramBuilder, from_expr_json,
                                              locate_span, split_sentences)
from vi_math_verified_aug.symbolic.units import QType, resolve_unit, requires_integer
from vi_math_verified_aug.symbolic.validate import validate_program

ROOT = Path(__file__).resolve().parents[1]
ALL_IDS = GSM8K_IDS + ALGEBRA_IDS
CANDY = "Lan có 3 cái kẹo. Lan cho bạn một nửa số kẹo. Hỏi Lan cho bạn bao nhiêu cái kẹo?"


# ------------------------------------------------------------------------------------------ fixtures
def test_fixture_counts():
    assert len(GSM8K_IDS) >= 20 and len(ALGEBRA_IDS) >= 5


@pytest.mark.parametrize("sid", ALL_IDS)
def test_fixture_accepted_by_formalizer_validator(sid):
    P = load_fixture(sid)
    assert validate_program(P, SEEDS[sid][1]) == []
    assert P.answer == F(SEEDS[sid][1])


def test_fixture_texts_match_dataset():
    gsm = ROOT / "data/translation/gsm8k/gsm8k_train_vietnamese.jsonl"
    alg = ROOT / "data/translation/MATH/train/algebra_vi.jsonl"
    if not gsm.exists() or not alg.exists():
        pytest.skip("seed data not present")
    g = {f"gsm8k-{d['index']}": d for d in map(json.loads, gsm.read_text(encoding="utf-8").splitlines())}
    a = {f"algebra-{d['index']}": d for d in map(json.loads, alg.read_text(encoding="utf-8").splitlines())}
    for sid, (text, gold) in SEEDS.items():
        d = g.get(sid) or a.get(sid)
        assert d["question" if sid.startswith("gsm8k") else "problem"] == text, sid
        assert str(d["ground_truth"]) == gold, sid


# ------------------------------------------------------------------------------------------ JSON
@pytest.mark.parametrize("sid", ALL_IDS)
def test_json_round_trip(sid):
    P = load_fixture(sid)
    s = P.to_json()
    Q = Program.from_json(s)
    assert Q.to_dict() == P.to_dict()
    assert execute(Q)[Q.query] == execute(P)[P.query]
    assert json.loads(s)["schema"] == SCHEMA_VERSION


def test_schema_version_enforced():
    d = load_fixture("gsm8k-0").to_dict()
    d["schema"] = "vi-math-ir/2.0"
    with pytest.raises(IRError):
        Program.from_dict(d)
    d["schema"] = "vi-math-ir/1.7"   # minor versions are forward compatible
    Program.from_dict(d)
    del d["schema"]
    with pytest.raises(IRError):
        Program.from_dict(d)


def test_structure_errors_detected():
    d = load_fixture("gsm8k-0").to_dict()
    d2 = json.loads(json.dumps(d))
    d2["ops"][0]["inputs"] = [999, 1]
    with pytest.raises(IRError):
        Program.from_dict(d2)
    d3 = json.loads(json.dumps(d))
    d3["ops"][0]["inputs"] = [d3["ops"][1]["output"], 2]   # cycle
    with pytest.raises(IRError):
        Program.from_dict(d3)
    d4 = json.loads(json.dumps(d))
    d4["ops"][0]["kind"] = "frobnicate"
    with pytest.raises(IRError):
        Program.from_dict(d4)


# ------------------------------------------------------------------------------------------ text helpers
def test_split_sentences_ignores_decimal_points():
    t = "Giá là 1.000 đồng. Lan mua 2.5 kg? Còn lại bao nhiêu"
    ss = split_sentences(t)
    assert [t[a:b] for a, b in ss] == ["Giá là 1.000 đồng.", "Lan mua 2.5 kg?", "Còn lại bao nhiêu"]


def test_locate_span_whole_token():
    t = "Có 25 trang và 2 tờ, 12 người."
    assert locate_span(t, "2") == (15, 16)
    assert locate_span(t, "12") == (21, 23)
    assert locate_span(t, "2", taken=[(15, 16)]) is None


# ------------------------------------------------------------------------------------------ types
def test_half_a_candy_is_impossible_derived():
    b = ProgramBuilder("candy", CANDY)
    n = b.given("3", "count", "cái kẹo", "số kẹo của Lan")
    h = b.given("một nửa", "ratio", "", "phần Lan cho bạn")
    g = b.op("ratio_part", [n, h], "count", "cái kẹo", "số kẹo Lan cho bạn")
    P = b.build(g)
    env = execute(P)
    assert env[g] == F(3, 2)
    assert any("not an integer" in v for v in type_violations(P, env))
    assert any(v.code == "type" for v in validate_program(P))


def test_half_a_candy_is_impossible_given():
    t = "Lan có 0,5 cái kẹo và 2 cái bánh. Hỏi Lan có bao nhiêu món?"
    b = ProgramBuilder("candy2", t)
    a = b.given("0,5", "count", "cái kẹo", "số kẹo")
    c = b.given("2", "count", "cái bánh", "số bánh")
    P = b.build(b.op("add", [a, c], "count", "món", "tổng"))
    assert type_violations(P, execute(P))


def test_unit_lexicon_classifier_and_divisibility():
    u = resolve_unit("cái kẹo")
    assert (u.lemma, u.classifier, u.divisible) == ("kẹo", "cái", False)
    assert requires_integer(None, u)
    r = resolve_unit("đô la/giờ")
    assert r.per == "giờ" and r.divisible
    assert resolve_unit("kg").divisible
    assert not resolve_unit("quả táo").divisible


def test_integrality_enforced_at_intermediate_nodes():
    """(5 kẹo / 2) * 2 = 5 is an integer answer, but the intermediate 2,5 kẹo is not allowed."""
    t = "Lan có 5 cái kẹo, chia đều cho 2 bạn, rồi mỗi bạn nhận gấp 2 lần. Hỏi?"
    b = ProgramBuilder("mid", t)
    n = b.given("5", "count", "cái kẹo", "số kẹo")
    k = b.given("2", "ratio", "", "số bạn", occurrence=0)
    f = b.given("2", "ratio", "", "hệ số", occurrence=1)
    each = b.op("div", [n, k], "count", "cái kẹo", "số kẹo mỗi bạn")
    P = b.build(b.op("mul", [each, f], "count", "cái kẹo", "kết quả"))
    env = execute(P)
    assert env[P.query] == 5
    assert any(f"q{each}" in v for v in type_violations(P, env))


def test_sign_constraint_remaining_nonnegative():
    t = "Lan có 3 cái kẹo và cho đi 5 cái. Hỏi còn lại bao nhiêu?"
    b = ProgramBuilder("neg", t)
    a = b.given("3", "count", "cái kẹo", "số kẹo")
    c = b.given("5", "count", "cái kẹo", "số kẹo cho đi")
    P = b.build(b.op("sub", [a, c], "count", "cái kẹo", "số kẹo còn lại"))
    assert any("negative" in v for v in check(P))


def test_inferred_type_of_untyped_node_is_checked():
    j = {"quantities": [
        {"id": "a", "role": "given", "kind": "count", "unit": "cái kẹo", "label": "số kẹo", "value": "3",
         "span": "3"},
        {"id": "h", "role": "derived", "kind": "count", "unit": "cái kẹo", "label": "kết quả", "expr": "a / 2 * 2"}],
        "query": "h"}
    P = from_expr_json("x", CANDY, j)
    # the anonymous node a/2 is inferred as COUNT 'kẹo' and must be an integer
    assert any("not an integer" in v for v in check(P))


def test_unit_mismatch_count_plus_money():
    t = "Lan có 3 quả táo và 5 đô la. Hỏi?"
    b = ProgramBuilder("mm", t)
    a = b.given("3", "count", "quả táo", "táo")
    m = b.given("5", "money", "đô la", "tiền")
    P = b.build(b.op("add", [a, m], "count", "quả táo", "tổng"))
    assert infer_types(P)
    assert any(v.code == "unit" for v in validate_program(P))


def test_auto_constraints_and_evaluator():
    P = load_fixture("gsm8k-80")
    cs = auto_constraints(P)
    assert any(c.startswith("is_int(") for c in cs)
    env = execute(P)
    assert all(eval_constraint(c, env) for c in cs)
    assert eval_constraint("q1 + q2 > 0 and is_int(q1)", {1: F(2), 2: F(1, 2)})
    with pytest.raises(IRError):
        eval_constraint("__import__('os')", {})


def test_exact_arithmetic_and_errors():
    t = "Có 7 và 0 và 2. Hỏi?"
    b = ProgramBuilder("z", t)
    a = b.given("7", "abstract", "", "a")
    z = b.given("0", "abstract", "", "z")
    P = b.build(b.op("div", [a, z], "abstract", "", "q"))
    with pytest.raises(ExecError):
        execute(P)
    b = ProgramBuilder("p", t)
    a = b.given("7", "abstract", "", "a")
    two = b.given("2", "abstract", "", "e")
    half = b.const(F(1, 2), "abstract", "", "một nửa")
    P = b.build(b.op("pow", [a, half], "abstract", "", "căn 7"))
    with pytest.raises(ExecError):      # sqrt(7) is irrational: outside the rational IR
        execute(P)


def test_unknowns_solved_exactly_and_domain_filter():
    P = load_fixture("gsm8k-150")
    assert execute(P)[P.query] == 3
    P = load_fixture("algebra-145")      # 2^x + 44 = 300 -> x = 8
    assert execute(P)[P.query] == 8


def test_expr_adapter_matches_builder():
    j = {"quantities": [
        {"id": "a", "role": "given", "kind": "count", "unit": "kẹp tóc", "label": "tháng 4", "value": "48",
         "span": "48", "context": "cho 48 người bạn"},
        {"id": "h", "role": "given", "kind": "ratio", "unit": "", "label": "nửa", "value": "1/2", "span": "một nửa"},
        {"id": "t", "role": "derived", "kind": "count", "unit": "kẹp tóc", "label": "tổng", "expr": "a + a * h"}],
        "query": "t"}
    P = from_expr_json("gsm8k-0", SEEDS["gsm8k-0"][0], j)
    assert validate_program(P, 72) == []
    assert P.canonical() == from_expr_json("gsm8k-0", SEEDS["gsm8k-0"][0], j).canonical()


def test_canonical_form_invariant_to_qid_renumbering_and_operand_order():
    P = load_fixture("gsm8k-20")
    d = P.to_dict()
    perm = {q["qid"]: 100 + i for i, q in enumerate(reversed(d["quantities"]))}
    for q in d["quantities"]:
        q["qid"] = perm[q["qid"]]
    for o in d["ops"]:
        o["inputs"] = [perm[i] for i in o["inputs"]]
        if o["kind"] == "add":
            o["inputs"].reverse()
        o["output"] = perm[o["output"]]
    d["query"] = perm[d["query"]]
    Q = Program.from_dict(d)
    assert Q.canonical() == P.canonical()


# ------------------------------------------------------------------------------------------ validators
def test_validator_rejections():
    P = load_fixture("gsm8k-0")
    assert [v.code for v in validate_program(P, 73)] == ["gold"]
    Q = P.copy()
    Q.quantities[1].value = F(49)                         # literal "48" no longer matches
    assert "span" in {v.code for v in validate_program(Q)}
    Q = P.copy()
    Q.quantities[1].span = None
    assert "span" in {v.code for v in validate_program(Q)}
    Q = P.copy()
    Q.query = 1                                           # query is a literal
    assert "query" in {v.code for v in validate_program(Q)}


def test_validator_unreachable_given_and_distractor():
    P = load_fixture("gsm8k-13")
    d = next(q for q in P.givens() if q.distractor)
    d.distractor = False
    assert "reach" in {v.code for v in validate_program(P)}


def test_validator_world_const_shadowing_text_number():
    t = "Lan có 12 quả táo và ăn 5 quả. Hỏi còn bao nhiêu?"
    b = ProgramBuilder("wc", t)
    a = b.given("12", "count", "quả táo", "số táo")
    five = b.const(5, "count", "quả táo", "số táo ăn")    # hard-coded although written in the text
    P = b.build(b.op("sub", [a, five], "count", "quả táo", "còn lại"))
    assert "world_const" in {v.code for v in validate_program(P, 7)}


def test_validator_query_depth_for_unknown_query():
    P = load_fixture("algebra-52")
    assert P.depth(P.query) >= 1
    assert validate_program(P, 21) == []


def test_transcendental_equation_irrational_root_rejected_quickly():
    """2^x + 44 = 158 -> x = log2(114): not rational. sympy.solve hangs on this in SymPy 1.14; the executor
    routes non-polynomial univariate equations to an exactly-verified scan instead."""
    import time
    P = load_fixture("algebra-145")
    v = next(q.qid for q in P.givens() if q.label_vi == "giá trị của F")
    t = time.time()
    with pytest.raises(ExecError):
        execute(P, values={v: F(158)})
    assert time.time() - t < 10
    assert execute(P, values={v: F(44 + 1024)})[P.query] == 10
