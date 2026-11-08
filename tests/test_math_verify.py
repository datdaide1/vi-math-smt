"""Tests for the typed Vietnamese grader ``vi_math_verified_aug/eval/math_verify.py`` (X1, 2026-11-07).

Labelled cases live in ``tests/grader_cases/``:
  synthetic.jsonl      191 hand-written cases: T2 §4.6 (60), T2 "still to add", legacy-scorer failure modes,
                       probe-found regressions.  Written together with the grader (not a generalisation estimate).
  real_vi_gsm8k.jsonl  102 REAL Qwen3 / Qwen3.5 base-model outputs on Vi-GSM8K test (jobs/outputs/w2-pilot-v1),
                       labelled by reading each output before the new grader was run on them.
  heldout_style.jsonl  40 MATH / exam-style responses written and labelled after the grader was frozen for the
                       dev cases; the first-run verdicts are the held-out numbers in exec/X1_grader.md.
Every case: {"id", "pred", "gold": {answer_type, answer_canonical, ...}, "expect": human label}.
Cases in KNOWN_ERRORS are documented grader errors (xfail, strict): they count in the reported FN/FP rates.
"""
import glob
import json
import os
import re
from pathlib import Path

import pytest

from vi_math_verified_aug.eval import math_verify as mv
from vi_math_verified_aug.eval.math_verify import extract_answer, grade, infer_gold, is_correct

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "tests" / "grader_cases"

KNOWN_ERRORS = {
    "H15": "cue-less prose 'Giá trị lớn nhất là 3, đạt được khi x = 1': last-number fallback takes 1 (FN)",
    "H19": "English '1,234' against a vi-locale gold with no votes: tie-break reads 1.234 (FN, flagged "
           "ambiguous-locale for adjudication)",
}


def _load(name):
    with open(CASES_DIR / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


ALL = [(f, c) for f in ("synthetic.jsonl", "real_vi_gsm8k.jsonl", "heldout_style.jsonl") for c in _load(f)]


def _param(f, c):
    marks = [pytest.mark.xfail(strict=True, reason=KNOWN_ERRORS[c["id"]])] if c["id"] in KNOWN_ERRORS else []
    return pytest.param(c, id=f"{f.split('.')[0]}:{c['id']}", marks=marks)


@pytest.mark.parametrize("case", [_param(f, c) for f, c in ALL])
def test_labelled_case(case):
    r = grade(case["pred"], case["gold"])
    assert r["correct"] is case["expect"], (case.get("note"), r)


def test_case_counts():
    real = _load("real_vi_gsm8k.jsonl")
    assert len(ALL) >= 150
    assert len(real) >= 50
    assert all(c["source"].startswith("real:") for c in real)
    assert len({c["id"] for _, c in ALL}) == len(ALL)


# ---------------------------------------------------------------------------------------------------------- #
# the four false positives the brief requires fixed (also inside synthetic.jsonl; restated here by name)
# ---------------------------------------------------------------------------------------------------------- #
def test_fp_125_vs_0125():
    assert not is_correct("125", {"answer_type": "decimal", "answer_canonical": "0.125"})
    assert not is_correct(r"\boxed{125}", "0.125")  # legacy string gold path too


def test_fp_pair_vs_decimal():
    assert not is_correct(r"\boxed{(1,2)}", {"answer_type": "decimal", "answer_canonical": "1.2"})
    assert not is_correct("(1,2)", "1.2")


def test_fp_partial_system_answer():
    gold = {"answer_type": "tuple", "answer_canonical": "7; 8", "vars": ["x", "y"]}
    assert not is_correct("8", gold)
    assert not is_correct("8", "x=7; y=8")
    assert is_correct("x = 7, y = 8", "x=7; y=8")


def test_fp_0625_never_625():
    for pred in ("0.625", r"\boxed{0.625}", "0,625", r"\boxed{0{,}625}", "Đáp số: 0.625 đồng"):
        for loc in ("vi", "en", "mixed"):
            assert not is_correct(pred, {"answer_type": "int", "answer_canonical": "625", "locale": loc}), pred


# ---------------------------------------------------------------------------------------------------------- #
# API / schema / compatibility
# ---------------------------------------------------------------------------------------------------------- #
def test_grade_fields_and_backend():
    r = grade(r"Đáp số: \boxed{3.500} đồng", {"answer_type": "int", "answer_canonical": "3500"})
    assert r["correct"] and r["method"] == "boxed" and r["matched"] == "gold"
    assert set(r) >= {"correct", "span", "method", "flag", "conv", "gold_type", "backend", "grader_version",
                      "matched"}
    assert r["backend"] in ("sympy-fallback", "math-verify-0.9.0")


def test_accept_also():
    gold = {"answer_canonical": "-3", "answer_type": "int", "locale": "en",
            "accept_also": [{"answer_canonical": "3", "answer_type": "int"}]}
    assert grade(r"\boxed{-3}", gold)["matched"] == "gold"
    r = grade(r"\boxed{3}", gold)
    assert r["correct"] and r["matched"] == "accept_also[0]"
    r = grade(r"\boxed{4}", gold)
    assert not r["correct"] and r["matched"] is None
    assert not grade(r"\boxed{-3} hoặc \boxed{3}", gold)["correct"]  # a hedge stays a hedge
    assert is_correct("3", gold)


@pytest.mark.skipif(not glob.glob(str(ROOT / "data/eval/*_math_algprealg.jsonl")), reason="eval sets not present")
def test_accept_also_algebra_0333_frozen():
    seen = 0
    for path in glob.glob(str(ROOT / "data/eval/*_math_algprealg.jsonl")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] == "math-test-algebra-0333":
                seen += 1
                assert grade(r"\boxed{-3}", r["gold"])["matched"] == "gold"
                assert grade(r"\boxed{3}", r["gold"])["matched"] == "accept_also[0]"
                assert not grade(r"\boxed{2}", r["gold"])["correct"]
    assert seen == 2  # vi and en


def test_t2_keys_and_aliases():
    assert is_correct("x \\ge 2", {"type": "ineq", "value": "[2,\\infty)"})
    with pytest.raises(ValueError):
        grade("1", {"answer_type": "float", "answer_canonical": "1"})


@pytest.mark.parametrize("raw,atype,canon", [
    ("18", "int", "18"), ("0.125", "decimal", "0.125"), (r"\frac{3}{4}", "rational", r"\frac{3}{4}"),
    (r"\dfrac{\sqrt{3}}{2}", "radical", r"\frac{\sqrt{3}}{2}"), ("(1,2)", "tuple", "1; 2"),
    (r"\{2;3\}", "set", "2; 3"), ("x=2 hoặc x=3", "set", "2; 3"), ("[1,3)", "interval", "[1,3)"),
    (r"\text{(C)}", "choice", "C"), (r"\text{Sunday}", "text", "Sunday"), ("10,\\!000", "int", "10000"),
    ("x^2+1", "expr", "x^2+1"), (r"20\%", "int", "20"), ("x = 5", "int", "5"),
])
def test_infer_gold_types(raw, atype, canon):
    g = infer_gold(raw)
    assert (g["answer_type"], g["answer_canonical"]) == (atype, canon)


def test_infer_gold_never_extracts():
    # a multi-part gold string stays multi-part: no last-number search inside it
    g = infer_gold("x=7; y=8")
    assert g["answer_type"] == "tuple" and g["vars"] == ["x", "y"]


def test_legacy_callers():
    # jobs/*/run.py: is_correct(extract_answer(body), ground_truth_string)
    body = "Bước 1...\nĐáp số: \\boxed{90.000}"
    assert extract_answer(body) == "90.000"
    assert is_correct(extract_answer(body), "90000")
    assert extract_answer("Em không biết.") is None
    assert is_correct("42", "42", rel_tol=1e-6, abs_tol=1e-6)  # old keywords accepted and ignored
    assert is_correct("x", None) is False
    # vi_exam/build.py helpers
    assert mv._to_float(mv.normalize_answer("3,5")) == 3.5
    assert mv._sympy_equal(r"\frac{\sqrt2}{2}", r"\frac{1}{\sqrt{2}}")


def test_hedge_and_degenerate_flags():
    assert grade(r"\boxed{2} hoặc \boxed{3}", "2")["flag"] == "hedged"
    loop = "Bước tính. " + "Anh ấy cần 2 hộp nước dùng bò, và " * 40
    r = grade(loop, "2")
    assert not r["correct"] and r["flag"] == "degenerate"


def test_backends_agree_on_labelled_cases():
    if mv._MV is None:
        pytest.skip("math-verify not importable in this environment (SymPy fallback only)")
    before = mv.BACKEND
    try:
        verdicts = {}
        for name in ("sympy", "math-verify"):
            mv.set_backend(name)
            verdicts[name] = [grade(c["pred"], c["gold"])["correct"] for _, c in ALL]
        assert verdicts["sympy"] == verdicts["math-verify"]
    finally:
        mv.set_backend("math-verify" if before.startswith("math-verify") else "sympy")


# ---------------------------------------------------------------------------------------------------------- #
# self-match: every gold of the project's test sets matches itself
# ---------------------------------------------------------------------------------------------------------- #
def _last_box(sol):
    b = mv._boxes(sol)
    return b[-1][0] if b else None


def _math_pairs():
    for f in sorted(glob.glob(str(ROOT / "data/test/math/*_vi.jsonl"))):
        subj = os.path.basename(f)[:-len("_vi.jsonl")]
        en_path = ROOT / f"data/generate/MATH/test/{subj}.json"
        vi = [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]
        en = json.load(open(en_path, encoding="utf-8"))
        assert len(vi) == len(en)
        yield from ((subj, i, v, e) for i, (v, e) in enumerate(zip(vi, en)))


@pytest.mark.skipif(not (ROOT / "data/test/math").exists(), reason="test data not present")
def test_self_match_math_vi_and_en():
    bad = []
    for subj, i, v, e in _math_pairs():
        for raw, prob, loc in ((_last_box(e["solution"]), e["problem"], "en"),
                               (_last_box(v["solution"]), v["problem"], "mixed")):
            assert raw is not None, (subj, i)
            if not grade("\\boxed{" + raw + "}", infer_gold(raw, problem=prob, locale=loc))["correct"]:
                bad.append((subj, i, loc, raw))
    assert not bad, bad[:20]


EVAL_SETS = sorted(glob.glob(str(ROOT / "data/eval/*.jsonl")))


def _is_localised_text(r):
    """VI rows whose text gold was localised (gold_raw stays English, e.g. east -> đông)."""
    return r["lang"] == "vi" and r["gold"]["answer_type"] == "text" and \
        infer_gold(r["gold_raw"])["answer_canonical"] != r["gold"]["answer_canonical"]


@pytest.mark.skipif(not EVAL_SETS, reason="frozen eval sets not present")
@pytest.mark.parametrize("path", EVAL_SETS, ids=[os.path.basename(p) for p in EVAL_SETS])
def test_self_match_frozen_eval_sets(path):
    """Eval-runner API: grade(response, record["gold"]).  The stored canonical must match its own gold, and so must
    gold_raw (the source string) except for localised VI text golds."""
    bad = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        g = r["gold"]
        if grade("\\boxed{" + g["answer_canonical"] + "}", g)["matched"] != "gold":
            bad.append((r["id"], "canonical"))
        for i, alt in enumerate(g.get("accept_also") or []):  # every alternative matches itself, via accept_also
            if grade("\\boxed{" + alt["answer_canonical"] + "}", g)["matched"] not in ("gold", f"accept_also[{i}]"):
                bad.append((r["id"], f"accept_also[{i}]"))
        if not _is_localised_text(r) and not grade("\\boxed{" + str(r["gold_raw"]) + "}", g)["correct"]:
            bad.append((r["id"], "gold_raw"))
        # the same gold without its locale field (as X3's base-check passes it) must still self-match
        nl = {k: v for k, v in g.items() if k != "locale"}
        if not _is_localised_text(r) and not grade("\\boxed{" + str(r["gold_raw"]) + "}", nl)["correct"]:
            bad.append((r["id"], "gold_raw/no-locale"))
    assert not bad, bad[:20]


@pytest.mark.skipif(not (ROOT / "data/test/gsm8k").exists(), reason="test data not present")
def test_self_match_gsm8k_vi():
    bad = []
    for line in open(ROOT / "data/test/gsm8k/gsm8k_test_vietnamese.jsonl", encoding="utf-8"):
        d = json.loads(line)
        g = infer_gold(d["ground_truth"], locale="mixed")
        if not (grade("\\boxed{" + d["ground_truth"] + "}", g)["correct"] and grade(d["answer"], g)["correct"]):
            bad.append(d["ground_truth"])
    assert not bad, bad[:20]
    assert re.fullmatch(r"-?\d+", d["ground_truth"])
