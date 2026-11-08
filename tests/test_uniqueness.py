"""T1.1: the uniqueness gate must call hardcoded vs. derived correctly."""
import pytest

from vi_math_verified_aug.verify.uniqueness import classify

HARDCODED = [
    # x pinned to an irrational GT value, answer aliased to it
    "(declare-fun x () Real)(declare-fun answer () Real)"
    "(assert (= x 1.4142135623730951))(assert (= answer x))",
    # answer written as a bare literal
    "(declare-fun answer () Real)(assert (= answer 124.0))",
    # a digit-lookup: a := 2, answer := a, gt was 2
    "(declare-fun a () Real)(declare-fun answer () Real)"
    "(assert (= a 2.0))(assert (= answer a))",
    # transcendental smuggled in as a decimal
    "(declare-fun pi () Real)(declare-fun answer () Real)"
    "(assert (= pi 3.141592653589793))(assert (= answer (* 40.0 pi)))",
]

DERIVED = [
    # GSM8K multi-step linear chain
    "(declare-fun a () Real)(declare-fun b () Real)(declare-fun c () Real)(declare-fun answer () Real)"
    "(assert (= a 5.0))(assert (= b 3.0))(assert (= c (* a 2.0)))(assert (= answer (+ c b)))",
    # answer coincidentally equals an input but is genuinely computed
    "(declare-fun friday () Real)(declare-fun saturday () Real)(declare-fun total () Real)(declare-fun answer () Real)"
    "(assert (= friday 16.0))(assert (= saturday 28.0))(assert (= total 60.0))"
    "(assert (= answer (- total (+ friday saturday))))",
    # a genuine linear equation solved for the unknown
    "(declare-fun p () Real)(declare-fun answer () Real)"
    "(assert (= (+ 15 (* 10 p)) (+ 90 25)))(assert (= answer p))",
]

UNKNOWN = [
    "(declare-fun answer () Real)(assert (> answer 0.0))",          # under-determined
    "(declare-fun x () Real)(assert (= x 1.0))",                    # no answer var
    "not valid smt at all",
]


@pytest.mark.parametrize("smt", HARDCODED)
def test_hardcoded(smt):
    assert classify(smt) == "hardcoded"


@pytest.mark.parametrize("smt", DERIVED)
def test_derived(smt):
    assert classify(smt) == "derived"


@pytest.mark.parametrize("smt", UNKNOWN)
def test_unknown(smt):
    assert classify(smt) == "unknown"
