"""The OLD answer-expression operators of src/mutation_informalize/mutation_engine.py, re-expressed in the IR.

They are reproduced (not imported) so that (a) the formal C3 claim can be tested — every one of them is an
answer tail under rules.is_answer_tail — and (b) the C3 attribution pilot can run an "old operator, everything
else best" cell (01_TECHNICAL_DESIGN §4). Old code, for reference:

    wrap(expr):           (* (+ a k) k) | (+ (* a k) k) | (* a (+ k k)) | (- (+ a k) k),  k ~ U(0.5, 5) rounded 2dp
    mutate_structure:     a + h*2            (h a fresh constant h_<rand>)
    mutate_difficulty:    d = a + 1 ; a * d  (d_<rand>)

Each function returns a new Program whose query is the old answer post-composed. With in_place=True the
encoding keeps the old query's qid as the new query and moves the old computation to a fresh node — the
renaming trick that a qid-based tail test would miss (the structural test does not).
"""
from __future__ import annotations

import random
from fractions import Fraction

from .ir import Program
from .units import QType, Unit

__all__ = ["LEGACY_OPS", "legacy_wrap", "legacy_structure", "legacy_difficulty", "apply_legacy"]

WRAP_VARIANTS = ("(a+k)*k", "a*k+k", "a*(k+k)", "(a+k)-k")


def _k(rng: random.Random) -> Fraction:
    return Fraction(str(round(rng.uniform(0.5, 5.0), 2)))


def _start(P: Program, in_place: bool) -> tuple[Program, int]:
    """Return (copy, node holding the old answer). in_place: the old computation moves to a new qid and the old
    query qid will be redefined on top of it."""
    P2 = P.copy()
    P2.meta = {**P.meta, "operator": "legacy", "parent": P.seed_id}
    q0 = P2.query
    if not in_place:
        return P2, q0
    q = P2.quantities[q0]
    if q.role != "derived":
        return P2, q0  # an unknown cannot be moved; fall back to the plain encoding
    o = P2.producer(q0)
    c = P2.add_quantity(None, q.qtype, q.unit, q.label_vi, "derived", origin="operator")
    o.output = c
    q.qtype, q.unit = QType.ABSTRACT, Unit()
    return P2, c


def _const(P2: Program, v: Fraction, name: str) -> int:
    return P2.add_quantity(v, QType.ABSTRACT, Unit(), name, "given", origin="inserted")


def _new(P2: Program, kind: str, ins: list[int], out: int | None = None) -> int:
    if out is None:
        out = P2.add_quantity(None, QType.ABSTRACT, Unit(), "", "derived", origin="operator")
    P2.add_op(kind, ins, out)
    return out


def legacy_wrap(P: Program, variant: str, k: Fraction, in_place: bool = False) -> Program:
    P2, a = _start(P, in_place)
    target = P.query if in_place and a != P.query else None
    kk = _const(P2, k, "k")
    if variant == "(a+k)*k":
        out = _new(P2, "mul", [_new(P2, "add", [a, kk]), kk], target)
    elif variant == "a*k+k":
        out = _new(P2, "add", [_new(P2, "mul", [a, kk]), kk], target)
    elif variant == "a*(k+k)":
        out = _new(P2, "mul", [a, _new(P2, "add", [kk, kk])], target)
    elif variant == "(a+k)-k":
        out = _new(P2, "sub", [_new(P2, "add", [a, kk]), kk], target)
    else:
        raise ValueError(variant)
    P2.query = out
    return P2


def legacy_structure(P: Program, h: Fraction, in_place: bool = False) -> Program:
    """a + h*2 (mutate_structure)."""
    P2, a = _start(P, in_place)
    target = P.query if in_place and a != P.query else None
    hh = _const(P2, h, "h")
    two = _const(P2, Fraction(2), "2")
    P2.query = _new(P2, "add", [a, _new(P2, "mul", [hh, two])], target)
    return P2


def legacy_difficulty(P: Program, in_place: bool = False) -> Program:
    """d = a + 1; a * d (mutate_difficulty)."""
    P2, a = _start(P, in_place)
    target = P.query if in_place and a != P.query else None
    one = _const(P2, Fraction(1), "1")
    d = _new(P2, "add", [a, one])
    P2.query = _new(P2, "mul", [a, d], target)
    return P2


LEGACY_OPS = [f"wrap:{v}" for v in WRAP_VARIANTS] + ["structure", "difficulty"]


def apply_legacy(name: str, P: Program, rng: random.Random, in_place: bool = False) -> Program:
    if name.startswith("wrap:"):
        return legacy_wrap(P, name.split(":", 1)[1], _k(rng), in_place)
    if name == "structure":
        return legacy_structure(P, _k(rng), in_place)
    if name == "difficulty":
        return legacy_difficulty(P, in_place)
    raise ValueError(name)
