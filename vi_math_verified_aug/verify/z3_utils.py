"""Shared Z3 helpers: parse an SMT-LIB string, solve, read a named value, test uniqueness."""
from __future__ import annotations

import re
from typing import Optional

import z3

_STRIP_CMD = re.compile(r"^\s*\((?:check-sat|get-value|get-model|get-info|set-option|exit|push|pop)\b")


def strip_commands(smt: str) -> str:
    return "\n".join(l for l in smt.splitlines() if not _STRIP_CMD.match(l))


def parse(smt: str, *, timeout_ms: int = 5000) -> Optional[z3.Solver]:
    """Return a Solver loaded with ``smt`` (commands stripped), or ``None`` on parse error."""
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    try:
        s.add(z3.parse_smt2_string(strip_commands(smt)))
    except z3.Z3Exception:
        return None
    return s


def value_of(model: z3.ModelRef, name: str) -> Optional[float]:
    for d in model.decls():
        if d.name() == name:
            v = model[d]
            try:
                if z3.is_rational_value(v):
                    return float(v.numerator_as_long()) / float(v.denominator_as_long())
                if z3.is_int_value(v):
                    return float(v.as_long())
                return float(str(v).replace("?", ""))
            except Exception:
                return None
    return None


def solve_value(smt: str, name: str = "answer", *, timeout_ms: int = 5000):
    """``(status, value)`` where status ∈ {"sat","unsat","unknown","parse_error","no_var"}."""
    s = parse(smt, timeout_ms=timeout_ms)
    if s is None:
        return "parse_error", None
    r = s.check()
    if r == z3.unsat:
        return "unsat", None
    if r == z3.unknown:
        return "unknown", None
    m = s.model()
    if not any(d.name() == name for d in m.decls()):
        return "no_var", None
    return "sat", value_of(m, name)


def is_unique(assertions: list, decls: dict, *, timeout_ms: int = 5000) -> Optional[bool]:
    """Given SMT-LIB assertion strings + a name→sort map, is the model unique over those vars?

    Returns None if the base system is unsat/unknown/unparseable.  Works on the
    z3 term API (no float round-trip), so exact rationals are preserved.
    """
    header = "\n".join(f"(declare-fun {n} () {srt})" for n, srt in decls.items())
    s = parse(header + "\n" + "\n".join(assertions), timeout_ms=timeout_ms)
    if s is None or s.check() != z3.sat:
        return None
    m = s.model()
    named = {d.name(): d for d in m.decls() if d.name() in decls and d.arity() == 0}
    if not named:
        return True
    block = z3.Or([d() != m[d] for d in named.values()])
    s.add(block)
    return s.check() == z3.unsat
