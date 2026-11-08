"""Z3 encoding of a Program, used only where Z3 is needed (01 §3.1): integer CSP sampling for coupled
integrality (M1) and solution-uniqueness of integer-domain equation systems.

encode(P, var) builds a z3 Real expression for every node. `var` maps qid -> z3 expression for the leaves
that are decision variables (resampled givens, unknowns); every other leaf is a constant. Unsupported op
kinds (floor/mod/min/max/abs, pow with a variable exponent) raise Unsupported.
"""
from __future__ import annotations

import ast
from fractions import Fraction

import z3

from .ir import Program
from .units import requires_integer, requires_nonneg

__all__ = ["Unsupported", "encode", "domain_constraints", "constraint_to_z3", "rv", "model_value"]


class Unsupported(Exception):
    pass


def rv(v: Fraction):
    v = Fraction(v)
    return z3.RealVal(f"{v.numerator}/{v.denominator}")


def model_value(m: z3.ModelRef, e) -> Fraction:
    val = m.eval(e, model_completion=True)
    if z3.is_rational_value(val):
        return Fraction(val.numerator_as_long(), val.denominator_as_long())
    if z3.is_int_value(val):
        return Fraction(val.as_long())
    raise Unsupported(f"non-rational model value {val}")


def _const_of(e):
    s = z3.simplify(e)
    if z3.is_rational_value(s):
        return Fraction(s.numerator_as_long(), s.denominator_as_long())
    if z3.is_int_value(s):
        return Fraction(s.as_long())
    return None


def encode(P: Program, var: dict) -> tuple[dict, list]:
    """Returns (expr per qid, side conditions such as non-zero divisors)."""
    env: dict = {}
    side: list = []
    for q in P.quantities.values():
        if q.qid in var:
            env[q.qid] = var[q.qid]
        elif q.role in ("given", "world_const"):
            env[q.qid] = rv(q.value)
    prod = P.producers()
    for n in P.topo():
        if n not in prod or n in var:
            continue
        o = prod[n]
        xs = [env[i] for i in o.inputs]
        k = o.kind
        if k == "add":
            e = z3.Sum(xs)
        elif k == "sub":
            e = xs[0] - z3.Sum(xs[1:]) if len(xs) > 2 else xs[0] - xs[1]
        elif k == "mul":
            e = xs[0]
            for x in xs[1:]:
                e = e * x
        elif k in ("div",):
            side.append(xs[1] != 0)
            e = xs[0] / xs[1]
        elif k == "pct_of":
            e = xs[0] * xs[1] / 100
        elif k == "pct_inc":
            e = xs[0] * (100 + xs[1]) / 100
        elif k == "pct_dec":
            e = xs[0] * (100 - xs[1]) / 100
        elif k == "ratio_part":
            e = xs[0] * xs[1]
        elif k == "avg":
            e = z3.Sum(xs) / len(xs)
        elif k == "neg":
            e = -xs[0]
        elif k == "pow":
            ex = _const_of(xs[1])
            if ex is None or ex.denominator != 1 or ex < 0 or ex > 12:
                raise Unsupported("pow with a variable / non-integer exponent")
            e = z3.RealVal(1)
            for _ in range(int(ex)):
                e = e * xs[0]
        else:
            raise Unsupported(f"op {k} not encoded")
        env[n] = e
    return env, side


def domain_constraints(P: Program, env: dict, aux_prefix: str = "n") -> list:
    """Integrality and sign at every node, as z3 constraints (integrality via an Int witness)."""
    cons = []
    for q in P.quantities.values():
        if q.qid not in env:
            continue
        e = env[q.qid]
        if requires_integer(q.qtype, q.unit):
            w = z3.Int(f"{aux_prefix}{q.qid}")
            cons.append(z3.ToReal(w) == e)
        if requires_nonneg(q.qtype):
            cons.append(e >= 0)
    return cons


def constraint_to_z3(c: str, env: dict):
    """Translate a seed constraint string (q<id> names) into z3."""
    tree = ast.parse(c, mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return rv(Fraction(str(n.value)))
        if isinstance(n, ast.Name):
            return env[int(n.id[1:])]
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else (z3.Not(v) if isinstance(n.op, ast.Not) else v)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            t = type(n.op)
            if t is ast.Add:
                return a + b
            if t is ast.Sub:
                return a - b
            if t is ast.Mult:
                return a * b
            if t is ast.Div:
                return a / b
            raise Unsupported(f"operator {t.__name__} in constraint")
        if isinstance(n, ast.BoolOp):
            vals = [ev(x) for x in n.values]
            return z3.And(vals) if isinstance(n.op, ast.And) else z3.Or(vals)
        if isinstance(n, ast.Compare):
            parts = []
            left = ev(n.left)
            for op, right in zip(n.ops, n.comparators):
                r = ev(right)
                t = type(op)
                parts.append({ast.Eq: left == r, ast.NotEq: left != r, ast.Lt: left < r, ast.LtE: left <= r,
                              ast.Gt: left > r, ast.GtE: left >= r}[t])
                left = r
            return z3.And(parts) if len(parts) > 1 else parts[0]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "is_int":
            return z3.IsInt(ev(n.args[0]))
        raise Unsupported(f"constraint syntax {ast.dump(n)}")

    return ev(tree)
