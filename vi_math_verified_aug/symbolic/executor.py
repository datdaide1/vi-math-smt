"""Exact executor, type/unit propagation and constraint checking (T1 §2.1).

execute(P, values=None)  -> {qid: Fraction}; forward Fraction arithmetic, SymPy only when there are unknowns.
                            Raises ExecError on division by zero, irrational results, no / several solutions.
type_violations(P, env)  -> list[str]: integrality and sign at EVERY node (indivisible nouns, ages), percent /
                            ratio ranges. This is what makes "0,5 cái kẹo" impossible.
infer_types(P, fill)     -> list[str]: unit algebra over the ops (COUNT - COUNT -> COUNT, MONEY / COUNT -> RATE,
                            RATE(a/b) x b -> a, ...). fill=True writes inferred types into untyped nodes; the
                            returned list holds hard unit mismatches (COUNT + MONEY, ...).
auto_constraints(P)      -> SymPy-style strings for the type constraints ("is_int(q3)", "q3 >= 0").
check(P, env)            -> list[str]: every violated auto or seed-specific constraint.
"""
from __future__ import annotations

import ast
import math
from collections import OrderedDict
from fractions import Fraction

import sympy as sp

from .ir import IRError, Program, OP_KINDS
from .units import QType, Unit, requires_integer, requires_nonneg, resolve_unit, lookup

__all__ = ["ExecError", "execute", "run", "apply_op", "type_violations", "infer_types", "infer_op_type",
           "auto_constraints", "eval_constraint", "check", "solve_unknowns", "sympy_env"]


class ExecError(ValueError):
    pass


# ----------------------------------------------------------------------------------- exact arithmetic
def _exact_pow(a: Fraction, b: Fraction) -> Fraction:
    if b.denominator == 1:
        if a == 0 and b < 0:
            raise ExecError("0 to a negative power")
        return a ** int(b)
    n = b.denominator
    if a < 0:
        raise ExecError("root of a negative number")

    def iroot(x: int) -> int | None:
        r = round(x ** (1.0 / n))
        for c in (r - 1, r, r + 1):
            if c >= 0 and c ** n == x:
                return c
        return None

    p, q = iroot(a.numerator), iroot(a.denominator)
    if p is None or q is None:
        raise ExecError(f"irrational power {a}**{b}")
    return Fraction(p, q) ** b.numerator


def apply_op(kind: str, xs: list[Fraction]) -> Fraction:
    """Fraction semantics of every op kind."""
    try:
        if kind == "add":
            return sum(xs, Fraction(0))
        if kind == "sub":
            return xs[0] - sum(xs[1:], Fraction(0))
        if kind == "mul":
            out = Fraction(1)
            for x in xs:
                out *= x
            return out
        if kind == "div":
            return xs[0] / xs[1]
        if kind == "floordiv":
            return Fraction(math.floor(xs[0] / xs[1]))
        if kind == "ceildiv":
            return Fraction(math.ceil(xs[0] / xs[1]))
        if kind == "mod":
            return xs[0] - xs[1] * math.floor(xs[0] / xs[1])
        if kind == "pct_of":
            return xs[0] * xs[1] / 100
        if kind == "pct_inc":
            return xs[0] * (100 + xs[1]) / 100
        if kind == "pct_dec":
            return xs[0] * (100 - xs[1]) / 100
        if kind == "ratio_part":
            return xs[0] * xs[1]
        if kind == "pow":
            return _exact_pow(xs[0], xs[1])
        if kind == "avg":
            return sum(xs, Fraction(0)) / len(xs)
        if kind == "min":
            return min(xs)
        if kind == "max":
            return max(xs)
        if kind == "neg":
            return -xs[0]
        if kind == "abs":
            return abs(xs[0])
        if kind == "floor":
            return Fraction(math.floor(xs[0]))
        if kind == "ceil":
            return Fraction(math.ceil(xs[0]))
    except ZeroDivisionError:
        raise ExecError(f"division by zero in {kind}")
    raise ExecError(f"unknown op kind {kind}")


def _sym_op(kind: str, xs: list):
    if kind == "add":
        return sp.Add(*xs)
    if kind == "sub":
        return xs[0] - sp.Add(*xs[1:])
    if kind in ("mul",):
        return sp.Mul(*xs)
    if kind == "div":
        return xs[0] / xs[1]
    if kind == "floordiv":
        return sp.floor(xs[0] / xs[1])
    if kind == "ceildiv":
        return sp.ceiling(xs[0] / xs[1])
    if kind == "mod":
        return sp.Mod(xs[0], xs[1])
    if kind == "pct_of":
        return xs[0] * xs[1] / 100
    if kind == "pct_inc":
        return xs[0] * (100 + xs[1]) / 100
    if kind == "pct_dec":
        return xs[0] * (100 - xs[1]) / 100
    if kind == "ratio_part":
        return xs[0] * xs[1]
    if kind == "pow":
        return xs[0] ** xs[1]
    if kind == "avg":
        return sp.Add(*xs) / len(xs)
    if kind == "min":
        return sp.Min(*xs)
    if kind == "max":
        return sp.Max(*xs)
    if kind == "neg":
        return -xs[0]
    if kind == "abs":
        return sp.Abs(xs[0])
    if kind == "floor":
        return sp.floor(xs[0])
    if kind == "ceil":
        return sp.ceiling(xs[0])
    raise ExecError(f"unknown op kind {kind}")


def _leaf_values(P: Program, values: dict | None) -> dict[int, Fraction]:
    env = {}
    for q in P.quantities.values():
        if q.role in ("given", "world_const"):
            v = values.get(q.qid, q.value) if values else q.value
            if v is None:
                raise ExecError(f"leaf {q.qid} has no value")
            env[q.qid] = Fraction(v)
    return env


def sympy_env(P: Program, values: dict | None = None, symbols: dict | None = None) -> dict:
    """SymPy expression for every node; unknowns (and any qid in `symbols`) stay symbolic."""
    env: dict[int, object] = {}
    symbols = symbols or {}
    for q in P.quantities.values():
        if q.qid in symbols:
            env[q.qid] = symbols[q.qid]
        elif q.role in ("given", "world_const"):
            v = Fraction(values.get(q.qid, q.value) if values else q.value)
            env[q.qid] = sp.Rational(v.numerator, v.denominator)
        elif q.role == "unknown":
            env[q.qid] = sp.Symbol(f"u{q.qid}", real=True)
    prod = P.producers()
    for n in P.topo():
        if n in prod and n not in symbols:
            o = prod[n]
            env[n] = _sym_op(o.kind, [env[i] for i in o.inputs])
    return env


def _forward(P: Program, env: dict[int, Fraction]) -> dict[int, Fraction]:
    prod = P.producers()
    for n in P.topo():
        if n in prod:
            o = prod[n]
            env[n] = apply_op(o.kind, [env[i] for i in o.inputs])
    return env


_SOLVE_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_SOLVE_CACHE_MAX = 4096


def _solve_key(P: Program, values: dict | None, domain: bool) -> tuple:
    qs = tuple((q.qid, q.role, (values or {}).get(q.qid, q.value), q.qtype.value, q.unit.divisible, q.unit.per)
               for q in sorted(P.quantities.values(), key=lambda x: x.qid))
    ops = tuple((o.kind, tuple(o.inputs), o.output) for o in P.ops)
    eqs = tuple((e.lhs, e.rhs) for e in P.equations)
    return qs, ops, eqs, tuple(P.constraints), domain


def solve_unknowns(P: Program, values: dict | None = None, domain: bool = True) -> list[dict[int, Fraction]]:
    """All rational solutions for the unknowns. domain=True keeps only solutions whose full execution satisfies
    every type constraint (integrality/sign at every node) and seed constraint. Raises ExecError if the system
    is underdetermined. Results are memoised on the program's structure and values (SymPy is the slow part)."""
    unk = P.unknowns()
    if not unk:
        return [{}]
    key = _solve_key(P, values, domain)
    hit = _SOLVE_CACHE.get(key)
    if hit is not None:
        _SOLVE_CACHE.move_to_end(key)
        if isinstance(hit, ExecError):
            raise ExecError(str(hit))
        return [dict(c) for c in hit]
    try:
        res = _solve_unknowns(P, values, domain, unk)
    except ExecError as e:
        _SOLVE_CACHE[key] = e
        raise
    _SOLVE_CACHE[key] = res
    if len(_SOLVE_CACHE) > _SOLVE_CACHE_MAX:
        _SOLVE_CACHE.popitem(last=False)
    return [dict(c) for c in res]


def _known_part(P: Program, values) -> dict[int, Fraction] | None:
    """Fraction values of every node that does not depend on an unknown (None on an arithmetic error)."""
    env = _leaf_values(P, values)
    prod = P.producers()
    for n in P.topo():
        if n in prod and all(i in env for i in prod[n].inputs):
            try:
                env[n] = apply_op(prod[n].kind, [env[i] for i in prod[n].inputs])
            except ExecError:
                return None
    return env


def _linear_solve(eqs: list, syms: list):
    """Exact solve when the system is linear and square-nonsingular; None -> use sympy.solve."""
    try:
        A, b = sp.linear_eq_to_matrix([e.lhs - e.rhs for e in eqs], syms)
    except (sp.polys.polyerrors.PolynomialError, ValueError, TypeError, NotImplementedError):
        return None
    if A.free_symbols or b.free_symbols or A.rows < A.cols:
        return None
    if A.rows > A.cols:
        r = A.rank()
        if r < A.cols:
            return None
    else:
        if A.det() == 0:
            return None
    try:
        x = A.solve_least_squares(b) if A.rows > A.cols else A.LUsolve(b)
    except Exception:
        return None
    if A * x != b:
        return []   # overdetermined and inconsistent
    return [dict(zip(syms, list(x)))]


def _numerator(expr):
    return sp.fraction(sp.together(expr))[0]


def _scan_univariate(eqs: list, x, lim: int = 10 ** 4, max_den: int = 10 ** 4) -> list[dict]:
    """Univariate NON-polynomial equations (unknown in an exponent, floor, min/max ...). sympy.solve can hang on
    these (e.g. 2**x = 114 in SymPy 1.14), so: scan [-lim, lim] for sign changes / exact zeros of the first
    equation, refine by bisection, rationalise each root (denominator <= max_den) and keep it only if EVERY
    equation holds exactly. A root that does not rationalise exactly is reported as irrational (excluded).
    This is a numeric search, not a proof of uniqueness (documented limit)."""
    import mpmath
    mpmath.mp.dps = 40
    f = sp.lambdify(x, eqs[0].lhs - eqs[0].rhs, modules="mpmath")

    def val(t):
        try:
            v = f(mpmath.mpf(t))
            return None if not mpmath.isfinite(v) or mpmath.im(v) != 0 else mpmath.re(v)
        except (ZeroDivisionError, ValueError, OverflowError, TypeError):
            return None

    pts = sorted({-lim, lim, 0} | {sgn * mpmath.mpf(10) ** (k / 8.0) for k in range(-24, 33) for sgn in (1, -1)}
                 | set(range(-60, 61)))
    pts = [p for p in pts if -lim <= p <= lim]
    roots = []
    prev_t, prev_v = None, None
    for t in pts:
        v = val(t)
        if v is not None and v == 0:
            roots.append(mpmath.mpf(t))
        elif v is not None and prev_v is not None and prev_v != 0 and (v > 0) != (prev_v > 0):
            a, b, fa = mpmath.mpf(prev_t), mpmath.mpf(t), prev_v
            for _ in range(200):
                m = (a + b) / 2
                fm = val(m)
                if fm is None:
                    break
                if fm == 0:
                    a = b = m
                    break
                if (fm > 0) == (fa > 0):
                    a, fa = m, fm
                else:
                    b = m
            roots.append((a + b) / 2)
        prev_t, prev_v = (t, v) if v is not None else (None, None)
    out = []
    irr = False
    for r in roots:
        q = Fraction(str(mpmath.nstr(r, 30))).limit_denominator(max_den)
        cand = sp.Rational(q.numerator, q.denominator)
        if all(sp.simplify((e.lhs - e.rhs).subs(x, cand)) == 0 for e in eqs):
            if {x: cand} not in out:
                out.append({x: cand})
        else:
            irr = True
    if not out and irr:
        raise ExecError("only irrational solutions (the IR is rational-valued)")
    return out


def _solve_unknowns(P: Program, values, domain: bool, unk) -> list[dict[int, Fraction]]:
    if domain:
        known = _known_part(P, values)
        if known is None or check_partial(P, known):
            return []   # a node not depending on the unknowns already violates its type: no domain solution
    env = sympy_env(P, values)
    syms = [env[q.qid] for q in unk]
    eqs = [sp.Eq(env[e.lhs], env[e.rhs]) for e in P.equations]
    if any(e is sp.false for e in eqs):
        return []
    eqs = [e for e in eqs if e is not sp.true]
    sols = _linear_solve(eqs, syms)
    if sols is None:
        numers = [_numerator(e.lhs - e.rhs) for e in eqs]
        if all(n.as_poly(*syms) is not None for n in numers):
            try:  # polynomial after clearing denominators: sympy.solve is reliable here
                sols = sp.solve(numers, syms, dict=True)
            except NotImplementedError as ex:
                raise ExecError(f"solve: {ex}")
        elif len(syms) == 1 and len(eqs) >= 1:
            sols = _scan_univariate(eqs, syms[0])
        else:
            raise ExecError("non-polynomial system in several unknowns (unsupported)")
    out = []
    irrational = 0
    for s in sols:
        if not all(sym in s for sym in syms):
            raise ExecError("underdetermined system")
        vals = [sp.nsimplify(s[sym]) for sym in syms]
        if any(v.free_symbols for v in vals):
            raise ExecError("underdetermined system")
        if not all(v.is_real for v in vals):
            continue
        if not all(v.is_Rational for v in vals):
            irrational += 1
            continue
        cand = {q.qid: Fraction(int(v.p), int(v.q)) for q, v in zip(unk, vals)}
        out.append(cand)
    if domain:
        good = []
        for cand in out:
            leaf = _leaf_values(P, values)
            leaf.update(cand)
            try:
                full = _forward(P, leaf)
            except ExecError:
                continue
            # the equations must hold exactly (floor/mod can make solve() over-approximate)
            if any(full[e.lhs] != full[e.rhs] for e in P.equations):
                continue
            if not check(P, full):  # type constraints at every node + seed constraints define the domain
                good.append(cand)
        out = good
    if not out and irrational:
        raise ExecError("only irrational solutions (the IR is rational-valued)")
    # dedupe
    uniq = []
    for c in out:
        if c not in uniq:
            uniq.append(c)
    return uniq


def execute(P: Program, values: dict | None = None) -> dict[int, Fraction]:
    """Exact value of every quantity. values overrides given/world_const values without copying P."""
    env = _leaf_values(P, values)
    if P.unknowns():
        sols = solve_unknowns(P, values)
        if len(sols) != 1:
            raise ExecError(f"{'no' if not sols else 'more than one'} solution in the domain ({len(sols)})")
        env.update(sols[0])
        return _forward(P, env)
    env = _forward(P, env)
    for e in P.equations:  # redundant equations must be consistent
        if env[e.lhs] != env[e.rhs]:
            raise ExecError("equation violated")
    return env


def run(P: Program) -> Fraction:
    env = execute(P)
    P.answer = env[P.query]
    return P.answer


# ----------------------------------------------------------------------------------- type checking
def type_violations(P: Program, env: dict[int, Fraction]) -> list[str]:
    """Integrality and sign at every node (T1 §2.1), plus value ranges for percents/ages/ratios."""
    bad = []
    for q in P.quantities.values():
        v = env.get(q.qid)
        if v is None:
            continue
        name = q.label_vi or f"q{q.qid}"
        if requires_integer(q.qtype, q.unit) and v.denominator != 1:
            bad.append(f"q{q.qid} ({name}): {q.qtype.value} '{q.unit}' is not an integer: {v}")
        if requires_nonneg(q.qtype) and v < 0:
            bad.append(f"q{q.qid} ({name}): {q.qtype.value} is negative: {v}")
        if q.qtype == QType.AGE and v > 120:
            bad.append(f"q{q.qid} ({name}): age {v} > 120")
        if q.qtype == QType.PERCENT and q.role == "given" and not (0 <= v <= 1000):
            bad.append(f"q{q.qid} ({name}): percent {v} out of range")
    return bad


def auto_constraints(P: Program) -> list[str]:
    out = []
    for q in sorted(P.quantities.values(), key=lambda x: x.qid):
        if requires_integer(q.qtype, q.unit):
            out.append(f"is_int(q{q.qid})")
        if requires_nonneg(q.qtype):
            out.append(f"q{q.qid} >= 0")
    return out


_CMP = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b}


def eval_constraint(c: str, env: dict[int, Fraction]) -> bool:
    """Safe evaluator: q<id> names, numbers, + - * / // % **, comparisons, and/or/not, is_int(x), abs(x)."""
    tree = ast.parse(c, mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return Fraction(str(n.value))
        if isinstance(n, ast.Name):
            if not n.id.startswith("q") or not n.id[1:].isdigit():
                raise IRError(f"bad name {n.id} in constraint {c!r}")
            k = int(n.id[1:])
            if k not in env:
                raise IRError(f"constraint {c!r} references unknown q{k}")
            return env[k]
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            if isinstance(n.op, ast.USub):
                return -v
            if isinstance(n.op, ast.Not):
                return not v
            return v
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
            if t is ast.FloorDiv:
                return Fraction(math.floor(a / b))
            if t is ast.Mod:
                return a - b * math.floor(a / b)
            if t is ast.Pow:
                return _exact_pow(a, b)
        if isinstance(n, ast.BoolOp):
            vals = [ev(x) for x in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, right in zip(n.ops, n.comparators):
                r = ev(right)
                if not _CMP[type(op)](left, r):
                    return False
                left = r
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and len(n.args) == 1:
            v = ev(n.args[0])
            if n.func.id == "is_int":
                return Fraction(v).denominator == 1
            if n.func.id == "abs":
                return abs(v)
        raise IRError(f"unsupported syntax in constraint {c!r}")

    return bool(ev(tree))


def check_partial(P: Program, env: dict[int, Fraction]) -> list[str]:
    """Type violations and seed constraints restricted to the nodes present in env."""
    bad = type_violations(P, env)
    for c in P.constraints:
        try:
            if not eval_constraint(c, env):
                bad.append(f"constraint failed: {c}")
        except (IRError, ZeroDivisionError, ExecError):
            pass  # references a node that is not computed yet
    return bad


def check(P: Program, env: dict[int, Fraction] | None = None) -> list[str]:
    """Every violated constraint: type constraints at every node + seed-specific constraint strings."""
    if env is None:
        env = execute(P)
    bad = type_violations(P, env)
    for c in P.constraints:
        try:
            if not eval_constraint(c, env):
                bad.append(f"constraint failed: {c}")
        except (ZeroDivisionError, ExecError):
            bad.append(f"constraint undefined: {c}")
    return bad


# ----------------------------------------------------------------------------------- unit algebra
_DIMENSIONAL = (QType.MONEY, QType.MEASURE, QType.RATE, QType.COUNT, QType.AGE, QType.TIME_POINT)
_SCALAR = (QType.RATIO, QType.ABSTRACT, QType.PERCENT)
_ADD_FAMILY = {QType.COUNT: "count", QType.MONEY: "money", QType.MEASURE: "measure", QType.AGE: "age",
               QType.TIME_POINT: "measure", QType.RATE: "rate", QType.PERCENT: "percent", QType.RATIO: "ratio"}
_HARD_CLASH = {frozenset({"count", "money"}), frozenset({"count", "measure"}), frozenset({"money", "measure"}),
               frozenset({"money", "age"}), frozenset({"count", "rate"}), frozenset({"money", "rate"})}


def infer_op_type(kind: str, ins: list[tuple[QType, Unit]]) -> tuple[tuple[QType, Unit], list[str]]:
    """Result type of an op from its input types, plus hard mismatch messages."""
    errs: list[str] = []
    dims = [(t, u) for t, u in ins if t not in _SCALAR]
    if kind in ("add", "sub", "min", "max", "avg", "neg", "abs", "floor", "ceil", "mod"):
        fams = {_ADD_FAMILY.get(t) for t, _ in ins if t != QType.ABSTRACT}
        fams.discard(None)
        for a in fams:
            for b in fams:
                if a != b and frozenset({a, b}) in _HARD_CLASH:
                    errs.append(f"{kind} mixes {a} and {b}")
        non_abs = [(t, u) for t, u in ins if t != QType.ABSTRACT]
        # ratio +/- count etc.: the dimensional operand wins
        pick = dims[0] if dims else (non_abs[0] if non_abs else ins[0])
        if kind in ("floor", "ceil") and pick[0] in _SCALAR:
            return (QType.ABSTRACT, Unit()), errs
        return pick, errs
    if kind in ("floordiv", "ceildiv"):
        a, b = ins
        if a[0] in _SCALAR or (b[0] not in _SCALAR and a[1].lemma == b[1].lemma):
            return (QType.COUNT, Unit(lemma="lần", divisible=False)), errs
        return (a[0] if a[0] in (QType.COUNT,) else QType.COUNT, a[1] if a[0] == QType.COUNT else
                Unit(lemma="", divisible=False)), errs
    if kind in ("pct_of", "pct_inc", "pct_dec", "ratio_part"):
        return ins[0], errs
    if kind == "pow":
        return ins[0], errs
    if kind == "mul":
        rates = [(t, u) for t, u in dims if t == QType.RATE]
        others = [(t, u) for t, u in dims if t != QType.RATE]
        if rates and others:
            r = rates[0][1]
            num = resolve_unit(r.lemma)
            e = lookup(r.lemma)
            qt = e.qtype if e else (QType.COUNT if not num.divisible else QType.MEASURE)
            return (qt, num), errs
        if rates:
            return rates[0], errs
        for pref in (QType.MONEY, QType.MEASURE, QType.COUNT, QType.AGE):
            for t, u in dims:
                if t == pref:
                    return (t, u), errs
        for pref in (QType.PERCENT, QType.RATIO):
            for t, u in ins:
                if t == pref:
                    return (QType.RATIO, Unit()), errs
        return (QType.ABSTRACT, Unit()), errs
    if kind == "div":
        a, b = ins
        if b[0] in _SCALAR:
            return a, errs
        if a[0] in _SCALAR:
            return (QType.RATE, Unit(lemma=a[1].lemma, per=b[1].lemma or "đơn vị")), errs
        if a[0] == b[0] and a[1].lemma == b[1].lemma and a[1].per == b[1].per:
            return (QType.RATIO, Unit()), errs
        if a[0] == QType.RATE and b[0] == QType.RATE:
            return (QType.RATE, Unit(lemma=b[1].per or "", per=a[1].per)), errs
        if b[0] == QType.RATE and a[1].lemma == b[1].lemma:
            per = resolve_unit(b[1].per or "")
            e = lookup(per.lemma)
            return ((e.qtype if e else QType.COUNT), per), errs
        return (QType.RATE, Unit(lemma=a[1].lemma, per=b[1].lemma or "đơn vị")), errs
    return (QType.ABSTRACT, Unit()), errs


def infer_types(P: Program, fill: bool = False) -> list[str]:
    """Propagate types along ops. fill=True writes the inferred type into derived nodes that are untyped
    (qtype ABSTRACT with an empty unit) in story programs. Returns hard unit mismatches."""
    prod = P.producers()
    mismatches = []
    for n in P.topo():
        if n not in prod:
            continue
        o = prod[n]
        ins = [(P.quantities[i].qtype, P.quantities[i].unit) for i in o.inputs]
        (qt, u), errs = infer_op_type(o.kind, ins)
        mismatches += [f"op {o.oid} -> q{n}: {e}" for e in errs]
        q = P.quantities[n]
        if fill and P.kind == "story" and q.qtype == QType.ABSTRACT and not q.unit.lemma:
            q.qtype, q.unit = qt, u
    return mismatches
