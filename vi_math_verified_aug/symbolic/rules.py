"""Rejection rules for mutated programs (T1 §2.2 step 2, §2.5).

is_answer_tail(P, P2)   the ancestor-difference rule (formal C3 criterion; see the docstring for the exact form)
is_identity(P, P2)      isomorphic DAG ∧ same givens ∧ same answer, plus the value-identity case ((a+k)-k)
clean_answer(P2, P)     integer counts, money on grid, <= 2 decimals for measures, small denominators, |a| < 1e7,
                        answer differs from the seed answer
plausible(P2)           every value inside its unit's lexicon range; stated quantities strictly positive
unique_solution(P2)     SymPy solve in the typed domain; Z3 integer enumeration when SymPy is underdetermined
is_semantic_tail(P, P2) OPTIONAL stricter, value-level check (not in the default gate): the new answer depends
                        on the old leaves only through the old answer (gradient-parallel test)
gate(P, P2, op)         runs the deterministic gates of T1 §2.2 step 2 in order; returns the first reason or None
"""
from __future__ import annotations

from fractions import Fraction

import sympy as sp
import z3

from .executor import ExecError, check, execute, solve_unknowns, sympy_env
from .ir import IRError, Program
from .units import QType, requires_integer, requires_nonneg, unit_class
from .z3enc import Unsupported, constraint_to_z3, domain_constraints, encode, model_value

__all__ = ["is_answer_tail", "is_identity", "clean_answer", "answer_issues", "plausible", "plausibility_issues",
           "unique_solution", "is_semantic_tail", "gate", "intact_images", "invented_leaves"]

MAX_ABS_ANSWER = Fraction(10 ** 7)


# ------------------------------------------------------------------------------------------- tail rule
def intact_images(P: Program, P2: Program) -> dict[int, set[int]]:
    """For every node of P, the P2 nodes that carry the SAME computation: a leaf maps to the P2 leaf with the
    same qid (any value: a resampled given is still the same story quantity); a derived node maps to every P2
    derived node whose op has the same kind and whose inputs are images of the original inputs (in order,
    or in any order for commutative ops). Matching is structural, so renaming the old answer node or
    re-defining the query qid in place does not hide a tail."""
    from .ir import OP_KINDS
    img: dict[int, set[int]] = {}
    for q in P.quantities.values():
        if q.role != "derived" and q.qid in P2.quantities and P2.quantities[q.qid].role in ("given", "world_const",
                                                                                              "unknown"):
            if P2.quantities[q.qid].role == q.role:
                img[q.qid] = {q.qid}
    prod1 = P.producers()
    ops2 = P2.ops
    for n in P.topo():
        if n not in prod1:
            continue
        o = prod1[n]
        if not all(i in img for i in o.inputs):
            continue
        hits = set()
        for o2 in ops2:
            if o2.kind != o.kind or len(o2.inputs) != len(o.inputs):
                continue
            if OP_KINDS[o.kind][2]:
                rem = list(o2.inputs)
                ok = True
                for i in o.inputs:
                    m = next((x for x in rem if x in img[i]), None)
                    if m is None:
                        ok = False
                        break
                    rem.remove(m)
            else:
                ok = all(x in img[i] for x, i in zip(o2.inputs, o.inputs))
            if ok:
                hits.add(o2.output)
        if hits:
            img[n] = hits
    return img


def invented_leaves(P: Program, P2: Program) -> set[int]:
    """Leaves of P2 that do not exist in P and are not story material from an M6 partner seed."""
    return {q.qid for q in P2.quantities.values()
            if q.role != "derived" and q.qid not in P.quantities and q.origin != "partner"}


def _extras(P: Program, P2: Program, anc_new: set[int], x: int, old_images: set[int]) -> set[int]:
    """Nodes the new query uses on top of x: leaves and old story nodes; new plumbing nodes are transparent."""
    diff = anc_new - P2.ancestors(x) - {x}
    return {n for n in diff if P2.quantities[n].role != "derived" or n in old_images}


def is_answer_tail(P: Program, P2: Program) -> bool:
    """T1 §2.2 ancestor-difference rule.

    Let q0 = the old query and q0' its intact image in P2 (same computation). Reject iff
        q_new != q0',  q0' is an ancestor of q_new,  and  A(q_new) \\ A(q0') \\ {q0'} ⊆ C_new
    where C_new are invented leaves (new constants that are not story material) and new derived "plumbing"
    nodes are looked through. Two refinements over the literal doc formula, both documented in the X4 report:
      * q0' must be an ancestor of q_new: a new query that does not consume the old answer (M5 asking for an
        intermediate) is not a post-composition of the answer;
      * the same test is applied to every other intact old derived node x when q_new is a NEW node and at
        least one invented constant sits on top of x ("intermediate tail": x + k for an intermediate x).
    If q0's computation is no longer intact (M2 makes it a given; M3/M4 change an upstream op) there is no
    old answer to post-compose, so the rule cannot fire."""
    try:
        img = intact_images(P, P2)
    except (IRError, KeyError):
        return False
    qn = P2.query
    old_images = set().union(*img.values()) if img else set()
    inv = invented_leaves(P, P2)
    anc_new = P2.ancestors(qn)
    q0_imgs = img.get(P.query, set())
    if P.quantities[P.query].role in ("derived", "unknown"):
        for q0i in q0_imgs:
            if q0i == qn:
                continue
            if q0i in anc_new and _extras(P, P2, anc_new, q0i, old_images) <= inv:
                return True
    # intermediate tails: a NEW query node = f(old intermediate x, invented constants)
    if qn not in old_images:
        for x_old, xs in img.items():
            if P.quantities[x_old].role not in ("derived", "unknown"):
                continue
            for x in xs:
                if x in anc_new:
                    ex = _extras(P, P2, anc_new, x, old_images)
                    if ex and ex <= inv:
                        return True
    return False


# ------------------------------------------------------------------------------------------- identity
def _given_multiset(P: Program) -> list:
    return sorted((q.qtype.value, q.value) for q in P.givens())


def is_identity(P: Program, P2: Program) -> bool:
    """Reject if P2 is P in disguise:
    (a) canonical DAG isomorphic ∧ same given values ∧ same answer (T1 §2.2), or
    (b) same answer ∧ every given of P kept with its value ∧ P2 adds no story material, only invented
        constants — catches the old '(e+k)-k' identities whose DAG differs."""
    try:
        a1, a2 = execute(P)[P.query], execute(P2)[P2.query]
    except (ExecError, IRError):
        return False
    if a1 != a2:
        return False
    if P.canonical() == P2.canonical() and _given_multiset(P) == _given_multiset(P2):
        return True
    for g in P.givens():
        g2 = P2.quantities.get(g.qid)
        if g2 is None or g2.role != "given" or g2.value != g.value:
            return False
    new_leaves = {q.qid for q in P2.quantities.values() if q.role != "derived" and q.qid not in P.quantities}
    return new_leaves <= invented_leaves(P, P2)


# ------------------------------------------------------------------------------------------- clean / plausible
_MONEY_GRID = {"vnd": Fraction(500), "vnd_k": Fraction(1, 2), "vnd_m": Fraction(1, 1000), "cent": Fraction(1)}


def answer_issues(P2: Program, P: Program | None = None, require_new: bool = True) -> list[str]:
    try:
        a = execute(P2)[P2.query]
    except (ExecError, IRError) as e:
        return [f"exec: {e}"]
    q = P2.quantities[P2.query]
    out = []
    if a == 0 and P2.kind == "story":
        out.append("answer is zero (degenerate story problem)")
    if abs(a) >= MAX_ABS_ANSWER:
        out.append(f"|answer| {a} >= 1e7")
    if requires_integer(q.qtype, q.unit) and a.denominator != 1:
        out.append(f"{q.qtype.value} answer not an integer: {a}")
    if requires_nonneg(q.qtype) and a < 0:
        out.append(f"{q.qtype.value} answer negative: {a}")
    if q.qtype == QType.MONEY:
        g = _MONEY_GRID.get(unit_class(q.unit, q.qtype), Fraction(1, 100))
        if (a / g).denominator != 1:
            out.append(f"money answer {a} not on grid {g}")
    elif q.qtype in (QType.MEASURE, QType.RATE, QType.PERCENT):
        if (a * 100).denominator != 1:
            out.append(f"{q.qtype.value} answer {a} has more than 2 decimals")
    elif q.qtype in (QType.RATIO, QType.ABSTRACT):
        lim = 12 if P2.kind == "story" else 100
        if a.denominator > lim:
            out.append(f"answer {a} denominator > {lim}")
    if require_new and P is not None:
        try:
            if execute(P)[P.query] == a:
                out.append("answer equals the seed answer")
        except (ExecError, IRError):
            pass
    return out


def clean_answer(P2: Program, P: Program | None = None, require_new: bool = True) -> bool:
    return not answer_issues(P2, P, require_new)


def plausibility_issues(P2: Program) -> list[str]:
    try:
        env = execute(P2)
    except (ExecError, IRError) as e:
        return [f"exec: {e}"]
    out = []
    for q in P2.quantities.values():
        v = env[q.qid]
        if q.unit.plausible is not None:
            lo, hi = q.unit.plausible
            if not (lo <= v <= hi):
                out.append(f"q{q.qid} ({q.label_vi}) = {v} outside [{lo}, {hi}] for '{q.unit}'")
        if q.role == "given" and q.qtype not in (QType.ABSTRACT, QType.TIME_POINT) and v <= 0 and not q.distractor:
            out.append(f"given q{q.qid} ({q.label_vi}) = {v} is not positive")
        if q.qtype == QType.PERCENT and q.role == "given" and v > 100 and q.origin == "inserted":
            out.append(f"inserted percent {v} > 100")
    return out


def plausible(P2: Program) -> bool:
    return not plausibility_issues(P2)


# ------------------------------------------------------------------------------------------- uniqueness
def unique_solution(P2: Program, bound: int = 10 ** 6, timeout_ms: int = 3000) -> tuple[bool, str]:
    """(True, how) iff the program has exactly one solution in its typed domain.
    No unknowns: the forward program is a function -> unique iff it executes and passes check().
    Unknowns: SymPy solve, solutions filtered by the domain (integrality/sign at every node + seed
    constraints). If SymPy reports an underdetermined system and every unknown is integer-typed, Z3 searches
    for two distinct integer solutions (model, then blocking clause) within [0, bound]."""
    if not P2.unknowns():
        try:
            env = execute(P2)
        except (ExecError, IRError) as e:
            return False, f"exec: {e}"
        bad = check(P2, env)
        return (not bad), ("forward" if not bad else f"domain: {bad[0]}")
    try:
        sols = solve_unknowns(P2)
        if len(sols) == 1:
            return True, "sympy"
        return False, f"sympy: {len(sols)} solutions in the domain"
    except ExecError as e:
        if "underdetermined" not in str(e):
            return False, f"sympy: {e}"
    unk = P2.unknowns()
    if not all(requires_integer(q.qtype, q.unit) for q in unk):
        return False, "underdetermined over a non-integer domain (infinitely many solutions)"
    try:
        xs = {q.qid: z3.Int(f"x{q.qid}") for q in unk}
        var = {k: z3.ToReal(v) for k, v in xs.items()}
        env, side = encode(P2, var)
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        s.add(side)
        s.add(domain_constraints(P2, env))
        s.add([env[e.lhs] == env[e.rhs] for e in P2.equations])
        s.add([constraint_to_z3(c, env) for c in P2.constraints])
        for v in xs.values():
            s.add(v >= 0, v <= bound)
        r = s.check()
        if r == z3.unsat:
            return False, "z3: no integer solution"
        if r != z3.sat:
            return False, "z3: unknown"
        m = s.model()
        s.add(z3.Or([v != m.eval(v, model_completion=True) for v in xs.values()]))
        r2 = s.check()
        if r2 == z3.unsat:
            return True, "z3"
        return False, "z3: second integer solution exists" if r2 == z3.sat else "z3: unknown"
    except Unsupported as e:
        return False, f"z3 unsupported: {e}"


# ------------------------------------------------------------------------------------------- semantic tail
def is_semantic_tail(P: Program, P2: Program) -> bool:
    """Optional value-level check: True if the new answer depends on P's leaves only through the old answer
    (E = f(Q0, new constants)): the gradients of E and Q0 w.r.t. P's leaves are parallel. Programs with
    unknowns are skipped (False). Pure value resampling (identical canonical DAG) is never a semantic tail."""
    if P.unknowns() or P2.unknowns():
        return False
    if P.canonical() == P2.canonical():
        return False
    leaves = [q.qid for q in P.quantities.values() if q.role in ("given", "world_const")]
    syms = {k: sp.Symbol(f"s{k}", positive=True) for k in leaves}
    shared = {k: s for k, s in syms.items() if k in P2.quantities and P2.quantities[k].role in ("given", "world_const")}
    try:
        e1 = sympy_env(P, symbols=syms)[P.query]
        e2 = sympy_env(P2, symbols=shared)[P2.query]
    except Exception:
        return False
    g1 = [sp.diff(e1, s) for s in shared.values()]
    g2 = [sp.diff(e2, s) for s in shared.values()]
    if all(sp.simplify(x) == 0 for x in g2):
        return False  # does not use the old story at all
    n = len(g1)
    for i in range(n):
        for j in range(i + 1, n):
            if sp.simplify(g2[i] * g1[j] - g2[j] * g1[i]) != 0:
                return False
    return True


# ------------------------------------------------------------------------------------------- gate
def gate(P: Program, P2: Program, op: str = "") -> str | None:
    """Deterministic symbolic gates of T1 §2.2 step 2, in order. Returns a reject reason or None."""
    try:
        P2.check_structure()
        env = execute(P2)
    except (ExecError, IRError) as e:
        return f"exec: {e}"
    bad = check(P2, env)
    if bad:
        return f"types: {bad[0]}"
    ok, why = unique_solution(P2)
    if not ok:
        return f"non_unique: {why}"
    if is_answer_tail(P, P2):
        return "tail"
    if is_identity(P, P2):
        return "identity"
    iss = answer_issues(P2, P, require_new=op in ("", "M1", "M2", "M3", "M4", "M5"))
    if iss:
        return f"ugly: {iss[0]}"
    pl = plausibility_issues(P2)
    if pl:
        return f"implausible: {pl[0]}"
    return None
