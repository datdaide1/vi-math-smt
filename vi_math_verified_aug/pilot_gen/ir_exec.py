"""Minimal typed quantity-DAG IR + exact executor (Fraction arithmetic) for the G1 pilot (T1 §2.1).

IR JSON: {"quantities": [{id, role, kind, unit, label, value?, span?, context?, expr?, clause?, distractor?}],
          "equations": ["lhs == rhs", ...], "query": id}
roles: given (literal in the text, span-anchored) | world_const (implicit fact) | derived (expr over ids) |
       unknown (fixed by LINEAR equations; solved exactly by Gaussian elimination over Fractions).

Formalization pass (task spec): executes exactly to gold AND every given is span-anchored (span found in the
Vietnamese text, non-overlapping, and its literal parses to the given's value) AND the query is derived
(role derived/unknown, depth >= 1). Secondary flags (reported, not part of pass): hard-coded text literal in
an expr, unreachable non-distractor given, type violations (non-integer/negative counts...).
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

from .numerals import numbers_in_text, parse_literal

KINDS = {"count", "money", "measure", "rate", "percent", "ratio", "age", "time_point", "abstract"}
INT_KINDS = {"count", "age"}
NONNEG = {"count", "money", "measure", "rate", "age", "percent"}
FUNCS = {"floor", "ceil", "min", "max", "abs"}
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant, ast.Call,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd)


class IRError(Exception):
    pass


def frac(v) -> Fraction:
    if isinstance(v, Fraction):
        return v
    if isinstance(v, bool):
        raise IRError("bool value")
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        return Fraction(str(v))
    s = str(v).strip().replace(" ", "")
    if "/" in s:
        a, b = s.split("/", 1)
        return Fraction(frac(a), frac(b))
    if s.count(",") == 1 and "." not in s:
        s = s.replace(",", ".")
    return Fraction(s.replace(",", ""))


def parse_expr(e: str) -> ast.Expression:
    try:
        t = ast.parse(str(e).strip().replace("^", "**"), mode="eval")
    except SyntaxError as ex:
        raise IRError(f"bad expr {e!r}: {ex}")
    for n in ast.walk(t):
        if not isinstance(n, _ALLOWED):
            raise IRError(f"disallowed {type(n).__name__} in {e!r}")
        if isinstance(n, ast.Call) and (not isinstance(n.func, ast.Name) or n.func.id not in FUNCS):
            raise IRError(f"disallowed call in {e!r}")
        if isinstance(n, ast.Constant) and (isinstance(n.value, bool) or not isinstance(n.value, (int, float))):
            raise IRError(f"non-numeric constant in {e!r}")
    return t


def names(e: str) -> set:
    return {n.id for n in ast.walk(parse_expr(e)) if isinstance(n, ast.Name) and n.id not in FUNCS}


def literals(e: str) -> list:
    return [n.value for n in ast.walk(parse_expr(e)) if isinstance(n, ast.Constant)]


def ev(node, env):
    if isinstance(node, ast.Expression):
        return ev(node.body, env)
    if isinstance(node, ast.Constant):
        return frac(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise IRError(f"unknown id {node.id}")
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        v = ev(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Call):
        a = [ev(x, env) for x in node.args]
        f = node.func.id
        if f == "floor":
            return Fraction(math.floor(a[0]))
        if f == "ceil":
            return Fraction(math.ceil(a[0]))
        if f == "abs":
            return abs(a[0])
        return (min if f == "min" else max)(a)
    a, b = ev(node.left, env), ev(node.right, env)
    op = node.op
    if isinstance(op, ast.Add):
        return a + b
    if isinstance(op, ast.Sub):
        return a - b
    if isinstance(op, ast.Mult):
        return a * b
    if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)) and b == 0:
        raise IRError("division by zero")
    if isinstance(op, ast.Div):
        return a / b
    if isinstance(op, ast.FloorDiv):
        return Fraction(math.floor(a / b))
    if isinstance(op, ast.Mod):
        return a - b * math.floor(a / b)
    if isinstance(op, ast.Pow):
        if b.denominator != 1 or abs(b) > 12:
            raise IRError("non-integer or huge exponent")
        return a ** int(b)
    raise IRError("bad node")


@dataclass
class Q:
    id: str
    role: str
    kind: str = "abstract"
    unit: str = ""
    label: str = ""
    value: Fraction | None = None
    expr: str | None = None
    span: str | None = None
    context: str | None = None
    clause: str | None = None
    distractor: bool = False
    start: int | None = None
    end: int | None = None


@dataclass
class Program:
    text: str
    qs: dict
    query: str
    equations: list = field(default_factory=list)
    answer: Fraction | None = None

    @staticmethod
    def from_json(text: str, j: dict) -> "Program":
        if not isinstance(j, dict) or not isinstance(j.get("quantities"), list):
            raise IRError("no quantities list")
        qs = {}
        for q in j["quantities"]:
            if not isinstance(q, dict):
                raise IRError("quantity not an object")
            qid = str(q.get("id", "")).strip()
            if not qid.isidentifier() or qid in FUNCS or qid in qs:
                raise IRError(f"bad/duplicate id {qid!r}")
            role = q.get("role")
            if role not in ("given", "world_const", "derived", "unknown"):
                raise IRError(f"bad role {role!r}")
            kind = q.get("kind") if q.get("kind") in KINDS else "abstract"
            val = None
            if role in ("given", "world_const"):
                try:
                    val = frac(q.get("value"))
                except Exception:
                    raise IRError(f"bad value for {qid}: {q.get('value')!r}")
            expr = None
            if role == "derived":
                expr = str(q.get("expr") or "")
                if not expr:
                    raise IRError(f"derived {qid} has no expr")
                parse_expr(expr)
            qs[qid] = Q(qid, role, kind, str(q.get("unit") or ""), str(q.get("label") or ""), val, expr,
                        None if q.get("span") is None else str(q.get("span")),
                        None if q.get("context") is None else str(q.get("context")),
                        None if q.get("clause") is None else str(q.get("clause")), bool(q.get("distractor")))
        eqs = []
        for e in j.get("equations") or []:
            e = str(e)
            if "==" not in e:
                if e.count("=") != 1:
                    raise IRError(f"bad equation {e!r}")
                e = e.replace("=", "==")
            l, r = e.split("==", 1)
            parse_expr(l), parse_expr(r)
            eqs.append(e)
        query = str(j.get("query", ""))
        if query not in qs:
            raise IRError(f"query {query!r} not a quantity")
        P = Program(text, qs, query, eqs)
        for q in qs.values():
            if q.role == "derived":
                for n in names(q.expr):
                    if n not in qs:
                        raise IRError(f"{q.id} references unknown id {n}")
        for e in eqs:
            for s in e.split("=="):
                for n in names(s):
                    if n not in qs:
                        raise IRError(f"equation references unknown id {n}")
        P.topo()
        return P

    def to_json(self) -> dict:
        out = []
        for q in self.qs.values():
            d = {"id": q.id, "role": q.role, "kind": q.kind, "unit": q.unit, "label": q.label}
            if q.value is not None:
                d["value"] = str(q.value)
            for k in ("expr", "span", "context", "clause", "start", "end"):
                if getattr(q, k) is not None:
                    d[k] = getattr(q, k)
            if q.distractor:
                d["distractor"] = True
            out.append(d)
        return {"quantities": out, "equations": self.equations, "query": self.query,
                "answer": None if self.answer is None else str(self.answer)}

    # ----- graph
    def parents(self, qid):
        q = self.qs[qid]
        return names(q.expr) if q.role == "derived" else set()

    def topo(self):
        order, st = [], {}

        def visit(n):
            if st.get(n) == 1:
                raise IRError("cycle")
            if st.get(n) == 2:
                return
            st[n] = 1
            for p in self.parents(n):
                visit(p)
            st[n] = 2
            order.append(n)
        for n in self.qs:
            visit(n)
        return order

    def ancestors(self, qid) -> set:
        eqn = [set().union(*[names(s) for s in e.split("==")]) for e in self.equations]
        seen, stack = set(), [qid]
        while stack:
            n = stack.pop()
            nb = set(self.parents(n))
            if self.qs[n].role == "unknown":
                for s in eqn:
                    if n in s:
                        nb |= s - {n}
            for p in nb:
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def depth(self, qid, memo=None) -> int:
        memo = {} if memo is None else memo
        if qid in memo:
            return memo[qid]
        q = self.qs[qid]
        if q.role == "unknown":
            d = 1
        elif q.role != "derived":
            d = 0
        else:
            d = 1 + max([self.depth(p, memo) for p in self.parents(qid)], default=0)
        memo[qid] = d
        return d

    def givens(self):
        return [q for q in self.qs.values() if q.role == "given"]

    # ----- execution
    def _forward(self, env):
        env = dict(env)
        for qid in self.topo():
            q = self.qs[qid]
            if q.role == "derived":
                env[qid] = ev(parse_expr(q.expr), env)
        return env

    def execute(self) -> dict:
        base = {q.id: q.value for q in self.qs.values() if q.role in ("given", "world_const")}
        unk = [q.id for q in self.qs.values() if q.role == "unknown"]
        if not unk:
            env = self._forward(base)
            for e in self.equations:
                l, r = e.split("==", 1)
                if ev(parse_expr(l), env) != ev(parse_expr(r), env):
                    raise IRError("redundant equation violated")
            return env
        if len(self.equations) < len(unk):
            raise IRError("fewer equations than unknowns")

        def resid(u):
            env = dict(base)
            env.update(dict(zip(unk, u)))
            # derived nodes that depend on unknowns are computed lazily via forward pass
            env = self._forward(env)
            out = []
            for e in self.equations:
                l, r = e.split("==", 1)
                out.append(ev(parse_expr(l), env) - ev(parse_expr(r), env))
            return out
        n = len(unk)
        z = [Fraction(0)] * n
        f0 = resid(z)
        cols = []
        for i in range(n):
            e = list(z)
            e[i] = Fraction(1)
            cols.append([a - b for a, b in zip(resid(e), f0)])
        # linearity check at a second point
        pt = [Fraction(7 + 3 * i, 2) for i in range(n)]
        pred = [f0[k] + sum(cols[i][k] * pt[i] for i in range(n)) for k in range(len(f0))]
        if resid(pt) != pred:
            raise IRError("equations not linear in the unknowns")
        # solve A u = -f0 (A: m x n), require rank n and consistency
        m = len(f0)
        A = [[cols[i][k] for i in range(n)] + [-f0[k]] for k in range(m)]
        row = 0
        piv = []
        for c in range(n):
            pr = next((r for r in range(row, m) if A[r][c] != 0), None)
            if pr is None:
                raise IRError("non-unique solution (rank deficient)")
            A[row], A[pr] = A[pr], A[row]
            pv = A[row][c]
            A[row] = [x / pv for x in A[row]]
            for r in range(m):
                if r != row and A[r][c] != 0:
                    fct = A[r][c]
                    A[r] = [x - fct * y for x, y in zip(A[r], A[row])]
            piv.append(c)
            row += 1
        for r in range(row, m):
            if A[r][n] != 0:
                raise IRError("inconsistent equations")
        u = [A[i][n] for i in range(n)]
        env = dict(base)
        env.update(dict(zip(unk, u)))
        return self._forward(env)

    def run(self) -> Fraction:
        env = self.execute()
        self.answer = env[self.query]
        return self.answer

    def type_violations(self, env) -> list:
        bad = []
        for q in self.qs.values():
            v = env.get(q.id)
            if v is None:
                continue
            if q.kind in INT_KINDS and v.denominator != 1:
                bad.append(f"{q.id}:{q.kind} non-integer {v}")
            if q.kind in NONNEG and v < 0:
                bad.append(f"{q.id}:{q.kind} negative {v}")
        return bad


def locate_spans(P: Program) -> list:
    errs, taken = [], []
    for q in P.givens():
        if not q.span:
            errs.append(f"{q.id}: no span")
            continue
        pos = -1
        cands = []
        if q.context and q.context in P.text and q.span in q.context:
            c0 = P.text.find(q.context)
            while c0 >= 0:
                cands.append(c0 + q.context.find(q.span))
                c0 = P.text.find(q.context, c0 + 1)
        i = P.text.find(q.span)
        while i >= 0:
            cands.append(i)
            i = P.text.find(q.span, i + 1)
        L = len(q.span)
        for i in cands:
            b = P.text[i - 1] if i > 0 else " "
            a = P.text[i + L] if i + L < len(P.text) else " "
            ok_b = not (b.isalnum() and q.span[0].isalnum())
            ok_a = not (a.isalnum() and q.span[-1].isalnum())
            if q.span[-1].isdigit() and a in ".," and i + L + 1 < len(P.text) and P.text[i + L + 1].isdigit():
                ok_a = False
            if ok_b and ok_a and not any(s < i + L and i < e for s, e in taken):
                pos = i
                break
        if pos < 0:
            errs.append(f"{q.id}: span {q.span!r} not found as a token in the text")
            continue
        q.start, q.end = pos, pos + L
        taken.append((pos, pos + L))
        vals = parse_literal(q.span)
        if q.value not in vals and not (q.kind == "percent" and (q.value * 100 in vals or q.value / 100 in vals)):
            errs.append(f"{q.id}: span {q.span!r} parses to {sorted(map(str, vals))} != value {q.value}")
    return errs


def validate(P: Program, gold: Fraction | None) -> tuple[list, list]:
    """Return (errors, flags). errors empty == formalization pass."""
    try:
        env = P.execute()
    except IRError as e:
        return [f"exec: {e}"], []
    except Exception as e:
        return [f"exec: {e!r}"], []
    P.answer = env[P.query]
    errs, flags = [], []
    if gold is not None and P.answer != gold:
        errs.append(f"answer {P.answer} != gold {gold}")
    errs += locate_spans(P)
    q = P.qs[P.query]
    if q.role not in ("derived", "unknown") or P.depth(P.query) < 1:
        errs.append("query is not derived")
    anc = P.ancestors(P.query)
    for g in P.givens():
        if g.id not in anc and not g.distractor:
            flags.append(f"{g.id}: unreachable given")
    flags += P.type_violations(env)
    covered = [(g.start, g.end) for g in P.givens() if g.start is not None]
    free = set()
    for lit, vals, s0, e0 in numbers_in_text(P.text):
        if not any(s0 < e and s < e0 for s, e in covered):
            free |= vals
    for qq in P.qs.values():
        exprs = [qq.expr] if qq.role == "derived" else []
        for ex in exprs + [s for e in P.equations for s in e.split("==")]:
            for lit in literals(ex):
                fv = frac(lit)
                if fv in free and fv not in (0, 1):
                    flags.append(f"hard-coded text literal {lit}")
    return errs, sorted(set(flags))


_TAG = re.compile(r"<q\s+n\s*=\s*[\"']?([A-Za-z_][\w]*)[\"']?\s*>(.*?)</q>", re.S)


def tag_text(P: Program, ids=None) -> str:
    """Seed text with every (selected) given's span wrapped as <q n="id">literal</q>."""
    gs = sorted([g for g in P.givens() if g.start is not None and (ids is None or g.id in ids)],
                key=lambda g: g.start)
    out, last = [], 0
    for g in gs:
        out.append(P.text[last:g.start])
        out.append(f'<q n="{g.id}">{P.text[g.start:g.end]}</q>')
        last = g.end
    out.append(P.text[last:])
    return "".join(out)


def parse_tags(tagged: str) -> list:
    return [(m.group(1), m.group(2).strip()) for m in _TAG.finditer(tagged or "")]


def strip_tags(tagged: str) -> str:
    return _TAG.sub(lambda m: m.group(2), tagged or "")


def untagged_numerals(tagged: str) -> list:
    rest = _TAG.sub(" ", tagged or "")
    return sorted(m.group(0) for m in re.finditer(r"\d+(?:[.,]\d+)*", rest))
