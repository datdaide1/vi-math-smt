"""Typed quantity DAG IR (T1 §2.1, 01_TECHNICAL_DESIGN §3.1).

A Program is a set of typed Quantities (nodes) and Ops (one op produces one derived quantity), optional
Equations (for unknowns: inverse / algebra problems), seed-specific constraint strings and one query.

Roles
    given        literal stated in the text; must carry a span (char offsets into Program.text)
    world_const  implicit fact (60 phút/giờ, 7 ngày/tuần, the "4" people in "Ann, Bill, Cate và Dale")
    unknown      not stated; fixed by equations (M2 inverse, algebra seeds)
    derived      output of exactly one Op

Origins (provenance, used by the answer-tail rule)
    seed         came from the formalized seed text
    partner      came from the partner seed of an M6 composition
    inserted     a constant invented by an operator (M3 step constant, legacy wrap constants, ...)
    stated_fact  the old answer restated as a given by M2
    operator     plumbing node created by an operator (M3 output x', M5 combination node, M2 copy node)

Op kinds (exact semantics in execute.py): add, sub, mul, div, floordiv, ceildiv, mod, pct_of, pct_inc,
pct_dec, ratio_part, pow, avg, min, max, neg, abs, floor, ceil.

JSON schema: see SCHEMA_VERSION and Program.to_dict(); loaders accept the current major version and the
lenient G1/formalizer "expression" format through from_expr_json().
"""
from __future__ import annotations

import ast
import copy
import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable

from .numerals import parse_literal, normalize_text
from .units import QType, Unit, resolve_unit, default_qtype

__all__ = ["SCHEMA_VERSION", "ROLES", "ORIGINS", "OP_KINDS", "IRError", "Quantity", "Op", "Equation", "Program",
           "ProgramBuilder", "split_sentences", "locate_span", "from_expr_json", "to_fraction"]

SCHEMA_VERSION = "vi-math-ir/1.0"
ROLES = ("given", "world_const", "unknown", "derived")
ORIGINS = ("seed", "partner", "inserted", "stated_fact", "operator")
OP_KINDS = {
    # kind: (min_inputs, max_inputs or None, commutative)
    "add": (2, None, True), "sub": (2, None, False), "mul": (2, None, True), "div": (2, 2, False),
    "floordiv": (2, 2, False), "ceildiv": (2, 2, False), "mod": (2, 2, False), "pct_of": (2, 2, False),
    "pct_inc": (2, 2, False), "pct_dec": (2, 2, False), "ratio_part": (2, 2, False), "pow": (2, 2, False),
    "avg": (2, None, True), "min": (2, None, True), "max": (2, None, True), "neg": (1, 1, False),
    "abs": (1, 1, False), "floor": (1, 1, False), "ceil": (1, 1, False),
}


class IRError(ValueError):
    """Structural problem in a Program (bad reference, cycle, unknown op...)."""


def to_fraction(v) -> Fraction:
    if isinstance(v, Fraction):
        return v
    if isinstance(v, bool):
        raise IRError("boolean is not a number")
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        return Fraction(str(v))
    s = str(v).strip().replace(" ", "")
    if "/" in s:
        a, b = s.split("/", 1)
        return Fraction(to_fraction(a), to_fraction(b))
    if s.count(",") == 1 and "." not in s:
        s = s.replace(",", ".")
    try:
        return Fraction(s.replace(",", ""))
    except ValueError as e:
        raise IRError(f"not a number: {v!r}") from e


# ------------------------------------------------------------------------------------------ sentences
def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence char ranges. A '.' between digits ('1.000', '0.5') never ends a sentence."""
    out = []
    start = 0
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch in ".?!":
            if ch == "." and 0 < i < n - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
                i += 1
                continue
            j = i + 1
            while j < n and text[j] in ".?!\"'”)":
                j += 1
            if j == n or text[j].isspace():
                out.append((start, j))
                while j < n and text[j].isspace():
                    j += 1
                start = j
                i = j
                continue
        i += 1
    if start < n and text[start:].strip():
        out.append((start, n))
    return out or [(0, n)]


def _token_ok(text: str, s: int, e: int) -> bool:
    lit = text[s:e]
    before = text[s - 1] if s > 0 else " "
    after = text[e] if e < len(text) else " "
    if lit[:1].isalnum() and before.isalnum():
        return False
    if lit[:1].isdigit() and (before.isdigit() or (before in ".," and s > 1 and text[s - 2].isdigit())):
        return False
    if lit[-1:].isdigit():
        if after.isdigit():
            return False
        if after in ".," and e + 1 < len(text) and text[e + 1].isdigit():
            return False
    elif lit[-1:].isalpha() and after.isalpha():
        return False
    return True


def locate_span(text: str, literal: str, context: str | None = None, occurrence: int | None = None,
                taken: Iterable[tuple[int, int]] = ()) -> tuple[int, int] | None:
    """Find a whole-token occurrence of literal in text, optionally inside an exact context quote,
    skipping occurrences overlapping 'taken'. occurrence=k picks the k-th whole-token match."""
    taken = list(taken)
    if not literal:
        return None
    hay = text
    low_text = text.lower()
    low_lit = literal.lower()

    def free(s, e):
        return not any(a < e and s < b for a, b in taken)

    cands = []
    i = low_text.find(low_lit)
    while i >= 0:
        if _token_ok(hay, i, i + len(literal)):
            cands.append((i, i + len(literal)))
        i = low_text.find(low_lit, i + 1)
    if context:
        c0 = text.find(context)
        ctx_cands = []
        while c0 >= 0:
            ctx_cands += [c for c in cands if c0 <= c[0] and c[1] <= c0 + len(context)]
            c0 = text.find(context, c0 + 1)
        if ctx_cands:
            cands = ctx_cands
    if occurrence is not None:
        return cands[occurrence] if 0 <= occurrence < len(cands) else None
    for c in cands:
        if free(*c):
            return c
    return None


# ------------------------------------------------------------------------------------------ dataclasses
@dataclass
class Quantity:
    qid: int
    value: Fraction | None
    qtype: QType
    unit: Unit
    label_vi: str
    role: str
    span: tuple[int, int] | None = None
    sentence_id: int | None = None
    origin: str = "seed"
    distractor: bool = False
    fixed: bool = False          # never resampled (lexical or structural literal)

    def to_dict(self) -> dict:
        return {"qid": self.qid, "role": self.role, "value": None if self.value is None else str(self.value),
                "qtype": self.qtype.value, "unit": self.unit.to_dict(), "label_vi": self.label_vi,
                "span": None if self.span is None else list(self.span), "sentence_id": self.sentence_id,
                "origin": self.origin, "distractor": self.distractor, "fixed": self.fixed}

    @staticmethod
    def from_dict(d: dict) -> "Quantity":
        role = d.get("role", "given")
        if role == "query":
            role = "derived"
        if role not in ROLES:
            raise IRError(f"bad role {role!r}")
        qt = QType.parse(d.get("qtype", d.get("kind")))
        unit = Unit.from_dict(d.get("unit")) if not isinstance(d.get("unit"), str) else resolve_unit(d["unit"], qt)
        v = d.get("value")
        sp = d.get("span")
        return Quantity(qid=int(d["qid"]), value=None if v is None else to_fraction(v), qtype=qt, unit=unit,
                        label_vi=str(d.get("label_vi", d.get("label", "")) or ""), role=role,
                        span=None if sp is None else (int(sp[0]), int(sp[1])), sentence_id=d.get("sentence_id"),
                        origin=d.get("origin", "seed"), distractor=bool(d.get("distractor", False)),
                        fixed=bool(d.get("fixed", False)))


@dataclass
class Op:
    oid: int
    kind: str
    inputs: list[int]
    output: int
    sentence_id: int | None = None   # the clause that expresses this relation
    verb_vi: str | None = None       # "cho", "mua thêm", "chia đều", "gấp đôi"

    def to_dict(self) -> dict:
        return {"oid": self.oid, "kind": self.kind, "inputs": list(self.inputs), "output": self.output,
                "sentence_id": self.sentence_id, "verb_vi": self.verb_vi}

    @staticmethod
    def from_dict(d: dict) -> "Op":
        return Op(oid=int(d["oid"]), kind=str(d["kind"]), inputs=[int(x) for x in d["inputs"]],
                  output=int(d["output"]), sentence_id=d.get("sentence_id"), verb_vi=d.get("verb_vi"))


@dataclass
class Equation:
    lhs: int
    rhs: int

    def to_dict(self) -> dict:
        return {"lhs": self.lhs, "rhs": self.rhs}


@dataclass
class Program:
    seed_id: str
    text: str
    quantities: dict[int, Quantity]
    ops: list[Op]
    query: int
    sentences: list[tuple[int, int]] = field(default_factory=list)
    equations: list[Equation] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    answer: Fraction | None = None
    kind: str = "story"            # "story" | "symbolic"
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- construction / copying
    def __post_init__(self):
        if not self.sentences:
            self.sentences = split_sentences(self.text)

    @property
    def q(self) -> dict[int, Quantity]:
        return self.quantities

    def copy(self) -> "Program":
        return copy.deepcopy(self)

    def next_qid(self) -> int:
        return max(self.quantities, default=0) + 1

    def next_oid(self) -> int:
        return max((o.oid for o in self.ops), default=0) + 1

    # ---------------------------------------------------------------- structure
    def producer(self, qid: int) -> Op | None:
        for o in self.ops:
            if o.output == qid:
                return o
        return None

    def producers(self) -> dict[int, Op]:
        return {o.output: o for o in self.ops}

    def consumers(self, qid: int) -> list[Op]:
        return [o for o in self.ops if qid in o.inputs]

    def givens(self) -> list[Quantity]:
        return [q for q in self.quantities.values() if q.role == "given"]

    def unknowns(self) -> list[Quantity]:
        return [q for q in self.quantities.values() if q.role == "unknown"]

    def derived(self) -> list[Quantity]:
        return [q for q in self.quantities.values() if q.role == "derived"]

    def leaves(self) -> list[Quantity]:
        return [q for q in self.quantities.values() if q.role != "derived"]

    def check_structure(self) -> None:
        """Raise IRError on dangling references, double producers, bad arities, cycles."""
        if self.query not in self.quantities:
            raise IRError(f"query {self.query} is not a quantity")
        seen_out = set()
        for o in self.ops:
            if o.kind not in OP_KINDS:
                raise IRError(f"unknown op kind {o.kind!r}")
            lo, hi, _ = OP_KINDS[o.kind]
            if len(o.inputs) < lo or (hi is not None and len(o.inputs) > hi):
                raise IRError(f"op {o.oid} ({o.kind}) has {len(o.inputs)} inputs")
            for x in o.inputs + [o.output]:
                if x not in self.quantities:
                    raise IRError(f"op {o.oid} references unknown qid {x}")
            if o.output in seen_out:
                raise IRError(f"qid {o.output} produced twice")
            seen_out.add(o.output)
            if self.quantities[o.output].role != "derived":
                raise IRError(f"op {o.oid} outputs non-derived qid {o.output}")
            if o.output in o.inputs:
                raise IRError(f"op {o.oid} consumes its own output")
        for q in self.quantities.values():
            if q.role == "derived" and q.qid not in seen_out:
                raise IRError(f"derived qid {q.qid} has no producing op")
            if q.role in ("given", "world_const") and q.value is None:
                raise IRError(f"{q.role} qid {q.qid} has no value")
        for e in self.equations:
            if e.lhs not in self.quantities or e.rhs not in self.quantities:
                raise IRError("equation references unknown qid")
        if self.unknowns() and not self.equations:
            raise IRError("unknowns without equations")
        self.topo()

    def topo(self) -> list[int]:
        """Quantities in dependency order (raises IRError on a cycle)."""
        prod = self.producers()
        order: list[int] = []
        state: dict[int, int] = {}

        def visit(n: int):
            stack = [(n, iter(prod[n].inputs) if n in prod else iter(()))]
            if state.get(n) == 2:
                return
            state[n] = 1
            while stack:
                node, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    stack.pop()
                    state[node] = 2
                    order.append(node)
                    continue
                s = state.get(nxt)
                if s == 1:
                    raise IRError("cycle in ops")
                if s == 2:
                    continue
                state[nxt] = 1
                stack.append((nxt, iter(prod[nxt].inputs) if nxt in prod else iter(())))

        for n in sorted(self.quantities):
            if state.get(n) != 2:
                visit(n)
        return order

    def parents(self, qid: int) -> list[int]:
        o = self.producer(qid)
        return list(o.inputs) if o else []

    def ancestors(self, qid: int) -> set[int]:
        """Ancestors via op inputs; an unknown's ancestors include everything in the equations that pin it."""
        prod = self.producers()
        eq_sets = [self._closure({e.lhs, e.rhs}, prod) for e in self.equations]
        seen: set[int] = set()
        stack = [qid]
        while stack:
            n = stack.pop()
            nb = set(prod[n].inputs) if n in prod else set()
            if self.quantities[n].role == "unknown":
                for s in eq_sets:
                    if n in s:
                        nb |= s - {n}
            for p in nb:
                if p not in seen and p != qid:
                    seen.add(p)
                    stack.append(p)
        return seen

    @staticmethod
    def _closure(roots: set[int], prod: dict[int, Op]) -> set[int]:
        seen = set(roots)
        stack = list(roots)
        while stack:
            n = stack.pop()
            if n in prod:
                for p in prod[n].inputs:
                    if p not in seen:
                        seen.add(p)
                        stack.append(p)
        return seen

    def descendants(self, qid: int) -> set[int]:
        out = set()
        frontier = [qid]
        while frontier:
            n = frontier.pop()
            for o in self.ops:
                if n in o.inputs and o.output not in out:
                    out.add(o.output)
                    frontier.append(o.output)
        return out

    def depth(self, qid: int) -> int:
        """Longest op chain from a leaf to qid. Unknowns count 1 (solved through >= 1 equation)."""
        prod = self.producers()
        memo: dict[int, int] = {}
        for n in self.topo():
            q = self.quantities[n]
            if n in prod:
                memo[n] = 1 + max((memo[p] for p in prod[n].inputs), default=0)
            else:
                memo[n] = 1 if q.role == "unknown" else 0
        return memo[qid]

    def on_query_path(self, qid: int) -> bool:
        return qid == self.query or qid in self.ancestors(self.query)

    def sentence_of(self, pos: int) -> int | None:
        for i, (s, e) in enumerate(self.sentences):
            if s <= pos < e:
                return i
        return None

    def question_sentence_id(self) -> int:
        for i in range(len(self.sentences) - 1, -1, -1):
            s, e = self.sentences[i]
            if "?" in self.text[s:e]:
                return i
        return len(self.sentences) - 1

    def literal(self, qid: int) -> str | None:
        q = self.quantities[qid]
        return None if q.span is None else self.text[q.span[0]:q.span[1]]

    # ---------------------------------------------------------------- modification helpers
    def with_values(self, vals: dict[int, Fraction]) -> "Program":
        P = self.copy()
        for k, v in vals.items():
            P.quantities[k].value = Fraction(v)
        P.answer = None
        return P

    def add_quantity(self, value, qtype, unit, label_vi, role, origin="operator", span=None, sentence_id=None,
                     fixed=False) -> int:
        qid = self.next_qid()
        qt = QType.parse(qtype)
        u = unit if isinstance(unit, Unit) else resolve_unit(unit, qt)
        self.quantities[qid] = Quantity(qid=qid, value=None if value is None else Fraction(value), qtype=qt, unit=u,
                                        label_vi=label_vi, role=role, span=span, sentence_id=sentence_id,
                                        origin=origin, fixed=fixed)
        return qid

    def add_op(self, kind, inputs, output, sentence_id=None, verb_vi=None) -> Op:
        o = Op(oid=self.next_oid(), kind=kind, inputs=list(inputs), output=output, sentence_id=sentence_id,
               verb_vi=verb_vi)
        self.ops.append(o)
        return o

    # ---------------------------------------------------------------- canonical form
    def canonical(self, qid: int | None = None, values: bool = False) -> str:
        """Canonical string of the DAG under qid (ids abstracted; commutative ops sorted).
        Leaves render as role:qtype (plus the value if values=True). Unknowns are expanded through the
        equations that pin them, so isomorphic programs get identical strings."""
        qid = self.query if qid is None else qid
        prod = self.producers()
        memo: dict[int, str] = {}

        def rec(n: int, stack: tuple) -> str:
            if n in memo:
                return memo[n]
            q = self.quantities[n]
            if n in prod:
                o = prod[n]
                kids = [rec(i, stack + (n,)) for i in o.inputs]
                if OP_KINDS[o.kind][2]:
                    kids.sort()
                s = f"{o.kind}({','.join(kids)})"
            else:
                s = f"{q.role}:{q.qtype.value}" + (f"={q.value}" if values and q.value is not None else "")
                if q.role == "unknown" and n not in stack:
                    eqs = []
                    for e in self.equations:
                        if n in self._closure({e.lhs, e.rhs}, prod):
                            sides = sorted([rec(e.lhs, stack + (n,)), rec(e.rhs, stack + (n,))])
                            eqs.append("==".join(sides))
                    s += "{" + ";".join(sorted(eqs)) + "}"
            memo[n] = s
            return s

        return rec(qid, ())

    # ---------------------------------------------------------------- JSON
    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION, "seed_id": self.seed_id, "kind": self.kind, "text": self.text,
            "sentences": [list(s) for s in self.sentences],
            "quantities": [self.quantities[k].to_dict() for k in sorted(self.quantities)],
            "ops": [o.to_dict() for o in self.ops], "equations": [e.to_dict() for e in self.equations],
            "constraints": list(self.constraints), "query": self.query,
            "answer": None if self.answer is None else str(self.answer), "meta": self.meta,
        }

    def to_json(self, **kw) -> str:
        kw.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), **kw)

    @staticmethod
    def from_dict(d: dict) -> "Program":
        schema = d.get("schema")
        if schema is None:
            raise IRError("missing 'schema' (use from_expr_json for formalizer output)")
        m = re.fullmatch(r"vi-math-ir/(\d+)\.(\d+)", str(schema))
        if not m:
            raise IRError(f"unrecognised schema {schema!r}")
        if int(m.group(1)) != 1:
            raise IRError(f"unsupported schema major version {schema!r} (this library reads 1.x)")
        qs = {}
        for qd in d.get("quantities", []):
            q = Quantity.from_dict(qd)
            if q.qid in qs:
                raise IRError(f"duplicate qid {q.qid}")
            qs[q.qid] = q
        P = Program(seed_id=str(d.get("seed_id", "")), text=str(d.get("text", "")), quantities=qs,
                    ops=[Op.from_dict(o) for o in d.get("ops", [])], query=int(d["query"]),
                    sentences=[(int(a), int(b)) for a, b in d.get("sentences") or []],
                    equations=[Equation(int(e["lhs"]), int(e["rhs"])) for e in d.get("equations", [])],
                    constraints=[str(c) for c in d.get("constraints", [])],
                    answer=None if d.get("answer") is None else to_fraction(d["answer"]),
                    kind=str(d.get("kind", "story")), meta=dict(d.get("meta") or {}))
        P.check_structure()
        return P

    @staticmethod
    def from_json(s: str) -> "Program":
        return Program.from_dict(json.loads(s))


# ------------------------------------------------------------------------------------------ builder
class ProgramBuilder:
    """Convenient, span-safe construction (fixtures, G1-style code, tests).

        b = ProgramBuilder("gsm8k-0", text)
        a = b.given("48", "count", "kẹp tóc", "số kẹp tóc bán trong tháng 4")
        h = b.given("một nửa", "ratio", "", "tỉ lệ so với tháng 4")
        m = b.op("ratio_part", [a, h], "count", "kẹp tóc", "số kẹp tóc bán trong tháng 5")
        t = b.op("add", [a, m], "count", "kẹp tóc", "tổng số kẹp tóc bán trong hai tháng")
        P = b.build(query=t)
    """

    def __init__(self, seed_id: str, text: str, kind: str = "story"):
        self.P = Program(seed_id=seed_id, text=text, quantities={}, ops=[], query=0, kind=kind)
        self._taken: list[tuple[int, int]] = []

    def _qt_unit(self, qtype, unit):
        qt = QType.parse(qtype) if qtype is not None else None
        u = unit if isinstance(unit, Unit) else resolve_unit(unit, qt)
        if qt is None:
            qt = default_qtype(u)
        return qt, u

    def given(self, literal: str, qtype, unit, label_vi: str, value=None, occurrence: int | None = None,
              context: str | None = None, distractor: bool = False, fixed: bool = False) -> int:
        span = locate_span(self.P.text, literal, context=context, occurrence=occurrence, taken=self._taken)
        if span is None:
            raise IRError(f"literal {literal!r} not found (free) in text")
        lit = self.P.text[span[0]:span[1]]
        qt, u = self._qt_unit(qtype, unit)
        if value is None:
            vals = parse_literal(lit)
            if len(vals) != 1:
                raise IRError(f"literal {lit!r} is ambiguous/unparseable {vals}; pass value=")
            value = next(iter(vals))
        self._taken.append(span)
        qid = self.P.add_quantity(value, qt, u, label_vi, "given", origin="seed", span=span,
                                  sentence_id=self.P.sentence_of(span[0]), fixed=fixed)
        self.P.quantities[qid].distractor = distractor
        return qid

    def const(self, value, qtype, unit, label_vi: str) -> int:
        qt, u = self._qt_unit(qtype, unit)
        return self.P.add_quantity(value, qt, u, label_vi, "world_const", origin="seed", fixed=True)

    def unknown(self, qtype, unit, label_vi: str) -> int:
        qt, u = self._qt_unit(qtype, unit)
        return self.P.add_quantity(None, qt, u, label_vi, "unknown", origin="seed")

    def op(self, kind: str, inputs: list[int], qtype, unit, label_vi: str, verb_vi: str | None = None,
           sentence_id: int | None = None) -> int:
        qt, u = self._qt_unit(qtype, unit)
        out = self.P.add_quantity(None, qt, u, label_vi, "derived", origin="seed")
        if sentence_id is None:
            sids = [self.P.quantities[i].sentence_id for i in inputs if self.P.quantities[i].sentence_id is not None]
            sentence_id = max(sids) if sids else None
        self.P.add_op(kind, inputs, out, sentence_id=sentence_id, verb_vi=verb_vi)
        return out

    def equation(self, lhs: int, rhs: int) -> None:
        self.P.equations.append(Equation(lhs, rhs))

    def constraint(self, c: str) -> None:
        self.P.constraints.append(c)

    def build(self, query: int, gold=None) -> Program:
        self.P.query = query
        self.P.check_structure()
        if gold is not None:
            self.P.meta["gold"] = str(to_fraction(gold))
        return self.P


# ------------------------------------------------------------------------------ expression adapter
_EXPR_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant, ast.Call,
                 ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd)
_EXPR_FUNCS = {"floor": "floor", "ceil": "ceil", "min": "min", "max": "max", "abs": "abs"}
_BIN = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.FloorDiv: "floordiv",
        ast.Mod: "mod", ast.Pow: "pow"}


def _parse_expr(e: str) -> ast.Expression:
    try:
        t = ast.parse(str(e).strip().replace("^", "**"), mode="eval")
    except SyntaxError as ex:
        raise IRError(f"bad expr {e!r}: {ex}")
    for n in ast.walk(t):
        if not isinstance(n, _EXPR_ALLOWED):
            raise IRError(f"disallowed syntax {type(n).__name__} in {e!r}")
        if isinstance(n, ast.Call) and (not isinstance(n.func, ast.Name) or n.func.id not in _EXPR_FUNCS):
            raise IRError(f"disallowed call in {e!r}")
        if isinstance(n, ast.Constant) and (isinstance(n.value, bool) or not isinstance(n.value, (int, float))):
            raise IRError(f"non-numeric constant in {e!r}")
    return t


def from_expr_json(seed_id: str, text: str, j: dict, kind: str = "story") -> Program:
    """Adapter for the formalizer / G1 pilot JSON:
        {"quantities": [{"id","role","kind","unit","label","value","span","context","expr","distractor"}],
         "equations": ["lhs == rhs"], "query": id}
    Each expression is compiled into ops (n-ary + and * are flattened); anonymous intermediate nodes get
    label "" and an inferred type. Literal numbers inside expressions become world_const nodes."""
    if not isinstance(j, dict):
        raise IRError("IR is not a JSON object")
    P = Program(seed_id=seed_id, text=text, quantities={}, ops=[], query=0, kind=kind)
    ids: dict[str, int] = {}
    taken: list[tuple[int, int]] = []
    raw = j.get("quantities") or []
    for d in raw:
        name = str(d.get("id", "")).strip()
        if not name or name in ids:
            raise IRError(f"bad or duplicate id {name!r}")
        role = d.get("role")
        if role == "query":
            role = "derived"
        if role not in ROLES:
            raise IRError(f"bad role {role!r} for {name}")
        qt = QType.parse(d.get("qtype", d.get("kind")))
        u = resolve_unit(d.get("unit") or "", qt)
        val = None
        if role in ("given", "world_const"):
            if d.get("value") is None:
                raise IRError(f"{role} {name} has no value")
            val = to_fraction(d["value"])
        span = None
        if role == "given":
            sp = d.get("span")
            if isinstance(sp, (list, tuple)) and len(sp) == 2:
                span = (int(sp[0]), int(sp[1]))
            elif isinstance(sp, str):
                span = locate_span(text, sp, context=d.get("context"), taken=taken)
            if span is not None:
                taken.append(span)
        qid = P.add_quantity(val, qt, u, str(d.get("label_vi", d.get("label", "")) or ""), role, origin="seed",
                             span=span, sentence_id=None if span is None else P.sentence_of(span[0]),
                             fixed=role == "world_const")
        P.quantities[qid].distractor = bool(d.get("distractor", False))
        if role == "given" and span is None:
            P.meta.setdefault("unanchored", []).append(qid)
        ids[name] = qid
    exprs = {ids[str(d["id"]).strip()]: d.get("expr") for d in raw if d.get("role") in ("derived", "query")}
    for qid, e in exprs.items():
        if not e:
            raise IRError(f"derived {qid} has no expr")
    consts: dict[Fraction, int] = {}

    def compile_node(node, target: int | None) -> int:
        if isinstance(node, ast.Expression):
            return compile_node(node.body, target)
        if isinstance(node, ast.Name):
            if node.id not in ids:
                raise IRError(f"unknown id {node.id}")
            q = ids[node.id]
            if target is None:
                return q
            # alias "c = a": materialise as a + 0 is ugly; use max(a, a)
            P.add_op("max", [q, q], target)
            return target
        if isinstance(node, ast.Constant):
            v = to_fraction(node.value)
            if v not in consts:
                consts[v] = P.add_quantity(v, QType.ABSTRACT, Unit(), f"hằng số {v}", "world_const", origin="seed",
                                           fixed=True)
            if target is None:
                return consts[v]
            P.add_op("max", [consts[v], consts[v]], target)
            return target
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.UAdd):
                return compile_node(node.operand, target)
            inner = compile_node(node.operand, None)
            out = target if target is not None else P.add_quantity(None, QType.ABSTRACT, Unit(), "", "derived",
                                                                   origin="seed")
            P.add_op("neg", [inner], out)
            return out
        if isinstance(node, ast.Call):
            args = [compile_node(a, None) for a in node.args]
            kind_ = _EXPR_FUNCS[node.func.id]
            out = target if target is not None else P.add_quantity(None, QType.ABSTRACT, Unit(), "", "derived",
                                                                   origin="seed")
            P.add_op(kind_, args, out)
            return out
        if isinstance(node, ast.BinOp):
            k = _BIN[type(node.op)]
            if k in ("add", "mul"):  # flatten
                operands = []

                def flat(n):
                    if isinstance(n, ast.BinOp) and _BIN[type(n.op)] == k:
                        flat(n.left)
                        flat(n.right)
                    else:
                        operands.append(n)
                flat(node)
                ins = [compile_node(x, None) for x in operands]
            elif k == "sub":
                operands = []
                n = node
                while isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub):
                    operands.insert(0, n.right)
                    n = n.left
                operands.insert(0, n)
                ins = [compile_node(x, None) for x in operands]
            else:
                ins = [compile_node(node.left, None), compile_node(node.right, None)]
            out = target if target is not None else P.add_quantity(None, QType.ABSTRACT, Unit(), "", "derived",
                                                                   origin="seed")
            P.add_op(k, ins, out)
            return out
        raise IRError(f"cannot compile {ast.dump(node)}")

    for qid, e in exprs.items():
        compile_node(_parse_expr(e), qid)
    for e in j.get("equations") or []:
        e = str(e)
        if "==" not in e:
            if e.count("=") != 1:
                raise IRError(f"bad equation {e!r}")
            e = e.replace("=", "==")
        l, r = e.split("==", 1)
        P.equations.append(Equation(compile_node(_parse_expr(l), None), compile_node(_parse_expr(r), None)))
    qname = str(j.get("query", ""))
    if qname not in ids:
        raise IRError(f"query {qname!r} is not a quantity")
    P.query = ids[qname]
    P.check_structure()
    # Sentence of an op = latest sentence among its inputs; a derived input contributes its producer's sentence
    # (G3 fix: ops over derived nodes only used to get None, so M3/M4 edits carried no sentence anchor).
    prod = P.producers()
    for qid in P.topo():
        o = prod.get(qid)
        if o is None or o.sentence_id is not None:
            continue
        sids = []
        for i in o.inputs:
            s = P.quantities[i].sentence_id
            if s is None and i in prod:
                s = prod[i].sentence_id
            if s is not None:
                sids.append(s)
        o.sentence_id = max(sids) if sids else None
    from .executor import infer_types
    infer_types(P, fill=True)
    return P
