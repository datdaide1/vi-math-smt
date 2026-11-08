"""Mutation operators M1-M6 (T1 §2.2, 01_TECHNICAL_DESIGN §3.3). They act on premises and structure, never on
the answer. Each operator returns (Program, list[TextEdit]) or raises NotApplicable(reason); every returned
program has already passed rules.gate() (types at every node, uniqueness, tail, identity, clean answer,
plausibility). Programs keep the SEED text; edits carry spans into it. M1 is rendered without an LLM by
render.render_substitutions(); M2-M6 edits go to the span-grounded editor.

    m1_resample(P, rng)            log-uniform [v/3, 3v] snapped to the type grid; Z3 Int CSP fallback
    m2_inverse(P, rng)             a given on the query path becomes the unknown; the old answer becomes a fact
    m3_insert_step(P, rng)         typed step x' = f(x, c) strictly upstream of the query (template library)
    m4_relation_change(P, rng)     add<->sub, pct_inc<->pct_dec, gấp k lần<->bằng 1/k, factor k->k'
    m5_query_shift(P, rng)         ask for another derived node (depth >= 2) or a difference/ratio of two
    m6_compose(P, rng, partners)   the old answer replaces a unit-compatible given of a partner seed
    mutate(name, P, rng, **kw)     dispatcher -> MutationResult(program, edits, reason)
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from fractions import Fraction

import z3

from .executor import ExecError, check, execute
from .ir import Equation, IRError, Program, Quantity
from .numerals import fmt_like, fmt_vi, parse_literal
from .render import TextEdit
from .rules import gate
from .units import QType, Unit, grid_for, requires_integer, unit_class
from .z3enc import Unsupported, constraint_to_z3, domain_constraints, encode, model_value, rv

__all__ = ["NotApplicable", "MutationResult", "OPERATORS", "resample_value", "m1_resample", "m2_inverse",
           "m3_insert_step", "m4_relation_change", "m5_query_shift", "m6_compose", "mutate", "TEMPLATES",
           "StepTemplate"]


class NotApplicable(Exception):
    pass


@dataclass
class MutationResult:
    program: Program | None
    edits: list[TextEdit] = field(default_factory=list)
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.program is not None


# ------------------------------------------------------------------------------------------ helpers
def _seed_dec(P: Program) -> str:
    for q in P.givens():
        lit = P.literal(q.qid) or ""
        if "." in lit and len(lit.split(".")[-1]) != 3:
            return "."
        if "," in lit and len(lit.split(",")[-1]) != 3:
            return ","
    return ","


def _fmt_new(P: Program, v: Fraction, qtype: QType | None = None) -> str:
    s = fmt_vi(v, dec=_seed_dec(P), group=".", group_min=5)
    return s + ("%" if qtype == QType.PERCENT else "")


def _loguniform(rng: random.Random, lo: Fraction, hi: Fraction) -> float:
    lo_f, hi_f = float(lo), float(hi)
    if lo_f <= 0:
        lo_f = min(hi_f, 1e-9) if hi_f > 0 else 1e-9
    return math.exp(rng.uniform(math.log(lo_f), math.log(max(hi_f, lo_f))))


def _snap(x: float, grid: Fraction, lo: Fraction | None = None, hi: Fraction | None = None) -> Fraction:
    """Nearest grid point, clamped to the grid points inside [lo, hi] when they exist."""
    k = max(1, round(x / float(grid)))
    if lo is not None:
        k = max(k, math.ceil(lo / grid))
    if hi is not None:
        k_hi = math.floor(hi / grid)
        if k_hi >= 1 and (lo is None or k_hi >= math.ceil(lo / grid)):
            k = min(k, k_hi)
    return grid * max(1, k)


_NICE_RATIOS = sorted({Fraction(p, q) for q in range(2, 13) for p in range(1, q)})


def resample_value(q: Quantity, rng: random.Random) -> Fraction:
    """One draw for a given (T1 §2.5): log-uniform [v/3, 3v] snapped to the type grid; ratios < 1 are drawn
    from p/q with q <= 12; multipliers stay integers >= 2; percents on a 5-grid within [5, 100]."""
    v = q.value
    if v is None or v == 0:
        raise NotApplicable("zero or missing value")
    if v < 0:
        return -resample_value(Quantity(**{**q.__dict__, "value": -v}), rng)
    if q.qtype == QType.RATIO:
        if v < 1:
            pool = [r for r in _NICE_RATIOS if v / 3 <= r <= min(3 * v, Fraction(19, 20)) and r != v]
            return rng.choice(pool) if pool else v
        if v.denominator == 1:
            return Fraction(rng.randint(2, max(3, int(3 * v))))
    if q.qtype == QType.PERCENT:
        hi = min(Fraction(100), 3 * v) if v <= 100 else 3 * v
        g = grid_for(q.qtype, q.unit, v)
        lo = max(g, v / 3)
        return _snap(_loguniform(rng, lo, hi), g, lo, hi)
    g = grid_for(q.qtype, q.unit, v)
    lo, hi = max(v / 3, g), 3 * v
    return _snap(_loguniform(rng, lo, hi), g, lo, hi)


def _resampleable(P: Program, q: Quantity) -> bool:
    if q.role != "given" or q.fixed or q.span is None or q.value is None or q.value == 0:
        return False
    if q.qtype == QType.TIME_POINT:
        return False
    return True


def _render_ok(v: Fraction, literal: str) -> str | None:
    s = fmt_like(v, literal)
    return s if v in parse_literal(s) else None


def _qsent(P: Program, qid: int) -> int | None:
    q = P.quantities[qid]
    if q.sentence_id is not None:
        return q.sentence_id
    o = P.producer(qid)
    return o.sentence_id if o else None


# ------------------------------------------------------------------------------------------ M1
def _fast_ok(P: Program, P2: Program, old_answer: Fraction) -> bool:
    try:
        env = execute(P2)
    except (ExecError, IRError, ZeroDivisionError):
        return False
    return not check(P2, env) and env[P2.query] != old_answer


def _m1_z3(P: Program, G: list[Quantity], rng: random.Random, timeout_ms: int = 2000) -> dict | None:
    """Coupled-integrality sampling: each resampled given = grid * k (k Int) within [v/3, 3v]; integrality and
    sign at every node, equations and seed constraints as z3 constraints; random targets with a widening
    window give varied (not always extremal) solutions. Ratios are kept fixed here."""
    Gz = [g for g in G if g.qtype != QType.RATIO]
    if not Gz:
        return None
    ks, var = {}, {}
    for g in Gz:
        k = z3.Int(f"k{g.qid}")
        ks[g.qid] = k
        var[g.qid] = rv(grid_for(g.qtype, g.unit, g.value)) * z3.ToReal(k)
    for u in P.unknowns():
        var[u.qid] = z3.Real(f"u{u.qid}")
    try:
        env, side = encode(P, var)
        cons = list(side) + domain_constraints(P, env)
        cons += [env[e.lhs] == env[e.rhs] for e in P.equations]
        cons += [constraint_to_z3(c, env) for c in P.constraints]
    except Unsupported:
        return None
    old = execute(P)[P.query]
    cons.append(env[P.query] != rv(old))
    for g in Gz:
        gr = grid_for(g.qtype, g.unit, g.value)
        lo, hi = max(g.value / 3, gr), 3 * g.value
        cons += [var[g.qid] >= rv(lo), var[g.qid] <= rv(hi), var[g.qid] != rv(g.value)]
    targets = {g.qid: _snap(_loguniform(rng, max(g.value / 3, grid_for(g.qtype, g.unit, g.value)), 3 * g.value),
                            grid_for(g.qtype, g.unit, g.value)) for g in Gz}
    for width in (Fraction(1, 10), Fraction(1, 3), None):
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        s.set("random_seed", rng.randint(0, 2 ** 31 - 1))
        s.add(cons)
        if width is not None:
            for g in Gz:
                t = targets[g.qid]
                s.add(var[g.qid] >= rv(t * (1 - width)), var[g.qid] <= rv(t * (1 + width)))
        if s.check() == z3.sat:
            m = s.model()
            try:
                return {g.qid: model_value(m, var[g.qid]) for g in Gz}
            except Unsupported:
                return None
    return None


def _constraint_closure(P: Program, G: list[Quantity], cands: list[Quantity]) -> list[Quantity]:
    """Givens linked to a chosen given by a seed constraint are resampled together (100 + 75 == 175)."""
    import re as _re
    by_id = {q.qid: q for q in cands}
    ids = {q.qid for q in G}
    changed = True
    while changed:
        changed = False
        for c in P.constraints:
            names = {int(x) for x in _re.findall(r"q(\d+)", c)}
            if names & ids:
                for n in names & set(by_id):
                    if n not in ids:
                        ids.add(n)
                        changed = True
    return [by_id[i] for i in sorted(ids) if i in by_id]


def m1_resample(P: Program, rng: random.Random, tries: int = 200, frac: float = 0.5, use_z3: bool = True,
                only: list[int] | None = None, time_budget: float = 5.0, redraw_every: int = 50):
    """M1 value-resample: >= 50% of the resampleable givens, >= 1 on the query path, each drawn log-uniformly
    in [v/3, 3v] and snapped to its type grid; givens tied by a seed constraint move together. The subset is
    re-drawn every `redraw_every` failed tries. Rejection sampling stops after `tries` draws or `time_budget`
    seconds; then the Z3 Int CSP handles coupled integrality (divisibility chains, equality constraints)."""
    cands = [q for q in P.givens() if _resampleable(P, q)]
    on_path = [q for q in cands if P.on_query_path(q.qid)]
    if only is None and not on_path:
        raise NotApplicable("no resampleable given on the query path")

    def pick() -> list[Quantity]:
        if only is not None:
            return [P.quantities[i] for i in only]
        op = list(on_path)
        rng.shuffle(op)
        others = [q for q in cands if q not in op]
        k = max(1, math.ceil(frac * len(cands)))
        G = op[:max(1, min(len(op), k))]
        if len(G) < k:
            G += rng.sample(others, min(len(others), k - len(G)))
        return _constraint_closure(P, G, cands)

    old = execute(P)[P.query]

    def finish(vals: dict):
        edits = []
        for qid, v in vals.items():
            s = _render_ok(v, P.literal(qid))
            if s is None:
                return None
            edits.append(TextEdit("SUBST", span=P.quantities[qid].span, text=s, qid=qid, value=s))
        P2 = P.with_values(vals)
        P2.meta = {**P.meta, "operator": "M1", "parent": P.seed_id}
        if gate(P, P2, "M1") is not None:
            return None
        P2.answer = execute(P2)[P2.query]
        return P2, edits

    t_end = time.monotonic() + time_budget
    G = pick()
    for t in range(tries):
        if time.monotonic() > t_end:
            break
        if t and t % redraw_every == 0:
            G = pick()
        vals = {}
        for g in G:
            v = resample_value(g, rng)
            if v == g.value:
                v = resample_value(g, rng)
            if v != g.value:
                vals[g.qid] = v
        if not any(P.on_query_path(q) for q in vals):
            continue
        P2 = P.with_values(vals)
        if not _fast_ok(P, P2, old):
            continue
        out = finish(vals)
        if out:
            return out
    if use_z3:
        for _ in range(3 if only is None else 1):
            vals = _m1_z3(P, G, rng)
            if vals:
                out = finish(vals)
                if out:
                    out[0].meta["m1_solver"] = "z3"
                    return out
            G = pick()
    raise NotApplicable("no valid resample within the try budget")


# ------------------------------------------------------------------------------------------ M2
def _make_inverse(P1: Program, g: int, q0v: Fraction) -> Program:
    P2 = P1.copy()
    q0 = P2.quantities[P2.query]
    if q0.role == "derived":
        o = P2.producer(q0.qid)
        cid = P2.add_quantity(None, q0.qtype, q0.unit, q0.label_vi, "derived", origin="operator")
        o.output = cid
        P2.equations.append(Equation(cid, q0.qid))
    q0.role, q0.value, q0.origin, q0.span, q0.sentence_id = "given", q0v, "stated_fact", None, None
    gq = P2.quantities[g]
    gq.role, gq.value = "unknown", None
    P2.query = g
    for q in P2.givens():  # givens no longer needed are distractors of the new problem
        if q.qid not in P2.ancestors(g) and q.qid != g:
            q.distractor = True
    return P2


def m2_inverse(P: Program, rng: random.Random, tries: int = 30, time_budget: float = 8.0):
    """M2 generalized FOBAR: resample a given g on the query path, state the old query's (new) value as a fact,
    ask for g. The unique-solution check runs in gate()."""
    cands = [q for q in P.givens() if q.span is not None and not q.distractor and P.on_query_path(q.qid)
             and q.value not in (None, 0)]
    if not cands:
        raise NotApplicable("no given on the query path")
    rng.shuffle(cands)
    t_end = time.monotonic() + time_budget
    for g in cands:
        for _ in range(tries):
            if time.monotonic() > t_end:
                raise NotApplicable("time budget exhausted")
            v = resample_value(g, rng)
            if v == g.value:
                continue
            P1 = P.with_values({g.qid: v})
            try:
                env1 = execute(P1)
            except (ExecError, IRError):
                continue
            if check(P1, env1):
                continue
            q0v = env1[P.query]
            P2 = _make_inverse(P1, g.qid, q0v)
            P2.meta = {**P.meta, "operator": "M2", "parent": P.seed_id}
            if gate(P, P2, "M2") is not None:
                continue
            P2.answer = execute(P2)[P2.query]
            q0 = P.quantities[P.query]
            qs = P.question_sentence_id()
            edits = [TextEdit("UNKNOWN", span=g.span, qid=g.qid, label=g.label_vi),
                     TextEdit("STATE_FACT", qid=q0.qid, value=_fmt_new(P, q0v, q0.qtype), label=q0.label_vi,
                              after_sentence=max(0, qs - 1)),
                     TextEdit("NEW_QUESTION", qid=g.qid, label=g.label_vi, sentence_id=qs)]
            return P2, edits
    raise NotApplicable("no invertible given with a unique, clean solution")


# ------------------------------------------------------------------------------------------ M3
@dataclass(frozen=True)
class StepTemplate:
    kind: str          # op kind applied as x' = kind(x, c)
    verb_vi: str       # Vietnamese verb phrase for the editor
    const: str         # "part" (fraction of x on x's grid) | "pct" | "factor" | "divisor"
    label_suffix: str  # appended to x's label for x'


_COMMON_COUNT = [StepTemplate("sub", "cho đi", "part", "sau khi cho đi"),
                 StepTemplate("sub", "làm mất", "part", "sau khi làm mất"),
                 StepTemplate("add", "mua thêm", "part", "sau khi mua thêm"),
                 StepTemplate("add", "được tặng thêm", "part", "sau khi được tặng thêm"),
                 StepTemplate("mul", "tăng gấp", "factor", "sau khi tăng gấp"),
                 StepTemplate("div", "chia đều cho", "divisor", "mỗi phần sau khi chia đều")]
TEMPLATES: dict[tuple[str, str], list[StepTemplate]] = {
    ("count", "count"): _COMMON_COUNT,
    ("count", "person"): [StepTemplate("add", "có thêm", "part", "sau khi có thêm người đến"),
                          StepTemplate("sub", "có người rời đi", "part", "sau khi có người rời đi")],
    ("money", "usd"): [StepTemplate("pct_dec", "được giảm giá", "pct", "sau khi giảm giá"),
                       StepTemplate("pct_inc", "phải trả thêm thuế", "pct", "sau thuế"),
                       StepTemplate("add", "trả thêm phí", "part", "cộng thêm phí"),
                       StepTemplate("sub", "được bớt", "part", "sau khi được bớt")],
    ("measure", "*"): [StepTemplate("add", "thêm", "part", "sau khi thêm"),
                       StepTemplate("sub", "bớt đi", "part", "sau khi bớt đi"),
                       StepTemplate("pct_dec", "hao hụt", "pct", "sau hao hụt")],
    ("rate", "*"): [StepTemplate("pct_inc", "tăng", "pct", "sau khi tăng"),
                    StepTemplate("pct_dec", "giảm", "pct", "sau khi giảm")],
}
for _c in ("vnd", "vnd_k", "vnd_m", "cent"):
    TEMPLATES[("money", _c)] = TEMPLATES[("money", "usd")]


def _templates_for(q: Quantity) -> list[StepTemplate]:
    cls = unit_class(q.unit, q.qtype)
    return TEMPLATES.get((q.qtype.value, cls)) or TEMPLATES.get((q.qtype.value, "*")) or \
        (TEMPLATES[("count", "count")] if q.qtype == QType.COUNT else [])


def _step_constant(t: StepTemplate, xq: Quantity, xv: Fraction, rng: random.Random) -> tuple[Fraction, QType, Unit]:
    if t.const == "pct":
        return Fraction(rng.choice([5, 10, 15, 20, 25, 30, 40, 50])), QType.PERCENT, Unit(lemma="%")
    if t.const == "factor":
        return Fraction(rng.choice([2, 3])), QType.RATIO, Unit()
    if t.const == "divisor":
        return Fraction(rng.choice([2, 3, 4, 5, 6])), QType.RATIO, Unit()
    g = grid_for(xq.qtype, xq.unit, xv if xv else Fraction(1))
    hi = max(g, xv / 2) if xv > 0 else g
    c = _snap(_loguniform(rng, max(g, xv / 10), hi), g)
    return c, xq.qtype, xq.unit


def m3_insert_step(P: Program, rng: random.Random, tries: int = 6, x: int | None = None):
    """M3: x' = f(x, c) for a given or derived x strictly upstream of the query; consumers of x use x'."""
    if P.kind != "story":
        raise NotApplicable("M3 is defined for story programs")
    anc = P.ancestors(P.query)
    xs = [P.quantities[x]] if x is not None else \
        [q for q in P.quantities.values() if q.qid in anc and q.role in ("given", "derived") and q.qid != P.query
         and q.label_vi]   # G3 fix: an unlabelled (anonymous) node cannot be described to the editor
    if not xs:
        raise NotApplicable("no upstream node")
    env = execute(P)
    pairs = [(xq, t) for xq in xs for t in _templates_for(xq)]
    if not pairs:
        raise NotApplicable("no template for the node types")
    rng.shuffle(pairs)
    for xq, t in pairs:
        for _ in range(tries):
            c, cqt, cu = _step_constant(t, xq, env[xq.qid], rng)
            P2 = P.copy()
            cid = P2.add_quantity(c, cqt, cu, f"lượng {t.verb_vi}" if t.const == "part" else
                                  (f"phần trăm {t.verb_vi}" if t.const == "pct" else f"số lần ({t.verb_vi})"),
                                  "given", origin="inserted")
            xp = P2.add_quantity(None, xq.qtype, xq.unit, f"{xq.label_vi} {t.label_suffix}".strip(), "derived",
                                 origin="operator")
            sid = _qsent(P, xq.qid)
            for o in P2.ops:
                o.inputs = [xp if i == xq.qid else i for i in o.inputs]
            for e in P2.equations:
                e.lhs = xp if e.lhs == xq.qid else e.lhs
                e.rhs = xp if e.rhs == xq.qid else e.rhs
            P2.add_op(t.kind, [xq.qid, cid], xp, sentence_id=sid, verb_vi=t.verb_vi)
            P2.meta = {**P.meta, "operator": "M3", "parent": P.seed_id}
            if gate(P, P2, "M3") is not None:
                continue
            P2.answer = execute(P2)[P2.query]
            cv = _fmt_new(P, c, cqt)
            return P2, [TextEdit("INSERT_STEP", qid=cid, after_sentence=sid, template=t.verb_vi,
                                 operands=[xq.label_vi], value=cv, label=P2.quantities[xp].label_vi)]
    raise NotApplicable("no feasible step insertion")


# ------------------------------------------------------------------------------------------ M4
_FLIP = {"nhiều hơn": "ít hơn", "ít hơn": "nhiều hơn", "mua thêm": "trả lại", "thêm": "bớt", "tăng": "giảm",
         "giảm": "tăng", "cộng": "trừ", "trừ": "cộng", "tổng cộng": "hiệu", "còn lại": "thêm vào"}
_DEFAULT_VERB = {"add": "thêm", "sub": "bớt", "pct_inc": "tăng", "pct_dec": "giảm", "mul": "gấp", "div": "bằng một phần"}


_OP_VI = {"add": "phép cộng (thêm, nhiều hơn, tổng)", "sub": "phép trừ (bớt, ít hơn, còn lại, hiệu)",
          "pct_inc": "tăng theo phần trăm", "pct_dec": "giảm theo phần trăm",
          "mul": "phép nhân (gấp)", "div": "phép chia (bằng một phần)"}


def _flip_verb(verb: str | None, old_kind: str, new_kind: str) -> tuple[str, str]:
    if not verb:
        # G3 fix: the op has no verb from the text (every from_expr_json program). Describe the operation instead
        # of inventing a literal ("bớt") that is not in the sentence and that the editor then cannot find.
        return _OP_VI.get(old_kind, old_kind), _OP_VI.get(new_kind, new_kind)
    old = verb
    for a, b in _FLIP.items():
        if a in old:
            return old, old.replace(a, b)
    return old, _DEFAULT_VERB.get(new_kind, new_kind)


def m4_relation_change(P: Program, rng: random.Random):
    """M4: swap one relation on the query path for a type-compatible sibling, then re-execute."""
    path = P.ancestors(P.query) | {P.query}
    ops = [o for o in P.ops if o.output in path]
    rng.shuffle(ops)
    cands = []
    for o in ops:
        # G3 fix: the edit must be describable (every operand labelled, no anonymous/world constant operand)
        if any(not P.quantities[i].label_vi or P.quantities[i].role == "world_const" for i in o.inputs):
            continue
        if o.kind in ("add", "sub") and len(o.inputs) == 2:
            cands.append((o, "sub" if o.kind == "add" else "add", None))
        elif o.kind in ("pct_inc", "pct_dec"):
            cands.append((o, "pct_dec" if o.kind == "pct_inc" else "pct_inc", None))
        elif o.kind in ("mul", "div") and len(o.inputs) == 2:
            f = P.quantities[o.inputs[1]]
            if f.role == "given" and f.qtype == QType.RATIO and f.value is not None and f.value.denominator == 1 \
                    and f.value >= 2:
                cands.append((o, "div" if o.kind == "mul" else "mul", None))
                if f.role == "given" and o.kind == "mul":
                    for k in rng.sample([2, 3, 4, 5], 4):
                        if k != f.value:
                            cands.append((o, "mul", Fraction(k)))
    for o, new_kind, new_factor in cands:
        P2 = P.copy()
        o2 = next(x for x in P2.ops if x.oid == o.oid)
        old_kind = o2.kind
        o2.kind = new_kind
        if {old_kind, new_kind} == {"mul", "div"}:
            fq0 = P.quantities[o.inputs[1]]
            k = fq0.value
            verb_old = o.verb_vi or (P.literal(fq0.qid) or f"gấp {k} lần")
            verb_new = f"bằng 1/{k}" if new_kind == "div" else f"gấp {k} lần"
        else:
            verb_old, verb_new = _flip_verb(o.verb_vi, old_kind, new_kind)
        edit_val = None
        if new_factor is not None:
            fq = P2.quantities[o2.inputs[1]]
            lit = P.literal(fq.qid) or ""
            fq.value = new_factor
            verb_old = lit or verb_old
            verb_new = fmt_like(new_factor, lit) if lit else f"gấp {new_factor} lần"
            edit_val = verb_new
        o2.verb_vi = verb_new
        P2.meta = {**P.meta, "operator": "M4", "parent": P.seed_id}
        if gate(P, P2, "M4") is not None:
            continue
        P2.answer = execute(P2)[P2.query]
        # G3 fix: a factor change rewrites the factor's tag, so the edit names the factor's qid (it used to name
        # the derived output, which has no tag in the text)
        ed = TextEdit("REPLACE_RELATION", sentence_id=o.sentence_id, verb_old=verb_old, verb_new=verb_new,
                      operands=[P.quantities[i].label_vi for i in o.inputs],
                      qid=o2.inputs[1] if new_factor is not None else o.output, value=edit_val,
                      label=P.quantities[o.output].label_vi)
        return P2, [ed]
    raise NotApplicable("no type-compatible relation swap")


# ------------------------------------------------------------------------------------------ M5
def m5_query_shift(P: Program, rng: random.Random):
    """M5: ask for another derived node of depth >= 2, or for the difference / ratio of two incomparable derived
    story nodes of the same type and unit."""
    q0 = P.query
    env = execute(P)
    plumbing = (QType.RATIO, QType.ABSTRACT) if P.kind == "story" else ()
    single = [q for q in P.derived() if q.qid != q0 and q.label_vi and P.depth(q.qid) >= 2
              and q0 not in P.ancestors(q.qid) and q.qtype not in plumbing]
    der = [q for q in P.derived() if q.qid != q0 and q.label_vi and q0 not in P.ancestors(q.qid)]
    pairs = []
    for i, a in enumerate(der):
        for b in der[i + 1:]:
            if a.qtype != b.qtype or a.unit.lemma != b.unit.lemma or a.qtype in (QType.RATIO, QType.ABSTRACT):
                continue
            if a.qid in P.ancestors(b.qid) or b.qid in P.ancestors(a.qid):
                continue
            hi, lo = (a, b) if env[a.qid] >= env[b.qid] else (b, a)
            pairs.append((hi, lo))
    options = [("node", q) for q in single] + [("diff", p) for p in pairs] + [("ratio", p) for p in pairs]
    rng.shuffle(options)
    for kind, x in options:
        P2 = P.copy()
        if kind == "node":
            P2.query = x.qid
            label = x.label_vi
        else:
            hi, lo = x
            if kind == "diff":
                label = f"số {hi.unit.lemma} mà {hi.label_vi} nhiều hơn {lo.label_vi}".replace("  ", " ")
                nid = P2.add_quantity(None, hi.qtype, hi.unit, label, "derived", origin="operator")
                P2.add_op("sub", [hi.qid, lo.qid], nid, sentence_id=P.question_sentence_id())
            else:
                label = f"số lần {hi.label_vi} so với {lo.label_vi}"
                nid = P2.add_quantity(None, QType.RATIO, Unit(), label, "derived", origin="operator")
                P2.add_op("div", [hi.qid, lo.qid], nid, sentence_id=P.question_sentence_id())
                try:
                    if execute(P2)[nid].denominator != 1:
                        continue
                except (ExecError, IRError):
                    continue
            P2.query = nid
        anc = P2.ancestors(P2.query)
        for g in P2.givens():
            g.distractor = g.qid not in anc
        P2.meta = {**P.meta, "operator": "M5", "parent": P.seed_id}
        if gate(P, P2, "M5") is not None:
            continue
        P2.answer = execute(P2)[P2.query]
        return P2, [TextEdit("NEW_QUESTION", qid=P2.query, label=label, sentence_id=P.question_sentence_id())]
    raise NotApplicable("no derived target of depth >= 2 / no comparable pair")


# ------------------------------------------------------------------------------------------ M6
def _compatible(q0: Quantity, g: Quantity) -> bool:
    if q0.qtype != g.qtype or q0.qtype in (QType.RATIO, QType.ABSTRACT, QType.PERCENT):
        return False
    if q0.qtype == QType.MONEY:
        return unit_class(q0.unit, q0.qtype) == unit_class(g.unit, g.qtype)
    return q0.unit.lemma == g.unit.lemma or (q0.unit.per == g.unit.per and q0.qtype == QType.RATE
                                             and q0.unit.lemma == g.unit.lemma)


def m6_compose(P: Program, rng: random.Random, partners: list[Program]):
    """M6 (MathFusion-sequential): pick a partner seed with a given g2 whose (qtype, unit) matches the old
    query q0; g2 := q0 as an unstated intermediate; the new query is the partner's query."""
    if P.kind != "story":
        raise NotApplicable("M6 is GSM8K-only")
    q0 = P.quantities[P.query]
    plist = [x for x in partners if x.seed_id != P.seed_id and x.kind == "story"]
    rng.shuffle(plist)
    for S in plist:
        gs = [g for g in S.givens() if not g.distractor and S.on_query_path(g.qid) and _compatible(q0, g)]
        rng.shuffle(gs)
        for g2 in gs:
            C = P.copy()
            off = (max(C.quantities) // 1000 + 1) * 1000
            shift = len(P.text) + 1
            C.text = P.text + " " + S.text
            m = {qid: qid + off for qid in S.quantities}
            m[g2.qid] = q0.qid
            for q in S.quantities.values():
                if q.qid == g2.qid:
                    continue
                nq = Quantity(**{**q.__dict__})
                nq.qid = m[q.qid]
                nq.origin = "partner" if q.role != "derived" else "operator"
                nq.span = None if q.span is None else (q.span[0] + shift, q.span[1] + shift)
                C.quantities[nq.qid] = nq
            base_oid = C.next_oid()
            for o in S.ops:
                C.add_op(o.kind, [m[i] for i in o.inputs], m[o.output],
                         sentence_id=None if o.sentence_id is None else o.sentence_id + len(P.sentences),
                         verb_vi=o.verb_vi)
            C.equations += [Equation(m[e.lhs], m[e.rhs]) for e in S.equations]
            import re as _re
            C.constraints += [_re.sub(r"q(\d+)", lambda mm: f"q{m.get(int(mm.group(1)), int(mm.group(1)))}", c)
                              for c in S.constraints]
            C.query = m[S.query]
            from .ir import split_sentences
            C.sentences = split_sentences(C.text)
            for q in C.quantities.values():
                if q.span is not None:
                    q.sentence_id = C.sentence_of(q.span[0])
            for g in C.givens():  # P's givens that no longer feed the new query are distractors
                g.distractor = g.qid not in C.ancestors(C.query)
            C.seed_id = f"{P.seed_id}+{S.seed_id}"
            C.meta = {**P.meta, "operator": "M6", "parent": P.seed_id, "partner": S.seed_id}
            try:
                C.check_structure()
            except IRError:
                continue
            if gate(P, C, "M6") is not None:
                continue
            C.answer = execute(C)[C.query]
            qs = P.question_sentence_id()
            keep = [g.qid for g in P.givens() if g.sentence_id == qs]
            edits = [TextEdit("JOIN", sentence_id=qs, payload={"first_text": [0, len(P.text)],
                                                               "second_text": [shift, len(C.text)],
                                                               "drop_question_of_first": qs,
                                                               "keep_givens_from_dropped_question": keep}),
                     TextEdit("REFERENCE", span=(g2.span[0] + shift, g2.span[1] + shift), qid=q0.qid,
                              label=q0.label_vi)]
            return C, edits
    raise NotApplicable("no unit-compatible partner given")


# ------------------------------------------------------------------------------------------ dispatcher
OPERATORS = {"M1": m1_resample, "M2": m2_inverse, "M3": m3_insert_step, "M4": m4_relation_change,
             "M5": m5_query_shift, "M6": m6_compose}


def mutate(name: str, P: Program, rng: random.Random, **kw) -> MutationResult:
    try:
        P2, edits = OPERATORS[name](P, rng, **kw)
        return MutationResult(P2, edits, None)
    except NotApplicable as e:
        return MutationResult(None, [], str(e))
    except (ExecError, IRError) as e:
        return MutationResult(None, [], f"error: {e}")
