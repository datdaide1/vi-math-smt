"""Formalizer acceptance (T1 §2.1 (i)-(v), 01_TECHNICAL_DESIGN §3.2).

validate_program(P, gold) -> list[Violation]   (empty list = accepted)
    (i)   executes, and execute(P)[query] == gold exactly (Fraction)
    (ii)  every given has a span inside the text, spans do not overlap, and the literal at the span
          normalizes (Vietnamese numerals) to the given's value
    (iii) the query is derived (not a literal) with depth >= 1
    (iv)  every given is reachable to the query unless marked distractor
    (v)   every type constraint (integrality/sign at every node) and every seed constraint holds;
          no hard unit mismatch (COUNT + MONEY ...); no world_const that shadows a number written in the text
Each check is also exported on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .executor import ExecError, check, execute, infer_types
from .ir import IRError, Program, to_fraction
from .numerals import numbers_in_text, parse_literal
from .units import QType

__all__ = ["Violation", "validate_program", "check_executes_to_gold", "check_spans", "check_query_derived",
           "check_reachability", "check_types", "check_world_consts", "unanchored_numbers", "accepted"]


@dataclass(frozen=True)
class Violation:
    code: str       # exec | gold | span | query | reach | type | unit | world_const
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def check_executes_to_gold(P: Program, gold) -> tuple[dict | None, list[Violation]]:
    try:
        env = execute(P)
    except (ExecError, IRError, ZeroDivisionError) as e:
        return None, [Violation("exec", str(e))]
    P.answer = env[P.query]
    if gold is not None and env[P.query] != to_fraction(gold):
        return env, [Violation("gold", f"answer {env[P.query]} != gold {to_fraction(gold)}")]
    return env, []


def _literal_matches(q, lit: str) -> bool:
    vals = parse_literal(lit)
    if q.value in vals:
        return True
    if q.qtype == QType.RATIO and lit.strip().endswith("%") and q.value * 100 in vals:
        return True  # "20%" formalized as the ratio 1/5
    return False


def check_spans(P: Program) -> list[Violation]:
    out = []
    spans = []
    n = len(P.text)
    for q in P.givens():
        if q.span is None:
            out.append(Violation("span", f"given q{q.qid} ({q.label_vi}) has no span"))
            continue
        s, e = q.span
        if not (0 <= s < e <= n):
            out.append(Violation("span", f"given q{q.qid} span {q.span} outside the text"))
            continue
        lit = P.text[s:e]
        if not _literal_matches(q, lit):
            out.append(Violation("span", f"given q{q.qid}: literal {lit!r} -> {sorted(map(str, parse_literal(lit)))}"
                                         f" != value {q.value}"))
        for (a, b, k) in spans:
            if a < e and s < b:
                out.append(Violation("span", f"given q{q.qid} span overlaps q{k}"))
        spans.append((s, e, q.qid))
    return out


def check_query_derived(P: Program) -> list[Violation]:
    q = P.quantities[P.query]
    if q.role in ("given", "world_const"):
        return [Violation("query", f"query q{q.qid} is a literal ({q.role}), not derived")]
    if P.depth(P.query) < 1:
        return [Violation("query", "query depth < 1")]
    return []


def check_reachability(P: Program) -> list[Violation]:
    anc = P.ancestors(P.query)
    return [Violation("reach", f"given q{g.qid} ({g.label_vi}) does not reach the query and is not a distractor")
            for g in P.givens() if g.qid not in anc and not g.distractor]


def check_types(P: Program, env: dict | None) -> list[Violation]:
    out = [Violation("unit", m) for m in infer_types(P)]
    if env is not None:
        out += [Violation("type", m) for m in check(P, env)]
    return out


def unanchored_numbers(P: Program) -> list[tuple[int, int, str]]:
    """Digit literals in the text not covered by any given span (dates, descriptors, or missed givens)."""
    cov = [q.span for q in P.givens() if q.span]
    return [(m.start, m.end, m.literal) for m in numbers_in_text(P.text)
            if not any(s < m.end and m.start < e for s, e in cov)]


def check_world_consts(P: Program) -> list[Violation]:
    """A world_const whose value is written as a digit literal in the text, uncovered by any given span, is a
    hard-coded given (the R3 'hard-coded answer' failure)."""
    free = set()
    for _, _, lit in unanchored_numbers(P):
        free |= parse_literal(lit)
    out = []
    anc = P.ancestors(P.query)
    for q in P.quantities.values():
        if q.role == "world_const" and q.qid in anc and q.value in free and q.value not in (0, 1):
            out.append(Violation("world_const", f"world_const q{q.qid}={q.value} is written in the text; "
                                                f"make it a span-anchored given"))
    return out


def validate_program(P: Program, gold=None) -> list[Violation]:
    """Full formalizer acceptance. gold=None skips the gold comparison (blind back-formalization)."""
    try:
        P.check_structure()
    except IRError as e:
        return [Violation("exec", f"structure: {e}")]
    env, out = check_executes_to_gold(P, gold)
    out += check_spans(P)
    out += check_query_derived(P)
    out += check_reachability(P)
    out += check_types(P, env)
    out += check_world_consts(P)
    return out


def accepted(P: Program, gold=None) -> bool:
    return not validate_program(P, gold)
