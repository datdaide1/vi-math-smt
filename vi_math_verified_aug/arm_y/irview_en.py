"""English blueprint of a program for ``prompts/ir_render_en.v1`` (labels stay Vietnamese; markers, formulas and
number formats are English: '.' decimal, ',' thousands)."""
from __future__ import annotations

from ..symbolic import Program
from ..symbolic.numerals import fmt_vi
from ..symbolic.units import QType

_OPF = {"add": " + ", "sub": " − ", "mul": " × ", "div": " ÷ ", "ratio_part": " × "}


def fmt_en(v) -> str:
    return fmt_vi(v, dec=".", group=",", group_min=4)


def _formula(o) -> str:
    a = [f"[{i}]" for i in o.inputs]
    k = o.kind
    if k in _OPF:
        return _OPF[k].join(a)
    return {"floordiv": f"whole part of {a[0]} ÷ {a[-1]}", "ceildiv": f"{a[0]} ÷ {a[-1]} rounded up",
            "mod": f"remainder of {a[0]} ÷ {a[-1]}", "pct_of": f"{a[-1]}% of {a[0]}",
            "pct_inc": f"{a[0]} increased by {a[-1]}%", "pct_dec": f"{a[0]} decreased by {a[-1]}%",
            "pow": f"{a[0]} to the power {a[-1]}", "avg": "average of " + ", ".join(a),
            "min": "smallest of " + ", ".join(a), "max": "largest of " + ", ".join(a), "neg": f"−{a[0]}",
            "abs": f"|{a[0]}|", "floor": f"{a[0]} rounded down", "ceil": f"{a[0]} rounded up"}[k]


def _unit(q) -> str:
    if q.qtype == QType.PERCENT:
        return ""
    u = str(q.unit)
    return f" (unit: {u})" if u else ""


def _val(q) -> str:
    return fmt_en(q.value) + ("%" if q.qtype == QType.PERCENT else "")


def ir_view_en(P: Program) -> dict:
    facts, consts, rels = [], [], []
    for q in sorted(P.givens(), key=lambda q: (q.sentence_id if q.sentence_id is not None else 99, q.qid)):
        tag = " (extra: not needed for the question)" if q.distractor else ""
        facts.append(f'- [{q.qid}] {q.label_vi or "(unnamed)"} = {_val(q)}{_unit(q)}{tag}  -> write <q n="{q.qid}">…</q>')
    for q in P.quantities.values():
        if q.role == "world_const":
            consts.append(f"- [{q.qid}] {q.label_vi or 'constant'} = {fmt_en(q.value)}")
    prod = P.producers()
    for n in P.topo():
        o = prod.get(n)
        if o is None:
            continue
        q = P.quantities[o.output]
        rels.append(f"- [{q.qid}] {q.label_vi or '(intermediate quantity)'}{_unit(q)} = {_formula(o)}")
    for u in P.unknowns():
        rels.append(f"- UNKNOWN: [{u.qid}] {u.label_vi}{_unit(u)}, value not given")
    for e in P.equations:
        rels.append(f"- EQUATION: [{e.lhs}] = [{e.rhs}]")
    qq = P.quantities[P.query]
    return {"facts": "\n".join(facts), "consts": "\n".join(consts) or "(none)", "relations": "\n".join(rels),
            "question": f"[{qq.qid}] {qq.label_vi}{_unit(qq)}"}


__all__ = ["ir_view_en", "fmt_en"]
