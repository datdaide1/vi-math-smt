"""IR-level localization for arm Y (T1 §2.2 ``localize_surface``, G3 §6 fix 1) and the IR view used by R2.

* ``localize(P2)``   USD -> đồng (×1000) and cent -> đồng (×10) on every money quantity of the mutated program, then
                     re-execute. Accepted only if the program still executes, passes the type/sign checks, and every
                     node scales exactly by its expected factor (money ×1000, money-per-x ×1000, x-per-money ÷1000,
                     everything else ×1). A non-homogeneous program (e.g. a 100 xu/đô la world constant) keeps its
                     original currency. The localized program is the TARGET of R1/R2 (tag values, cycle check, solve).
* ``money_map``      for R1: "thẻ n: <old literal> → <new literal> đồng" for each rescaled money given.
* ``keep_units``     units tied by a unit-conversion world constant (R1/R2 must not relabel those).
* ``ir_view``        R2 input: labelled givens with values, world constants, relations over [ids], unknown/equation,
                     and the question label. Derived values and the answer are never shown.
* ``card_for``       one localization card per seed (G1 cards_v1, Random(f"{seed_id}:20261108")).
"""
from __future__ import annotations

import json
import random
from fractions import Fraction

from ..pilot_gen.util import card_text
from ..symbolic import Program, execute
from ..symbolic.executor import check
from ..symbolic.numerals import fmt_vi
from ..symbolic.units import QType, Unit, lookup
from ..pilot_gen import seeds as G1S

VND = Unit(lemma="đồng", plausible=(Fraction(0), Fraction(10 ** 11)))
_CARDS = {c["id"]: c for c in json.load(open(G1S.CARDS, encoding="utf-8"))}


def card_for(seed_id: str) -> tuple[str, str]:
    cid = random.Random(f"{seed_id}:20261108").choice(sorted(_CARDS))
    return cid, card_text(_CARDS[cid])


def _cls(lemma: str | None) -> str | None:
    e = lookup(lemma or "")
    return e.cls if e is not None and e.cls in ("usd", "cent") else None


_MULT = {"usd": Fraction(1000), "cent": Fraction(10)}          # R1/R2 (run 1): 1 USD -> 1.000 đồng
SCALES = {1000: {"usd": Fraction(1000), "cent": Fraction(10)},
          10000: {"usd": Fraction(10000), "cent": Fraction(100)}}   # R1b: 1 USD -> 10.000 đồng (realistic prices)


def _factor(q, mult=None) -> Fraction:
    mult = mult or _MULT
    f = Fraction(1)
    c = _cls(q.unit.lemma)
    if c:
        f *= mult[c]
    c2 = _cls(q.unit.per)
    if c2:
        f /= mult[c2]
    return f


def _new_unit(u: Unit) -> Unit:
    lemma = "đồng" if _cls(u.lemma) else u.lemma
    per = "đồng" if _cls(u.per) else u.per
    if _cls(u.lemma) and not u.per:
        return VND
    return Unit(lemma=lemma, classifier=u.classifier, divisible=True if _cls(u.lemma) else u.divisible,
                plausible=None if _cls(u.lemma) or _cls(u.per) else u.plausible, per=per)


def money_fmt(v: Fraction) -> str:
    return fmt_vi(v, dec=",", group=".", group_min=4)


def localize(P2: Program, usd: int = 1000) -> tuple[Program, dict]:
    """(program to render, info). info: rescaled (bool), why (if not), changed {qid: (old, new)} for givens.
    usd = đồng per USD (1000 in the R1/R2 run, 10000 in R1b)."""
    fac = {q.qid: _factor(q, SCALES[usd]) for q in P2.quantities.values()}
    if all(f == 1 for f in fac.values()):
        return P2, {"rescaled": False, "why": "no USD/cent quantity", "changed": {}}
    env0 = execute(P2)
    PL = P2.copy()
    changed = {}
    for q in PL.quantities.values():
        f = fac[q.qid]
        if f != 1:
            q.unit = _new_unit(q.unit)
            if q.role in ("given", "world_const") and q.value is not None:
                changed[q.qid] = (str(q.value), str(q.value * f))
                q.value = q.value * f
    try:
        env = execute(PL)
        bad = check(PL, env)
    except Exception as e:                       # noqa: BLE001 - any failure = keep the original currency
        return P2, {"rescaled": False, "why": f"exec: {type(e).__name__}", "changed": {}}
    if bad:
        return P2, {"rescaled": False, "why": f"check: {bad[0]}", "changed": {}}
    for qid, f in fac.items():
        if env0.get(qid) is not None and env.get(qid) != env0[qid] * f:
            return P2, {"rescaled": False, "why": f"non-homogeneous at q{qid}", "changed": {}}
    PL.answer = env[PL.query]
    PL.meta = {**PL.meta, "localized": "usd->vnd"}
    return PL, {"rescaled": True, "why": "", "changed": {k: v for k, v in changed.items()
                                                         if PL.quantities[k].role == "given"}}


def money_map(PL: Program, info: dict) -> str:
    if not info.get("rescaled"):
        return "(không đổi)"
    lines = []
    for qid, (old, new) in sorted(info["changed"].items()):
        lines.append(f'- thẻ {qid}: {fmt_vi(Fraction(old))} (đô la/xu) → {money_fmt(Fraction(new))} đồng')
    return "\n".join(lines) or "(không đổi)"


def keep_units(P: Program) -> str:
    units = set()
    for o in P.ops:
        ins = [P.quantities[i] for i in o.inputs]
        if any(q.role == "world_const" for q in ins):
            for q in ins + [P.quantities[o.output]]:
                for x in (q.unit.lemma, q.unit.per):
                    if x:
                        units.add(x)
    return ", ".join(sorted(units)) or "(không có)"


# ------------------------------------------------------------------------------------------------ R2 view
_OPF = {"add": " + ", "sub": " − ", "mul": " × ", "div": " : ", "ratio_part": " × "}


def _ref(P: Program, i: int) -> str:
    return f"[{i}]"


def _formula(P: Program, o) -> str:
    a = [_ref(P, i) for i in o.inputs]
    k = o.kind
    if k in _OPF:
        return _OPF[k].join(a)
    return {"floordiv": lambda: f"phần nguyên của {a[0]} : {a[1]}",
            "ceildiv": lambda: f"{a[0]} : {a[1]}, làm tròn lên số nguyên",
            "mod": lambda: f"số dư của {a[0]} : {a[1]}",
            "pct_of": lambda: f"{a[1]}% của {a[0]}",
            "pct_inc": lambda: f"{a[0]} tăng thêm {a[1]}%",
            "pct_dec": lambda: f"{a[0]} giảm đi {a[1]}%",
            "pow": lambda: f"{a[0]} lũy thừa {a[1]}",
            "avg": lambda: "trung bình cộng của " + ", ".join(a),
            "min": lambda: "số nhỏ nhất trong " + ", ".join(a),
            "max": lambda: "số lớn nhất trong " + ", ".join(a),
            "neg": lambda: f"−{a[0]}", "abs": lambda: f"|{a[0]}|",
            "floor": lambda: f"{a[0]} làm tròn xuống", "ceil": lambda: f"{a[0]} làm tròn lên"}[k]()


def _val(q) -> str:
    v = q.value
    if q.qtype == QType.PERCENT:
        return fmt_vi(v) + "%"
    if q.qtype == QType.MONEY or str(q.unit.lemma) == "đồng":
        return money_fmt(v)
    return fmt_vi(v)


def _unit(q) -> str:
    if q.qtype == QType.PERCENT:
        return ""
    u = str(q.unit)
    return f" {u}" if u else ""


def _paren(q) -> str:
    u = _unit(q).strip()
    return f" (đơn vị: {u})" if u else ""


def ir_view(PL: Program) -> dict:
    facts, consts, rels = [], [], []
    for q in sorted(PL.givens(), key=lambda q: (q.sentence_id if q.sentence_id is not None else 99, q.qid)):
        tag = " (phụ: không cần cho câu hỏi)" if q.distractor else ""
        facts.append(f'- [{q.qid}] {q.label_vi or "(không tên)"} = {_val(q)}{_unit(q)}{tag}  → viết <q n="{q.qid}">…</q>')
    for q in PL.quantities.values():
        if q.role == "world_const":
            consts.append(f"- [{q.qid}] {q.label_vi or 'hằng số'} = {fmt_vi(q.value)}")
    order = PL.topo()
    prod = PL.producers()
    for n in order:
        o = prod.get(n)
        if o is None:
            continue
        q = PL.quantities[o.output]
        rels.append(f"- [{q.qid}] {q.label_vi or '(đại lượng trung gian)'}{_paren(q)} = {_formula(PL, o)}")
    for u in PL.unknowns():
        rels.append(f"- CHƯA_BIẾT: [{u.qid}] {u.label_vi}{_paren(u)}, không cho biết giá trị")
    for e in PL.equations:
        rels.append(f"- PHƯƠNG_TRÌNH: [{e.lhs}] = [{e.rhs}]")
    qq = PL.quantities[PL.query]
    return {"facts": "\n".join(facts), "consts": "\n".join(consts) or "(không có)",
            "relations": "\n".join(rels), "question": f"[{qq.qid}] {qq.label_vi}{_paren(qq)}"}


__all__ = ["localize", "money_map", "keep_units", "ir_view", "card_for", "money_fmt"]
