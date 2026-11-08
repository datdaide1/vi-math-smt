"""Deterministic M2–M5 edit lists from a correct (formalization-pass) IR, for the renderer test (c).

Each builder returns {"edits": [...], "expected": {tag_id: value-as-str}, "new_answer": str, "info": str} or None.
The renderer gets only the tagged seed text + the edit list (labels and GIVEN values); never new_answer.
Simplified versions of T1 §2.2 operators (pilot scope: straight-line programs, no unknowns):
  M2 inverse      resample g (on the query path, affine in g), state the old query's new value, ask for g
  M3 insert-step  x' = x − c | x + c | x·(100 − p)/100 on a given x upstream of the query
  M4 relation     flip a top-level + / − of a derived node on the query path (both operands are ids)
  M5 query-shift  ask for another derived node (depth ≥ 2 preferred)
All results must satisfy type constraints (integer/non-negative counts...), have a clean answer and differ
from the seed answer.
"""
from __future__ import annotations

import ast
import copy
import math
import random
import re
from fractions import Fraction

from .ir_exec import IRError, Program, Q, parse_expr
from .numerals import fmt_like


def _clean(v: Fraction, kind: str) -> bool:
    if abs(v) >= 10 ** 7 or v <= 0:
        return False
    if kind in ("count", "age"):
        return v.denominator == 1
    return (v * 100).denominator == 1


def _ok(P: Program):
    try:
        env = P.execute()
    except (IRError, ZeroDivisionError, OverflowError):
        return None
    if P.type_violations(env):
        return None
    for q in P.qs.values():  # every intermediate of an integer kind must be integral, all non-negative
        if q.role == "derived" and env[q.id] < 0:
            return None
    if not _clean(env[P.query], P.qs[P.query].kind):
        return None
    return env


def _resample(q: Q, rng: random.Random):
    v = q.value
    if v is None or v <= 0 or q.kind in ("ratio", "abstract", "time_point"):
        return None
    x = math.exp(rng.uniform(math.log(v / 3), math.log(v * 3)))
    if q.kind in ("count", "age"):
        n = Fraction(max(1, round(x)))
    elif q.kind == "percent":
        n = Fraction(min(95, max(5, 5 * round(x / 5))))
    elif v.denominator == 1:
        grid = 500 if (v >= 1000 and v % 500 == 0) else (5 if (v >= 50 and v % 5 == 0) else 1)
        n = Fraction(max(grid, grid * round(x / grid)))
    else:
        n = Fraction(round(x * 2), 2) if (v * 2).denominator == 1 else Fraction(round(x * 100), 100)
        if n <= 0:
            return None
    return n if n != v else None


def _digit_style(P: Program) -> str:
    for g in P.givens():
        if g.span and any(ch.isdigit() for ch in g.span) and re.search(r"\d[.,]\d", g.span):
            return g.span
    return "1"


def _replace_name(expr: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, expr)


def m2(P: Program, rng):
    if any(q.role == "unknown" for q in P.qs.values()):
        return None
    anc = P.ancestors(P.query)
    cands = [g for g in P.givens() if g.id in anc and g.span and any(c.isdigit() for c in g.span)
             and g.kind not in ("ratio", "abstract", "time_point")]
    rng.shuffle(cands)
    for g in cands:
        for _ in range(60):
            v = _resample(g, rng)
            if v is None:
                continue
            P1 = copy.deepcopy(P)
            P1.qs[g.id].value = v
            env = _ok(P1)
            if env is None:
                continue
            q0v = env[P.query]
            vals = []
            for d in (1, 2):  # affine (invertible) in g?
                P2 = copy.deepcopy(P1)
                P2.qs[g.id].value = v + d
                try:
                    vals.append(P2.execute()[P.query])
                except Exception:
                    vals.append(None)
            if None in vals or vals[0] - q0v == 0 or (vals[1] - vals[0]) != (vals[0] - q0v):
                break  # not invertible through this given
            if q0v == v:
                continue
            ql = P.qs[P.query]
            return {"edits": [{"type": "UNKNOWN", "q": g.id},
                              {"type": "STATE_FACT", "q": "s1", "label": ql.label,
                               "value": fmt_like(q0v, _digit_style(P)), "after": "cuối câu chuyện, trước câu hỏi"},
                              {"type": "NEW_QUESTION", "target_label": g.label}],
                    "expected": {**{x.id: str(x.value) for x in P.givens() if x.id != g.id}, "s1": str(q0v)},
                    "new_answer": str(v), "info": f"inverse on {g.id}"}
    return None


_M3_TPL = {
    "count": [("sub", "bị bớt đi (cho đi / bị hỏng / bị mất) một số {unit}"), ("add", "có thêm một số {unit}")],
    "money": [("sub", "được bớt (giảm trực tiếp) một khoản tiền"), ("pct", "được giảm giá theo phần trăm")],
    "measure": [("add", "được thêm một lượng {unit}"), ("sub", "bị bớt đi một lượng {unit}")],
}


def m3(P: Program, rng):
    if any(q.role == "unknown" for q in P.qs.values()):
        return None
    anc = P.ancestors(P.query)
    cands = [g for g in P.givens() if g.id in anc and g.kind in _M3_TPL and g.span
             and any(c.isdigit() for c in g.span) and g.value and g.value > 0]
    rng.shuffle(cands)
    for x in cands:
        tpls = list(_M3_TPL[x.kind])
        rng.shuffle(tpls)
        for kind, phrase in tpls:
            for _ in range(30):
                v = x.value
                if kind == "pct":
                    c = Fraction(rng.choice([10, 15, 20, 25, 30]))
                    expr = f"{x.id} * (100 - c1) / 100"
                elif x.kind == "count":
                    hi = max(1, int(v // 3) if kind == "sub" else int(max(2, v // 2)))
                    c = Fraction(rng.randint(1, hi))
                    if kind == "sub" and c >= v:
                        continue
                    expr = f"{x.id} - c1" if kind == "sub" else f"{x.id} + c1"
                else:
                    f = Fraction(rng.choice([10, 15, 20, 25, 30, 40, 50]), 100)
                    c = v * f
                    c = Fraction(round(c)) if v.denominator == 1 and c >= 1 else Fraction(round(c * 2), 2)
                    if c <= 0 or (kind == "sub" and c >= v):
                        continue
                    expr = f"{x.id} - c1" if kind == "sub" else f"{x.id} + c1"
                P2 = copy.deepcopy(P)
                for q in P2.qs.values():
                    if q.role == "derived":
                        q.expr = _replace_name(q.expr, x.id, "x_adj")
                P2.qs["c1"] = Q("c1", "given", "percent" if kind == "pct" else x.kind, x.unit, "", c)
                P2.qs["x_adj"] = Q("x_adj", "derived", x.kind, x.unit, x.label + " (sau tình tiết mới)",
                                   expr=expr)
                env = _ok(P2)
                if env is None or env[P.query] == P.answer:
                    continue
                desc = phrase.format(unit=x.unit or "")
                lit = fmt_like(c, x.span)
                return {"edits": [{"type": "INSERT_STEP", "after_q": x.id, "new_q": "c1", "value": lit,
                                   "description": f"Thêm tình tiết: '{x.label}' {desc}; giá trị mới là "
                                                  f"<q n=\"c1\">{lit}</q>{'%' if kind == 'pct' else ''}. Mọi phép "
                                                  f"tính sau đó dùng giá trị sau khi đã thay đổi. Câu hỏi giữ nguyên."}],
                        "expected": {**{g.id: str(g.value) for g in P.givens()}, "c1": str(c)},
                        "new_answer": str(env[P.query]), "info": f"{kind} on {x.id}"}
    return None


def m4(P: Program, rng):
    if any(q.role == "unknown" for q in P.qs.values()):
        return None
    path = (P.ancestors(P.query) | {P.query})
    cands = []
    for qid in path:
        q = P.qs[qid]
        if q.role != "derived":
            continue
        t = parse_expr(q.expr).body
        if isinstance(t, ast.BinOp) and isinstance(t.op, (ast.Add, ast.Sub)) and \
                isinstance(t.left, ast.Name) and isinstance(t.right, ast.Name):
            cands.append((q, t))
    rng.shuffle(cands)
    for q, t in cands:
        new_op = "-" if isinstance(t.op, ast.Add) else "+"
        P2 = copy.deepcopy(P)
        P2.qs[q.id].expr = f"{t.left.id} {new_op} {t.right.id}"
        env = _ok(P2)
        if env is None or env[P.query] == P.answer:
            continue
        L, R = P.qs[t.left.id].label, P.qs[t.right.id].label
        old_w, new_w = ("cộng", "trừ") if new_op == "-" else ("trừ", "cộng")
        return {"edits": [{"type": "REPLACE_RELATION", "clause": q.clause if (q.clause and q.clause in P.text) else None,
                           "description": f"Trước đây: '{q.label}' = '{L}' {old_w} '{R}'. Bây giờ phải là: "
                                          f"'{q.label}' = '{L}' {new_w} '{R}'. Viết lại câu chuyện cho hợp lý "
                                          f"(ví dụ đổi 'thêm/nhận' thành 'bớt/cho đi' hoặc ngược lại); giữ nguyên "
                                          f"các thẻ số và câu hỏi."}],
                "expected": {g.id: str(g.value) for g in P.givens()},
                "new_answer": str(env[P.query]), "info": f"flip {q.id} to {new_op}"}
    return None


def m5(P: Program, rng):
    if any(q.role == "unknown" for q in P.qs.values()):
        return None
    try:
        env = P.execute()
    except Exception:
        return None
    ds = [q for q in P.qs.values() if q.role == "derived" and q.id != P.query and q.label
          and _clean(env[q.id], q.kind) and env[q.id] != P.answer
          and not any(env[q.id] == g.value for g in P.givens())]
    deep = [q for q in ds if P.depth(q.id) >= 2]
    pool = deep or ds
    if not pool:
        return None
    t = rng.choice(sorted(pool, key=lambda q: q.id))
    return {"edits": [{"type": "NEW_QUESTION", "target_label": t.label}],
            "expected": {g.id: str(g.value) for g in P.givens()},
            "new_answer": str(env[t.id]), "info": f"query -> {t.id} (depth {P.depth(t.id)})"}


BUILD = {"M2": m2, "M3": m3, "M4": m4, "M5": m5}
