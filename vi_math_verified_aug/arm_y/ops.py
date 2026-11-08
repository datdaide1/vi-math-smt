"""Arm-Y v3 operators: constraints applied AT MUTATION TIME (not only as filters after the fact).

G4 evidence (exec/G4_symbolic_v2.md): Haiku judged story_meaning = false on 107/125 M3 and 11/15 M4 candidates; M4
add<->sub swaps almost never give a sensible story; R1 survivors were absurd on 21%.

* M6: dropped.
* M3 template library, restricted by entity role and domain (``m3_ok_v3``, patched into X4's ``_templates_for``):
  - G4 rules (``roles.m3_template_ok``): count templates only on count stocks, price templates never on income, no
    fee on a per-item price, nothing on a difference/ratio;
  - entity role: a count/person/measure step (mua thêm, cho đi, làm mất, được tặng, có thêm/rời đi, thêm/bớt) acts
    only on a STATED stock (a given that someone owns or has), never on a computed aggregate ("tổng", "sau khi",
    "còn lại", per-item "mỗi", averages, differences);
  - domain: giảm giá / thuế / phí / được bớt only on a price or bill (label names a price/cost);
  - "chia đều cho" and "tăng gấp" are removed: they change ownership / have no story cause (G3 §4, G4 checks).
* M4: only changes that keep the story's causal direction: a factor change k -> k' ("gấp đôi" -> "gấp ba") and
  pct_inc <-> pct_dec. add <-> sub and "gấp k lần" <-> "bằng 1/k" swaps are rejected (they invert give/receive,
  more/less, whole/part).
* Domain ranges on every node of the mutated program (``domain_issues``): ages <= 100, a count per container/person
  ("mỗi …") <= 200, hours per day <= 24, percents of a given <= 100, money per item > 0.
* pi_v3 (re-weighted over the operators that survive, from G4 per-operator accepted yield per slot, R1:
  M1 0.85 (not new by definition), M2 0.72·0.58, M3 0.32·0.57, M4 0.11, M5 0.45·0.20):
  M1 .25, M2 .40, M3 .25, M4 .05, M5 .05.
"""
from __future__ import annotations

import random
import re

from ..pilot_symbolic_v2 import roles as RL
from ..symbolic import execute, mutate
from ..symbolic import operators as OPS
from ..symbolic.units import QType

PI_V3 = {"M1": .25, "M2": .40, "M3": .25, "M4": .05, "M5": .05}
SEED = 20261108
MAX_DRAWS = 12
MAX_CANDS = 3

AGG_RE = re.compile(r"tổng|sau khi|còn lại|\bmỗi\b|trung bình|chênh lệch|\bhiệu\b|tất cả|cả hai|cả ba")
PRICE_RE = re.compile(r"giá|chi phí|hóa đơn|phải trả|tiền mua|tiền vé|học phí|cước|phí")
STOCK_VERBS = RL.COUNT_VERBS | {"thêm", "bớt đi"}
DROP_VERBS = {"chia đều cho", "tăng gấp"}


def m3_ok_v3(q, t) -> tuple[bool, str]:
    ok, why = RL.m3_template_ok(q, t)
    if not ok:
        return ok, why
    lab = (q.label_vi or "").lower()
    if t.verb_vi in DROP_VERBS:
        return False, "template removed in v3 (ownership change / no story cause)"
    if t.verb_vi in STOCK_VERBS:
        if q.role != "given":
            return False, "entity: step on a computed aggregate, not a stated stock"
        if AGG_RE.search(lab):
            return False, "entity: aggregate or per-item quantity"
    if t.verb_vi in RL.MONEY_VERBS and not PRICE_RE.search(lab):
        return False, "domain: price step on a non-price quantity"
    return True, ""


def templates_v3(q):
    return [t for t in RL.ORIG_TEMPLATES(q) if m3_ok_v3(q, t)[0]]


def m4_ok_v3(P, P2) -> tuple[bool, str]:
    p, o = RL._changed_op(P, P2)
    if p is None:
        return False, "changed op not found"
    if {p.kind, o.kind} in ({"add", "sub"}, {"mul", "div"}):
        return False, "causal direction: add/sub or gấp/bằng-1/k swap"
    return True, ""


def _clean_literal(v) -> bool:
    """Terminating decimal (<= 2 dp), or a simple proper fraction such as 1/3, 2/3 (denominator <= 12, < 1)."""
    from ..symbolic.numerals import fmt_vi
    return "/" not in fmt_vi(v) or (v < 1 and v.denominator <= 12)


def domain_issues(P2, P=None) -> list[str]:
    """Per-QType world-plausibility of the mutated program (C3: 8/13 R1b absurd items are operator VALUE choices).
    With the seed program P: every node that exists in both must stay within [1/4, 4] x its seed value (catches
    48/2^15 cans, a 3.6x egg price), and every given (stated facts included) must be a clean literal (no 400/7)."""
    try:
        env = execute(P2)
    except Exception as e:                         # noqa: BLE001
        return [f"exec: {e}"]
    out = []
    if P is not None:
        env0 = execute(P)
        for qid, v0 in env0.items():
            q = P2.quantities.get(qid)
            v = env.get(qid)
            if q is None or v is None or v0 is None or v0 <= 0 or v <= 0 or q.qtype in (QType.RATIO, QType.PERCENT):
                continue
            if not (v0 / 4 <= v <= 4 * v0) and max(v, v0) > 12:
                out.append(f"value drift q{qid} {v0} -> {v} (> 4x)")
        for q in P2.givens():
            if q.value is not None and q.qtype not in (QType.RATIO, QType.ABSTRACT) and not _clean_literal(q.value):
                out.append(f"unclean given {q.value}")
    for q in P2.quantities.values():
        v = env.get(q.qid)
        if v is None:
            continue
        lab = (q.label_vi or "").lower()
        if q.qtype == QType.AGE and v > 100:
            out.append(f"age {v} > 100")
        if q.qtype == QType.COUNT and "mỗi" in lab and v > 200:
            out.append(f"count per unit {v} > 200 ({lab})")
        if "giờ" in str(q.unit) and "mỗi ngày" in lab and v > 24:
            out.append(f"hours per day {v} > 24")
        if q.qtype == QType.PERCENT and q.role == "given" and v > 100 and "tăng" not in lab:
            out.append(f"percent {v} > 100")
        if q.qtype == QType.MONEY and v <= 0 and not q.distractor:
            out.append(f"money {v} <= 0")
    return out


def det_check_v3(op, P, P2, edits) -> tuple[bool, str]:
    ok, why = RL.det_check(op, P, P2, edits)
    if not ok:
        return ok, why
    if op == "M3":
        e = next(e for e in edits if e.kind == "INSERT_STEP")
        new_op = next(o for o in P2.ops if e.qid in o.inputs)
        x = P.quantities[new_op.inputs[0]]
        t = next((t for t in RL.ORIG_TEMPLATES(x) if t.verb_vi == e.template), None)
        if t is None:
            return False, "template not found"
        ok, why = m3_ok_v3(x, t)
        if not ok:
            return ok, why
    if op == "M4":
        ok, why = m4_ok_v3(P, P2)
        if not ok:
            return ok, why
    d = domain_issues(P2, P)
    if d:
        return False, "domain: " + d[0]
    return True, ""


def candidates(op: str, P, key: str) -> tuple[list, list]:
    """Up to MAX_CANDS distinct candidates passing the deterministic v3 checks (M1/M2: first passing draw).
    Draw 0 uses the G3/G4 slot RNG Random(f"{key}:20261108"); later draws f"{key}:20261108:g5r{r}"."""
    out, log, seen = [], [], set()
    OPS._templates_for = templates_v3
    try:
        for r in range(MAX_DRAWS):
            rng = random.Random(f"{key}:{SEED}" if r == 0 else f"{key}:{SEED}:g5r{r}")
            res = mutate(op, P, rng)
            if not res.ok:
                log.append({"draw": r, "na": res.reason})
                if op in ("M1", "M2") and r >= 4:
                    break
                continue
            sig = repr([e.to_editor_dict() for e in res.edits])
            if sig in seen:
                continue
            seen.add(sig)
            ok, why = det_check_v3(op, P, res.program, res.edits)
            log.append({"draw": r, "det_ok": ok, "why": why})
            if ok:
                out.append((f"r{r}", res.program, res.edits))
                if op in ("M1", "M2") or len(out) >= MAX_CANDS:
                    break
            elif op in ("M1", "M2") and r >= 4:
                break
    finally:
        OPS._templates_for = RL.ORIG_TEMPLATES
    return out, log


__all__ = ["PI_V3", "candidates", "det_check_v3", "domain_issues", "templates_v3", "m3_ok_v3"]
