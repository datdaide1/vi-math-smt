"""G4 operator fixes (applied to every renderer variant):

* M6 compose is dropped (G3: judged tail 6/6, 0 teacher-acceptable; X4: semantic tail 100/100 by construction).
* M3/M4/M5: a deterministic story-role filter, then a Haiku story-role check (prompts/story_check.v1).
  - M3: the template library is filtered by the ROLE of the node it acts on (``role_templates``, patched into X4's
    ``operators._templates_for`` for the duration of a mutate call, the same read-only wrapping G3 used for
    ``gate``): count/person templates (mua thêm, cho đi, chia đều …) only on a count STOCK (not a rate, a per-item
    quantity, a difference or a ratio); money templates (giảm giá, thuế, phí) never on income/savings, and "phí"
    never on a per-item price; no template on a difference/comparison quantity.
  - M4: no swap on a ratio literal / ratio_part, no swap with an unlabelled output, add<->sub only between
    operands of the same type and unit.
  - M5: the target must be a story quantity (count, money, measure, rate, age), not ratio/abstract/percent/time
    point plumbing and not a unit conversion (an op with a world-constant input).
  - Haiku check (effort low): type_ok AND story_meaning, else reject. A semantic-tail M3 (new answer = old ± c or
    k·old, ``rules.is_semantic_tail``) is thereby rejected whenever its step has no story meaning.
* Candidate order per slot: G3's own program first (so unchanged slots stay paired with G3 byte-for-byte), then
  fresh X4 draws ``Random(f"{key}:20261108:g4r{r}")``; at most ``MAX_CHECKS`` distinct candidates go to Haiku.
"""
from __future__ import annotations

import re

from ..symbolic import operators as OPS
from ..symbolic.units import QType, unit_class

ORIG_TEMPLATES = OPS._templates_for
MAX_CHECKS = 4
MAX_DRAWS = 12

RATE_RE = re.compile(r"\bmỗi\b|/|tốc độ|năng suất|tách được|một giờ|một phút|một ngày|một tuần|mỗi giờ")
COMPARE_RE = re.compile(r"chênh lệch|nhiều hơn|ít hơn|cao hơn|thấp hơn|hơn .*bao nhiêu|\bhiệu\b|tỷ lệ|tỉ lệ|"
                        r"hệ số|số lần")
INCOME_RE = re.compile(r"kiếm|lương|thu nhập|tiết kiệm|tiền công|được trả|thưởng|thu từ|thu được|bán được|"
                       r"\blãi\b|lợi nhuận|tiền boa|\btip\b|quyên góp|cho thêm")
COUNT_VERBS = {"cho đi", "làm mất", "mua thêm", "được tặng thêm", "tăng gấp", "chia đều cho", "có thêm",
               "có người rời đi"}
MONEY_VERBS = {"được giảm giá", "phải trả thêm thuế", "trả thêm phí", "được bớt"}


def m3_template_ok(q, t) -> tuple[bool, str]:
    lab = (q.label_vi or "").lower()
    if q.role == "world_const":
        return False, "world constant"
    if COMPARE_RE.search(lab):
        return False, "difference/ratio quantity"
    if t.verb_vi in COUNT_VERBS:
        if q.qtype != QType.COUNT or q.unit.per or RATE_RE.search(lab):
            return False, "count template on a non-stock (rate/per-item) quantity"
    if t.verb_vi in MONEY_VERBS:
        if INCOME_RE.search(lab):
            return False, "price template on income/savings"
        if t.verb_vi == "trả thêm phí" and ("mỗi" in lab or q.unit.per):
            return False, "fee on a per-item price"
    return True, ""


def role_templates(q):
    return [t for t in ORIG_TEMPLATES(q) if m3_template_ok(q, t)[0]]


def _changed_op(P, P2):
    old = {o.oid: o for o in P.ops}
    for o in P2.ops:
        p = old.get(o.oid)
        if p is not None and (p.kind != o.kind or [P2.quantities[i].value for i in o.inputs] !=
                              [P.quantities[i].value for i in p.inputs]):
            return p, o
    return None, None


def det_check(op: str, P, P2, edits) -> tuple[bool, str]:
    """Deterministic story-role filter on a candidate mutation (P -> P2)."""
    if op == "M3":
        e = next(e for e in edits if e.kind == "INSERT_STEP")
        new_op = next(o for o in P2.ops if e.qid in o.inputs)
        x = P.quantities.get(new_op.inputs[0])
        t = next((t for t in ORIG_TEMPLATES(x) if t.verb_vi == e.template), None) if x else None
        if x is None or t is None:
            return False, "inserted step not found"
        return m3_template_ok(x, t)
    if op == "M4":
        p, o = _changed_op(P, P2)
        if p is None:
            return False, "changed op not found"
        ins = [P.quantities[i] for i in p.inputs]
        if p.kind == "ratio_part" or any(re.search(r"tỷ lệ|tỉ lệ", (q.label_vi or "").lower()) for q in ins):
            return False, "swap on a ratio literal"
        if not P.quantities[p.output].label_vi:
            return False, "unlabelled output"
        if {p.kind, o.kind} == {"add", "sub"}:
            a, b = ins[0], ins[1]
            if a.qtype != b.qtype or (a.unit.lemma or "") != (b.unit.lemma or "") or a.unit.per != b.unit.per:
                return False, "add/sub between different types or units"
        return True, ""
    if op == "M5":
        q = P2.quantities[P2.query]
        if q.qtype in (QType.RATIO, QType.ABSTRACT, QType.PERCENT, QType.TIME_POINT):
            return False, f"target is {q.qtype.value} plumbing"
        prod = P2.producer(q.qid)
        if prod is not None and any(P2.quantities[i].role == "world_const" for i in prod.inputs):
            return False, "target is a unit conversion"
        return True, ""
    return True, ""


def _sent(P, sid) -> str:
    if sid is None or sid >= len(P.sentences):
        return ""
    s, e = P.sentences[sid]
    return P.text[s:e].strip()


def describe(op: str, P, P2, edits) -> str:
    """Plain Vietnamese description of the change for the story check (no answer, no derived value)."""
    if op == "M3":
        e = next(e for e in edits if e.kind == "INSERT_STEP")
        c = P2.quantities[e.qid]
        unit = "" if c.qtype in (QType.PERCENT, QType.RATIO) else (" " + str(c.unit) if str(c.unit) else "")
        how = {"tăng gấp": f"tăng gấp {e.value} lần", "chia đều cho": f"chia đều cho {e.value} (người/phần)"}.get(
            e.template, f"{e.template} {e.value}{unit}")
        return (f'Chèn một sự việc ngay sau câu: "{_sent(P, e.after_sentence)}"\n'
                f'Sự việc: đại lượng "{e.operands[0]}" thay đổi: {how}.\n'
                f'Từ đó trở đi bài toán dùng giá trị sau thay đổi. Câu hỏi giữ nguyên.')
    if op == "M4":
        e = next(e for e in edits if e.kind == "REPLACE_RELATION")
        extra = f' (giá trị mới trong câu: "{e.value}")' if e.value else ""
        return (f'Trong câu: "{_sent(P, e.sentence_id)}"\nĐổi cách tính "{e.label}" (từ: '
                + "; ".join(f'"{x}"' for x in e.operands) + f') từ "{e.verb_old}" thành "{e.verb_new}"{extra}.\n'
                "Câu hỏi giữ nguyên.")
    if op == "M5":
        e = next(e for e in edits if e.kind == "NEW_QUESTION")
        return (f'Giữ nguyên câu chuyện. Thay câu hỏi "{_sent(P, P.question_sentence_id())}" bằng câu hỏi về: '
                f'"{e.label}".')
    return ""


__all__ = ["role_templates", "det_check", "describe", "m3_template_ok", "MAX_CHECKS", "MAX_DRAWS", "ORIG_TEMPLATES"]
