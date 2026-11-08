"""Editor-facing view of an X4 edit list (M2-M6) and the post-edit tag check.

What the editor sees (T1 §2.2 step 3; never the new answer, never IR op names, never ids beyond tag numbers):
* ``text_tagged``  the base text with every given wrapped as <q n="qid">literal</q> (``symbolic.tagged_text``).
                   M2-M5: base = the seed text. M6: base = seed text + " " + partner text (the X4 compose program's
                   text); partner givens keep their offset qids; the partner literal that the old query replaces
                   (the REFERENCE span) is wrapped as <ref>literal</ref>.
* ``sentences``    the base text split into numbered sentences "[i] ..." (tagged), because X4 edits refer to
                   sentence indices (after_sentence, sentence_id, drop_question_of_first).
* ``edits``        ``[e.to_editor_dict() for e in edits]`` verbatim, except that JOIN's character offsets
                   (first_text / second_text) are dropped: they index the raw program text, which the editor never
                   sees, and the <ref> tag carries the same information.

``check_output`` applies ``symbolic.tag_issues`` (tag multiset == givens of P2; no untagged numeral except world
constants and numerals already untagged in the seed). For M6 the partner seed's own untagged numerals are allowed
too (``tag_issues`` takes one seed), and a leftover <ref> tag is an issue.
"""
from __future__ import annotations

import json
import re

from ..symbolic import Program, TextEdit, strip_tags, tag_issues
from ..symbolic.render import _seed_untagged_values, untagged_numerals

_REF = re.compile(r"</?ref>")


def _tag_spans(P: Program, ref_span: tuple[int, int] | None) -> list[tuple[int, int, str, str]]:
    out = [(q.span[0], q.span[1], f'<q n="{q.qid}">', "</q>") for q in P.givens() if q.span is not None]
    if ref_span is not None:
        out.append((ref_span[0], ref_span[1], "<ref>", "</ref>"))
    return sorted(out)


def _tag_range(P: Program, lo: int, hi: int, spans) -> str:
    out = P.text[lo:hi]
    for s, e, a, b in sorted(spans, reverse=True):
        if lo <= s and e <= hi:
            s2, e2 = s - lo, e - lo
            out = out[:s2] + a + out[s2:e2] + b + out[e2:]
    return out


def editor_view(P: Program, P2: Program, edits: list[TextEdit], op: str) -> dict:
    """Inputs for prompts/editor.v2 (text_tagged, sentences, edits as JSON)."""
    base = P2 if op == "M6" else P
    ref = next((e.span for e in edits if e.kind == "REFERENCE"), None) if op == "M6" else None
    spans = _tag_spans(base, ref)
    text_tagged = _tag_range(base, 0, len(base.text), spans)
    sents = [f"[{i}] {_tag_range(base, s, e, spans)}" for i, (s, e) in enumerate(base.sentences)]
    ed = []
    for e in edits:
        d = e.to_editor_dict()
        if e.kind == "JOIN":
            d.pop("first_text", None)
            d.pop("second_text", None)
        ed.append(d)
    return {"text_tagged": text_tagged, "sentences": "\n".join(sents),
            "edits": json.dumps(ed, ensure_ascii=False, indent=1)}


def check_output(tagged: str, P: Program, P2: Program, op: str, partner: Program | None = None) -> list[str]:
    """Tag-check issues of an editor output (empty = pass)."""
    issues = []
    if _REF.search(tagged or ""):
        issues.append("leftover <ref> tag")
        tagged = _REF.sub("", tagged)
    raw = tag_issues(tagged, P2, seed=P)
    if op == "M6" and partner is not None:
        extra_ok = _seed_untagged_values(partner)
        bad_lits = {m.literal for m in untagged_numerals(tagged) if not (set(m.values) & extra_ok)}
        raw = [i for i in raw if not i.startswith("untagged numeral") or i.split("'")[1] in bad_lits]
    return issues + raw


def final_text(tagged: str) -> str:
    return re.sub(r"\s+", " ", strip_tags(_REF.sub("", tagged or ""))).strip()


__all__ = ["editor_view", "check_output", "final_text"]
