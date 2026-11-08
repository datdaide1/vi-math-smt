"""Renderer support (T1 §2.2 step 3, 01_TECHNICAL_DESIGN §3.4). No LLM here.

TextEdit                         one edit of the seed text, as produced by an operator. to_editor_dict() is what
                                 an LLM editor may see: labels and GIVEN values only, never the new answer.
apply_substitutions(text, edits) deterministic M1 rendering: replace each SUBST span (right to left)
render_substitutions(P2, edits)  same, and re-anchor P2 (text, spans, sentences) so it validates on the new text
tagged_text(P)                   'Lan có <q n="1">12</q> cái kẹo...' (every given span wrapped in a tag)
parse_tags / strip_tags          read the editor output
tag_multiset_ok(tagged, P2, P)   multiset {(qid, value)} of tags == givens of P2, and the untagged text has no
                                 numeral except world constants and numerals already untagged in the seed
leaks_answer(text, P2)           the answer value appears in the text (any VN numeric surface form)
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from fractions import Fraction

from .ir import Program, split_sentences
from .numerals import numbers_in_text, parse_literal, contains_value

__all__ = ["TextEdit", "EDIT_KINDS", "apply_substitutions", "render_substitutions", "tagged_text", "parse_tags",
           "strip_tags", "tag_issues", "tag_multiset_ok", "leaks_answer", "untagged_numerals"]

EDIT_KINDS = ("SUBST", "UNKNOWN", "STATE_FACT", "NEW_QUESTION", "INSERT_STEP", "REPLACE_RELATION", "JOIN",
              "REFERENCE")


@dataclass
class TextEdit:
    kind: str
    span: tuple[int, int] | None = None   # char offsets in the SEED text (SUBST, UNKNOWN, REFERENCE)
    text: str | None = None               # replacement surface (SUBST) or verb/template text
    qid: int | None = None                # quantity this edit is about
    value: str | None = None              # formatted given value (never the answer)
    label: str | None = None              # label_vi of the quantity (STATE_FACT, NEW_QUESTION, REFERENCE)
    after_sentence: int | None = None     # STATE_FACT / INSERT_STEP placement
    sentence_id: int | None = None        # REPLACE_RELATION target sentence
    template: str | None = None           # INSERT_STEP template (Vietnamese verb phrase)
    operands: list[str] = field(default_factory=list)
    verb_old: str | None = None
    verb_new: str | None = None
    payload: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in EDIT_KINDS:
            raise ValueError(f"unknown edit kind {self.kind}")

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()
                if v not in (None, [], {})}

    def to_editor_dict(self) -> dict:
        """The editor-facing view: no spans of the answer, no IR op names beyond the edit kind."""
        d = {"type": self.kind}
        if self.qid is not None:
            d["q"] = self.qid
        for k in ("value", "label", "after_sentence", "sentence_id", "template", "verb_old", "verb_new"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.operands:
            d["operands"] = list(self.operands)
        if self.kind == "SUBST":
            d["new_text"] = self.text
        if self.kind == "NEW_QUESTION":
            d["target_label"] = self.label
        if self.kind == "JOIN":
            d.update({k: v for k, v in self.payload.items()})
        return d


def apply_substitutions(text: str, edits: list[TextEdit]) -> str:
    """Deterministic rendering of SUBST edits (M1). Spans must not overlap."""
    subs = sorted([e for e in edits if e.kind == "SUBST"], key=lambda e: e.span[0])
    for a, b in zip(subs, subs[1:]):
        if a.span[1] > b.span[0]:
            raise ValueError("overlapping SUBST spans")
    out = text
    for e in reversed(subs):
        s, t = e.span
        out = out[:s] + e.text + out[t:]
    return out


def render_substitutions(P2: Program, edits: list[TextEdit]) -> Program:
    """Apply SUBST edits to P2.text and shift every span so P2 is anchored in the new text."""
    subs = sorted([e for e in edits if e.kind == "SUBST"], key=lambda e: e.span[0])
    new_text = apply_substitutions(P2.text, subs)
    R = P2.copy()

    def shift(pos: int) -> int:
        d = 0
        for e in subs:
            if e.span[1] <= pos:
                d += len(e.text) - (e.span[1] - e.span[0])
        return pos + d

    by_span = {e.span: e for e in subs}
    for q in R.quantities.values():
        if q.span is None:
            continue
        s, t = q.span
        if (s, t) in by_span:
            ns = shift(s)
            q.span = (ns, ns + len(by_span[(s, t)].text))
        else:
            q.span = (shift(s), shift(t))
    R.text = new_text
    R.sentences = split_sentences(new_text)
    for q in R.quantities.values():
        if q.span is not None:
            q.sentence_id = R.sentence_of(q.span[0])
    return R


_TAG = re.compile(r"<q\s+n\s*=\s*[\"']?(\d+)[\"']?\s*>(.*?)</q>", re.S)


def tagged_text(P: Program, qids=None) -> str:
    """Seed text with every given (or the qids listed) wrapped as <q n="qid">literal</q>."""
    gs = [q for q in P.givens() if q.span is not None and (qids is None or q.qid in qids)]
    out = P.text
    for q in sorted(gs, key=lambda q: q.span[0], reverse=True):
        s, e = q.span
        out = out[:s] + f'<q n="{q.qid}">' + out[s:e] + "</q>" + out[e:]
    return out


def parse_tags(tagged: str) -> list[tuple[int, str]]:
    return [(int(m.group(1)), m.group(2)) for m in _TAG.finditer(tagged)]


def strip_tags(tagged: str) -> str:
    return _TAG.sub(lambda m: m.group(2), tagged)


def untagged_numerals(tagged: str) -> list:
    """Numerals (digits and number words) outside tags."""
    blanked = _TAG.sub(lambda m: " " * len(m.group(0)), tagged)
    return numbers_in_text(blanked, words=True)


def _seed_untagged_values(P: Program) -> set:
    vals: set = set()
    for m in untagged_numerals(tagged_text(P)):
        vals |= set(m.values)
    return vals


def tag_issues(tagged: str, P2: Program, seed: Program | None = None) -> list[str]:
    """Why the editor output fails the tag check (empty list = OK)."""
    issues = []
    got = []
    for qid, lit in parse_tags(tagged):
        vals = parse_literal(lit)
        q = P2.quantities.get(qid)
        if q is None or q.role != "given":
            issues.append(f"tag q{qid} is not a given of the mutated program")
            continue
        if q.value not in vals:
            issues.append(f"tag q{qid} literal {lit!r} != value {q.value}")
            continue
        got.append((qid, q.value))
    want = [(q.qid, q.value) for q in P2.givens()]
    if Counter(got) != Counter(want) and not issues:
        missing = Counter(want) - Counter(got)
        extra = Counter(got) - Counter(want)
        issues.append(f"tag multiset mismatch: missing {sorted(missing)} extra {sorted(extra)}")
    allowed = {q.value for q in P2.quantities.values() if q.role == "world_const"}
    if seed is not None:
        allowed |= _seed_untagged_values(seed)
    for m in untagged_numerals(tagged):
        if not (set(m.values) & allowed):
            issues.append(f"untagged numeral {m.literal!r}")
    return issues


def tag_multiset_ok(tagged: str, P2: Program, seed: Program | None = None) -> bool:
    return not tag_issues(tagged, P2, seed)


def leaks_answer(text: str, P2: Program, answer: Fraction | None = None) -> bool:
    from .executor import execute
    a = answer if answer is not None else execute(P2)[P2.query]
    return contains_value(text, a)
