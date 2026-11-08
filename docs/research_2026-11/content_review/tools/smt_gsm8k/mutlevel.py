"""Story-level mutation support per program (heuristic; validated against the 150 read items).

L3 story-mutable : no hardcode/closed form; every digit-quantity of the story that the program uses is
                   bound to its own variable (pin or relation); no hand-precomputed literals.
L2 partial       : >=1 story quantity bound to a variable, but others inline as bare literals or
                   folded/precomputed by hand.
L1 answer-expr   : closed form, or no story quantity bound to a variable -> only an arithmetic
                   expression to perturb.
L0 none          : answer hard-coded.
Also: semantic names (>=50% of non-answer vars have names of >=3 chars), and retargetable
(>=1 named intermediate variable defined by a relation, i.e. a sub-question exists).
"""
import json, re, sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(sys.argv[1]); FEAT = sys.argv[2]; OUT = sys.argv[3]
sys.path.insert(0, str(ROOT))
from vi_math_verified_aug.verify.uniqueness import _balanced, _DECL_RE  # noqa

NUM = r"-?\d+(?:\.\d+)?"
PIN = re.compile(rf"^\(assert \(= ([A-Za-z_][\w.$]*) ({NUM})\)\)$")
PIN2 = re.compile(rf"^\(assert \(= ([A-Za-z_][\w.$]*) \(\s*[-+*/](?:\s+{NUM})+\s*\)\)\)$")  # v = (op lits) e.g. (* 3.0 12.0)
LIT = re.compile(rf"(?<![\w.])({NUM})(?![\w.])")


def qdigits(q):
    s = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", q):
        try:
            v = Fraction(m.group(0).replace(",", "").rstrip("."))
        except Exception:
            continue
        s.add(v)
    return s


def level(smt, q, feat):
    if feat.get("cls") == "hardcoded":
        return "L0", False, False
    if feat.get("status") != "sat":
        return "ERR", False, False
    decls = {(m.group(1) or m.group(2)): m.group(3) for m in _DECL_RE.finditer(smt)}
    asserts = _balanced(smt, "assert")
    qd = qdigits(q)
    bound, inline = set(), set()
    pinned_vars, defined_vars = set(), set()
    for a in asserts:
        m = PIN.match(a)
        if m and m.group(1) != "answer":
            pinned_vars.add(m.group(1))
            v = Fraction(m.group(2))
            if v in qd:
                bound.add(v)
            continue
        m2 = PIN2.match(a)
        if m2 and m2.group(1) != "answer":   # (= total (* 4.0 24.0)) -> named product of story numbers
            pinned_vars.add(m2.group(1))
            for x in LIT.findall(a):
                if Fraction(x) in qd:
                    bound.add(Fraction(x))
            continue
        lhs = re.match(r"^\(assert \(= ([A-Za-z_][\w.$]*) ", a)
        if lhs and lhs.group(1) != "answer":
            defined_vars.add(lhs.group(1))
        tgt = inline if re.search(r"(?<![\w.])answer(?![\w.])", a) else bound
        for x in LIT.findall(a):
            if Fraction(x) in qd:
                tgt.add(Fraction(x))
    inline -= bound
    nonans = [n for n in decls if n != "answer"]
    used = [n for n in nonans if re.search(rf"(?<![\w.]){re.escape(n)}(?![\w.])", "\n".join(asserts))]
    semantic = bool(used) and sum(len(n) >= 3 for n in used) / len(used) >= 0.5
    retarget = len(defined_vars - pinned_vars) >= 1
    precomp = bool(feat.get("unexplained_lits"))
    if feat.get("closed_form") or not bound:
        return "L1", semantic, retarget
    if not inline and not precomp:
        return "L3", semantic, retarget
    return "L2", semantic, retarget


F = {}
for l in open(FEAT, encoding="utf-8"):
    x = json.loads(l); F[x["index"]] = x
Q = {json.loads(l)["index"]: json.loads(l)["question"] for l in open(ROOT / "data/generate/gsm8k/train.jsonl", encoding="utf-8")}
with open(ROOT / "data/formalize_output/gsm8k/gsm8k-smt-nvidia-FULL.jsonl", encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as o:
    for l in f:
        r = json.loads(l)
        lv, sem, rt = level(r["smt_lib"] or "", Q[r["index"]], F[r["index"]])
        o.write(json.dumps({"index": r["index"], "mut_level": lv, "semantic_names": sem, "retargetable": rt}) + "\n")
