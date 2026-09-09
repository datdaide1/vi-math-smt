"""The uniqueness gate + the N2 audit (T1.1 — also a C3 result).

``classify(smt) -> "derived" | "hardcoded" | "unknown"``

* **hardcoded** — the answer value is written into the program as a literal
  (``(assert (= x 1.4142135)) (assert (= answer x))``): the solver checks a
  tautology the model built by looking at the ground truth.
* **derived** — the non-answer constraints uniquely determine the answer through
  ≥ 2 relational asserts (mostly GSM8K-style linear chains).
* **unknown** — unsat / solver timeout / parse error / under-determined / fewer
  than 2 relational asserts / non-linear (QF_LRA can't express it faithfully).

Run the audit:  ``python -m vi_math_verified_aug.verify.uniqueness --audit``
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from .z3_utils import is_unique, solve_value

_TOL = 1e-4

_DECL_RE = re.compile(r"\(declare-(?:fun\s+([A-Za-z_][\w.$]*)\s*\(\s*\)|const\s+([A-Za-z_][\w.$]*))\s*([A-Za-z]+)\s*\)")
_NUM = r"-?\d+(?:\.\d+)?"
_PIN_RE = re.compile(rf"^\(assert\s+\(=\s+([A-Za-z_][\w.$]*)\s+({_NUM})\s*\)\s*\)$")
_ALIAS_RE = re.compile(r"^\(assert\s+\(=\s+([A-Za-z_][\w.$]*)\s+([A-Za-z_][\w.$]*)\s*\)\s*\)$")
_OP_RE = re.compile(r"\(\s*[-+*/]\s|\bmod\b|\bdiv\b")
_CMP_RE = re.compile(r"\(\s*(?:<=|>=|<|>|distinct)\s")


def _balanced(code: str, kw: str) -> list[str]:
    out, i = [], 0
    pat = f"({kw}"
    while (j := code.find(pat, i)) != -1:
        depth = 0
        for k in range(j, len(code)):
            if code[k] == "(":
                depth += 1
            elif code[k] == ")":
                depth -= 1
                if depth == 0:
                    out.append(" ".join(code[j : k + 1].split()))
                    i = k + 1
                    break
        else:
            break
    return out


def _messy(v: float) -> bool:
    """A clean word-problem input is round; ``1.4142135`` is a ground-truth constant."""
    s = f"{v:.10f}".rstrip("0")
    dec = s.split(".")[1] if "." in s else ""
    return len(dec) >= 4


def classify_verbose(smt: str) -> tuple[str, str]:
    if not smt or "answer" not in smt:
        return "unknown", "no_answer"

    decls = {(m.group(1) or m.group(2)): m.group(3) for m in _DECL_RE.finditer(smt)}
    if "answer" not in decls:
        return "unknown", "no_answer_decl"
    asserts = _balanced(smt, "assert")
    if not asserts:
        return "unknown", "no_asserts"

    status, a_star = solve_value(smt, "answer")
    if status != "sat" or a_star is None:
        return "unknown", f"solve_{status}"

    answer_defs, pins, aliases, relational = [], {}, {}, []
    for a in asserts:
        if (m := _PIN_RE.match(a)) and m.group(1) != "answer":
            pins[m.group(1)] = float(m.group(2))
            continue
        if "answer" in a:
            answer_defs.append(a)
            # the answer-definition itself is a computation step if its RHS has an op
            if _OP_RE.search(a) and len(set(re.findall(r"[A-Za-z_][\w.$]*", a)) & set(decls)) >= 2:
                relational.append(a)
            continue
        if (m := _ALIAS_RE.match(a)):
            aliases[m.group(1)] = m.group(2)
            continue
        if _OP_RE.search(a) or _CMP_RE.search(a):
            relational.append(a)

    # ---- hardcoded: the answer value (not a problem input) is written into the program ----
    for adef in answer_defs:
        if (m := re.match(r"\(assert \(= answer ([A-Za-z_][\w.$]*)\)\)$", adef)):
            v = m.group(1)
            val = pins.get(v, pins.get(aliases.get(v, "\x00")))
            if val is not None and abs(val - a_star) < _TOL:
                return "hardcoded", "answer_is_pinned_var"
        if (m := re.match(rf"\(assert \(= answer ({_NUM})\)\)$", adef)) and abs(float(m.group(1)) - a_star) < _TOL:
            return "hardcoded", "answer_pinned_literal"
    if any(_messy(v) for v in pins.values()):     # irrational / transcendental approximated in-place
        return "hardcoded", "messy_constant"

    # ---- derived: answer follows from the (clean) pinned inputs via a unique relational system ----
    if relational:
        non_answer = {n: s for n, s in decls.items() if n != "answer"}
        core = [a for a in asserts if a not in answer_defs] or asserts
        if is_unique(core, non_answer) is True:
            return "derived", f"unique_{len(relational)}rel"
        return "unknown", "underdetermined"
    return "unknown", "no_relational"


def classify(smt: str) -> str:
    return classify_verbose(smt)[0]


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #

_DEFAULT_SRC = Path("data/finetune/mine/train_vietnamese_final.jsonl")


def audit(src: Path = _DEFAULT_SRC, limit: Optional[int] = None, out: Optional[Path] = None) -> dict:
    rows = []
    with open(src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_subject: dict[str, Counter] = defaultdict(Counter)
    overall = Counter()
    reasons = Counter()
    examples: dict[str, list] = {"hardcoded": [], "derived": [], "unknown": []}
    for r in rows:
        subj = (r.get("subject") or r.get("type") or "?").lower()
        # GSM8K rows are tagged "arithmetic"
        cls, reason = classify_verbose(r.get("smt") or r.get("smt_code") or "")
        by_subject[subj][cls] += 1
        overall[cls] += 1
        reasons[f"{cls}:{reason}"] += 1
        if len(examples[cls]) < 20:
            examples[cls].append({"index": r.get("index"), "subject": subj, "reason": reason,
                                   "ground_truth": r.get("ground_truth"),
                                   "smt": (r.get("smt") or "")[:600]})

    n = sum(overall.values())
    # N2's stricter definition: "genuinely derive it through >=2 relational constraints"
    derived_strict = sum(v for k, v in reasons.items()
                         if k.startswith("derived:unique_") and int(k.split("unique_")[1][:-3]) >= 2)
    report = {
        "source": str(src),
        "n": n,
        "overall": dict(overall),
        "overall_pct": {k: round(100 * v / n, 1) for k, v in overall.items()} if n else {},
        "derived_strict_ge2rel": derived_strict,
        "derived_strict_pct": round(100 * derived_strict / n, 1) if n else 0,
        "reasons": dict(reasons.most_common()),
        "by_subject": {
            s: {**dict(c), "derived_pct": round(100 * c["derived"] / sum(c.values()), 1),
                "hardcoded_pct": round(100 * c["hardcoded"] / sum(c.values()), 1)}
            for s, c in sorted(by_subject.items())
        },
        "examples": examples,
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--src", type=Path, default=_DEFAULT_SRC)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/uniqueness_audit.json"))
    a = ap.parse_args()
    if a.audit:
        rep = audit(a.src, a.limit, a.out)
        print(json.dumps({"n": rep["n"], "overall_pct": rep["overall_pct"]}, indent=1))
        print("by subject (derived% / hardcoded%):")
        for s, c in rep["by_subject"].items():
            print(f"  {s:26} n={sum(v for k,v in c.items() if isinstance(v,int)):5}  "
                  f"derived={c['derived_pct']:5}%  hardcoded={c['hardcoded_pct']:5}%")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    _main()
