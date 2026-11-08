"""Re-type every candidate key and re-check key agreement with the project grader (3.0.2).

  .venv/Scripts/python.exe -m vi_math_verified_aug.vi_exam_qa.keys prepare      # keys_table.jsonl + selection.json

Keys per Core item (T5 §3.1 stage 2):
  sympy   — X2's SymPy program key (non-Claude). Typed into a grader gold directly from its structured values.
  page    — answer extracted by X2's regex from the third-party worked solution (non-Claude, ~80 % precise).
  claude1 — X2's blind Opus solve (effort low, Vietnamese "ĐÁP ÁN:" prompt).
  claude2 — Q1's blind Opus solve (effort high, different prompt; filled in after ``llm run solve2``).

Agreement is decided by ``math_verify.grade`` only: key A agrees with a typed gold G iff grade(A, G) is correct.
Two raw keys agree iff a gold typed from one (round-tripping on its own source) grades the other correct.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from typing import Any, Optional

from ..eval import math_verify as mv
from . import data as D

SOURCES = ("sympy", "page", "claude1", "claude2", "audit")
NON_CLAUDE = ("sympy", "page")


# ----------------------------------------------------------------------------------------------- raw key text
_DELIMS = [(r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$")]


_BARE_BEFORE = set("([;,=") | {"{"}


def _drop_bare_braces(s: str) -> str:
    """Remove MathType-style grouping braces that are not command arguments: ``\\left( {x;y} \\right)``,
    ``\\{ { - 2; 0} \\}``, ``= {\\rm{ }} - 1``. A brace is bare when the previous non-space char is one of ``([;,=``
    (or it opens the string, or follows ``\\{``)."""
    for _ in range(6):
        out, i, changed = [], 0, False
        while i < len(s):
            ch = s[i]
            if ch == "{" and (i == 0 or s[i - 1] != "\\"):
                prev = "".join(out).rstrip()
                bare = (not prev) or prev[-1] in "([;,=" or prev.endswith("\\{")
                if bare:
                    j = mv._close_brace(s, i)  # noqa: SLF001
                    if j is not None:
                        out.append(" " + s[i + 1:j] + " ")
                        i = j + 1
                        changed = True
                        continue
            out.append(ch)
            i += 1
        s = "".join(out)
        if not changed:
            break
    return s


def clean_key(s: str) -> str:
    """Undo X2-extraction / MathType artefacts in a *key string* (never applied to model predictions)."""
    s = re.sub(r"\{\\rm\{\s*\}\}", " ", s)
    s = re.sub(r"\{\\rm\{([^{}]*)\}\}", r"\\text{\1}", s)
    s = re.sub(r"\\(?:left|right)(?![A-Za-z])\s*", "", s)
    s = re.sub(r"\\displaystyle", "", s)
    s = _drop_bare_braces(s)
    return re.sub(r"\s+", " ", s).strip()


def strip_math_delims(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\\\(|\\\)|\\\[|\\\]", " ", s)
    s = s.replace("$", " ")
    s = clean_key(s)
    return re.sub(r"\s+", " ", s).strip().rstrip(".").strip()


def boxed(s: Optional[str]) -> Optional[str]:
    """Present a bare key string to the grader as a final answer."""
    if s is None or not str(s).strip():
        return None
    return "\\boxed{" + strip_math_delims(str(s)) + "}"


def last_box(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    bx = mv._boxes(text)  # noqa: SLF001 — the grader's own brace-balanced box finder
    return bx[-1][0] if bx else None


# ----------------------------------------------------------------------------------------------- SymPy → gold
def _latex(v: str) -> str:
    import sympy
    try:
        return sympy.latex(sympy.sympify(v))
    except Exception:
        return v


def sympy_gold(sk: Optional[dict]) -> Optional[dict]:
    """Typed gold from X2's structured SymPy key. ``simplify_equivalence`` keys hold the *unsimplified* expression:
    they are an equivalence reference (``role: equivalence``), never a canonical form."""
    if not sk or sk.get("values") is None:
        return None
    kt, vals = sk.get("key_type"), sk.get("values") or []
    g: dict[str, Any] = {"locale": "vi"}
    if kt == "solve_system":
        names = [str(v) for v in sk.get("variables") or []]
        sols = [[_latex(str(x)) for x in s] for s in vals]
        if not sols:
            return {**g, "answer_type": "set", "answer_canonical": "\\emptyset"}
        if len(sols) == 1:
            return {**g, "answer_type": "tuple", "answer_canonical": "; ".join(sols[0]),
                    **({"vars": names} if len(names) == len(sols[0]) else {})}
        return {**g, "answer_type": "set", "answer_canonical": "; ".join("(" + "; ".join(s) + ")" for s in sols)}
    lat = [_latex(str(v)) for v in vals]
    if kt == "solve_equation" or (kt == "solve_for_value" and len(lat) != 1):
        return {**g, "answer_type": "set", "answer_canonical": "; ".join(lat) if lat else "\\emptyset"}
    if kt == "simplify_equivalence":
        return {**g, "answer_type": "expr", "answer_canonical": lat[0], "role": "equivalence"} if lat else None
    if len(lat) != 1:
        return {**g, "answer_type": "set", "answer_canonical": "; ".join(lat)}
    out = mv.infer_gold(lat[0], locale="vi")
    out.pop("target_var", None)
    return out


# ----------------------------------------------------------------------------------------------- comparisons
SCALAR_TYPES = ("int", "rational", "decimal", "radical", "expr", "text")


def strict_verdict(res: dict, gold: dict) -> bool:
    """Key-vs-key agreement is stricter than model grading. The grader (built for model outputs) reads a prose span
    such as "Đội I: 48 ngày; Đội II: 24 ngày" or "max A = 16 khi x = y = -1" as its last value, so a partial page
    key ("24", "-1") would 'agree' with a complete key. For agreement we reject prose/hedged spans and, for a
    scalar gold, spans that hold several numeric items or an equality-case clause ("khi ...")."""
    if not res.get("correct"):
        return False
    if res.get("flag") in ("prose-span", "hedged", "degenerate"):
        return False
    span = res.get("span") or ""
    if gold.get("answer_type") in SCALAR_TYPES:
        if re.search(r"(?<![A-Za-zÀ-ỹ])khi(?![A-Za-zÀ-ỹ])|\bwhen\b", span):
            return False
        try:
            parts = [p for p in mv._split_top(span, (";",)) if re.search(r"\d", p)]  # noqa: SLF001
        except Exception:
            parts = []
        if len(parts) >= 2:
            return False
        # "Bình: 20 km/h, An: 25 km/h" / "9A = 42, 9B = 40": several labelled values for a one-value gold
        # (found by the Q1 audit: a partial page key "25" had been confirmed by such a prose key)
        if len(re.findall(r"(?:=|:)\s*\$?\s*[-+]?\s*(?:\\[dt]?frac|\\sqrt|\d)", span)) >= 2:
            return False
    return True


def grade_ok(pred: Optional[str], gold: Optional[dict]) -> Optional[bool]:
    """None if either side is missing; else the grader's verdict made strict for key agreement (see
    ``strict_verdict``; exceptions count as no-match)."""
    if pred is None or gold is None or gold.get("answer_canonical") in (None, ""):
        return None
    g = {k: v for k, v in gold.items() if k != "role"}
    try:
        res = mv.grade(pred, g)
    except Exception:
        return False
    if res["correct"] and res.get("matched") not in (None, "gold"):
        alts = g.get("accept_also") or []
        idx = int(re.search(r"\d+", res["matched"]).group()) if re.search(r"\d+", res["matched"] or "") else None
        return strict_verdict(res, alts[idx] if idx is not None and idx < len(alts) else g)
    return strict_verdict(res, g)


def type_gold(raw: Optional[str], problem: str) -> Optional[dict]:
    """Gold typed from a raw key string, kept only if it round-trips on its own source."""
    if raw is None or not strip_math_delims(raw):
        return None
    s = strip_math_delims(raw)
    m = re.fullmatch(r"\(\s*([A-Za-z])\s*[;,]\s*([A-Za-z])\s*\)\s*=\s*\((.+)\)", s)
    if m:                                        # "(x;y)=(2;1)": infer_gold would type it as a set
        vals = mv._split_top(m.group(3), (";", ","))  # noqa: SLF001
        if len(vals) == 2:
            s = f"{m.group(1)} = {vals[0].strip()}; {m.group(2)} = {vals[1].strip()}"
    labeled = _labeled_items(s)
    if labeled:                                  # "\text{Đội I} = 48; \text{Đội II} = 24" → positional tuple
        names, vals = zip(*labeled)
        try:
            canon = [mv.infer_gold(v, locale="vi")["answer_canonical"] for v in vals]
        except Exception:
            return None
        g = {"locale": "vi", "answer_type": "tuple", "answer_canonical": "; ".join(canon), "quantities": list(names)}
        return g if grade_ok("\\boxed{" + "; ".join(vals) + "}", g) else None
    try:
        g = mv.infer_gold(s, problem=problem, locale="vi")
    except Exception:
        return None
    if g.get("answer_type") in ("set", "tuple") and re.search(r"(^|;)\s*=", g.get("answer_canonical", "")):
        return None                              # a label was stripped and left "= v": not a usable gold
    return g if grade_ok(boxed(raw), g) else None


_UNIT = r"(?:\s*\\text\{[^{}]*\}|\s*(?:km/h|km|cm|m|ngày|giờ|phút|đồng|sản phẩm|học sinh|người|trận|chiếc|cây|quyển|kg|tấn|lít|%))*"


def _labeled_items(s: str) -> Optional[list]:
    """["name", "value"] pairs when every ';'-item is "<word label> = <value>" (word labels, not math vars)."""
    try:
        items = [x.strip() for x in mv._split_top(s, (";",)) if x.strip()]  # noqa: SLF001
    except Exception:
        return None
    if len(items) < 2:
        return None
    out = []
    for it in items:
        m = re.fullmatch(r"(\\text\{[^{}]*\}|[^=;]*?[A-Za-zÀ-ỹ][^=;]*?)\s*=\s*(.+?)" + _UNIT, it)
        if not m:
            return None
        name = re.sub(r"\\text\{([^{}]*)\}", r"\1", m.group(1)).strip()
        val = m.group(2).strip()
        if not re.search(r"\d", val) or re.fullmatch(r"[A-Za-z](_\{?\w+\}?)?", name):
            return None                          # plain math variables are handled by infer_gold (vars kept)
        out.append((name, val))
    return out


def raw_agree(a: Optional[str], b: Optional[str], problem: str) -> Optional[bool]:
    """Symmetric raw-key agreement: True if a gold typed from either key grades the other correct."""
    if a is None or b is None:
        return None
    ga, gb = type_gold(a, problem), type_gold(b, problem)
    if ga is None and gb is None:
        return None
    return bool((gb is not None and grade_ok(boxed(a), gb)) or (ga is not None and grade_ok(boxed(b), ga)))


def roundtrip(gold: Optional[dict]) -> Optional[bool]:
    if not gold or gold.get("answer_canonical") in (None, ""):
        return None
    return grade_ok("\\boxed{" + str(gold["answer_canonical"]) + "}", gold)


# ----------------------------------------------------------------------------------------------- prepare
def key_strings(it: dict, ck: Optional[dict]) -> dict[str, Optional[str]]:
    prov = it.get("provisional_answer") or {}
    page = prov.get("value_final") if prov.get("value_final") not in (None, "") else None
    return {"page": page, "claude1": (ck or {}).get("answer") or ((it.get("claude_blind_key") or {}).get("answer"))}


def consensus_gold(it: dict, ks: dict, sg: Optional[dict]) -> dict:
    """Try to confirm X2's consensus with the grader and derive a typed gold from the agreeing keys."""
    prob = it["problem"]
    page, c1 = boxed(ks["page"]), boxed(ks["claude1"])
    if sg is not None and sg.get("role") != "equivalence":
        m = {"page": grade_ok(page, sg), "claude1": grade_ok(c1, sg)}
        agreeing = ["sympy"] + [k for k, v in m.items() if v]
        if len(agreeing) >= 2:
            return {"gold": sg, "gold_from": "sympy", "agreeing": agreeing}
    if sg is not None and sg.get("role") == "equivalence":
        # simplified form from a Claude/page key, confirmed equivalent to the SymPy-parsed original expression
        for src in ("claude1", "page"):
            g = type_gold(ks[src], prob)
            if g and g.get("answer_type") == "expr" and grade_ok(boxed(ks[src]), sg):
                g.pop("target_var", None)
                agreeing = ["sympy", src]
                other = "page" if src == "claude1" else "claude1"
                if grade_ok(boxed(ks[other]), g):
                    agreeing.append(other)
                return {"gold": g, "gold_from": src, "agreeing": agreeing}
    for src, other in (("claude1", "page"), ("page", "claude1")):
        g = type_gold(ks[src], prob)
        if g is not None and grade_ok(boxed(ks[other]), g):
            agreeing = [src, other] + (["sympy"] if sg is not None and grade_ok(boxed(ks[src]), sg) else [])
            return {"gold": g, "gold_from": src, "agreeing": sorted(set(agreeing))}
    return {"gold": None, "gold_from": None, "agreeing": []}


def prepare() -> None:
    items = D.core_items()
    ids = {it["id"] for it in items}
    ck = D.claude_keys_by_id(ids)
    rows, why = [], Counter()
    for it in items:
        ks = key_strings(it, ck.get(it["id"]))
        sg = sympy_gold(it.get("sympy_key"))
        pair = {"sympy_vs_page": grade_ok(boxed(ks["page"]), sg) if sg and sg.get("role") != "equivalence" else None,
                "sympy_vs_claude1": grade_ok(boxed(ks["claude1"]), sg) if sg else None,
                "page_vs_claude1": raw_agree(ks["page"], ks["claude1"], it["problem"])}
        st = it["key_status"]
        cons = consensus_gold(it, ks, sg) if st in ("consensus_2", "consensus_3") else {"gold": None, "agreeing": []}
        if st in ("conflict", "unverified"):
            adj, reason = True, st
        elif cons["gold"] is None:
            adj, reason = True, "consensus_not_confirmed_by_grader"
        else:
            adj, reason = False, "grader_confirmed_consensus"
        why[reason] += 1
        rows.append({"id": it["id"], "x2_key_status": st, "keys": ks, "sympy_gold": sg,
                     "sympy_key_type": (it.get("sympy_key") or {}).get("key_type"),
                     "grader_pairs": pair, "consensus": cons, "needs_adjudication": adj, "route_reason": reason,
                     "gold_roundtrip": roundtrip(cons["gold"])})
    D.write_jsonl(D.KEYS_TABLE, rows)
    adj_ids = [r["id"] for r in rows if r["needs_adjudication"]]
    # queue order (conflicts first, as frozen by X2), then the grader-disputed consensus items
    qrank = {q["id"]: q["queue_rank"] for q in D.iter_jsonl(D.QUEUE)}
    adj_ids.sort(key=lambda i: (0 if qrank.get(i, 10 ** 6) <= 631 else 1, qrank.get(i, 10 ** 6)))
    all_ids = sorted(ids)
    audit = random.Random(D.AUDIT_SEED).sample(all_ids, D.AUDIT_N + 30)   # 30 reserves replace INVALID items
    sel = {"n_core": len(items), "adjudicate": adj_ids, "audit_sample": audit, "audit_n": D.AUDIT_N,
           "audit_seed": D.AUDIT_SEED, "route_counts": dict(why),
           "queue_freeze_sha256_lf": D.sha256_file(D.QUEUE, normalize_eol=True),
           "queue_freeze_recorded": D.load_json(D.QUEUE_FREEZE, {}).get("sha256")}
    D.OUT.mkdir(parents=True, exist_ok=True)
    D.SELECTION.write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in sel.items() if k not in ("adjudicate", "audit_sample")}, ensure_ascii=False))
    print("adjudicate:", len(adj_ids), "| pair agreement:",
          {k: Counter(r["grader_pairs"][k] for r in rows) for k in rows[0]["grader_pairs"]})
    print("consensus gold_from:", Counter(r["consensus"].get("gold_from") for r in rows),
          "roundtrip:", Counter(r["gold_roundtrip"] for r in rows))


TRIAGE = D.OUT / "triage.jsonl"


def triage() -> None:
    """After solve #2: order the adjudication set by need (budget triage, see the report §Budget).

    T1 no_agreement   — no two keys agree under the grader            → adjudicate first
    T2 two_claude     — claude1 ≈ claude2, no non-Claude key agrees     → adjudicate next (VERIFIED needs it)
    T3 keys_verified  — claude2 (or claude1) ≈ a non-Claude key        → adjudicate last (rule already met)
    """
    items = {it["id"]: it for it in D.core_items()}
    sel = D.load_json(D.SELECTION)
    ktab = {r["id"]: r for r in D.iter_jsonl(D.KEYS_TABLE)}
    s2 = D.stage_rows("solve2")
    rows = []
    for i in sel["adjudicate"]:
        k, prob = ktab[i], items[i]["problem"]
        sg = k.get("sympy_gold")
        sgv = sg if sg and sg.get("role") != "equivalence" else None
        c2row = s2.get(i) if (s2.get(i) or {}).get("ok") else None
        c2 = c2row.get("final_answer") if c2row else None
        c1, page = k["keys"]["claude1"], k["keys"]["page"]
        ag = {"c2_sympy": grade_ok(c2row["text"], sg) if (c2row and sg) else None,
              "c2_page": raw_agree(c2, page, prob), "c2_c1": raw_agree(c2, c1, prob),
              "c1_sympy": grade_ok(boxed(c1), sgv) if sgv else None, "c1_page": raw_agree(c1, page, prob),
              "page_sympy": grade_ok(boxed(page), sgv) if sgv else None}
        if ag["c2_sympy"] or ag["c2_page"] or ag["c1_sympy"] or ag["c1_page"]:
            tier = "T3_keys_verified"
        elif ag["c2_c1"]:
            tier = "T2_two_claude"
        else:
            tier = "T1_no_agreement"
        rows.append({"id": i, "tier": tier, "agree": ag, "x2_key_status": k["x2_key_status"],
                     "solve2_ok": c2row is not None})
    D.write_jsonl(TRIAGE, rows)
    order = [r["id"] for t in ("T1_no_agreement", "T2_two_claude", "T3_keys_verified") for r in rows if r["tier"] == t]
    sel["adjudicate_order"] = order
    D.SELECTION.write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    print(Counter(r["tier"] for r in rows), Counter((r["x2_key_status"], r["tier"]) for r in rows),
          "solve2 missing:", sum(1 for r in rows if not r["solve2_ok"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "triage"])
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "triage":
        triage()


if __name__ == "__main__":
    main()
