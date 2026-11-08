"""Q1 final statuses, audit, contamination flags, frozen ``core_v1.jsonl`` + datacard + SHA256 + stats.

  .venv/Scripts/python.exe -m vi_math_verified_aug.vi_exam_qa.finalize audit-prep   # audit mismatches → judge
  .venv/Scripts/python.exe -m vi_math_verified_aug.vi_exam_qa.finalize freeze       # core_v1 + stats

Status rule (Q1 brief; human raters replaced by LLM judges):
  VERIFIED   — adjudicator confidence high/medium AND (≥2 keys agree with the final gold under the grader, at least
               one of them non-Claude (SymPy or page)), OR (both Claude solves agree with the gold AND the adjudicator
               is highly confident). Grader-confirmed X2 consensus items (≥2 keys, ≥1 non-Claude, re-checked with
               grader 3.0.2) are VERIFIED without adjudication.
  ADJUDICATED_LOW_CONFIDENCE — valid per the adjudicator but the rule above is not met (or confidence low, or the
               audit judge revised the key).
  INVALID    — the adjudicator (or the audit judge: item_ambiguous) found the item not well-posed / not gradable;
               dropped from Core (kept in ``invalid_v1.jsonl``).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from typing import Any, Optional

from ..eval import math_verify as mv
from . import data as D
from .keys import boxed, grade_ok, raw_agree, roundtrip, strip_math_delims

STATUS_V, STATUS_L, STATUS_I = "VERIFIED", "ADJUDICATED_LOW_CONFIDENCE", "INVALID"
GOLD_KEYS = ("answer_type", "answer_canonical", "vars", "target_var", "round_dp", "percent")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def parse_json(text: Optional[str]) -> Any:
    """Last fenced JSON block, else the last balanced {...} object (mirrors common.llm.parse_json, last-wins)."""
    if not text:
        return None
    for m in reversed(list(_FENCE.finditer(text))):
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()
    found = None
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                found = dec.raw_decode(text, i)[0]
            except json.JSONDecodeError:
                continue
    return found


_VAR = r"[A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?"


def _in_problem(v: str, problem: Optional[str]) -> bool:
    """A variable name the problem itself uses (x, y, m, x_1 ...), so a model is likely to name it the same way."""
    if problem is None:
        return True
    base = re.escape(str(v).replace("{", "").replace("}", ""))
    plain = re.sub(r"\\text\s*\{[^{}]*\}", " ", problem).replace("{", "").replace("}", "")
    plain = re.sub(r"\\[A-Za-z]+", " ", plain)                       # drop LaTeX command names (\frac, \sqrt)
    return re.search(r"(?<![A-Za-zÀ-ỹ])" + base + r"(?![A-Za-zÀ-ỹ])", plain) is not None


def make_gold(d: Optional[dict], problem: Optional[str] = None) -> Optional[dict]:
    """Typed grader gold from an LLM decision; None if it does not validate under ``math_verify.as_gold``."""
    if not isinstance(d, dict) or d.get("answer_canonical") in (None, ""):
        return None
    g: dict[str, Any] = {"locale": "vi"}
    for k in GOLD_KEYS:
        v = d.get(k)
        if v not in (None, [], ""):
            g[k] = v
    if g.get("percent") is False:
        g.pop("percent")
    g["answer_canonical"] = str(g["answer_canonical"])
    # word-problem quantity names ("hoa hồng", "gioi") make the grader match by name and fail "x = 15; y = 10":
    # keep vars only when they are math variables (x, y, m, x_1, ...); otherwise the tuple is positional.
    if g.get("vars") and not all(re.fullmatch(_VAR, str(v)) and _in_problem(v, problem) for v in g["vars"]):
        g["quantities"] = g.pop("vars")
    if g.get("target_var") and not (re.fullmatch(_VAR, str(g["target_var"])) and _in_problem(g["target_var"], problem)):
        g.pop("target_var")
    if g.get("target_var") and g["answer_type"] in ("set", "tuple", "expr", "interval"):
        g.pop("target_var")
    try:
        mv.as_gold(g)
    except Exception:
        return None
    alts = []
    for a in d.get("accept_also") or []:
        ga = make_gold(a, problem) if isinstance(a, dict) else None
        if ga:
            ga.pop("accept_also", None)
            alts.append(ga)
    if alts:
        g["accept_also"] = alts
    return g


def gmatch(pred: Optional[str], gold: Optional[dict], *, full_text: bool = False) -> Optional[bool]:
    if pred is None or gold is None:
        return None
    return grade_ok(pred if full_text else boxed(pred), gold)


def sympy_match(sg: Optional[dict], gold: Optional[dict]) -> Optional[bool]:
    if not sg or not gold:
        return None
    if sg.get("role") == "equivalence":       # gold must be equivalent to the parsed original expression
        if gold.get("answer_type") != "expr":
            return False
        return grade_ok("\\boxed{" + gold["answer_canonical"] + "}", sg)
    return gmatch(sg["answer_canonical"], gold) or grade_ok("\\boxed{" + gold["answer_canonical"] + "}", sg)


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper–Pearson upper bound for a binomial proportion (bisection on the binomial CDF)."""
    from math import comb
    if k >= n:
        return 1.0

    def cdf(p: float) -> float:
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if cdf(mid) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return hi


# ----------------------------------------------------------------------------------------------- decisions
def unadjudicated_gold(it: dict, k: dict, sg: Optional[dict], c2: Optional[dict]) -> Optional[dict]:
    """Gold for an adjudication-set item the budget did not reach: the first typed key (SymPy, then Claude #2's
    boxed answer, then Claude #1, then the page key) that another key confirms under the grader; else Claude #2's
    typed answer (unconfirmed). ``_from`` names the source."""
    from .keys import type_gold
    prob = it["problem"]
    c1, page = k["keys"]["claude1"], k["keys"]["page"]
    c2box = c2.get("final_answer") if c2 else None
    others = {"claude2": (c2["text"], True) if c2 else (None, False), "claude1": (c1, False), "page": (page, False)}

    def confirmed(g, skip):
        return any(gmatch(t, g, full_text=ft) for name, (t, ft) in others.items() if name != skip and t)
    if sg and sg.get("role") != "equivalence" and confirmed(sg, None):
        return {**sg, "_from": "keys:sympy"}
    for name, raw in (("claude2", c2box), ("claude1", c1), ("page", page)):
        g = type_gold(raw, prob)
        if g is not None and confirmed(g, name):
            g.pop("target_var", None)
            return {**g, "_from": f"keys:{name}"}
    g = type_gold(c2box, prob) or type_gold(c1, prob)
    if g is not None:
        g.pop("target_var", None)
        return {**g, "_from": "keys:unconfirmed_claude"}
    return None


def decide_all() -> tuple[list[dict], dict]:
    items = D.core_items()
    ktab = {r["id"]: r for r in D.iter_jsonl(D.KEYS_TABLE)}
    s2 = D.stage_rows("solve2")
    adj = D.stage_rows("adjudicate")
    tiers = {r["id"]: r["tier"] for r in D.iter_jsonl(D.OUT / "triage.jsonl")}
    dec: list[dict] = []
    misc = Counter()
    for it in items:
        i = it["id"]
        k = ktab[i]
        sg = k.get("sympy_gold")
        c2 = s2.get(i) if (s2.get(i) or {}).get("ok") else None
        d: dict[str, Any] = {"id": i, "x2_key_status": k["x2_key_status"], "route_reason": k["route_reason"],
                             "adjudicated": False, "adjudicator": None, "confidence": None, "invalid_reason": None,
                             "tier": tiers.get(i)}
        if not k["needs_adjudication"]:
            gold = dict(k["consensus"]["gold"])
            gold.pop("target_var", None) if gold.get("answer_type") in ("set", "tuple", "expr") else None
            d.update(gold=gold, gold_from=f"keys:{k['consensus']['gold_from']}", confidence="consensus")
        else:
            a = adj.get(i) if (adj.get(i) or {}).get("ok") else None
            j = parse_json(a["text"]) if a else None
            if not isinstance(j, dict):
                # not adjudicated (budget): gold from the keys that agree after solve #2 (triage tier)
                misc["not_adjudicated"] += 1
                d["tier"] = tiers.get(i)
                d.update(gold=unadjudicated_gold(it, k, sg, c2), confidence=None)
                d["gold_from"] = d["gold"].pop("_from") if d["gold"] else None
            else:
                d["adjudicated"] = True
        if d["adjudicated"]:
            d["adjudicator"] = {kk: j.get(kk) for kk in ("valid", "invalid_reason", "well_posed", "figure_needed",
                                                          "multiple_answers", "confidence", "key_verdicts",
                                                          "certifiable", "justification")}
            d["confidence"] = j.get("confidence")
            gold = make_gold(j, it["problem"])
            if j.get("answer_canonical") not in (None, "") and gold is None:
                misc["adjudicated_gold_failed_schema"] += 1
            d.update(gold=gold, gold_from="adjudicator")
            if j.get("valid") is False:
                d["invalid_reason"] = j.get("invalid_reason") or "other"
        gold = d.get("gold")
        km = {"sympy": sympy_match(sg, gold), "page": gmatch(k["keys"]["page"], gold),
              "claude1": gmatch(k["keys"]["claude1"], gold),
              "claude2": gmatch(c2["text"], gold, full_text=True) if c2 else None}
        d["key_matches"] = km
        d["agreeing_keys"] = [s for s, v in km.items() if v]
        d["gold_roundtrip"] = roundtrip(gold)
        nonc = sum(1 for s in ("sympy", "page") if km[s])
        if not d["adjudicated"] and not k["needs_adjudication"]:
            d["status"], d["status_route"] = STATUS_V, "x2_consensus_grader_confirmed"
        elif not d["adjudicated"]:
            if gold is None:
                d["status"], d["status_route"] = STATUS_L, "not_adjudicated_no_usable_key"
            elif nonc >= 1 and len(d["agreeing_keys"]) >= 2:
                d["status"], d["status_route"] = STATUS_V, "keys_agree_after_solve2_not_adjudicated"
            else:
                d["status"], d["status_route"] = STATUS_L, ("two_claude_agree_not_adjudicated" if km["claude1"]
                                                            and km["claude2"] else "not_adjudicated")
        elif d["invalid_reason"]:
            d["status"], d["status_route"] = STATUS_I, f"adjudicator_invalid:{d['invalid_reason']}"
        elif gold is None:
            d["status"], d["status_route"] = STATUS_L, "adjudicator_gold_unusable"
        elif d["confidence"] == "low":
            d["status"], d["status_route"] = STATUS_L, "adjudicator_low_confidence"
        elif nonc >= 1 and len(d["agreeing_keys"]) >= 2:
            d["status"], d["status_route"] = STATUS_V, "independent_keys_incl_non_claude"
        elif km["claude1"] and km["claude2"] and d["confidence"] == "high":
            d["status"], d["status_route"] = STATUS_V, "two_claude_solves_plus_adjudicator"
        else:
            d["status"], d["status_route"] = STATUS_L, "insufficient_agreeing_keys"
        dec.append(d)
    return dec, dict(misc)


# ----------------------------------------------------------------------------------------------- audit
def audit_sample(dec: list[dict]) -> list[str]:
    sel = D.load_json(D.SELECTION)
    st = {d["id"]: d["status"] for d in dec}
    return [i for i in sel["audit_sample"] if st.get(i) != STATUS_I][: sel["audit_n"]]


AUDITS = {   # audit #1 (seed 20261107, keys of core_v1) and audit #2 (seed 20261108, keys of core_v1.1)
    "audit": {"solve": "audit", "judge": "audit_judge", "sample": "audit_sample.frozen.json",
              "mism": "audit_mismatches", "results": "audit_results"},
    "audit2": {"solve": "audit2", "judge": "audit2_judge", "sample": "audit2_sample.frozen.json",
               "mism": "audit2_mismatches", "results": "audit2_results"},
}


def audit_compare(dec: list[dict], ids: Optional[list] = None, solve_stage: str = "audit") -> list[dict]:
    by = {d["id"]: d for d in dec}
    aud = D.stage_rows(solve_stage)
    out = []
    for i in (ids if ids is not None else audit_sample(dec)):
        a = aud.get(i)
        g = by[i].get("gold")
        ok = gmatch(a["text"], g, full_text=True) if (a and a.get("ok") and g) else None
        out.append({"id": i, "status": by[i]["status"], "route": by[i]["status_route"],
                    "x2_key_status": by[i]["x2_key_status"], "audit_ok": bool(a and a.get("ok")), "match": ok,
                    "audit_answer": (a or {}).get("final_answer"), "gold": g})
    return out


def audit_prep() -> None:
    dec, _ = decide_all()
    cmp = audit_compare(dec)
    mism = [c for c in cmp if c["match"] is False or (c["audit_ok"] and c["gold"] is None)]
    D.write_jsonl(D.OUT / "audit_mismatches.jsonl", mism)
    print(json.dumps({"audit_n": len(cmp), "audit_done": sum(c["audit_ok"] for c in cmp),
                      "match": sum(1 for c in cmp if c["match"]), "mismatch": len(mism),
                      "pending": sum(1 for c in cmp if not c["audit_ok"])}))


def apply_audit(dec: list[dict], name: str = "audit") -> dict:
    """Audit statistics from the FROZEN audit record (sample, grader mismatches and judge verdicts as measured
    against the keys of that moment: ``audit_sample.frozen.json`` + ``audit_mismatches.frozen.jsonl``), then the
    corrections the judge asked for. Later pipeline fixes therefore cannot hide an error the audit found."""
    by = {d["id"]: d for d in dec}
    A = AUDITS[name]
    fr = D.load_json(D.OUT / A["sample"])
    if fr is None or "grader_match_at_audit" not in fr:   # before the audit was frozen
        return {"n": 0, "note": f"{name} not frozen yet"}
    mism_rows = {r["id"]: r for r in D.iter_jsonl(D.OUT / f"{A['mism']}.frozen.jsonl")}
    judg = {i: (parse_json(r["text"]) or {}) for i, r in D.stage_rows(A["judge"]).items() if r.get("ok")}
    now = {c["id"]: c for c in audit_compare(dec, fr["ids"], A["solve"])}   # post-fix agreement
    solves = D.stage_rows(A["solve"])
    verdicts, rows = Counter(), []
    for i in fr["ids"]:
        d = by[i]
        j = judg.get(i) or {}
        v = (j.get("verdict") or "pending") if i in mism_rows else "match"
        verdicts[v] += 1
        d.setdefault("audits", {})[name] = {"grader_match_at_audit": fr["grader_match_at_audit"].get(i),
                                            "verdict": v}
        d["audit"] = {"in_sample": True, "name": name, "grader_match_at_audit": fr["grader_match_at_audit"].get(i),
                      "verdict": v}
        if v == "key_wrong":
            g = make_gold(j.get("corrected_gold"), None)
            if g and g.get("answer_type") == "tuple":
                g["answer_canonical"] = re.sub(r"^\((.*)\)$", lambda m: m.group(1), g["answer_canonical"].strip())
                if g.get("vars"):
                    g["quantities"] = g.pop("vars")
            d["audit"]["gold_at_audit"] = (mism_rows[i] or {}).get("gold")
            if g and g.get("answer_type") == "text" and re.search(r"(?:^|\s)[a-d]\)", g["answer_canonical"]):
                # the item text holds several asked sub-parts and the judge's "gold" is a prose list of them:
                # not gradable as one typed answer -> drop (segmentation defect, back to X2)
                d["status"], d["status_route"], d["invalid_reason"] = STATUS_I, "audit_multi_part_item", "not_gradable"
                continue_flag = True
            else:
                continue_flag = False
            if g and not continue_flag:
                d["gold"] = g
                d["gold_roundtrip"] = roundtrip(g)
            if not continue_flag:
                d["status"], d["status_route"] = STATUS_L, "audit_revised_key"
        elif v == "item_ambiguous":
            d["status"], d["status_route"], d["invalid_reason"] = STATUS_I, "audit_item_ambiguous", "ill_posed"
        elif v == "both_acceptable":
            g = make_gold(j.get("accept_also"), None)
            if g and d.get("gold"):
                g.pop("accept_also", None)
                d["gold"] = {**d["gold"], "accept_also": (d["gold"].get("accept_also") or []) + [g]}
        rows.append({"id": i, "verdict": v, "status_at_audit": fr["status_at_audit"].get(i),
                     "route_at_audit": fr["route_at_audit"].get(i), "x2_key_status": fr["x2_at_audit"].get(i),
                     "grader_match_at_audit": fr["grader_match_at_audit"].get(i),
                     "grader_match_now": (now.get(i) or {}).get("match"), "judge": j,
                     "audit_effort": (solves.get(i) or {}).get("effort")})
    n = len(fr["ids"])
    mism = len(mism_rows)
    errs = verdicts["key_wrong"] + verdicts["item_ambiguous"]
    by_status = defaultdict(Counter)
    for r in rows:
        by_status[r["status_at_audit"]][r["verdict"]] += 1
        by_status["route:" + r["route_at_audit"]][r["verdict"]] += 1
    D.write_jsonl(D.OUT / f"{A['results']}.jsonl", rows)
    return {"name": name, "seed": fr.get("seed", D.AUDIT_SEED), "n": n, "verdicts": dict(verdicts), "grader_mismatches": mism, "label_errors": errs,
            "error_rate": errs / n if n else None,
            "upper95_label_errors": cp_upper(errs, n) if n else None,
            "upper95_all_mismatches_conservative": cp_upper(mism, n) if n else None,
            "rule_of_three_if_zero": 3 / n if n else None,
            "grader_mismatches_now": sum(1 for r in rows if r["grader_match_now"] is False),
            "stratum_bounds": {st: {"n": sum(c.values()), "errors": c["key_wrong"] + c["item_ambiguous"],
                                    "upper95": cp_upper(c["key_wrong"] + c["item_ambiguous"], sum(c.values()))}
                               for st, c in by_status.items() if not st.startswith("route:")},
            "by_status": {k: dict(v) for k, v in by_status.items()},
            "by_audit_effort": {e: dict(Counter(r["verdict"] for r in rows if r["audit_effort"] == e))
                                for e in sorted({r["audit_effort"] for r in rows if r["audit_effort"]})}}


def current_decisions() -> tuple[list[dict], dict, dict]:
    """decide_all + the corrections of audit #1 (the state core_v1.1 starts from)."""
    dec, misc = decide_all()
    a1 = apply_audit(dec, "audit")
    return dec, misc, a1


def audit2_sample() -> None:
    """Fresh audit #2: 150 items uniformly at random from all Core (new seed), INVALID items skipped."""
    import random
    dec, _, _ = current_decisions()
    st = {d["id"]: d["status"] for d in dec}
    drawn = random.Random(D.AUDIT2_SEED).sample(sorted(st), D.AUDIT_N + 30)
    ids = [i for i in drawn if st[i] != STATUS_I][: D.AUDIT_N]
    rec = {"seed": D.AUDIT2_SEED, "drawn": drawn, "ids": ids,
           "frozen_sample_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (D.OUT / AUDITS["audit2"]["sample"]).write_text(json.dumps(rec, ensure_ascii=False, indent=0), encoding="utf-8")
    a1 = D.load_json(D.OUT / AUDITS["audit"]["sample"])["ids"]
    print(json.dumps({"n": len(ids), "seed": D.AUDIT2_SEED, "overlap_with_audit1": len(set(ids) & set(a1))}))


def audit2_prep() -> None:
    """Compare audit #2 solves with the core_v1.1 keys; freeze the mismatch record for the judge."""
    dec, _, _ = current_decisions()
    path = D.OUT / AUDITS["audit2"]["sample"]
    rec = D.load_json(path)
    cmp = audit_compare(dec, rec["ids"], "audit2")
    mism = [c for c in cmp if c["match"] is False or (c["audit_ok"] and c["gold"] is None)]
    D.write_jsonl(D.OUT / "audit2_mismatches.jsonl", mism)
    D.write_jsonl(D.OUT / "audit2_mismatches.frozen.jsonl", mism)
    rec.update(grader_match_at_audit={c["id"]: c["match"] for c in cmp},
               route_at_audit={c["id"]: c["route"] for c in cmp},
               status_at_audit={c["id"]: c["status"] for c in cmp},
               x2_at_audit={c["id"]: c["x2_key_status"] for c in cmp},
               frozen_mismatch_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps({"audit_n": len(cmp), "audit_done": sum(c["audit_ok"] for c in cmp),
                      "match": sum(1 for c in cmp if c["match"]), "mismatch": len(mism),
                      "pending": sum(1 for c in cmp if not c["audit_ok"])}))


# ----------------------------------------------------------------------------------------------- contamination
def contamination() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in D.iter_jsonl(D.FLAGS):
        c = out.setdefault(r["item_id"], {"flag": False, "g4_any": False, "signals": [], "max_similarity": 0.0})
        c["g4_any"] = True
        c["flag"] = c["flag"] or bool(r.get("review_rule"))
        sig = f"{r['corpus']}:{r['signal']}"
        if sig not in c["signals"]:
            c["signals"].append(sig)
        c["max_similarity"] = max(c["max_similarity"], float(r.get("similarity") or 0.0))
    return out


# ----------------------------------------------------------------------------------------------- freeze
def freeze(version: str = "1.1", force: bool = False) -> None:
    """Write core_v<version>.jsonl (+ invalid, decisions, stats, datacard, sha256). v1 is never overwritten unless
    ``force``: it stays frozen for traceability. v1.1 = v1 + adjudication of the two-Claude-only items + audit #2."""
    vt = f"v{version}"
    core_path, inv_path = D.OUT / f"core_{vt}.jsonl", D.OUT / f"invalid_{vt}.jsonl"
    if version == "1" and (D.OUT / "core_v1.sha256").exists() and not force:
        raise SystemExit("core_v1 is frozen (core_v1.sha256 exists); freeze a new version instead")
    items = {it["id"]: it for it in D.core_items()}
    dec, misc, audit1 = current_decisions()
    audit = apply_audit(dec, "audit2") if version != "1" else audit1
    if audit.get("n", 0) == 0:
        audit = audit1
    cont = contamination()
    core_rows, invalid_rows = [], []
    for d in dec:
        it = items[d["id"]]
        pv = it.get("provenance") or {}
        c = cont.get(d["id"], {"flag": False, "g4_any": False, "signals": [], "max_similarity": 0.0})
        row = {
            "id": d["id"], "schema": f"vi-exammath-core/{version} (Q1, 2026-11-07)", "canary": it.get("canary"),
            "exam_id": it["exam_id"], "parent_id": it.get("parent_id"), "question_label": it.get("question_label"),
            "sub_label": it.get("sub_label"), "problem": it["problem"],
            "gold": d.get("gold"), "status": d["status"],
            "qa": {"status_route": d["status_route"], "confidence": d.get("confidence"),
                   "invalid_reason": d.get("invalid_reason"), "adjudicated": d["adjudicated"],
                   "agreeing_keys": d.get("agreeing_keys", []), "key_matches": d.get("key_matches", {}),
                   "x2_key_status": d["x2_key_status"], "gold_from": d.get("gold_from"),
                   "gold_roundtrip": d.get("gold_roundtrip"), "audit": d.get("audits"),
                   "grader_version": mv.GRADER_VERSION, "grader_backend": mv.BACKEND,
                   "adjudicator_certifiable": (d.get("adjudicator") or {}).get("certifiable")},
            "exam_year": pv.get("exam_year"), "first_seen": pv.get("first_seen"),
            "temporal": it.get("temporal"),
            "certifiable": it.get("symbolically_certifiable"),
            "topic": it.get("topic"), "topics": it.get("topics"), "answer_type_tag": it.get("answer_type"),
            "contamination": {"flag": c["flag"], "rule": "X5 §5.3 review rule (data/contam/vi_exam_flags.jsonl)",
                              "g4_any": c["g4_any"], "signals": c["signals"],
                              "max_similarity": round(c["max_similarity"], 4)},
            "provenance": {k: pv.get(k) for k in ("site", "source_url", "page_title", "province", "school",
                                                  "exam_year", "exam_kind", "issuer_type", "official_mark",
                                                  "fetched_at", "first_seen", "first_seen_evidence")},
            "license_basis": it.get("license_basis"), "release_eligible": it.get("release_eligible"),
        }
        (invalid_rows if d["status"] == STATUS_I else core_rows).append(row)
    D.write_jsonl(core_path, core_rows)
    D.write_jsonl(inv_path, invalid_rows)
    D.write_jsonl(D.OUT / f"decisions_{vt}.jsonl", dec)
    sha = D.sha256_file(core_path)
    with (D.OUT / f"core_{vt}.sha256").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{sha}  {core_path.name}\n{D.sha256_file(inv_path)}  {inv_path.name}\n")
    stats = compute_stats(dec, core_rows, invalid_rows, audit, misc, sha)
    stats["version"] = version
    stats["audit_v1"] = audit1
    stats["core_v1_sha256"] = "48bfd1c9287d88d5b7f22556e0914b033048410e7d8232aa420e3e1688c534dc"
    (D.OUT / f"q1_stats_{vt}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    write_datacard(stats, sha, version)
    print(json.dumps({k: stats[k] for k in ("status_counts", "audit", "post_cutoff", "cost")}, ensure_ascii=False,
                     indent=1))


def compute_stats(dec, core_rows, invalid_rows, audit, misc, sha) -> dict:
    ktab = {r["id"]: r for r in D.iter_jsonl(D.KEYS_TABLE)}
    s2 = D.stage_rows("solve2")
    aud = D.stage_rows("audit")
    st = Counter(d["status"] for d in dec)
    st_by_x2 = defaultdict(Counter)
    for d in dec:
        st_by_x2[d["x2_key_status"]][d["status"]] += 1
    routes = Counter(d["status_route"] for d in dec)
    conf = Counter(d.get("confidence") for d in dec if d["adjudicated"])
    inv = Counter(d.get("invalid_reason") for d in dec if d["status"] == STATUS_I)
    # key vs final gold (all Core with a gold, and the adjudicated subset)
    kvg = {}
    for scope, pred in (("all", lambda d: True), ("adjudicated", lambda d: d["adjudicated"])):
        m = {}
        for s in ("sympy", "page", "claude1", "claude2"):
            vals = [d["key_matches"].get(s) for d in dec if pred(d) and d.get("gold") and d["status"] != STATUS_I]
            m[s] = {"agree": sum(1 for v in vals if v), "disagree": sum(1 for v in vals if v is False),
                    "absent": sum(1 for v in vals if v is None)}
        kvg[scope] = m
    # pairwise key agreement over ALL Core items where both keys exist (strict grader agreement, symmetric typing;
    # SymPy is a typed gold so it is compared one-way). Plus every key vs the final gold.
    srcs = ("sympy", "page", "claude1", "claude2", "audit")
    items = {it["id"]: it for it in D.core_items()}
    pair = {}
    vals_by_id = {}
    for d in dec:
        i = d["id"]
        k = ktab[i]
        sg = k.get("sympy_gold")
        vals_by_id[i] = {"sympy": sg if sg and sg.get("role") != "equivalence" else None,
                         "page": k["keys"]["page"], "claude1": k["keys"]["claude1"],
                         "claude2": (s2.get(i) or {}).get("final_answer") if (s2.get(i) or {}).get("ok") else None,
                         "audit": (aud.get(i) or {}).get("final_answer") if (aud.get(i) or {}).get("ok") else None}
    for idx, a in enumerate(srcs):
        for b in srcs[idx + 1:]:
            cnt = Counter()
            for i, v in vals_by_id.items():
                if v[a] is None or v[b] is None:
                    continue
                if a == "sympy":
                    r = grade_ok(boxed(v[b]), v[a])
                else:
                    r = raw_agree(v[a], v[b], items[i]["problem"])
                cnt[{True: "agree", False: "disagree", None: "untyped"}[r]] += 1
            n = cnt["agree"] + cnt["disagree"]
            pair[f"{a}~{b}"] = {**cnt, "rate": round(cnt["agree"] / n, 4) if n else None}
    # adjudicator key verdicts vs grader verdicts (how often the grader's strict typing disagrees with the judge)
    vg = Counter()
    for d in dec:
        if not d["adjudicated"] or not d.get("adjudicator") or d["status"] == STATUS_I:
            continue
        kv = (d["adjudicator"] or {}).get("key_verdicts") or {}
        for s in srcs:
            g = d["key_matches"].get(s)
            j = kv.get(s)
            if g is None or j in (None, "absent"):
                continue
            vg[f"grader_{'agree' if g else 'disagree'}|judge_{j}"] += 1
    cert_agree = Counter()
    for d in dec:
        ac = (d.get("adjudicator") or {}).get("certifiable")
        if d["adjudicated"] and ac:
            it_c = next((r["certifiable"] for r in core_rows + invalid_rows if r["id"] == d["id"]), None)
            cert_agree["same" if ac == it_c else f"x2={it_c}->adj={ac}"] += 1

    def subset(rows, key):
        return [r for r in rows if (r.get("temporal") or {}).get(key)]
    post = {}
    for name, key in (("exam_2026", "exam_2026"), ("exam_2025_plus", "exam_2025_plus"),
                      ("first_seen_2026_06_plus", "first_seen_2026_06_plus")):
        rows = subset(core_rows, key)
        clean = [r for r in rows if not r["contamination"]["flag"]]
        post[name] = {"core_after_qa": len(rows),
                      "after_qa_and_contamination": len(clean),
                      "verified_and_clean": sum(1 for r in clean if r["status"] == STATUS_V),
                      "g4_clean_sensitivity": sum(1 for r in rows if not r["contamination"]["g4_any"])}
    x2_before = Counter()
    for it in D.core_items():
        for key in ("exam_2026", "exam_2025_plus", "first_seen_2026_06_plus"):
            if (it.get("temporal") or {}).get(key):
                x2_before[key] += 1
    for name in post:
        post[name]["x2_core_before_qa"] = x2_before[name]
    cert = {}
    for scope, rows in (("all_core", core_rows), ("clean", [r for r in core_rows if not r["contamination"]["flag"]]),
                        ("verified_clean", [r for r in core_rows if not r["contamination"]["flag"]
                                            and r["status"] == STATUS_V])):
        c3 = Counter(r["certifiable"] for r in rows)
        cert[scope] = {"yes": c3.get("yes", 0), "maybe": c3.get("maybe", 0), "no": c3.get("no", 0),
                       "certifiable(yes)": c3.get("yes", 0), "non_certifiable(maybe+no)": c3.get("maybe", 0) + c3.get("no", 0)}
    for name, key in (("exam_2025_plus_clean", "exam_2025_plus"), ("exam_2026_clean", "exam_2026")):
        rows = [r for r in subset(core_rows, key) if not r["contamination"]["flag"]]
        c3 = Counter(r["certifiable"] for r in rows)
        cert[name] = {"yes": c3.get("yes", 0), "maybe": c3.get("maybe", 0), "no": c3.get("no", 0)}
    status_by_cert = defaultdict(Counter)
    for r in core_rows + invalid_rows:
        status_by_cert[r["certifiable"]][r["status"]] += 1
    cost = {}
    for stage in D.STAGE_OUT:
        rows = list(D.iter_jsonl(D.STAGE_OUT[stage]))
        cost[stage] = {"rows": len(rows), "ok": sum(1 for r in rows if r.get("ok")),
                       "usd": round(sum(r.get("cost_usd") or 0 for r in rows), 4),
                       "input_tokens": sum((r.get("usage") or {}).get("input_tokens", 0) for r in rows),
                       "output_tokens": sum((r.get("usage") or {}).get("output_tokens", 0) for r in rows),
                       "truncated": sum(1 for r in rows if r.get("finish_reason") == "max_tokens"),
                       "batches": sorted({r.get("batch_id") for r in rows if r.get("batch_id")})}
    cost["total_usd"] = round(sum(v["usd"] for v in cost.values() if isinstance(v, dict)), 4)
    return {"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sha256_core_v1": sha,
            "n_core_x2": len(dec), "n_core_v1": len(core_rows), "n_invalid": len(invalid_rows),
            "status_counts": dict(st), "status_by_x2_key_status": {k: dict(v) for k, v in st_by_x2.items()},
            "status_routes": dict(routes),
            "status_by_tier": {t: dict(Counter(d["status"] for d in dec if d.get("tier") == t))
                               for t in ("T1_no_agreement", "T2_two_claude", "T3_keys_verified")},
            "adjudicated_by_tier": dict(Counter(d.get("tier") for d in dec if d["adjudicated"])), "adjudicator_confidence": dict(conf), "invalid_reasons": dict(inv),
            "misc": misc, "key_vs_final_gold": kvg, "pairwise_key_agreement_all_core": pair,
            "grader_vs_judge_key_verdicts": dict(vg), "certifiable_x2_vs_adjudicator": dict(cert_agree),
            "audit": audit, "post_cutoff": post, "certifiable_split": cert,
            "status_by_certifiable": {k: dict(v) for k, v in status_by_cert.items()},
            "contaminated_in_core_v1": sum(1 for r in core_rows if r["contamination"]["flag"]),
            "gold_roundtrip_fail": sum(1 for r in core_rows if r["qa"]["gold_roundtrip"] is False),
            "cost": cost, "grader": {"version": mv.GRADER_VERSION, "backend": mv.BACKEND}}


def write_datacard(s: dict, sha: str, version: str = "1") -> None:
    sc = s["status_counts"]
    pc = s["post_cutoff"]
    txt = f"""# Vi-ExamMath Core v1 — datacard (Q1, frozen {s['frozen_utc']})

**File** `core_v1.jsonl` ({s['n_core_v1']} items) · SHA256 `{sha}` · invalid/dropped items: `invalid_v1.jsonl`
({s['n_invalid']}). Canary: every row carries X2's canary GUID; do not train on this file.

## What it is
Native Vietnamese grade-10 entrance-exam (Toán vào 10) sub-questions with an auto-gradable final answer, collected
by X2 (`docs/research_2026-11/exec/X2_benchmark.md`) from loigiaihay.com (state-issued exam texts, 2017–2026) and
answer-key-checked by Q1 (`docs/research_2026-11/exec/Q1_benchmark_qa.md`). Human raters were replaced by LLM judges
(Claude Opus 5.5) by the user's decision; read every status with that in mind.

## Status (field `status`)
| status | n | meaning |
|---|---|---|
| VERIFIED | {sc.get('VERIFIED', 0)} | ≥2 keys agree with the gold under grader {s['grader']['version']}, ≥1 non-Claude (SymPy program or page solution); or two independent Claude solves + a highly confident adjudicator |
| ADJUDICATED_LOW_CONFIDENCE | {sc.get('ADJUDICATED_LOW_CONFIDENCE', 0)} | gold decided by the Opus adjudicator without enough independent agreement (or low confidence, or revised by the audit) |
| INVALID (not in this file) | {sc.get('INVALID', 0)} | not well-posed / needs a figure / not gradable — dropped |

Audit: {s['audit']['n']} random items, {s['audit']['label_errors']} label errors → 95 % upper bound on the label-error
rate {100 * (s['audit']['upper95_label_errors'] or 0):.1f} % (counting every grader mismatch as an error:
{100 * (s['audit']['upper95_all_mismatches_conservative'] or 0):.1f} %). By status at audit time: VERIFIED
{s['audit']['stratum_bounds']['VERIFIED']['errors']}/{s['audit']['stratum_bounds']['VERIFIED']['n']} (upper bound
{100 * s['audit']['stratum_bounds']['VERIFIED']['upper95']:.1f} %), ADJUDICATED_LOW_CONFIDENCE
{s['audit']['stratum_bounds']['ADJUDICATED_LOW_CONFIDENCE']['errors']}/{s['audit']['stratum_bounds']['ADJUDICATED_LOW_CONFIDENCE']['n']}
(upper bound {100 * s['audit']['stratum_bounds']['ADJUDICATED_LOW_CONFIDENCE']['upper95']:.1f} %). The errors found were corrected.

## Fields
- `id`, `exam_id`, `parent_id`, `question_label`, `sub_label`, `problem` (context + this part; LaTeX verbatim).
- `gold`: typed gold for `vi_math_verified_aug/eval/math_verify.py` (grader {s['grader']['version']}):
  `answer_type` ∈ int/rational/decimal/radical/tuple/set/interval/expr/text, `answer_canonical`, optional `vars`,
  `target_var`, `round_dp`, `percent`, `accept_also`; `quantities` = names of the asked quantities of a word problem
  (documentation only: such tuples are positional, in the order the problem asks). Grade with
  `math_verify.grade(response, gold)`.
- `status`, `qa` (route, adjudicator confidence, keys agreeing with the gold, X2 key status, audit verdict,
  `gold_roundtrip` = the gold grades its own canonical form correct).
- `exam_year` (printed paper header), `first_seen` (min of Wayback / sitemap lastmod / og:image date; an upper bound
  on publication), `temporal` {{exam_2026, exam_2025_plus, first_seen_2026_06_plus}}.
- `certifiable` (X2 rule tag yes/maybe/no; H4 uses yes vs maybe+no), `topic`, `topics`, `answer_type_tag`.
- `contamination`: `flag` = X5 §5.3 review rule (likely real copy in a public corpus — exclude from post-cutoff
  claims); `g4_any` = any G4 signal (type-level similarity; sensitivity analysis only); `signals`, `max_similarity`.
  Contaminated items are kept, flagged.
- `provenance` (source URL, province, school, exam kind, issuer, fetched_at, first_seen evidence), `license_basis`,
  `release_eligible` (False for university-school papers: research use only).

## Post-cutoff slices (after QA and removal of review-rule contamination)
| slice | X2 Core | after QA | after QA + contamination | of which VERIFIED |
|---|---|---|---|---|
| exam year 2026 | {pc['exam_2026']['x2_core_before_qa']} | {pc['exam_2026']['core_after_qa']} | {pc['exam_2026']['after_qa_and_contamination']} | {pc['exam_2026']['verified_and_clean']} |
| exam year ≥ 2025 | {pc['exam_2025_plus']['x2_core_before_qa']} | {pc['exam_2025_plus']['core_after_qa']} | {pc['exam_2025_plus']['after_qa_and_contamination']} | {pc['exam_2025_plus']['verified_and_clean']} |
| first_seen ≥ 2026-06 | {pc['first_seen_2026_06_plus']['x2_core_before_qa']} | {pc['first_seen_2026_06_plus']['core_after_qa']} | {pc['first_seen_2026_06_plus']['after_qa_and_contamination']} | {pc['first_seen_2026_06_plus']['verified_and_clean']} |

## Not included / caveats
- Third-party worked solutions are NOT in this file (QA-only, `data/vi_exam/items_private.jsonl`).
- Text fidelity against official Sở PDFs is unchecked; keys are LLM-adjudicated, not human-verified.
- `certifiable` is X2's regex tag; the adjudicator's re-tag is in `qa.adjudicator_certifiable` (adjudicated items).
"""
    txt = txt.replace("core_v1.jsonl", f"core_v{version}.jsonl").replace("invalid_v1.jsonl",
                                                                          f"invalid_v{version}.jsonl")
    txt = txt.replace("# Vi-ExamMath Core v1 ", f"# Vi-ExamMath Core v{version} ")
    if version != "1":
        a1 = s["audit_v1"]
        txt += (f"\n## Version history\n- v1 (`core_v1.jsonl`, SHA256 `{s['core_v1_sha256']}`): first freeze; audit #1 "
                f"(seed {a1.get('seed')}) found {a1['label_errors']}/{a1['n']} label errors (95 % upper bound "
                f"{100 * a1['upper95_label_errors']:.1f} %), all corrected in v1.1.\n"
                f"- v{version}: the items where only the two Claude solves agreed were adjudicated; audit #2 "
                f"(seed {s['audit'].get('seed')}) is the bound quoted above.\n"
                "- Recommended use (T5): **VERIFIED = primary analysis set; all of Core = sensitivity analysis.**\n")
    (D.OUT / f"core_v{version}.datacard.md").write_text(txt, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit-prep", "freeze", "decide", "audit2-sample", "audit2-prep"])
    ap.add_argument("--version", default="1.1")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "audit-prep":
        audit_prep()
    elif a.cmd == "audit2-sample":
        audit2_sample()
    elif a.cmd == "audit2-prep":
        audit2_prep()
    elif a.cmd == "freeze":
        freeze(a.version, a.force)
    else:
        dec, misc = decide_all()
        print(json.dumps({"status": Counter(d["status"] for d in dec), "routes": Counter(d["status_route"] for d in dec),
                          "misc": misc}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
