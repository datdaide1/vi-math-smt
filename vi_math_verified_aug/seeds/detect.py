"""Stage ``detect``: deterministic repairs (T4 §3.8 step 1) + every defect trigger on the existing translations.

Triggers that send a row to MTA-vi re-translation (``retranslate`` = True):
  * R1 §6 must-fix: the 14 corrupt GSM8K rows + the 2 half-English rows;
  * R1/R2 hand annotations of major/critical severity that fall in the pool (520, 529, 2597, 3661, 6922; algebra
    387; algebra 327 is escape damage only and is repaired deterministically);
  * any G0/G1 failure that remains *after* the deterministic repairs (numbers, final answer, glossary/units,
    CR-1/2/3/5, annotations, LaTeX/structure, English residue).
Rows whose only defect was repaired deterministically are kept with provenance ``det_repair``.
"""
from __future__ import annotations

import json
from collections import Counter

from ..data_fix.common import CTRL_RE
from .checks import deterministic_repair, fail_kind, g0, g1
from .common import ANN, WORK, read_jsonl, write_jsonl

R1_CORRUPT = {220, 221, 222, 223, 224, 225, 2577, 3077, 2345, 3539, 5085, 7454, 1801, 2911}
R1_HALF_EN = {3424, 6465}


def _annotated() -> dict[tuple[str, int], str]:
    out = {}
    for r in read_jsonl(ANN / "vi_gsm8k_train.jsonl"):
        if r.get("severity") in ("major", "critical") and "translation" in (r.get("origin") or []):
            out[("vi_gsm8k_train", int(r["index"]))] = f"R1 annotation {r['severity']}: {r['note'][:160]}"
    for r in read_jsonl(ANN / "vi_math_train.jsonl"):
        if r.get("severity") in ("major", "critical") and r.get("subject") in ("algebra", "prealgebra"):
            out[(f"vi_math_train_{r['subject']}", int(r["index"]))] = \
                f"R2 annotation {r['severity']} {r['codes']}: {r['note'][:160]}"
    return out


def run() -> dict:
    rows = read_jsonl(WORK / "aligned.jsonl")
    ann = _annotated()
    out, kinds, det_kinds = [], Counter(), Counter()
    for r in rows:
        fixed, changes = deterministic_repair(r)
        for c in changes:
            det_kinds[c["kind"]] += 1
        vp, vs = fixed["problem_vi"], fixed["solution_vi"]
        fails = g0(vp, vs, r["problem_en"], r["solution_en"], r["dataset"]) + \
            g1(vp, vs, r["problem_en"], r["solution_en"], r["gold_raw_en"], r["dataset"])
        if CTRL_RE.search(fixed["gold_raw_vi"]):
            fails.append("G0 control characters in VI ground_truth")
        must = []
        if r["source"] == "vi_gsm8k_train" and r["index"] in R1_CORRUPT:
            must.append("R1 §5 corrupt row (misaligned / dropped or changed condition)")
        if r["source"] == "vi_gsm8k_train" and r["index"] in R1_HALF_EN:
            must.append("R1 §5 half-English row")
        if (r["source"], r["index"]) in ann:
            must.append(ann[(r["source"], r["index"])])
        for f in fails:
            kinds[fail_kind(f)] += 1
        out.append({"seed_id": r["seed_id"], "source": r["source"], "index": r["index"], "dev": r["dev"],
                    "problem_vi": vp, "solution_vi": vs, "gold_raw_vi": fixed["gold_raw_vi"],
                    "det_changes": changes, "must": must, "gate_fails": fails,
                    "retranslate": bool(must or fails)})
    write_jsonl(WORK / "detect.jsonl", out)
    n_rt = sum(o["retranslate"] for o in out)
    summ = {"n": len(out), "det_change_kinds": dict(det_kinds),
            "rows_with_det_changes": sum(bool(o["det_changes"]) for o in out),
            "retranslate": n_rt, "retranslate_by_source": dict(Counter(o["source"] for o in out if o["retranslate"])),
            "must": sum(bool(o["must"]) for o in out),
            "fail_kinds_rows": dict(Counter(k for o in out for k in {fail_kind(f) for f in o["gate_fails"]})),
            "det_only_rows": sum(bool(o["det_changes"]) and not o["retranslate"] for o in out)}
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return summ
