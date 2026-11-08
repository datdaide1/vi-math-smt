"""Current text of every seed after the repair stages (aligned -> detect deterministic repairs -> MTA-vi)."""
from __future__ import annotations

from .common import WORK, read_jsonl


def current_rows() -> list[dict]:
    """One dict per seed: aligned fields + the current VI text + provenance + every change with its reason."""
    aligned = read_jsonl(WORK / "aligned.jsonl")
    det = {d["seed_id"]: d for d in read_jsonl(WORK / "detect.jsonl")}
    mta = {}
    if (WORK / "mta.jsonl").exists():
        mta = {r["key"]: r for r in read_jsonl(WORK / "mta.jsonl")}
    sc = {}
    if (WORK / "solve_batch.jsonl").exists():          # Haiku-batch checks (remainder); Gemini verdicts win
        sc = {r["key"]: r for r in read_jsonl(WORK / "solve_batch.jsonl")
              if r.get("verdict") in ("PASS", "FAIL", "UNINF", "PENDING2")}  # PENDING2 = VI missed once, 2nd check queued
    if (WORK / "solve.jsonl").exists():
        for r in read_jsonl(WORK / "solve.jsonl"):
            if r.get("verdict") in ("PASS", "FAIL", "UNINF"):
                # runs 1-2 (before per-attempt model logging) used only gemini-3.1-flash-lite
                r.setdefault("solver_models", ["gemini-3.1-flash-lite"])
                sc[r["key"]] = r
    out = []
    for a in aligned:
        d = det[a["seed_id"]]
        r = dict(a)
        r["problem_vi"], r["solution_vi"], r["gold_raw_vi"] = d["problem_vi"], d["solution_vi"], d["gold_raw_vi"]
        r["changes"] = [{"stage": "deterministic", **c} for c in d["det_changes"]]
        r["provenance"] = "det_repair" if d["det_changes"] else "original"
        r["must"] = d["must"]
        r["gate_fails"] = d["gate_fails"]
        r["solvecheck"] = sc.get(a["seed_id"], {}).get("verdict")
        r["solvecheck_models"] = sc.get(a["seed_id"], {}).get("solver_models")
        m = mta.get(a["seed_id"])
        r["mta_status"] = m.get("status") if m else None
        if m and m.get("status") == "ACCEPT":
            why = "; ".join(d["must"] + d["gate_fails"]) or ("solve-check FAIL (English solved, Vietnamese not)"
                                                           if r["solvecheck"] == "FAIL" else "")
            r["changes"] = [{"stage": "mta_vi", "field": "problem_vi+solution_vi", "kind": "retranslated",
                             "reason": f"re-translated from English with MTA-vi ({m['accepted_by']}); "
                                       f"trigger: {why[:600]}",
                             "old_problem_vi": d["problem_vi"]}]
            r["problem_vi"], r["solution_vi"] = m["problem_vi"], m["solution_vi"]
            r["provenance"] = "mta_vi"
            r["mta"] = {"accepted_by": m["accepted_by"], "rounds": len(m["history"]),
                        "glossary_fired": m.get("glossary_fired"), "rules_fired": m.get("rules_fired")}
        out.append(r)
    return out
