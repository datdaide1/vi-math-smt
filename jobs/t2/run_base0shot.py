"""Base Qwen3-0.6B 0-shot eval in the arms' exact template + decoding (lead 2026-11-08 08:1x).

Runs ev-base-qwen3-0.6b-0shot-vi-primary (vi_primary_1000, K=4 + greedy) and ev-base-qwen3-0.6b-0shot-vi-exam-greedy
(Vi-ExamMath VERIFIED, greedy) on Modal L40S, batch-invariant, cap 1,024 -- ONLY if the Modal workspace
month-to-date (all apps) + EST_USD <= CAP_USD, and only once no tranche work is still unbilled.  Otherwise it prints
REFUSED and does nothing (the item stays a successor with its cost).

    python jobs/t2/run_base0shot.py            # check + launch (blocks until the eval finishes, then pulls preds)
    python jobs/t2/run_base0shot.py --check    # print the decision only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vi_math_verified_aug.train import launcher as L  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B-Base"
EST_USD = 0.30
CAP_USD = 29.5
TAG = "ev-base-qwen3-0.6b-0shot-vi"


def configs() -> list[dict]:
    r = {"run_id": "base-qwen3-0.6b-0shot-vi", "arm": "base", "model": MODEL, "size": "0.6B", "seed": 0,
         "lang": "vi"}
    out = L.eval_configs(r, None)          # shots 0, no few-shot template = the arms' template and decoding
    assert [c["run_id"] for c in out] == [f"{TAG}-primary", f"{TAG}-exam-greedy"], [c["run_id"] for c in out]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    st = json.loads((REPO / "jobs" / "queue" / "t2_state.json").read_text(encoding="utf-8"))
    inflight = [r for r, s in st["runs"].items() if s.get("status") == "LAUNCHED"]
    unbilled = (L._spent_now(st) - st["modal_spent_usd"]) if st.get("modal_spent_usd") is not None else 0.0
    mtd = L.modal_month_total()
    ok = mtd == mtd and not inflight and unbilled <= 1e-6 and mtd + EST_USD <= CAP_USD
    print(json.dumps({"workspace_mtd_usd": mtd, "est_usd": EST_USD, "cap_usd": CAP_USD, "training_in_flight": inflight,
                      "tranche_unbilled_usd": round(unbilled, 3), "decision": "RUN" if ok else "REFUSED"}))
    if a.check or not ok:
        return 0 if ok else 2
    c = L.code_stamp()
    if c["dirty"]:
        raise SystemExit(f"refusing to launch from a dirty tree: {c['dirty_files']}")
    L._CODE["stamp"] = c
    cfgs = configs()
    spec = REPO / "jobs" / "t2" / f"{TAG}.json"
    spec.write_text(json.dumps({"base": MODEL, "tag": TAG, "configs": cfgs}, indent=1), encoding="utf-8")
    log = REPO / "jobs" / "logs" / "T2" / f"{TAG}.log"
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run([L.MODAL, "run", "vi_math_verified_aug/evalrun/modal_eval.py::eval_main", "--spec",
                             str(spec)], cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, timeout=5400,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8", MSYS_NO_PATHCONV="1")).returncode
    print("modal run rc", rc)
    dest_dir = REPO / "jobs" / "outputs" / "t2-eval"
    dest_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for cfg in cfgs:
        dest = dest_dir / f"{cfg['run_id']}.jsonl"
        L.sh([L.MODAL, "volume", "get", "--force", "t1r-eval", f"/eval/{cfg['run_id']}/preds.jsonl", str(dest)],
             timeout=600)
        res[cfg["run_id"]] = L.eval_metrics(dest) if dest.exists() else None
        print(cfg["run_id"], json.dumps(res[cfg["run_id"]])[:400])
    (REPO / "jobs" / "t2" / f"{TAG}.metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("month-to-date after:", L.modal_month_total())
    return rc


if __name__ == "__main__":
    sys.exit(main())
