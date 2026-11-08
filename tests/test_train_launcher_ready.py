"""READY rule for D5's final training files (lead 2026-11-08) + the EN seeds 4-5 hold in the queue.  Torch-free."""
import hashlib
import json
from pathlib import Path

import pytest

from vi_math_verified_aug.train import grid as GR
from vi_math_verified_aug.train import launcher as L

ARMS = {("Y", "plus"), ("Y", "minus"), ("N", "plus"), ("N", "minus")}


def write_final(root: Path, lang="vi", n=5, rows=None, manifest=True, sums=True, drop=None, sel_only=False):
    final = root / "data" / "train" / "final"
    files = {}
    for x, s in ARMS:
        d = final / x / lang
        d.mkdir(parents=True, exist_ok=True)
        k = (rows or {}).get((x, s), n)
        p = d / f"train_{s}.jsonl"
        if (x, s) != drop:
            p.write_text("".join(json.dumps({"id": f"{x}{s}{i}", "question": f"q{x}{s}{i}?", "solution": f"s {i}",
                                             "answer": str(i)}) + "\n" for i in range(k)), encoding="utf-8")
            files[f"data/train/final/{x}/{lang}/train_{s}.jsonl"] = {
                "rows": k, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    if sums:
        for x in ("Y", "N"):
            d = final / x / lang
            d.joinpath("SHA256SUMS").write_text("".join(
                f"{files[k]['sha256']}  {Path(k).name}\n" for k in files if f"/{x}/{lang}/" in k), encoding="utf-8")
    if sel_only:   # D5's part-1 file must never be read for n
        (final / f"selection_{lang}_v1.json").write_text(json.dumps({"n": 3000}), encoding="utf-8")
    if manifest:
        (final / f"manifest_{lang}.json").write_text(json.dumps({"n_effective": n, "files": files}), encoding="utf-8")
    return final


def test_ready_when_manifest_files_and_sums_agree(tmp_path):
    final = write_final(tmp_path, n=5)
    n, why, det = L.final_set_status(final, "vi")
    assert n == 5 and why == "" and set(det) == {"Y+", "Y-", "N+", "N-"}


@pytest.mark.parametrize("kw,needle", [
    ({"manifest": False, "sel_only": True}, "manifest_vi.json missing"),   # selection_* is never a fallback
    ({"sums": False}, "SHA256SUMS missing"),
    ({"drop": ("N", "minus")}, "missing"),
    ({"rows": {("Y", "plus"): 4}}, "!= n_effective"),                     # every file must have EXACTLY n rows
])
def test_not_ready(tmp_path, kw, needle):
    final = write_final(tmp_path, n=5, **kw)
    n, why, _ = L.final_set_status(final, "vi")
    assert n is None and needle in why


def test_hash_mismatch_is_not_ready(tmp_path):
    final = write_final(tmp_path, n=5)
    p = final / "Y" / "vi" / "train_plus.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace("q", "Q"), encoding="utf-8")
    n, why, _ = L.final_set_status(final, "vi")
    assert n is None and "sha256" in why


def test_tranche_and_holds_match_the_budget_decision():
    """Lead 2026-11-08 (Modal $6.5): VI 0.6B seeds 0-3 + EN 0.6B seeds 0-1 in the order VI s0, EN s0, VI s1-3, EN s1;
    VI seed 4, EN seeds 2-4 and the S0-EN LR check are HOLD rows (priority 12).  Lead 08:19: VI s3 was already
    training, so EN s1 became the first successor (HOLD, priority 11)."""
    rows = GR.build()
    tranche = [r for r in rows if r["priority"] <= 3 and r["kind"] == "sft"]
    assert {(r["lang"], r["seed"]) for r in tranche} == {("vi", s) for s in range(4)} | {("en", 0)}
    cells = []
    for r in tranche:
        c = (r["lang"], r["seed"])
        if c not in cells:
            cells.append(c)
    assert cells == [("vi", 0), ("en", 0), ("vi", 1), ("vi", 2), ("vi", 3)]
    assert all(r["lr"] == GR.LR and r["epochs"] == GR.EPOCHS for r in tranche)
    hold = [r for r in rows if r.get("status") == "hold"]
    assert {(r["lang"], r["seed"]) for r in hold if "-lr" not in r["run_id"]} ==         {("vi", 4)} | {("en", s) for s in (1, 2, 3, 4)}
    assert all(r["priority"] > 3 for r in hold)


def test_eval_metrics_and_vi_s0_summary(tmp_path):
    preds = tmp_path / "p.jsonl"
    rows = []
    for i in range(20):
        for k in (-1, 0, 1, 2, 3):
            rows.append({"dataset": "gsm8k_platinum" if i < 12 else "math_algprealg", "item_id": str(i),
                         "sample_k": k, "correct": int((i + k) % 3 == 0)})
    preds.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    m = L.eval_metrics(preds)
    assert set(m) == {"gsm8k_platinum", "math_algprealg"} and m["gsm8k_platinum"]["n"] == 12
    assert m["gsm8k_platinum"]["pass1_ci"][0] <= m["gsm8k_platinum"]["pass1"] <= m["gsm8k_platinum"]["pass1_ci"][1]
    q = {r["run_id"]: r for r in GR.build()}
    vi0 = [rid for rid, r in q.items() if r["lang"] == "vi" and r["seed"] == 0 and r["priority"] <= 3]
    st = {"runs": {rid: {"status": "TRAINED", "check": {"clean": True}, "evals": [f"ev-{rid}-primary"],
                         "eval_metrics": {}} for rid in vi0}}
    out = tmp_path / "vi_s0_summary.md"
    assert not L.write_cell_summary(st, q, "vi", 0, out)          # evals not finished -> nothing written
    for rid in vi0:
        st["runs"][rid]["eval_metrics"][f"ev-{rid}-primary"] = m
    assert L.write_cell_summary(st, q, "vi", 0, out)
    txt = out.read_text(encoding="utf-8")
    assert txt.count("| S0 |") == 1 and "gsm8k_platinum" in txt and "Y+" in txt
