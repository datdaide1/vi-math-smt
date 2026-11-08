"""D5 R1 results wrapper on synthetic predictions (real Core ids, planted effects; no API, light)."""
import json
import random

import pytest

from vi_math_verified_aug.analysis import c3_r1_results as R

pytestmark = pytest.mark.skipif(not R.CORE.exists(), reason="no core_v1.1")


def _rows(n_core=60, n_par=30, seeds=(0, 1), effect=0.25):
    meta, allowed = R.core_meta()
    core_ids = sorted(allowed)[:n_core]
    rng = random.Random(0)
    rows = []
    base_p = {"base": .2, "S0": .3, "Y+": .45, "Y-": .4, "N+": .3, "N-": .25}
    for lang in ("vi", "en"):
        for arm, p in base_p.items():
            for s in seeds:
                rid = f"{arm}-{lang}-s{s}"
                common = {"run_id": rid, "arm": arm, "model": "qwen3-0.6b", "size": "0.6B", "seed": s, "lang": lang}
                for i in range(n_par):
                    for k in (-1, 0, 1):
                        rows.append({**common, "dataset": "gsm8k_platinum", "item_id": f"g{i}", "sample_k": k,
                                     "decode": "greedy" if k < 0 else "sample", "correct": int(rng.random() < p),
                                     "tokens": rng.randint(50, 300)})
                if lang == "vi":
                    for cid in core_ids:
                        pp = p + (effect if (arm.startswith("Y") and meta[cid]["certifiable"]) else 0)
                        rows.append({**common, "dataset": "vi_exam_core", "item_id": cid, "sample_k": -1,
                                     "decode": "greedy", "correct": int(rng.random() < min(pp, .95)),
                                     "tokens": rng.randint(50, 300)})
                    rows.append({**common, "dataset": "vi_exam_core", "item_id": "not-a-core-id", "sample_k": -1,
                                 "decode": "greedy", "correct": 1, "tokens": 10})
    return rows


def test_r1_end_to_end(tmp_path):
    p = tmp_path / "preds.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in _rows()), encoding="utf-8")
    res = R.run([str(p)], B=200)
    assert res["inputs"]["greedy_as_k1_rows"] > 0
    assert res["inputs"]["exam_dropped_not_verified_or_contaminated"] == 12          # 6 arms x 2 seeds x 1 bogus id
    names = {r["name"]: r for r in res["confirmatory"]}
    assert any(n.startswith("H2") for n in names) and any(n.startswith("H3") for n in names)
    h2 = next(r for n, r in names.items() if n.startswith("H2"))
    assert h2["est_pp"] > 0 and h2["mde80_pp"] > 0
    ds = {r["name"].split("|")[0] for r in res["per_dataset"]}
    assert ds == {"gsm8k_platinum", "vi_exam_core"}                                   # per dataset, never pooled only
    assert any(r["arm"] == "S0" for r in res["arm_table"])
    md = R.tables(res)
    assert "Confirmatory family" in md and "Per-dataset contrasts" in md
