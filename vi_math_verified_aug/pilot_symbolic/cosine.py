"""G3: bge-m3 cosine signals for the arm-Y items (X5 calibration, exec/X5_decontam.md §5).

Needs numpy (run with the project .venv; ~60 MB RSS). Inputs: <run>/emb.json from ``embed_modal`` and X5's target
embeddings (same encoder settings): data/contam/embed_modal/main/target_emb.npy (Vi-GSM8K test 1,319, Vi-MATH test
5,000, Vi-ExamMath raw 3,024) and data/contam/embed_modal/vex/target_emb.npy (Vi-ExamMath items 4,209).

Per item: cos_eval_max (+ which eval item), cos_seed (vs its own seed; M6: max over both seeds), and the max cosine
to an item of ANOTHER seed (cross-seed near-duplicate; same-seed siblings are by design, X5 §5 G5 note).

    .venv/Scripts/python -m vi_math_verified_aug.pilot_symbolic.cosine   -> <run>/cosine.jsonl
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SYM = ROOT / "data" / "pilot" / "symbolic"
X5 = ROOT / "data" / "contam" / "embed_modal"
TAU_VI = 0.85


def main():
    run = SYM / os.environ.get("G3_RUN", "run_b")
    d = json.load(open(run / "emb.json"))
    keys = d["keys"]
    E = np.asarray(d["emb"], dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    idx = {k: i for i, k in enumerate(keys)}
    T, tids = [], []
    for sub in ("main", "vex"):
        T.append(np.load(X5 / sub / "target_emb.npy").astype(np.float32))
        tids += [f"{a}:{b}" for a, b in json.load(open(X5 / sub / "target_ids.json"))]
    T = np.concatenate(T)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    mut = {}
    for l in open(run / "mutate.jsonl", encoding="utf-8"):
        r = json.loads(l)
        mut[r["key"]] = r
    items = [k for k in keys if k.startswith("item:")]

    def fam(key: str) -> set:          # the seeds an item is made from (M6: seed + partner)
        return {key.split(":")[0]} | ({mut[key]["partner"]} if mut[key].get("partner") else set())
    I = E[[idx[k] for k in items]]
    S_eval = I @ T.T
    S_items = I @ I.T
    out = []
    for n, k in enumerate(items):
        key = k[5:]
        sid = key.split(":")[0]
        seeds = [sid] + ([mut[key]["partner"]] if mut[key].get("partner") else [])
        cs = max(float(E[idx["seed:" + s]] @ I[n]) for s in seeds if "seed:" + s in idx)
        j = int(S_eval[n].argmax())
        other = [m for m, k2 in enumerate(items) if not (fam(k2[5:]) & set(seeds))]
        jo = max(other, key=lambda m: S_items[n, m]) if other else None
        out.append({"key": key, "op": mut[key]["op"], "cos_eval_max": round(float(S_eval[n, j]), 4),
                    "cos_eval_id": tids[j], "cos_seed": round(cs, 4),
                    "cos_other_seed_item_max": None if jo is None else round(float(S_items[n, jo]), 4),
                    "cos_other_seed_item": None if jo is None else items[jo][5:]})
    with open(run / "cosine.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ev = sum(r["cos_eval_max"] >= TAU_VI for r in out)
    print(f"items {len(out)}; cos_eval >= {TAU_VI}: {ev}; max cos_eval {max(r['cos_eval_max'] for r in out):.3f}")


if __name__ == "__main__":
    main()
