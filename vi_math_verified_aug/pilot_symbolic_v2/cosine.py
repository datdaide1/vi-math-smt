"""G4: bge-m3 cosine signals (X5 tau 0.85 Vi-Vi) for E/R1/R2 items. Needs numpy (project .venv, ~60 MB RSS).

Per item "<V>:<slot>": cos_eval_max vs X5's 13,552 Vietnamese eval targets, cos_seed (own seed), and the max cosine to
an item of ANOTHER seed within the same variant (cross-seed near-duplicate).

    .venv/Scripts/python -m vi_math_verified_aug.pilot_symbolic_v2.cosine  -> data/pilot/symbolic_v2/cosine.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "pilot" / "symbolic_v2"
X5 = ROOT / "data" / "contam" / "embed_modal"
TAU_VI = 0.85


def main():
    d = json.load(open(OUT / "emb.json"))
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
    items = [k for k in keys if k.startswith("item:")]
    I = E[[idx[k] for k in items]]
    S_eval = I @ T.T
    S_items = I @ I.T
    out = []
    for n, k in enumerate(items):
        _, v, slot = k.split(":", 2)
        sid = slot.split(":")[0]
        cs = float(E[idx["seed:" + sid]] @ I[n])
        j = int(S_eval[n].argmax())
        other = [m for m, k2 in enumerate(items) if k2.split(":")[1] == v and k2.split(":")[2].split(":")[0] != sid]
        jo = max(other, key=lambda m: S_items[n, m]) if other else None
        out.append({"key": f"{v}:{slot}", "cos_eval_max": round(float(S_eval[n, j]), 4), "cos_eval_id": tids[j],
                    "cos_seed": round(cs, 4),
                    "cos_other_seed_item_max": None if jo is None else round(float(S_items[n, jo]), 4),
                    "cos_other_seed_item": None if jo is None else items[jo][5:]})
    with open(OUT / "cosine.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for v in ("E", "R1", "R2", "R1b"):
        rs = [r for r in out if r["key"].startswith(v + ":")]
        print(v, len(rs), "cos_eval>=tau", sum(r["cos_eval_max"] >= TAU_VI for r in rs),
              "max", max(r["cos_eval_max"] for r in rs), "median cos_seed", float(np.median([r["cos_seed"] for r in rs])),
              "xseed>=tau", sum((r["cos_other_seed_item_max"] or 0) >= TAU_VI for r in rs))


if __name__ == "__main__":
    main()
