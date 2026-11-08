"""G3 seeds: 60 NEW Vi-GSM8K train dev seeds and the T1 §2.0 operator schedule (J = 3 slots per seed).

* Dev pool: T1 §2.0 DEV_SEEDS = seed_id hash ≡ 0 (mod 20), seed_id = sha1("vi_gsm8k_train||index")[:12]
  (same definition as G1 ``pilot_gen/seeds.py``).
* Excluded: the 14 R1 corrupt rows and 2 half-English rows (R1 §5), the 100 G1 seeds
  (data/pilot/generator/seeds.jsonl) and the 40 G2 seeds (data/pilot/g2/seeds40.jsonl).
* 60 seeds: random.Random(20261108).sample over the remaining dev pool sorted by index.
* Schedule: for each seed (sorted by index) and slot j in 1..3, o ~ π = {M1 .20, M2 .20, M3 .25, M4 .15, M5 .10,
  M6 .10} with random.Random(20261108) (a separate stream from the sampler).

python -m vi_math_verified_aug.pilot_symbolic.seeds  -> data/pilot/symbolic/seeds.jsonl, slots.jsonl, seeds_info.json
"""
from __future__ import annotations

import json
import random
from collections import Counter

from ..pilot_gen import seeds as G1S
from .common import OUT, ROOT

SEED = 20261108
N_SEEDS = 60
J = 3
PI = {"M1": .20, "M2": .20, "M3": .25, "M4": .15, "M5": .10, "M6": .10}


def _indices(path) -> set[int]:
    return {json.loads(l)["index"] for l in open(path, encoding="utf-8") if l.strip()}


def build() -> dict:
    vi = [json.loads(l) for l in open(G1S.VI, encoding="utf-8")]
    en = [json.loads(l) for l in open(G1S.EN, encoding="utf-8")]
    assert len(vi) == len(en) == 7473
    g1 = _indices(ROOT / "data/pilot/generator/seeds.jsonl")
    g2 = _indices(ROOT / "data/pilot/g2/seeds40.jsonl")
    excl = G1S.R1_CORRUPT | G1S.R1_HALF_ENGLISH | g1 | g2
    dev = []
    for v, e in zip(vi, en):
        assert v["index"] == e["index"]
        sid = G1S.seed_id(v["index"])
        if int(sid, 16) % 20:
            continue
        dev.append({"seed_id": sid, "index": v["index"], "text": v["question"],
                    "gold": str(e["ground_truth"]).replace(",", ""), "en_text": e["question"]})
    pool = sorted([d for d in dev if d["index"] not in excl], key=lambda d: d["index"])
    seeds = sorted(random.Random(SEED).sample(pool, N_SEEDS), key=lambda d: d["index"])
    srng = random.Random(SEED)
    ops, w = list(PI), list(PI.values())
    slots = []
    for s in seeds:
        for j in range(1, J + 1):
            slots.append({"key": f"{s['seed_id']}:{j}", "seed_id": s["seed_id"], "index": s["index"], "slot": j,
                          "op": srng.choices(ops, w)[0]})
    with open(OUT / "seeds.jsonl", "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(OUT / "slots.jsonl", "w", encoding="utf-8") as f:
        for s in slots:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    info = {"dev_pool": len(dev), "excluded_in_dev_pool": len(dev) - len(pool), "remaining_pool": len(pool),
            "n_seeds": len(seeds), "n_slots": len(slots), "ops": dict(sorted(Counter(s["op"] for s in slots).items())),
            "overlap_with_g1_g2": sorted({s["index"] for s in seeds} & (g1 | g2)), "indices": [s["index"] for s in seeds]}
    json.dump(info, open(OUT / "seeds_info.json", "w"), indent=1)
    return info


def load_seeds() -> dict[str, dict]:
    return {json.loads(l)["seed_id"]: json.loads(l) for l in open(OUT / "seeds.jsonl", encoding="utf-8")}


def load_slots() -> list[dict]:
    return [json.loads(l) for l in open(OUT / "slots.jsonl", encoding="utf-8")]


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
