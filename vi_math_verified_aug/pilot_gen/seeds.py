"""G1 pilot seeds: 100 dev seeds from Vi-GSM8K train, one operator per seed from T1's π.

* Pool: data/translation/gsm8k/gsm8k_train_vietnamese.jsonl aligned by index with data/generate/gsm8k/train.jsonl.
* Excluded: the 14 corrupt rows of content_review/R1 §5 (220-225, 2577, 3077, 2345, 3539, 5085, 7454, 1801,
  2911) and the 2 half-English rows R1 also flags (3424, 6465).
* DEV_SEEDS = seeds whose seed_id hash is 0 mod 20 (5%, T1 §2.0; pilot/calibration only).
* 100 seeds: random.Random(20261107).sample over the sorted dev pool.
* Operator: o ~ π = {M1 .20, M2 .20, M3 .25, M4 .15, M5 .10, M6 .10} with random.Random(20261107)
  (one slot per seed; every generator gets identical (seed, operator) pairs). M6 partner seed and the
  localization card are drawn from the same RNG stream.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VI = ROOT / "data/translation/gsm8k/gsm8k_train_vietnamese.jsonl"
EN = ROOT / "data/generate/gsm8k/train.jsonl"
OUT = ROOT / "data/pilot/generator"
CARDS = Path(__file__).with_name("cards_v1.json")

R1_CORRUPT = {220, 221, 222, 223, 224, 225, 2577, 3077, 2345, 3539, 5085, 7454, 1801, 2911}
R1_HALF_ENGLISH = {3424, 6465}
SEED = 20261107
PI = {"M1": .20, "M2": .20, "M3": .25, "M4": .15, "M5": .10, "M6": .10}


def seed_id(index: int) -> str:
    return hashlib.sha1(f"vi_gsm8k_train||{index}".encode()).hexdigest()[:12]


def build(n: int = 100) -> dict:
    vi = [json.loads(l) for l in open(VI, encoding="utf-8")]
    en = [json.loads(l) for l in open(EN, encoding="utf-8")]
    assert len(vi) == len(en) == 7473
    pool = []
    for v, e in zip(vi, en):
        assert v["index"] == e["index"]
        if v["index"] in R1_CORRUPT | R1_HALF_ENGLISH:
            continue
        sid = seed_id(v["index"])
        if int(sid, 16) % 20:
            continue
        pool.append({"seed_id": sid, "index": v["index"], "text": v["question"],
                     "gold": str(e["ground_truth"]).replace(",", ""), "en_text": e["question"],
                     "en_solution": e["answer"]})
    pool.sort(key=lambda p: p["index"])
    rng = random.Random(SEED)
    seeds = sorted(rng.sample(pool, n), key=lambda p: p["index"])
    cards = json.load(open(CARDS, encoding="utf-8"))
    ops, w = list(PI), list(PI.values())
    srng = random.Random(SEED)
    for s in seeds:
        s["op"] = srng.choices(ops, w)[0]
        s["card"] = srng.choice(cards)["id"]
        s["partner"] = None
        if s["op"] == "M6":
            cand = [p for p in pool if p["seed_id"] != s["seed_id"]]
            s["partner"] = srng.choice(cand)["seed_id"]
    partners = {p["seed_id"]: p for p in pool}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "seeds.jsonl", "w", encoding="utf-8") as f:
        for s in seeds:
            if s["partner"]:
                p = partners[s["partner"]]
                s["partner_text"], s["partner_gold"], s["partner_index"] = p["text"], p["gold"], p["index"]
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    info = {"dev_pool": len(pool), "n": len(seeds), "ops": dict(sorted(Counter(s["op"] for s in seeds).items())),
            "excluded": sorted(R1_CORRUPT | R1_HALF_ENGLISH)}
    json.dump(info, open(OUT / "seeds_info.json", "w"), indent=1)
    return info


def load() -> list:
    return [json.loads(l) for l in open(OUT / "seeds.jsonl", encoding="utf-8")]


def cards() -> dict:
    return {c["id"]: c for c in json.load(open(CARDS, encoding="utf-8"))}


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
