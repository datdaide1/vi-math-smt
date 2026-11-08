"""W5 pilot seeds + shared operator schedule (T1 §2.0).

* Pool: Vi-GSM8K train (data/translation/gsm8k/gsm8k_train_vietnamese.jsonl), aligned by index with
  the English data/generate/gsm8k/train.jsonl. R1 corrupt rows are excluded.
* DEV_SEEDS = 5 % of seeds by hash (T1: pilot/calibration only, never used for training items).
* 100 seeds sampled with random.Random(20261107), stratified (proportional allocation) by the number
  of reasoning steps = non-empty lines before '####' in the English solution.
* Schedule: J = 3 slots per seed, operator o ~ π = {M1 .20, M2 .20, M3 .25, M4 .15, M5 .10, M6 .10}
  with random.Random(20261107); both arms get exactly the same (seed, slot, op) triples.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VI = ROOT / "data/translation/gsm8k/gsm8k_train_vietnamese.jsonl"
EN = ROOT / "data/generate/gsm8k/train.jsonl"
OUT = ROOT / "data/pilot_gen"

# R1 §5/§6: 14 corrupt rows + 2 half-English rows
R1_EXCLUDE = {220, 221, 222, 223, 224, 225, 2577, 3077, 2345, 3539, 5085, 7454, 1801, 2911, 3424, 6465}
SEED = 20261107
PI = {"M1": .20, "M2": .20, "M3": .25, "M4": .15, "M5": .10, "M6": .10}
J = 3


def seed_id(index: int) -> str:
    return hashlib.sha1(f"vi_gsm8k_train||{index}".encode()).hexdigest()[:12]


def is_dev(sid: str) -> bool:
    return int(sid, 16) % 20 == 0


def n_steps(en_answer: str) -> int:
    body = en_answer.split("####")[0]
    return sum(1 for l in body.split("\n") if l.strip())


def step_stratum(n: int) -> str:
    return "<=2" if n <= 2 else ("6+" if n >= 6 else str(n))


def build(n: int = 100):
    vi = [json.loads(l) for l in open(VI, encoding="utf-8")]
    en = [json.loads(l) for l in open(EN, encoding="utf-8")]
    assert len(vi) == len(en)
    pool = []
    for v, e in zip(vi, en):
        assert v["index"] == e["index"]
        if v["index"] in R1_EXCLUDE:
            continue
        sid = seed_id(v["index"])
        if not is_dev(sid):
            continue
        gt = str(e["ground_truth"]).replace(",", "")
        st = n_steps(e["answer"])
        pool.append({"seed_id": sid, "index": v["index"], "text": v["question"], "gold": gt,
                     "en_text": e["question"], "en_solution": e["answer"], "vi_solution": v["answer"],
                     "steps": st, "stratum": step_stratum(st)})
    rng = random.Random(SEED)
    by = defaultdict(list)
    for p in pool:
        by[p["stratum"]].append(p)
    tot = len(pool)
    alloc = {k: n * len(v) / tot for k, v in by.items()}
    base = {k: int(a) for k, a in alloc.items()}
    rem = n - sum(base.values())
    for k in sorted(alloc, key=lambda k: -(alloc[k] - base[k]))[:rem]:
        base[k] += 1
    seeds = []
    for k in sorted(by):
        seeds += rng.sample(sorted(by[k], key=lambda p: p["index"]), base[k])
    seeds.sort(key=lambda p: p["index"])
    # schedule
    srng = random.Random(SEED)
    ops, w = list(PI), list(PI.values())
    sched = []
    for s in seeds:
        for j in range(1, J + 1):
            sched.append({"seed_id": s["seed_id"], "slot": j, "op": srng.choices(ops, w)[0]})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "seeds.jsonl", "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(OUT / "schedule.jsonl", "w", encoding="utf-8") as f:
        for r in sched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"pool_dev_size": tot, "pool_strata": {k: len(v) for k, v in sorted(by.items())},
            "alloc": base, "n_seeds": len(seeds), "n_slots": len(sched),
            "op_counts": dict(Counter(r["op"] for r in sched))}
    json.dump(info, open(OUT / "seeds_info.json", "w"), indent=1)
    return info


def load_seeds() -> list:
    return [json.loads(l) for l in open(OUT / "seeds.jsonl", encoding="utf-8")]


def load_schedule() -> list:
    return [json.loads(l) for l in open(OUT / "schedule.jsonl", encoding="utf-8")]


def gold_fraction(s: str) -> Fraction:
    return Fraction(str(s).replace(",", "").strip())


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
