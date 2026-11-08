"""Shared (seed, slot, operator) schedule for arms Y and N: ``data/train/slots_v1.jsonl``.

Rule (T1 §2.0, deterministic, independent of process / PYTHONHASHSEED):
* seeds = every row of S3's frozen ``data/seeds/pool_v1.jsonl`` (dev seeds are in dev_v1 and never used);
* J = 3 slots per seed; slot key = ``<seed_id>:<j>`` (same key format as G3/G4's pilot slots);
* per-slot RNG = ``random.Random(f"20261107|<seed_id>|<j>")`` (str seeding is deterministic), so a slot's draws do
  not depend on which other seeds are in the pool;
* op ~ π = {M1 .20, M2 .20, M3 .25, M4 .15, M5 .10, M6 .10}; M6 is GSM8K-only, for MATH seeds its mass is
  redistributed proportionally (T1 §2.0);
* card ~ uniform over the 12 localization cards (ids C01–C12; the VI card set ``pilot_gen/cards_v1.json`` and the
  EN mirror ``scale_gen/cards_en_v1.json`` share ids), drawn from the same per-slot RNG after op;
* M6 partner = the other GSM8K pool seed p minimising sha1("20261107|<seed_id>|<j>|<p>") (rendezvous hashing, so a
  slot's partner only changes if that partner is dropped from the pool);
* ``rank`` = global generation order: all j=1 slots first, then j=2, then j=3; within j by
  sha1("slots_v1|<key>"). An arm that generates K items takes the K lowest ranks, so arms with the same K use the
  same slots, and a 10k prefix covers ~one slot per seed.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

from .common import J, PI, POOL, RNG_SEED, SLOTS, SLOTS_META, read_jsonl, sha256_file

CARD_IDS = [f"C{i:02d}" for i in range(1, 13)]
VERSION = "slots_v1"


def _pi(dataset: str) -> dict[str, float]:
    if dataset == "gsm8k":
        return dict(PI)
    p = {k: v for k, v in PI.items() if k != "M6"}
    z = sum(p.values())
    return {k: v / z for k, v in p.items()}


def slot_draw(seed_id: str, dataset: str, j: int, partners: list[str]) -> dict:
    rng = random.Random(f"{RNG_SEED}|{seed_id}|{j}")
    pi = _pi(dataset)
    op = rng.choices(list(pi), list(pi.values()))[0]
    card = rng.choice(CARD_IDS)
    partner = None
    if op == "M6":                # rendezvous hashing: stable unless the chosen partner itself leaves the pool
        partner = min((p for p in partners if p != seed_id),
                      key=lambda p: hashlib.sha1(f"{RNG_SEED}|{seed_id}|{j}|{p}".encode()).hexdigest())
    return {"op": op, "card": card, "partner": partner}


def build_slots(seeds: list[dict]) -> list[dict]:
    gsm = sorted(s["seed_id"] for s in seeds if s.get("dataset") == "gsm8k")
    out = []
    for s in sorted(seeds, key=lambda s: (s["source"], s["index"])):
        for j in range(1, J + 1):
            key = f"{s['seed_id']}:{j}"
            d = slot_draw(s["seed_id"], s.get("dataset") or "gsm8k", j, gsm)
            out.append({"key": key, "seed_id": s["seed_id"], "source": s["source"], "dataset": s.get("dataset"),
                        "index": s["index"], "slot": j, **d,
                        "_h": hashlib.sha1(f"{VERSION}|{key}".encode()).hexdigest()})
    out.sort(key=lambda r: (r["slot"], r["_h"]))
    for i, r in enumerate(out):
        r["rank"] = i
        del r["_h"]
    return out


def create_shared(pool: Path = POOL, path: Path = SLOTS) -> dict:
    """Create data/train/slots_v1.jsonl exclusively (never overwrite: G4 may have created it first)."""
    if path.exists():
        return {"created": False, "path": str(path), "n": sum(1 for _ in open(path, encoding="utf-8"))}
    seeds = read_jsonl(pool)
    slots = build_slots(seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.d1tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in slots:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    try:
        os.link(tmp, path)            # atomic create-if-absent
    except FileExistsError:
        os.remove(tmp)
        return {"created": False, "path": str(path), "note": "created concurrently by another agent"}
    os.remove(tmp)
    meta = {"version": VERSION, "created_by": "D1 (vi_math_verified_aug/scale_gen/slots.py)", "rule": __doc__,
            "pool": str(pool.relative_to(pool.parents[2])).replace("\\", "/"), "pool_sha256": sha256_file(pool),
            "slots_sha256": sha256_file(path), "n_seeds": len(seeds), "n_slots": len(slots),
            "ops": dict(sorted(Counter(s["op"] for s in slots).items())),
            "ops_first_10k": dict(sorted(Counter(s["op"] for s in slots[:10000]).items())),
            "ops_first_20k": dict(sorted(Counter(s["op"] for s in slots[:20000]).items()))}
    SLOTS_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"created": True, **{k: v for k, v in meta.items() if k != "rule"}}


def load_slots(path: Path = SLOTS) -> list[dict]:
    return sorted(read_jsonl(path), key=lambda r: r["rank"])
