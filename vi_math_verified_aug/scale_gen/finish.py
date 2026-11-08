"""Decontaminate + freeze one language (run in the project .venv: needs numpy; embeddings run on Modal).

    .venv\\Scripts\\python -m vi_math_verified_aug.scale_gen.finish --lang vi --cfg A --n 13500 [--provisional] [--no-embed]
"""
from __future__ import annotations

import argparse
import json

from . import decontam as D
from . import finalize as F
from .common import POOL, SLOTS, WORK, Store, read_jsonl


def _slots(provisional: bool) -> list[dict]:
    if provisional:
        from .run import provisional_seeds
        from .slots import build_slots
        return build_slots(list(provisional_seeds().values()))
    from .slots import load_slots
    return load_slots(SLOTS)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--cfg", default="A")
    ap.add_argument("--n", type=int, default=13500)
    ap.add_argument("--provisional", action="store_true")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--k", type=int, default=F.K)
    ap.add_argument("--retry-slots", default=None, help="file of retry slot keys (attempt_no=2, plus-only)")
    ap.add_argument("--slots-from", default=None, help="restrict to the slots named in this file (D5 list), see "
                    "run.select_slots; applied to the first --n slots of slots_v1")
    a = ap.parse_args(argv)
    slots = _slots(a.provisional)[:a.n]
    if a.slots_from:
        from .run import select_slots
        slots = select_slots(slots, a.slots_from)
    gen = Store(WORK / a.lang / "gen.jsonl")
    solve = Store(WORK / a.lang / "solve.jsonl")
    dec = Store(WORK / a.lang / "decontam.jsonl")
    items = []
    for s in slots:
        k = f"{a.lang}|{a.cfg}|{s['key']}"
        g, sv = gen.get(k), solve.get(k) or {}
        if g and g.get("fmt_ok") and sv.get("api_ok") and sv.get("boxed"):
            items.append((k, g["problem"]))
    # every item is re-checked lexically (MinHash dedup depends on rank order); embeddings only for new texts
    tag = a.tag or f"{a.lang}_{a.cfg}_{a.n}"
    done = {k for k, _ in items if (dec.get(k) or {}).get("embedded")}
    print(json.dumps({"items": len(items), "already_embedded": len(done)}), flush=True)
    st = D.apply(a.lang, items, tag, embed=not a.no_embed)
    seeds = None
    if not a.provisional:
        from .run import pool_seeds
        seeds = pool_seeds()
    retry = set()
    if a.retry_slots:
        retry = {l.strip() for l in open(a.retry_slots, encoding="utf-8") if l.strip()}
    man = F.build(a.lang, a.cfg, slots, require_embed=not a.no_embed, k=a.k, seeds=seeds, retry_slots=retry)
    print(json.dumps({"counts": man["counts"], "subsample_k": man["subsample_k"], "files": man["files"]}, indent=1))


if __name__ == "__main__":
    main()
