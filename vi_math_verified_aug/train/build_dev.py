"""Arm-run dev sets from S3's DEV seed bucket (QA-T HIGH 1, lead 2026-11-08).

    python -m vi_math_verified_aug.train.build_dev [--src data/seeds/dev_v1.jsonl] [--out data/t1r]

The DEV bucket (``int(sha1(f"{source}||{index}")[:12], 16) % 20 == 0``) is excluded from every generator, so no
Y/N item is a mutation of a dev item (dev300 overlapped ~4 % of N/Y seeds).  Outputs, one row per dev seed, the same
ids in both languages (EN mirror):
  dev_seed_vi.jsonl / dev_seed_en.jsonl : id (= seed_id), question, solution (GSM8K calculator annotations removed),
                                          answer_canonical, gold (typed, from S3), dataset (dev_seed_<gsm8k|math>)
  dev_seed.MANIFEST.json                : source sha256, counts per dataset, eval-overlap check, output sha256
Rows whose question hash equals an eval / benchmark question (data/t1r/eval_qhashes.json) are refused (exit 1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from . import guards as G

REPO = Path(__file__).resolve().parents[2]
_ANN = re.compile(r"<<[^<>]*>>")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(src: Path, out: Path) -> dict:
    rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    hashes = set(json.loads((REPO / "data" / "t1r" / "eval_qhashes.json").read_text(encoding="utf-8")))
    vi, en, bad = [], [], []
    for r in rows:
        if r.get("split", "dev") != "dev":
            continue
        g = r["gold"]
        ans = g.get("answer_canonical", g.get("value")) if isinstance(g, dict) else str(g)
        ds = f"dev_seed_{r.get('dataset') or 'unknown'}"
        for lang, q, sol, lst in (("vi", r["problem"], r["solution"], vi),
                                  ("en", (r.get("en") or {}).get("problem"), (r.get("en") or {}).get("solution"), en)):
            if not q or not sol:
                continue
            if G.qhash(q) in hashes:
                bad.append({"seed_id": r["seed_id"], "lang": lang})
                continue
            gold = dict(g) if isinstance(g, dict) else {"answer_canonical": ans, "answer_type": "int"}
            gold["locale"] = "en" if lang == "en" else gold.get("locale", "mixed")
            lst.append({"id": r["seed_id"], "question": q.strip(), "solution": _ANN.sub("", sol).rstrip(),
                        "answer_canonical": str(ans), "gold": gold, "dataset": ds})
    common = {x["id"] for x in vi} & {x["id"] for x in en}   # EN mirror on the same ids
    vi = [x for x in vi if x["id"] in common]
    en = [x for x in en if x["id"] in common]
    out.mkdir(parents=True, exist_ok=True)
    for name, lst in (("dev_seed_vi.jsonl", vi), ("dev_seed_en.jsonl", en)):
        with (out / name).open("w", encoding="utf-8", newline="\n") as f:
            for x in lst:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    man = {"source": str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src), "source_sha256": _sha(src),
           "n": len(vi), "per_dataset": dict(Counter(x["dataset"] for x in vi)), "eval_overlap_refused": bad,
           "files": {n: _sha(out / n) for n in ("dev_seed_vi.jsonl", "dev_seed_en.jsonl")}}
    (out / "dev_seed.MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "data" / "seeds" / "dev_v1.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data" / "t1r"))
    a = ap.parse_args()
    m = build(Path(a.src), Path(a.out))
    print(json.dumps(m, indent=1))
    sys.exit(1 if m["eval_overlap_refused"] else 0)
