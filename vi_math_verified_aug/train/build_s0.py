"""Build the S0 seed SFT pool + a held-out dev300 for the first real runs (T1r, 2026-11-07).

    python -m vi_math_verified_aug.train.build_s0 [--out data/t1r]

Source: ``data/translation/gsm8k/gsm8k_train_vietnamese.jsonl`` (Vi-GSM8K train, 7,473 rows; content review R1).

* Excluded: the 14 corrupt rows of R1 §6 (220-225, 2577, 3077, 2345, 3539, 5085, 7454, 1801, 2911) and the 2
  half-English rows R1 also flags (3424, 6465) -- the same 16 rows G1/G2/G3 exclude.  Not hand-fixed.
* Solutions: GSM8K calculator annotations ``<<a*b=c>>`` are stripped (as in S2's pilot); the trailing ``#### x``
  line is replaced by the template's ``Đáp số: \\boxed{x}`` at tokenisation (``template.completion``), with
  ``x`` = the row's integer ``ground_truth`` (never the ``####`` text, which has 79 ``1,080``-style lines).
* S0 train excludes the DEV seed bucket (``sha1("vi_gsm8k_train||i")[:12] % 20 == 0`` = S3's dev_v1 bucket; QA-T
  HIGH 1, 2026-11-08): the arms' dev set (dev_v1) must never be trained on.
* dev300: ``random.Random(20261107)`` draw of 300 rows from the clean rows that are NOT in the augmentation-seed
  pool (``sha1("vi_gsm8k_train||i")[:12] % 20 == 0``, used by G1/G2/G3) and not S2's few-shot exemplars, so later
  augmented arms can never contain a dev item's seed.  dev300 is removed from the S0 train pool.

Outputs: ``s0_train.jsonl`` (id, question, solution, answer_canonical), ``dev300.jsonl`` (+ typed ``gold``),
``fewshot4_vi.jsonl`` (S2's 4 exemplars, for 4-shot base rows), ``MANIFEST.json`` (counts + sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "translation" / "gsm8k" / "gsm8k_train_vietnamese.jsonl"
SRC_EN = REPO / "data" / "generate" / "gsm8k" / "train.jsonl"   # English originals, aligned by `index`
FEWSHOT = REPO / "jobs" / "build" / "s2-modal-data" / "fewshot4.json"
R1_CORRUPT = {220, 221, 222, 223, 224, 225, 2577, 3077, 2345, 3539, 5085, 7454, 1801, 2911}
R1_HALF_ENGLISH = {3424, 6465}
SEED = 20261107
N_DEV = 300
_ANN = re.compile(r"<<[^<>]*>>")


def aug_seed_pool(i: int) -> bool:
    return int(hashlib.sha1(f"vi_gsm8k_train||{i}".encode()).hexdigest()[:12], 16) % 20 == 0


def clean_solution(ans: str) -> str:
    return _ANN.sub("", ans).rstrip()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in SRC.open(encoding="utf-8") if line.strip()]
    assert len(rows) == 7473 and [r["index"] for r in rows] == list(range(7473))
    shots = json.loads(FEWSHOT.read_text(encoding="utf-8"))
    shot_idx = {s["train_index"] for s in shots}
    clean = [r for r in rows if r["index"] not in R1_CORRUPT | R1_HALF_ENGLISH]
    for r in clean:
        assert re.fullmatch(r"-?\d+", str(r["ground_truth"])), r["index"]
    dev_pool = sorted(r["index"] for r in clean if not aug_seed_pool(r["index"]) and r["index"] not in shot_idx)
    dev = set(random.Random(SEED).sample(dev_pool, N_DEV))

    def item(r):
        return {"id": f"vi_gsm8k_train/{r['index']}", "question": r["question"].strip(),
                "solution": clean_solution(r["answer"]), "answer_canonical": str(r["ground_truth"])}

    with (out / "s0_train.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in clean:
            if r["index"] not in dev and not aug_seed_pool(r["index"]):  # DEV bucket (dev_v1) never trains
                f.write(json.dumps(item(r), ensure_ascii=False) + "\n")
    with (out / "dev300.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in clean:
            if r["index"] in dev:
                it = item(r)
                it.update(dataset="vi_gsm8k_train_dev",
                          gold={"answer_canonical": it["answer_canonical"], "answer_type": "int", "locale": "mixed"})
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with (out / "fewshot4_vi.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for s in shots:  # S2's solutions already end in "Đáp số: \boxed{x}" -> pass the answer explicitly
            m = re.search(r"\\boxed\{([^{}]*)\}\s*$", s["solution"])
            f.write(json.dumps({"id": f"vi_gsm8k_train/{s['train_index']}", "question": s["problem"],
                                "solution": s["solution"], "answer": m.group(1) if m else None},
                               ensure_ascii=False) + "\n")
    # English mirror (E1 R1: the same seed items in English): same indices, same dev300, same exemplars
    en = {r["index"]: r for r in (json.loads(x) for x in SRC_EN.open(encoding="utf-8") if x.strip())}
    assert len(en) == 7473
    with (out / "s0_train_en.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in clean:
            if r["index"] not in dev and not aug_seed_pool(r["index"]):  # DEV bucket (dev_v1) never trains
                f.write(json.dumps(item(en[r["index"]]), ensure_ascii=False) + "\n")
    with (out / "dev300_en.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in clean:
            if r["index"] in dev:
                it = item(en[r["index"]])
                it.update(dataset="vi_gsm8k_train_dev",
                          gold={"answer_canonical": it["answer_canonical"], "answer_type": "int", "locale": "en"})
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with (out / "fewshot4_en.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for s_ in shots:
            it = item(en[s_["train_index"]])
            f.write(json.dumps({"id": it["id"], "question": it["question"], "solution": it["solution"],
                                "answer": it["answer_canonical"]}, ensure_ascii=False) + "\n")
    man = {"source": str(SRC.relative_to(REPO)), "source_en": str(SRC_EN.relative_to(REPO)), "source_sha256": sha(SRC), "rows_source": len(rows),
           "excluded_r1": sorted(R1_CORRUPT | R1_HALF_ENGLISH), "rows_clean": len(clean),
           "n_train": len(clean) - N_DEV, "n_dev": N_DEV, "dev_seed": SEED,
           "dev_excludes": "augmentation-seed pool (sha1 % 20 == 0) and S2 few-shot exemplars",
           "fewshot_source": str(FEWSHOT.relative_to(REPO)),
           "files": {p.name: sha(p) for p in sorted(out.glob("*.jsonl"))}}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "t1r"))
    print(json.dumps(build(Path(ap.parse_args().out)), indent=1))
