"""Build the PRIVATE Kaggle input dataset for the W2 day-1 student pilot.

Contents (nothing else; Vi-GSM8K test is uploaded only as the fixed 300-item subset, private):
  train.jsonl      = data/finetune/gsm8k/gsm8k_train_vietnamese_final.jsonl (verbatim)
  test300.jsonl    = 300 items of data/test/gsm8k/gsm8k_test_vietnamese.jsonl, random.Random(20261107).sample
  fewshot4.json    = 4 exemplars drawn from TRAIN with the same seed, in the plain-text template
  math_verify.py   = frozen snapshot of vi_math_verified_aug/eval/math_verify.py (W1 owns the source)
  MANIFEST.json    = sha256 of every file + source paths + selection rule

Usage: python -m vi_math_verified_aug.infra.pilot.build_pilot_data OUT_DIR
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SEED = 20261107
TRAIN = REPO / "data/finetune/gsm8k/gsm8k_train_vietnamese_final.jsonl"
TEST = REPO / "data/test/gsm8k/gsm8k_test_vietnamese.jsonl"
GRADER = REPO / "vi_math_verified_aug/eval/math_verify.py"


def clean_solution(sol: str, gt: str) -> str:
    """GSM8K solution -> template body: drop <<calc>> annotations, '#### x' -> 'Đáp số: \\boxed{x}'."""
    s = re.sub(r"<<[^<>]*>>", "", sol)
    s = s.rsplit("####", 1)[0].rstrip()
    return f"{s}\nĐáp số: \\boxed{{{gt}}}"


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(out: str):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    test = read_jsonl(TEST)
    idx = sorted(random.Random(SEED).sample(range(len(test)), 300))
    with open(out / "test300.jsonl", "w", encoding="utf-8") as f:
        for i in idx:
            r = test[i]
            f.write(json.dumps({"test_index": i, "question": r["question"], "ground_truth": r["ground_truth"]},
                               ensure_ascii=False) + "\n")
    train = read_jsonl(TRAIN)
    rng = random.Random(SEED)
    # 4 exemplars, short-ish (<= 400 chars of solution) so the 4-shot prompt stays compact
    pool = [i for i, r in enumerate(train) if len(r["solution"]) <= 400]
    shots = [train[i] for i in rng.sample(pool, 4)]
    fewshot = [{"train_index": r["index"], "problem": r["problem"],
                "solution": clean_solution(r["solution"], r["ground_truth"])} for r in shots]
    (out / "fewshot4.json").write_text(json.dumps(fewshot, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy2(TRAIN, out / "train.jsonl")
    shutil.copy2(GRADER, out / "math_verify.py")
    man = {"seed": SEED, "test_source": str(TEST.relative_to(REPO)), "test_n_total": len(test),
           "test_indices": idx, "train_source": str(TRAIN.relative_to(REPO)), "train_n": len(train),
           "grader_source": str(GRADER.relative_to(REPO)),
           "files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.suffix in (".jsonl", ".json", ".py")
                     and p.name not in ("MANIFEST.json", "dataset-metadata.json")}}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "test_indices"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
