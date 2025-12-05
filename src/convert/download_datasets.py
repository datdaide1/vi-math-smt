"""
Download the four source math benchmarks used throughout this project.

This is reference code — it is not executed as part of the pipeline. Run it
manually once to (re)populate data/raw/ before running src/formalize/ or
src/translate/.

Sources
-------
GSM8K   — Cobbe et al. (2021), "Training Verifiers to Solve Math Word Problems"
          HF: openai/gsm8k, config "main". 7,473 train / 1,319 test.
MATH    — Hendrycks et al. (2021), "Measuring Mathematical Problem Solving
          With the MATH Dataset" (arXiv:2103.03874). 7,500 train / 5,000 test,
          7 subjects x 5 difficulty levels. Original release:
          https://github.com/hendrycks/math ; also mirrored on HF as
          hendrycks/competition_math (fields: problem, level, type, solution).
ASDiv   — Miao et al. (2020), "A Diverse Corpus for Evaluating and Developing
          English Math Word Problem Solvers". HF: EleutherAI/asdiv
          (validation split only, 2,305 examples).
SVAMP   — Patel et al. (2021), "Are NLP Models really able to Solve Simple
          Math Word Problems?". HF: ChilleD/SVAMP (700 train / 300 test).

Output layout (under data/raw/, gitignored):
  data/raw/gsm8k/{train,test}.jsonl
  data/raw/math/{subject}/{train,test}.json      (grouped like the original
                                                    Hendrycks release, since
                                                    src/formalize/ and
                                                    src/translate/ expect
                                                    one JSON array per subject)
  data/raw/asdiv/validation.jsonl
  data/raw/svamp/{train,test}.jsonl
"""

import json
import os

from datasets import load_dataset

# Assumes this script is run from its own directory (src/convert/)
RAW_DIR = os.path.join("..", "..", "data", "raw")


def _write_jsonl(dataset, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.to_json(path, force_ascii=False)
    print(f"wrote {path} ({len(dataset)} rows)")


def download_gsm8k():
    ds = load_dataset("openai/gsm8k", "main")
    _write_jsonl(ds["train"], os.path.join(RAW_DIR, "gsm8k", "train.jsonl"))
    _write_jsonl(ds["test"], os.path.join(RAW_DIR, "gsm8k", "test.jsonl"))


def download_asdiv():
    ds = load_dataset("EleutherAI/asdiv")
    _write_jsonl(ds["validation"], os.path.join(RAW_DIR, "asdiv", "validation.jsonl"))


def download_svamp():
    ds = load_dataset("ChilleD/SVAMP")
    _write_jsonl(ds["train"], os.path.join(RAW_DIR, "svamp", "train.jsonl"))
    _write_jsonl(ds["test"], os.path.join(RAW_DIR, "svamp", "test.jsonl"))


def download_math():
    """
    MATH ships as one JSON array per subject/split in this pipeline
    (matches the folder layout the original Hendrycks release uses, and what
    src/formalize/ and src/translate/ expect). The HF mirror
    "hendrycks/competition_math" exposes flat train/test splits with a
    "type" field for the subject — split it back into per-subject files here.
    """
    ds = load_dataset("hendrycks/competition_math")
    for split in ("train", "test"):
        by_subject = {}
        for row in ds[split]:
            subject = row["type"].lower().replace(" ", "_").replace("&", "and")
            by_subject.setdefault(subject, []).append(row)
        for subject, rows in by_subject.items():
            out_path = os.path.join(RAW_DIR, "math", split, f"{subject}.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    download_gsm8k()
    download_math()
    download_asdiv()
    download_svamp()
