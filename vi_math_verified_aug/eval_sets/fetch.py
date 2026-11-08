"""Fetch the English sources at pinned HF revisions and convert them to JSONL.

Run with the tools venv (needs ``pyarrow`` + ``requests``)::

    E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.eval_sets.fetch

Writes ``data/eval/_sources/<name>.jsonl`` and ``data/eval/_sources/manifest.json`` (URL, revision, raw-file
sha256, JSONL sha256, row count).  Parquet is parsed as data only; nothing downloaded is executed.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "eval" / "_sources"

# (name, repo, revision, path-in-repo)
SOURCES = [
    ("gsm8k_platinum_test", "madrylab/gsm8k-platinum", "e762492455a1cf7967de89f05b6bef72fc713b66",
     "main/test-00000-of-00001.parquet"),
    ("gsm8k_test", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
     "main/test-00000-of-00001.parquet"),
    ("hendrycks_math_algebra_test", "EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3",
     "algebra/test-00000-of-00001.parquet"),
    ("hendrycks_math_prealgebra_test", "EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3",
     "prealgebra/test-00000-of-00001.parquet"),
    ("math500_test", "HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test.jsonl"),
]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": []}
    for name, repo, rev, path in SOURCES:
        url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        raw = r.content
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq

            rows = pq.read_table(io.BytesIO(raw)).to_pylist()
        else:
            rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        dst = OUT / f"{name}.jsonl"
        text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        dst.write_text(text, encoding="utf-8", newline="\n")
        manifest["files"].append({
            "name": name, "repo": repo, "revision": rev, "path": path, "url": url,
            "raw_sha256": _sha(raw), "jsonl": dst.name, "jsonl_sha256": _sha(text.encode("utf-8")),
            "rows": len(rows), "columns": sorted(rows[0].keys()) if rows else [],
        })
        print(f"{name}: {len(rows)} rows, columns={sorted(rows[0].keys())}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
