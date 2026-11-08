"""Download the public English references used by the W3 builders.

Outputs (data only, never executed) go to ``data/clean/_sources/``:
  gsm8k_platinum_test.jsonl   madrylab/gsm8k-platinum (HF datasets-server, config main, split test)
  math500_test.jsonl          HuggingFaceH4/MATH-500 (HF datasets-server)
  SVAMP.json                  github.com/arkilpatel/SVAMP (raw)
  ASDiv.xml                   github.com/chaochun/nlu-asdiv-dataset (raw)
plus ``_sources/manifest.json`` with URL, sha256, bytes, row count and fetch time.

Run:  python -I -m vi_math_verified_aug.data_fix.fetch_sources
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "clean" / "_sources"

DS_SERVER = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length=100"
RAW = {
    "SVAMP.json": "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json",
    "ASDiv.xml": "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml",
}
HF = {
    "gsm8k_platinum_test.jsonl": ("madrylab/gsm8k-platinum", "main", "test"),
    "math500_test.jsonl": ("HuggingFaceH4/MATH-500", "default", "test"),
}


def _get(url: str, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vi-math-w3/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # pragma: no cover - network
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)
            print("retry", url, e)
    raise RuntimeError("unreachable")


def fetch_hf(name: str, ds: str, cfg: str, split: str) -> dict:
    rows, off, total = [], 0, None
    while total is None or off < total:
        url = DS_SERVER.format(ds=ds, cfg=cfg, split=split, off=off)
        page = json.loads(_get(url).decode("utf-8"))
        total = page["num_rows_total"]
        for r in page["rows"]:
            row = dict(r["row"])
            row["_row_idx"] = r["row_idx"]
            rows.append(row)
        off += 100
        time.sleep(0.5)
    assert len(rows) == total, (len(rows), total)
    p = SRC / name
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    b = p.read_bytes()
    return {"file": name, "url": f"hf://datasets/{ds} (config={cfg}, split={split}) via datasets-server",
            "rows": len(rows), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def fetch_raw(name: str, url: str) -> dict:
    b = _get(url)
    (SRC / name).write_bytes(b)
    return {"file": name, "url": url, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def main() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    man = {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": []}
    for name, (ds, cfg, split) in HF.items():
        man["files"].append(fetch_hf(name, ds, cfg, split))
        print(man["files"][-1])
    for name, url in RAW.items():
        man["files"].append(fetch_raw(name, url))
        print(man["files"][-1])
    (SRC / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
