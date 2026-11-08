"""G4: bge-m3 embeddings of the E/R1/R2 items and their seeds on Modal (one small L4/A10G call; same encoder
settings as X5 and G3: BAAI/bge-m3, CLS pooling, max_length 256, bf16, L2-normalised).

    modal run -m vi_math_verified_aug.pilot_symbolic_v2.embed_modal    # -> data/pilot/symbolic_v2/emb.json
Budget guard: timeout 600 s (<= $0.14 on L4 incl. cold start); a few hundred short texts take seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("g4-embed")
image = (                                          # identical to G3's pilot_symbolic/embed_modal.py image
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1", "transformers==5.22.2", "numpy==2.4.6", "huggingface_hub==2.4.0",
                 "sentencepiece==0.2.3")
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("vi_math_verified_aug")
)


@app.function(gpu=["L4", "A10G"], image=image, timeout=600, retries=0, memory=8192, cpu=2)
def embed(texts: list[str], model: str = "BAAI/bge-m3", max_length: int = 256) -> list:
    import subprocess

    import numpy as np

    from vi_math_verified_aug.eval import decontam as dc
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    print("ENVCHECK", smi, flush=True)
    enc = dc.Encoder(model, batch_size=64, max_length=max_length, dtype="bf16")
    E = enc.encode(texts).astype(np.float32)
    return [[round(float(x), 5) for x in row] for row in E]


@app.local_entrypoint()
def main():
    root = Path("data/pilot/symbolic_v2")
    seeds = [json.loads(l) for l in open("data/pilot/symbolic/seeds.jsonl", encoding="utf-8")]
    keys, texts = [], []
    for s in seeds:
        keys.append("seed:" + s["seed_id"])
        texts.append(s["text"])
    for v in ("E", "R1", "R2", "R1b"):
        rows = {}
        for l in open(root / f"render_{v}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            rows[r["key"]] = r
        for k, r in sorted(rows.items()):
            if r.get("tag_ok") and r.get("text"):
                keys.append(f"item:{v}:{k}")
                texts.append(r["text"])
    E = embed.remote(texts)
    json.dump({"model": "BAAI/bge-m3", "max_length": 256, "keys": keys, "emb": E}, open(root / "emb.json", "w"))
    print("embedded", len(E), len(E[0]))
