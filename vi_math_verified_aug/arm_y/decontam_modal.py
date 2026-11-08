"""G5 decontamination of the arm-Y pools on Modal (X5 tool: ``eval.decontam.Encoder``, BAAI/bge-m3, CLS, max_len 256,
bf16, L2-normalised; X5 tau = 0.85). Targets = S3's frozen eval-target list (data/seeds/_work/embed/inputs.json
"targets": Vi/EN GSM8K-Platinum, MATH alg/prealg, MATH-500, primary-1000 subsamples, Vi-ExamMath; 13,655 texts).
Each item is compared with the targets of its own language; only the per-item max cosine and argmax id come back.

    modal run -m vi_math_verified_aug.arm_y.decontam_modal --items <items.json> --out <out.json>
items.json: [{"id": ..., "text": ..., "lang": "vi"|"en"}, ...]. Cap: G4 decontam $1.5; timeout 1800 s, retries 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("g5-decontam")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1", "transformers==5.22.2", "numpy==2.4.6", "huggingface_hub==2.4.0",
                 "sentencepiece==0.2.3")
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("vi_math_verified_aug")
)


@app.function(gpu=["L4", "A10G"], image=image, timeout=1800, retries=0, memory=16384, cpu=4)
def scan(items: list[dict], targets: list[dict]) -> list[dict]:
    import subprocess

    import numpy as np

    from vi_math_verified_aug.eval import decontam as dc
    print("ENVCHECK", subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip(), flush=True)
    enc = dc.Encoder("BAAI/bge-m3", batch_size=128, max_length=256, dtype="bf16")
    T = enc.encode([t["text"] for t in targets]).astype(np.float32)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    I = enc.encode([i["text"] for i in items]).astype(np.float32)
    I /= np.linalg.norm(I, axis=1, keepdims=True)
    out = []
    for lang in sorted({i["lang"] for i in items}):
        ti = [k for k, t in enumerate(targets) if t["lang"] == lang]
        ii = [k for k, i in enumerate(items) if i["lang"] == lang]
        if not ti or not ii:
            continue
        S = I[ii] @ T[ti].T
        am = S.argmax(axis=1)
        for r, k in enumerate(ii):
            out.append({"id": items[k]["id"], "cos_max": round(float(S[r, am[r]]), 4),
                        "tid": targets[ti[int(am[r])]]["tid"]})
    return out


@app.local_entrypoint()
def main(items: str, out: str):
    tg = json.load(open("data/seeds/_work/embed/inputs.json", encoding="utf-8"))["targets"]
    its = json.load(open(items, encoding="utf-8"))
    res = scan.remote(its, [{"tid": t["tid"], "text": t["text"], "lang": t["lang"]} for t in tg])
    Path(out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print("scanned", len(res), "max", max(r["cos_max"] for r in res), ">=0.85:", sum(r["cos_max"] >= 0.85 for r in res))
