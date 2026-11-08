"""D1 decontamination embeddings on Modal (bge-m3, the X5-calibrated encoder, CLS, max_len 256, bf16).

    E:\\venv_tools\\Scripts\\modal run -m vi_math_verified_aug.scale_gen.embed_modal --inputs <inputs.json> --out <result.json>

Input JSON: {"items": [{"id", "lang", "text"}], "targets": [{"tid", "lang", "text"}]} (targets = S3/X5's list:
every data/eval/*.jsonl item + Vi-ExamMath). Output: per item, the top-1 cosine and target id against the VI targets and
against the EN targets. Image = X5's pinned image (cached build). GPU fallback list, retries=0, 900 s timeout
(<= $0.30 worst case on L4); D1 Modal cap $1.5.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("d1-n-decontam")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1", "transformers==5.22.2", "pyarrow==26.0.1", "numpy==2.4.6",
                 "huggingface_hub==2.4.0", "sentencepiece==0.2.3")
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    # ship ONLY X5's self-contained decontam.py (stdlib + numpy): shipping the whole package fails whenever another
    # workstream edits any file of it during the image build ("modified during build process")
    .add_local_file(str(Path(__file__).resolve().parents[1] / "eval" / "decontam.py"), "/opt/x5/eval/x5_decontam.py")  # its REPO = parents[2]
)


@app.function(gpu=["L4", "A10G", "L40S"], image=image, timeout=900, retries=0, memory=8192, cpu=2)
def run(inputs_json: bytes, model: str = "BAAI/bge-m3", max_length: int = 256, batch: int = 256) -> dict:
    import subprocess

    import numpy as np
    import torch

    import sys
    sys.path.insert(0, "/opt/x5/eval")
    import x5_decontam as dc

    t0 = time.time()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print(f"ENVCHECK gpu={smi} torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    inp = json.loads(inputs_json)
    items, targets = inp["items"], inp["targets"]
    enc = dc.Encoder(model, batch_size=batch, max_length=max_length, dtype="bf16")
    TE = {lang: torch.from_numpy(enc.encode([t["text"] for t in targets if t["lang"] == lang]).astype(np.float32)
                                 ).cuda() for lang in ("vi", "en")}
    tids = {lang: [t["tid"] for t in targets if t["lang"] == lang] for lang in ("vi", "en")}
    print(f"[{time.time() - t0:.0f}s] targets encoded", flush=True)
    IE = torch.from_numpy(enc.encode([i["text"] for i in items]).astype(np.float32)).cuda()
    print(f"[{time.time() - t0:.0f}s] items encoded n={len(items)}", flush=True)
    res = {}
    for lang in ("vi", "en"):
        S = IE @ TE[lang].T
        v, ix = torch.max(S, dim=1)
        res[lang] = {"cos": v.cpu().numpy().round(4).tolist(), "tid": [tids[lang][j] for j in ix.cpu().numpy()]}
    return {"model": model, "max_length": max_length, "env": smi, "ids": [i["id"] for i in items], "top1": res,
            "seconds": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(inputs: str, out: str):
    data = Path(inputs).read_bytes()
    r = run.remote(data)
    Path(out).write_text(json.dumps(r), encoding="utf-8")
    print(json.dumps({k: v for k, v in r.items() if k not in ("ids", "top1")}))
