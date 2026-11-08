"""S3 seed-pool embedding stage on Modal (bge-m3, the X5-calibrated encoder; L4, bf16; X5 ``Encoder`` reused).

    E:\\venv_tools\\Scripts\\modal run -m vi_math_verified_aug.seeds.embed_modal          # from the repo root

Input: ``data/seeds/_work/embed/inputs.json`` (``build embed-inputs``): seeds {seed_id, vi, en} and targets
{tid, lang, text} (data/eval/* + Vi-ExamMath). For each (seed side, target language) pair the job returns the top-3
target cosines of every seed text; the local ``decontam`` stage applies the X5 thresholds.

Budget guard: function timeout 900 s (L4 $0.80/h -> <= $0.20 worst case); expected ~4 min. Image = X5's pinned image
(torch 2.14.1, transformers 5.22.2, numpy 2.4.6), so the build is cached. The result is also committed to the
``s3-embed`` volume.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("s3-seed-embed")
vol = modal.Volume.from_name("s3-embed", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1", "transformers==5.22.2", "pyarrow==26.0.1", "numpy==2.4.6",
                 "huggingface_hub==2.4.0", "sentencepiece==0.2.3")
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("vi_math_verified_aug")
)


@app.function(gpu="L4", image=image, timeout=900, retries=0, volumes={"/out": vol}, memory=8192, cpu=2)
def run(inputs_json: bytes, model: str = "BAAI/bge-m3", max_length: int = 256, batch: int = 256, k: int = 3) -> dict:
    import subprocess

    import numpy as np
    import torch

    from vi_math_verified_aug.eval import decontam as dc

    t0 = time.time()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    import transformers
    print(f"ENVCHECK gpu={smi} torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"transformers={transformers.__version__} numpy={np.__version__} model={model}", flush=True)
    inp = json.loads(inputs_json)
    seeds, targets = inp["seeds"], inp["targets"]
    enc = dc.Encoder(model, batch_size=batch, max_length=max_length, dtype="bf16")
    print(f"[{time.time() - t0:.0f}s] encoder loaded", flush=True)
    tids = {"vi": [t["tid"] for t in targets if t["lang"] == "vi"],
            "en": [t["tid"] for t in targets if t["lang"] == "en"]}
    TE = {lang: torch.from_numpy(enc.encode([t["text"] for t in targets if t["lang"] == lang]).astype(np.float32)
                                 ).cuda() for lang in ("vi", "en")}
    print(f"[{time.time() - t0:.0f}s] targets {[(l, tuple(v.shape)) for l, v in TE.items()]}", flush=True)
    SE = {side: torch.from_numpy(enc.encode([s[side] for s in seeds]).astype(np.float32)).cuda()
          for side in ("vi", "en")}
    print(f"[{time.time() - t0:.0f}s] seeds encoded", flush=True)
    pairs = {}
    for side in ("vi", "en"):
        for lang in ("vi", "en"):
            S = SE[side] @ TE[lang].T
            v, i = torch.topk(S, k, dim=1)
            pairs[f"{side}>{lang}"] = {"cos": v.cpu().numpy().round(4).tolist(), "idx": i.cpu().numpy().tolist()}
    out = {"model": model, "max_length": max_length, "env": smi, "seed_ids": [s["seed_id"] for s in seeds],
           "target_ids": tids, "pairs": pairs, "seconds": round(time.time() - t0, 1),
           "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)}
    Path("/out/result.json").write_text(json.dumps(out), encoding="utf-8")
    vol.commit()
    print(f"[{time.time() - t0:.0f}s] done", flush=True)
    return out


@app.local_entrypoint()
def main(inputs: str = "data/seeds/_work/embed/inputs.json"):
    data = Path(inputs).read_bytes()
    out = run.remote(data)
    p = Path("data/seeds/_work/embed/result.json")
    p.write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("pairs", "seed_ids", "target_ids")}))
