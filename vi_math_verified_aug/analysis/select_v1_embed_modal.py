"""D5: bge-m3 embeddings of the selected training items (final/, final_natural/, final_shared/) and their seeds, for the
select_v1 diversity report. Same encoder settings as X5/G3/G4 (BAAI/bge-m3, CLS pooling, max_length 256, bf16,
L2-normalised). Training items only (no eval/benchmark text is sent).

    E:\\venv_tools\\Scripts\\modal run -m vi_math_verified_aug.analysis.select_v1_embed_modal --lang vi
      -> data/audit/d5/select_emb_<lang>.npy (float16, n x 1024) + select_emb_<lang>.keys.json
Budget guard: GPU fallback [L4, A10G], retries=0, timeout 900 s (<= $0.28 worst case on A10G; ~25k short texts take
~2 min). Log ``modal billing`` before/after (D5 cap for this step: $0.30).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("d5-select-embed")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1", "transformers==5.22.2", "numpy==2.4.6", "huggingface_hub==2.4.0",
                 "sentencepiece==0.2.3")
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("vi_math_verified_aug")
)


@app.function(gpu=["L4", "A10G"], image=image, timeout=900, retries=0, memory=8192, cpu=2)
def embed(texts: list[str], model: str = "BAAI/bge-m3", max_length: int = 256) -> bytes:
    import subprocess

    import numpy as np

    from vi_math_verified_aug.eval import decontam as dc
    print("ENVCHECK", subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip(), flush=True)
    enc = dc.Encoder(model, batch_size=64, max_length=max_length, dtype="bf16")
    return enc.encode(texts).astype(np.float16).tobytes()


def collect(lang: str) -> tuple[list[str], list[str]]:
    """keys = 'item:<sha1(text)>' / 'seed:<seed_id>', deduplicated."""
    from vi_math_verified_aug.analysis.select_v1 import FINAL, NATURAL, SHARED, read_jsonl
    from vi_math_verified_aug.analysis.select_v1_diversity import seed_texts
    keys, texts, seen, seeds = [], [], set(), set()
    for base in (FINAL, NATURAL, SHARED):
        for x in ("Y", "N"):
            for s in ("plus", "minus"):
                p = base / x / lang / f"train_{s}.jsonl"
                if not p.exists():
                    continue
                for r in read_jsonl(p):
                    q = r.get("question") or ""
                    k = "item:" + hashlib.sha1(q.encode("utf-8")).hexdigest()
                    seeds.add(r.get("seed_id"))
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
                        texts.append(q)
    st = seed_texts(lang, seeds)
    for sid in sorted(st):
        keys.append("seed:" + sid)
        texts.append(st[sid])
    return keys, texts


@app.local_entrypoint()
def main(lang: str = "vi"):
    import numpy as np
    keys, texts = collect(lang)
    print(f"{lang}: {len(texts)} texts to embed", flush=True)
    E = np.frombuffer(embed.remote(texts), dtype=np.float16).reshape(len(texts), -1)
    out = Path("data/audit/d5")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"select_emb_{lang}.npy", E)
    (out / f"select_emb_{lang}.keys.json").write_text(json.dumps({"model": "BAAI/bge-m3", "max_length": 256,
                                                                  "keys": keys}), encoding="utf-8")
    print("embedded", E.shape)
