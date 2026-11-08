"""D5: bge-m3 embeddings for the select_v1 diversity report on a PRIVATE Kaggle T4 kernel (lead 07:10: Modal is at its
credit limit). Training-split text only (selected items + their seeds); no eval/benchmark text.

    .venv\Scripts\python -m vi_math_verified_aug.analysis.select_v1_embed_kaggle prepare      # texts -> private dataset dir
    E:\venv_tools\Scripts\kaggle.exe datasets create -p data/audit/d5/kaggle_ds            # (version: datasets version -m ..)
    .venv\Scripts\python -m vi_math_verified_aug.analysis.select_v1_embed_kaggle kernel       # writes jobs/kernels/d5-select-embed
    E:\venv_tools\Scripts\kaggle.exe kernels push -p jobs/kernels/d5-select-embed --accelerator NvidiaTeslaT4
    E:\venv_tools\Scripts\kaggle.exe kernels output datuni/d5-select-embed -p data/audit/d5/kaggle_out
    .venv\Scripts\python -m vi_math_verified_aug.analysis.select_v1_embed_kaggle fetch        # -> select_emb_<lang>.npy
    .venv\Scripts\python -m vi_math_verified_aug.analysis.select_v1 diversity vi,en
Encoder: BAAI/bge-m3, CLS pooling, max_length 256, fp16 on T4, L2-normalised (X5/G3/G4 settings; fp16 instead of bf16
because T4 has no bf16 -- diversity metrics need no numeric parity with the decontam embeddings).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "audit" / "d5" / "kaggle_ds"
KDIR = ROOT / "jobs" / "kernels" / "d5-select-embed"
OUTD = ROOT / "data" / "audit" / "d5" / "kaggle_out"
OWNER = "datuni"

KERNEL = r'''
import json, glob, os, subprocess, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
print("ENVCHECK", subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(), torch.__version__, flush=True)
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3"); m = AutoModel.from_pretrained("BAAI/bge-m3").half().cuda().eval()
for f in sorted(glob.glob("/kaggle/input/**/texts_*.json", recursive=True)):
    lang = os.path.basename(f)[6:-5]
    d = json.load(open(f, encoding="utf-8")); texts = d["texts"]
    order = np.argsort([-len(t) for t in texts], kind="stable"); out = np.zeros((len(texts), 1024), np.float16)
    with torch.inference_mode():
        for s in range(0, len(texts), 64):
            idx = order[s:s+64]
            b = tok([texts[i][:4000] for i in idx], padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
            e = m(**b).last_hidden_state[:, 0].float(); e = torch.nn.functional.normalize(e, dim=-1)
            out[idx] = e.cpu().numpy().astype(np.float16)
    np.save(f"/kaggle/working/select_emb_{lang}.npy", out)
    json.dump({"model": "BAAI/bge-m3", "max_length": 256, "dtype": "fp16", "keys": d["keys"]}, open(f"/kaggle/working/select_emb_{lang}.keys.json", "w"))
    print("embedded", lang, out.shape, flush=True)
'''


def prepare(langs=("vi", "en")):
    from .select_v1_diversity import collect
    DS.mkdir(parents=True, exist_ok=True)
    for l in langs:
        keys, texts = collect(l)
        (DS / f"texts_{l}.json").write_text(json.dumps({"keys": keys, "texts": texts}, ensure_ascii=False), encoding="utf-8")
        print(l, len(texts))
    (DS / "dataset-metadata.json").write_text(json.dumps({"title": "d5-select-texts", "id": f"{OWNER}/d5-select-texts",
                                                          "licenses": [{"name": "other"}]}), encoding="utf-8")


def kernel():
    KDIR.mkdir(parents=True, exist_ok=True)
    (KDIR / "embed.py").write_text(KERNEL, encoding="utf-8")
    meta = {"id": f"{OWNER}/d5-select-embed", "title": "d5-select-embed", "code_file": "embed.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{OWNER}/d5-select-texts"], "kernel_sources": [],
            "model_sources": [], "competition_sources": []}
    (KDIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(KDIR)


def fetch():
    dst = ROOT / "data" / "audit" / "d5"
    for f in OUTD.glob("select_emb_*"):
        shutil.copy2(f, dst / f.name)
        print("copied", f.name)


if __name__ == "__main__":
    {"prepare": prepare, "kernel": kernel, "fetch": fetch}[sys.argv[1]]()
