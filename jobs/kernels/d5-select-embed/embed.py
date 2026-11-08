
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
