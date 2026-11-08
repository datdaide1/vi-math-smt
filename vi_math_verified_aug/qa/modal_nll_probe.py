"""QA-T: base-model completion NLL per data source (step-0 loss band and dev/train-ratio guard calibration).

    E:/venv_tools/Scripts/modal.exe run vi_math_verified_aug/qa/modal_nll_probe.py

Qwen3-0.6B-Base (and 1.7B), fp16 on T4 (the Kaggle precision), token-mean completion NLL exactly as train/data.encode +
train/modeling.completion_nll_sum compute it, on: S0-VI, S0-EN, dev300-VI, dev300-EN, and the N-arm VI pilot traces
(blind-solver traces, cleaned like scale_gen.finalize).  Budget: one T4 for a few minutes (< $0.10), retries=0.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import modal

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "transformers==5.20.2", "accelerate", "pyyaml")
         .add_local_dir(str(PKG), "/root/vi_math_verified_aug", ignore=["**/__pycache__/**"]))
app = modal.App("qa-t-nllprobe", image=image)


def _rows(path: Path, n: int | None = None) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if n and len(out) >= n:
                    break
    return out


@app.function(gpu=["T4", "L4"], timeout=1800, retries=0)
def probe(payload: dict) -> dict:
    import sys
    import torch
    sys.path.insert(0, "/root")
    from vi_math_verified_aug.train import data as D, modeling as M
    out = {"gpu": torch.cuda.get_device_name(0)}
    for model in ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base"):
        tok = M.load_tokenizer(model)
        m = M.load_model(model, "fp16", "cuda", "sdpa").eval()
        res = {}
        for name, (rows, lang, fields) in payload.items():
            items, st = D.prepare_pool(rows, tok, 1024, fields, lang)
            tot, n, per = 0.0, 0, []
            with torch.no_grad():
                for i in range(0, len(items), 8):
                    mb = items[i:i + 8]
                    b = D.collate(mb, tok.pad_token_id)
                    t = {k: torch.tensor(b[k], device="cuda") for k in ("input_ids", "attention_mask", "labels")}
                    with torch.autocast("cuda", dtype=torch.float16):
                        s = float(M.completion_nll_sum(m, t["input_ids"], t["attention_mask"], t["labels"]))
                    c = D.n_completion_tokens(mb)
                    tot, n = tot + s, n + c
                    per.append(s / c)
            comp = sorted(x["n_completion"] for x in items)
            res[name] = {"n_items": len(items), "nll_token_mean": round(tot / max(1, n), 4),
                         "batch_nll_min_max": [round(min(per), 4), round(max(per), 4)],
                         "completion_tokens_mean": round(sum(comp) / len(comp), 1),
                         "completion_tokens_p99": comp[min(len(comp) - 1, int(.99 * len(comp)))],
                         "drop_rate_1024": st["drop_rate"]}
        out[model] = res
        del m
        torch.cuda.empty_cache()
    return out


@app.local_entrypoint()
def main():
    d = REPO / "data"
    probs = {r["key"]: r.get("problem") for r in _rows(d / "train/N/_work/vi/gen.jsonl")}
    box = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
    n_rows = [{"id": r["key"], "question": probs[r["key"]], "solution": box.sub(lambda m: m.group(1), r["trace"]).strip(),
               "answer": r["boxed"]}
              for r in _rows(d / "train/N/_work/vi/solve.jsonl")
              if r.get("trace") and r.get("boxed") and probs.get(r["key"])]
    payload = {"S0_vi_400": (_rows(d / "t1r/s0_train.jsonl", 400), "vi", None),
               "S0_en_400": (_rows(d / "t1r/s0_train_en.jsonl", 400), "en", None),
               "dev300_vi": (_rows(d / "t1r/dev300.jsonl"), "vi", None),
               "dev300_en": (_rows(d / "t1r/dev300_en.jsonl"), "en", None),
               "N_pilot_vi": (n_rows, "vi", None)}
    res = probe.remote(payload)
    p = REPO / "jobs" / "outputs" / "QA_T" / "nll_probe.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
