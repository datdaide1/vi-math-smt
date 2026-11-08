"""Self-hosted cross-family back-formalizer for the arm-Y cycle check (lead 02:10: Gemini free tier exhausted; NIM
glm-5.3-flash agrees 91/92 with Gemini but runs at ~13 items/min). Qwen3-32B-AWQ (Qwen family, not Claude) on vLLM
0.33.0 (the repo's pinned eval stack), thinking disabled, temperature 0. Same prompt as every cycle check
(prompts/back_formalizer.v1 / back_formalizer_en.v1; rendered locally and sent as chat messages).

    modal run -m vi_math_verified_aug.arm_y.vllm_cycle_modal --inp <prompts.json> --out <texts.json>
inp: [{"id": ..., "system": ..., "user": ...}], out: [{"id": ..., "text": ...}]. Training/dev items only.
Cap (lead): $3 Modal. GPU fallback list, retries=0, hard timeout; model weights cached in a Volume.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3-32B-AWQ"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.33.0", "transformers==5.20.2")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_FLASHINFER_SAMPLER": "0",
               "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1"}))
hf_cache = modal.Volume.from_name("g5-hf-cache", create_if_missing=True)
app = modal.App("g5-vllm-cycle", image=image)


@app.function(gpu=["L40S", "A100-80GB", "H100"], timeout=3600, retries=0, memory=65536, cpu=8,
              volumes={"/root/.cache/huggingface": hf_cache})
def generate(items: list[dict], max_tokens: int = 4096) -> dict:
    import subprocess

    from vllm import LLM, SamplingParams
    gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip()
    print("ENVCHECK", gpu, flush=True)
    t0 = time.time()
    llm = LLM(model=MODEL, max_model_len=12288, gpu_memory_utilization=0.92, enable_prefix_caching=True)
    hf_cache.commit()
    t1 = time.time()
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    msgs = [[{"role": "system", "content": it["system"]}, {"role": "user", "content": it["user"]}] if it.get("system")
            else [{"role": "user", "content": it["user"]}] for it in items]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    res = [{"id": it["id"], "text": o.outputs[0].text, "finish": o.outputs[0].finish_reason,
            "n_out": len(o.outputs[0].token_ids)} for it, o in zip(items, outs)]
    t2 = time.time()
    print(f"load {t1 - t0:.0f}s, generate {t2 - t1:.0f}s for {len(items)} items", flush=True)
    return {"gpu": gpu, "model": MODEL, "load_s": round(t1 - t0), "gen_s": round(t2 - t1), "results": res}


@app.local_entrypoint()
def main(inp: str, out: str):
    items = json.load(open(inp, encoding="utf-8"))
    r = generate.remote(items)
    Path(out).write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    print("gpu", r["gpu"], "load_s", r["load_s"], "gen_s", r["gen_s"], "n", len(r["results"]))
