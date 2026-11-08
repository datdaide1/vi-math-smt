"""Chunked, resumable version of vllm_cycle_modal (same model, prompt and sampling): one warm container (modal.Cls,
model loaded once in @enter) serves chunks of ``chunk`` items; the local entrypoint writes every finished chunk to
<out>.partN.json at once, so a timeout or kill keeps all finished chunks and a re-run skips them.

    modal run -m vi_math_verified_aug.arm_y.vllm_cycle_modal2 --inp <prompts.json> --out <texts.json> [--chunk 400]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3-32B-AWQ"          # identical to vllm_cycle_modal (kept self-contained for the container)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.33.0", "transformers==5.20.2")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_FLASHINFER_SAMPLER": "0",
               "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1"}))
hf_cache = modal.Volume.from_name("g5-hf-cache", create_if_missing=True)
app = modal.App("g5-vllm-cycle2", image=image)


@app.cls(gpu=["L40S"], timeout=1800, retries=0, memory=65536, cpu=8, scaledown_window=60,
         volumes={"/root/.cache/huggingface": hf_cache})
class Gen:
    @modal.enter()
    def load(self):
        import subprocess

        from vllm import LLM, SamplingParams
        self.gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                                  capture_output=True, text=True).stdout.strip()
        t0 = time.time()
        self.llm = LLM(model=MODEL, max_model_len=12288, gpu_memory_utilization=0.92, enable_prefix_caching=True)
        self.sp = SamplingParams(temperature=0.0, max_tokens=4096)
        self.load_s = round(time.time() - t0)
        print("ENVCHECK", self.gpu, "load_s", self.load_s, flush=True)

    @modal.method()
    def run(self, items: list[dict]) -> dict:
        t0 = time.time()
        msgs = [[{"role": "system", "content": it["system"]}, {"role": "user", "content": it["user"]}]
                if it.get("system") else [{"role": "user", "content": it["user"]}] for it in items]
        outs = self.llm.chat(msgs, self.sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        res = [{"id": it["id"], "text": o.outputs[0].text, "finish": o.outputs[0].finish_reason,
                "n_out": len(o.outputs[0].token_ids)} for it, o in zip(items, outs)]
        return {"gpu": self.gpu, "load_s": self.load_s, "gen_s": round(time.time() - t0), "results": res}


@app.local_entrypoint()
def main(inp: str, out: str, chunk: int = 400):
    items = json.load(open(inp, encoding="utf-8"))
    g = Gen()
    parts = []
    for i in range(0, len(items), chunk):
        p = Path(f"{out}.part{i // chunk}.json")
        parts.append(p)
        if p.exists():
            continue
        r = g.run.remote(items[i:i + chunk])
        p.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print(f"chunk {i // chunk}: {len(r['results'])} items in {r['gen_s']} s on {r['gpu']}", flush=True)
    allr, meta = [], None
    for p in parts:
        r = json.load(open(p, encoding="utf-8"))
        meta = meta or r
        allr += r["results"]
    gen_s = sum(json.load(open(p, encoding="utf-8"))["gen_s"] for p in parts)
    Path(out).write_text(json.dumps({"gpu": meta["gpu"], "model": MODEL, "load_s": meta["load_s"], "gen_s": gen_s,
                                     "results": allr}, ensure_ascii=False), encoding="utf-8")
    print("done", len(allr))
