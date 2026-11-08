"""O1 throughput micro-benchmarks on Modal (SFT variants + vLLM multi-LoRA eval), L4 and T4.

    modal billing summary
    modal run bench/o1/modal_bench.py --plans sft_l4,eval_l4,sft_t4,eval_t4,unsloth_l4
    modal billing summary
Outputs: bench/o1/results/<plan>/<variant>.json (+ .log).  Data = train splits only (bench/o1/data).
Budget guard: every plan has a hard function timeout (worst case for all five ≈ $2.3).
"""
import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PINS = ["vllm==0.33.0", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0", "datasets"]
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(*PINS)
         .run_commands("pip uninstall -y torchao || true")
         .pip_install("kernels")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
         .add_local_file(str(HERE / "sft_worker.py"), "/app/sft_worker.py")
         .add_local_file(str(HERE / "eval_worker.py"), "/app/eval_worker.py")
         .add_local_dir(str(HERE / "data"), "/data"))
uimage = (modal.Image.debian_slim(python_version="3.12")
          .apt_install("git", "build-essential")
          .pip_install("unsloth")
          .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
          .add_local_file(str(HERE / "sft_worker.py"), "/app/sft_worker.py")
          .add_local_dir(str(HERE / "data"), "/data"))

app = modal.App("o1-throughput-bench")
Q17 = "Qwen/Qwen3-1.7B-Base"

SFT_L4 = ["pad_rand", "pad_group", "pad_group_mb16", "pad_group_nock", "pack_sdpa", "pack_flex", "pack_fa2"]
SFT_T4 = ["pad_rand", "pad_group", "pack_sdpa", "pack_flex", "pad_group_nock"]
# plan -> (gpu, kind, dtype, steps [(worker, variant, budget_s)], timeout_s, image)
PLANS = {
    "sft_l4": ("L4", "sft", "bf16", [("sft", v, 420) for v in SFT_L4], 1800, "hf"),
    "sft_t4": ("T4", "sft", "fp16", [("sft", v, 600) for v in SFT_T4], 2400, "hf"),
    "unsloth_l4": ("L4", "sft", "bf16", [("sft", v, 600) for v in ("unsloth_pad", "unsloth_pack")], 1500, "unsloth"),
    "unsloth_t4": ("T4", "sft", "fp16", [("sft", v, 800) for v in ("unsloth_pad", "unsloth_pack")], 1800, "unsloth"),
    "eval_l4": ("L4", "eval", "bf16", [("eval", m, 900) for m in ("mkadapters", "base", "multi", "seq")], 2100, "hf"),
    "eval_t4": ("T4", "eval", "fp16", [("eval", m, 1100) for m in ("mkadapters", "base", "multi", "seq")], 2700, "hf"),
}


def _run(name, dtype, steps):
    t0 = time.time()
    out = "/out"
    os.makedirs(out, exist_ok=True)
    smi = subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
                         shell=True, capture_output=True, text=True).stdout.strip()
    ledger = {"plan": name, "gpu": smi, "dtype": dtype, "tasks": []}
    for worker, v, budget in steps:
        logf = f"{out}/{v}.log"
        t = time.time()
        with open(logf, "a") as lf:
            p = subprocess.Popen(["python", f"/app/{worker}_worker.py", v, Q17, dtype, f"{out}/{v}.json"],
                                 env=dict(os.environ, O1_DATA="/data", PYTHONUNBUFFERED="1"),
                                 stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = p.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 9)
                p.wait()
                rc = "timeout"
        ledger["tasks"].append({"task": v, "rc": rc, "s": round(time.time() - t, 1)})
        print(name, v, rc, round(time.time() - t), flush=True)

    ledger["wall_s"] = round(time.time() - t0, 1)
    files = {"ledger.json": json.dumps(ledger, indent=1)}
    for p in Path(out).glob("*"):
        if p.is_file() and p.stat().st_size < 3_000_000:
            txt = p.read_text(encoding="utf-8", errors="replace")
            if p.suffix == ".log" and len(txt) > 20000:
                txt = txt[:4000] + "\n...\n" + txt[-12000:]
            files[p.name] = txt
    return files


@app.function(image=image, gpu="L4", timeout=1800, retries=0)
def run_hf(name: str, dtype: str, steps: list) -> dict:
    return _run(name, dtype, steps)


@app.function(image=uimage, gpu="L4", timeout=1500, retries=0)
def run_unsloth(name: str, dtype: str, steps: list) -> dict:
    return _run(name, dtype, steps)


@app.local_entrypoint()
def main(plans: str = "sft_l4,eval_l4"):
    calls = {}
    for n in plans.split(","):
        gpu, kind, dtype, steps, timeout, img = PLANS[n]
        fn = run_unsloth if img == "unsloth" else run_hf
        calls[n] = fn.with_options(gpu=gpu, timeout=timeout).spawn(n, dtype, steps)
        print("spawned", n, gpu, dtype, flush=True)
    for n, c in calls.items():
        try:
            files = c.get()
        except Exception as e:  # noqa
            files = {"ledger.json": json.dumps({"plan": n, "error": repr(e)[:2000]})}
        d = HERE / "results" / n
        d.mkdir(parents=True, exist_ok=True)
        for rel, txt in files.items():
            (d / rel).write_text(txt, encoding="utf-8")
        print("done", n, json.loads(files["ledger.json"]).get("tasks"), flush=True)
