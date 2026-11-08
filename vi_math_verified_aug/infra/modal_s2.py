"""S2 on Modal: fp16 (T4) vs bf16 (L4) parity + per-platform SFT throughput, reusing pilot/s2_pilot.py unchanged.

Run (from repo root; logs spend before/after):
    modal billing summary
    modal run vi_math_verified_aug/infra/modal_s2.py --plans parity,tput4b      # comma list of PLANS keys
    modal billing summary
Outputs: jobs/outputs/s2-modal/<plan>/results/<model>/{sft.json, eval_dev.json, *.log}

Same code path as Kaggle: each task is `python s2_pilot.py --task ...` in its own process with its own time budget.
Data = train.jsonl + fewshot4.json + math_verify.py only (no test items leave the laptop for this job).
Budget guard: every function has a hard `timeout`; the plans below were sized to ~$2 of the $5 S2 cap.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import modal

# BUG (2026-11-07, ap-9fjZI5SEmtD7QOrF1qMTF9): `Path(__file__).resolve().parents[2]` ran at import time INSIDE the
# container too, where this file is /root/modal_s2.py (only 2 parents) -> IndexError(2) during startup. Startup
# crashes are not covered by `retries=0` and Modal rescheduled the container in a loop. Fix: repo paths are
# resolved only on the laptop (modal.is_local()); the container never touches them.
if modal.is_local():
    REPO = Path(__file__).resolve().parents[2]
    PILOT = REPO / "vi_math_verified_aug" / "infra" / "pilot" / "s2_pilot.py"
    DATA = REPO / "jobs" / "build" / "s2-modal-data"
else:
    REPO = PILOT = DATA = None

PINS = ["vllm==0.33.0", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0", "datasets"]
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(*PINS)
         .run_commands("pip uninstall -y torchao || true")
         # 3rd bug (2026-11-07): on A10/L4 vLLM picks the FlashInfer top-k/top-p sampler, which JIT-compiles
         # at warm-up and needs nvcc (absent in debian_slim) -> "Could not find nvcc". Use the torch sampler.
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false",
               "VLLM_USE_FLASHINFER_SAMPLER": "0"}))
if modal.is_local():
    image = image.add_local_file(str(PILOT), "/app/s2_pilot.py").add_local_dir(str(DATA), "/data")
hf_cache = modal.Volume.from_name("s2-hf-cache", create_if_missing=True)
merged_vol = modal.Volume.from_name("s2-merged", create_if_missing=True)  # merged SFT models survive the container
app = modal.App("s2-student-pilot", image=image)

Q06, Q17, Q4 = "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"
GEMMA = "google/gemma-4-E2B"


def sft_eval(model, budget=1500, tag_=""):
    d = f"/merged/{tag_}{model.split('/')[-1]}"
    return [("sft", model, "", {"S2_SAVE_MERGED": d}, budget), ("eval_dev", model, d, {}, 900)]


# plan -> (gpu, dtype, extra env, [(task, model, arg, env, budget_s)], function timeout_s)
PLANS = {
    "parity_t4_fp16": ("T4", "fp16", {}, sft_eval(Q06, tag_="fp16_") + sft_eval(Q17, tag_="fp16_"), 3600),
    "parity_l4_bf16": (["L4", "A10G", "L40S"], "bf16", {}, sft_eval(Q06, tag_="bf16_") + sft_eval(Q17, tag_="bf16_"),
                       3000),
    "tput_l40s": ("L40S", "bf16", {"S2_GC_ABOVE": "1e12"},
                  [("sft", Q4, "", {}, 1200), ("sft", Q17, "", {}, 600), ("sft", Q06, "", {}, 600)], 2400),
    "tput_h100_4b": ("H100", "bf16", {"S2_GC_ABOVE": "1e12"}, [("sft", Q4, "", {}, 900)], 1200),
    "tput_l4_4b": (["L4", "A10G"], "bf16", {}, [("sft", Q4, "", {}, 1500)], 1800),
    "gemma_l4_bf16": (["L4", "A10G", "L40S"], "bf16", {}, [("sft", GEMMA, "", {}, 1800)], 2400),
}
GROUPS = {"parity": ["parity_t4_fp16", "parity_l4_bf16"], "tput4b": ["tput_l40s", "tput_h100_4b", "tput_l4_4b"]}


def _run_plan(name: str, dtype: str, env_extra: dict, steps: list) -> dict:
    t0 = time.time()
    res_dir = "/out/results"
    os.makedirs(res_dir, exist_ok=True)
    smi = subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
                         shell=True, capture_output=True, text=True).stdout.strip()
    ledger = {"plan": name, "gpu": smi, "dtype": dtype, "tasks": []}
    for task, model, arg, env_t, budget in steps:
        env = dict(os.environ, S2_DATA="/data", S2_RES=res_dir, S2_DTYPE=dtype, S2_PLATFORM=f"modal-{name}",
                   S2_TASK_BUDGET_S=str(budget), PYTHONUNBUFFERED="1", **env_extra, **env_t)
        logf = f"{res_dir}/{model.split('/')[-1]}/{task}.log"
        os.makedirs(os.path.dirname(logf), exist_ok=True)
        t = time.time()
        with open(logf, "a") as lf:
            p = subprocess.Popen(["python", "/app/s2_pilot.py", "--task", task, "--model", model, "--data", "/data",
                                  "--arg", arg], env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = p.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 9)
                p.wait()
                rc = "timeout"
        ledger["tasks"].append({"task": task, "model": model, "rc": rc, "s": round(time.time() - t, 1)})
        print(name, task, model, rc, round(time.time() - t), flush=True)
    hf_cache.commit()
    merged_vol.commit()
    ledger["wall_s"] = round(time.time() - t0, 1)
    files = {}
    for p in Path(res_dir).rglob("*"):
        if p.is_file() and p.suffix in (".json", ".log", ".jsonl") and p.stat().st_size < 5_000_000:
            files[str(p.relative_to(res_dir))] = p.read_text(encoding="utf-8", errors="replace")
    files["ledger.json"] = json.dumps(ledger, indent=1)
    return files


# One decorated function per GPU spec: Modal 1.6 accepts a GPU fallback LIST only in the decorator, not in
# Function.with_options (2nd bug of 2026-11-07: InvalidError "Value must be a string or None").
# retries=0 everywhere; the timeout is the per-plan hard cap (PLANS[...][4] must be <= it).
_KW = dict(volumes={"/root/.cache/huggingface": hf_cache, "/merged": merged_vol}, retries=0, timeout=3600)


@app.function(gpu="T4", **_KW)
def run_t4(name: str, dtype: str, env_extra: dict, steps: list) -> dict:
    return _run_plan(name, dtype, env_extra, steps)


@app.function(gpu=["L4", "A10G", "L40S"], **_KW)
def run_l4_fallback(name: str, dtype: str, env_extra: dict, steps: list) -> dict:
    return _run_plan(name, dtype, env_extra, steps)


@app.function(gpu=["L4", "A10G"], **_KW)
def run_l4_a10g(name: str, dtype: str, env_extra: dict, steps: list) -> dict:
    return _run_plan(name, dtype, env_extra, steps)


@app.function(gpu="L40S", **_KW)
def run_l40s(name: str, dtype: str, env_extra: dict, steps: list) -> dict:
    return _run_plan(name, dtype, env_extra, steps)


@app.function(gpu="H100", **_KW)
def run_h100(name: str, dtype: str, env_extra: dict, steps: list) -> dict:
    return _run_plan(name, dtype, env_extra, steps)


FN_BY_GPU = {"T4": run_t4, "L4|A10G|L40S": run_l4_fallback, "L4|A10G": run_l4_a10g, "L40S": run_l40s,
             "H100": run_h100}


@app.local_entrypoint()
def main(plans: str = "parity,tput4b"):
    names = []
    for x in plans.split(","):
        names += GROUPS.get(x, [x])
    calls = {}
    for n in names:
        gpu, dtype, env_extra, steps, timeout = PLANS[n]
        fn = FN_BY_GPU["|".join(gpu) if isinstance(gpu, list) else gpu]
        calls[n] = fn.spawn(n, dtype, env_extra, steps)
        print("spawned", n, gpu, dtype, flush=True)
    t0 = time.time()
    for n, c in calls.items():
        # bounded wait: function timeout + 20 min for scheduling/cold start; then CANCEL (no silent crash loop)
        left = PLANS[n][4] + 1200 - (time.time() - t0)
        try:
            files = c.get(timeout=max(60, left))
        except Exception as e:  # noqa  (TimeoutError, startup failure, remote exception)
            try:
                c.cancel()
            except Exception:  # noqa
                pass
            files = {"ledger.json": json.dumps({"plan": n, "error": repr(e)[:2000], "cancelled": True})}
        out = REPO / "jobs" / "outputs" / "s2-modal" / n
        for rel, txt in files.items():
            f = out / ("results" if rel != "ledger.json" else "") / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(txt, encoding="utf-8")
        print("done", n, json.loads(files["ledger.json"]).get("tasks"), flush=True)
