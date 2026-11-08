"""Kaggle environment probe (runs ON Kaggle, T4x2). Short: records the stack and whether vLLM 0.33 runs on sm_75.

Writes /kaggle/working/probe.json. No secrets, no test data used.
"""
import json
import os
import subprocess
import sys
import time

R = {"t0": time.time()}
OUT = "/kaggle/working/probe.json"


def sh(cmd, timeout=1800):
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return {"cmd": cmd, "rc": p.returncode, "s": round(time.time() - t, 1),
            "out": (p.stdout + p.stderr)[-3000:]}


def save():
    json.dump(R, open(OUT, "w"), indent=1)


R["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; nvidia-smi | head -5")
R["python"] = sys.version
R["preinstalled"] = sh("pip list 2>/dev/null | grep -iE '^(torch|transformers|peft|trl|accelerate|vllm|triton|flash|xformers|bitsandbytes) '")
R["inputs"] = sh("find /kaggle/input -maxdepth 4 | head -30")
save()

R["pip_vllm"] = sh("pip install -q 'vllm==0.33.0' 'peft' 'flash-linear-attention' 2>&1 | tail -20", timeout=2400)
R["post"] = sh("pip list 2>/dev/null | grep -iE '^(torch|transformers|peft|vllm|triton|flash-linear-attention|flashinfer-python|fla-core) '")
R["torch_cuda"] = sh("python -c \"import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); x=torch.randn(512,512,device='cuda',dtype=torch.float16); print((x@x).float().abs().mean().item())\"")
save()

VLLM_TEST = r'''
import time, sys, json
from vllm import LLM, SamplingParams
m = sys.argv[1]
t = time.time()
llm = LLM(model=m, dtype="float16", max_model_len=2048, gpu_memory_utilization=0.85, seed=0, enforce_eager=False)
load = time.time() - t
ps = ["Bài toán:\nAn có 3 quả táo, mua thêm 5 quả. Hỏi An có bao nhiêu quả táo?\n\nLời giải:\n"] * 64
t = time.time()
outs = llm.generate(ps, SamplingParams(temperature=0, max_tokens=128))
dt = time.time() - t
ntok = sum(len(o.outputs[0].token_ids) for o in outs)
print("RESULT", json.dumps({"model": m, "load_s": round(load,1), "gen_s": round(dt,2), "gen_tokens": ntok,
      "tok_per_s": round(ntok/dt,1), "sample": outs[0].outputs[0].text[:300]}, ensure_ascii=False))
'''
open("/kaggle/working/vllm_test.py", "w").write(VLLM_TEST)
# run the two families in parallel, one per GPU
procs = {}
for gpu, m in [(0, "Qwen/Qwen3-0.6B-Base"), (1, "Qwen/Qwen3.5-0.8B-Base")]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    procs[m] = subprocess.Popen([sys.executable, "/kaggle/working/vllm_test.py", m], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for m, p in procs.items():
    try:
        out, _ = p.communicate(timeout=1500)
    except subprocess.TimeoutExpired:
        p.kill(); out, _ = p.communicate()
    R[f"vllm::{m}"] = {"rc": p.returncode, "tail": out[-4000:]}
    save()
R["t_total_s"] = round(time.time() - R["t0"], 1)
save()
print(json.dumps(R, indent=1)[:20000])
