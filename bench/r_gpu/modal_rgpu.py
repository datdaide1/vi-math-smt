"""R-GPU Modal benchmark app (2026-11-07).  Same image layers as ``evalrun/modal_eval.py`` (vLLM 0.33.0, transformers
5.20.2, peft 0.21.2, Unsloth 2026.11.3 isolated in /uns_site) + torchao isolated in /ao_site.

    modal run bench/r_gpu/modal_rgpu.py::train_main --spec bench/r_gpu/specs/train_<x>.json
    modal run bench/r_gpu/modal_rgpu.py::eval_main  --spec bench/r_gpu/specs/eval_<x>.json

Data are read from the T1r Volume (``t1r-eval:/data``, ``/eval_data``; never written).  Outputs go to the
``r-gpu`` Volume.  One GPU type per function, NO fallback (a benchmark must know its GPU; the real GPU is recorded).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

if modal.is_local():
    REPO = Path(__file__).resolve().parents[2]
    PKG = REPO / "vi_math_verified_aug"

PINS = ["vllm==0.33.0", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0", "math-verify==0.9.0",
        "pyyaml", "datasets==4.8.5"]
UNS = ("pip install -q --no-deps --target /uns_site unsloth==2026.11.3 unsloth_zoo==2026.11.3 trl==1.17.1 "
       "cut_cross_entropy==25.1.1 bitsandbytes==0.50.2")
ENV = {"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_FLASHINFER_SAMPLER": "0",
       "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1"}
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(*PINS)
         .run_commands("pip uninstall -y torchao || true", UNS)
         .env(ENV)
         .run_commands("pip install -q --no-deps --target /ao_site torchao"))
if modal.is_local():
    _KEEP = {"train", "evalrun", "bench_gpu", "eval", "infra", "common"}
    _SKIP = [f"{d}/**" for d in os.listdir(PKG) if (PKG / d).is_dir() and d not in _KEEP] + ["*.md", "modal_app.py"]
    image = image.add_local_dir(str(PKG), "/root/vi_math_verified_aug",  # only what train/evalrun import
                                ignore=["**/__pycache__/**", "**/*.pyc"] + _SKIP)
t1r = modal.Volume.from_name("t1r-eval")
rg = modal.Volume.from_name("r-gpu", create_if_missing=True)
hf = modal.Volume.from_name("r-gpu-hf-cache", create_if_missing=True)
app = modal.App("r-gpu-bench", image=image)
VOLS = {"/t1r": t1r, "/rg": rg, "/root/.cache/huggingface": hf}


def _gpu() -> str:
    return subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", shell=True,
                          capture_output=True, text=True).stdout.strip()


def _cfg(task: dict, gpu_tag: str, seed: int, name: str) -> dict:
    eng = task.get("engine", "unsloth")
    return {"kind": "sft", "run_id": f"rgpu-{gpu_tag}-{name}", "arm": "S0",
            "model": task.get("model", "Qwen/Qwen3-0.6B-Base"), "size": task.get("size", "0.6B"), "seed": seed, "precision": "bf16", "engine": eng, "platform": f"modal_{gpu_tag}",
            "out_dir": f"/rg/train/{task.get('ver', 'v2')}/{gpu_tag}/{name}", "resume": "never",
            "budget": {"n_train": task.get("n_train"), "epochs": task.get("epochs", 2), "max_steps": task.get("steps")},
            "data": {"train": task.get("train", "/t1r/data/s0_train.jsonl")},
            "sft": {"lr": 2e-4, "micro_batch": task.get("mb", 8), "max_length": 1024,
                    "packing": eng == "unsloth" and task.get("packing", True),
                    "dev_path": "/t1r/data/dev300.jsonl" if task.get("dev", True) else None,
                    "dev_every_steps": task.get("dev_every", 100), "ckpt_every_min": 0, "keep_ckpts": 1,
                    "grad_ckpt": task.get("grad_ckpt", False),
                    "pack_gate_min_argmax": task.get("gate_argmax", 0.90)},
            "guards": {"enabled": False}}


def _collect(out_dir: str) -> dict:
    o = Path(out_dir)
    r: dict = {}
    for f in ("manifest.json", "rgpu_variant.json"):
        if (o / f).exists():
            r[f.split(".")[0]] = json.loads((o / f).read_text(encoding="utf-8"))
    for f in ("train_log.jsonl", "dev_log.jsonl"):
        if (o / f).exists():
            r[f.split(".")[0]] = [json.loads(x) for x in (o / f).read_text(encoding="utf-8").splitlines() if x.strip()]
    if (o / "stdout.log").exists():
        r["tail"] = (o / "stdout.log").read_text(encoding="utf-8", errors="replace")[-2500:]
    return r


def _launch(task: dict, gpu_tag: str, seed: int, name: str):
    conf = _cfg(task, gpu_tag, seed, name)
    Path(conf["out_dir"]).mkdir(parents=True, exist_ok=True)
    cpath = f"{conf['out_dir']}/cfg.json"
    Path(cpath).write_text(json.dumps(conf), encoding="utf-8")
    pp = ["/root"]
    if conf["engine"] == "unsloth":
        pp.append("/uns_site")
    if task.get("variant", "none") in ("unsloth_fp8", "ao_fp8", "ao_fp8_compile") or task.get("ao_path"):
        pp.append("/ao_site")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(pp), UNSLOTH_RETURN_LOGITS="0", **task.get("env", {}))
    logf = open(Path(conf["out_dir"], "stdout.log"), "w", encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-m", "vi_math_verified_aug.bench_gpu.sft_bench", "--config", cpath,
                          "--variant", task.get("variant", "none"), "--budget-s", str(task.get("budget_s", 1500))],
                         env=env, cwd="/tmp", stdout=logf, stderr=subprocess.STDOUT)
    return p, logf, conf["out_dir"]


def _run_tasks(tasks: list[dict], gpu_tag: str) -> dict:
    res = {"gpu": _gpu(), "container_entry_t": time.time(), "spawn_t": tasks[0].get("spawn_t"), "tasks": []}
    for task in tasks:
        k = int(task.get("concurrent", 1))
        names = [task["name"]] if k == 1 else [f"{task['name']}-c{i}" for i in range(k)]
        t0 = time.time()
        procs = [_launch(task, gpu_tag, task.get("seed", 0) + i, n) for i, n in enumerate(names)]
        smi = []
        while any(p.poll() is None for p, _, _ in procs):
            if time.time() - t0 > task.get("budget_s", 1500) + 240:
                for p, _, _ in procs:
                    p.kill()
                break
            smi.append(subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,"
                                      "nounits", shell=True, capture_output=True, text=True).stdout.strip())
            time.sleep(5)
        wall = round(time.time() - t0, 1)
        for p, lf, _ in procs:
            p.wait()
            lf.close()
        runs = [{"name": n, "rc": p.returncode, **_collect(od)} for (p, _, od), n in zip(procs, names)]
        res["tasks"].append({"task": task, "t0": t0, "t1": time.time(), "wall_s": wall, "smi": smi, "runs": runs})
        Path(f"/rg/results").mkdir(parents=True, exist_ok=True)
        Path(f"/rg/results/train_{tasks[0].get('ver', 'v2')}_{gpu_tag}_{tasks[0]['name']}.json").write_text(json.dumps(res, indent=1, default=str))
        rg.commit()
        print(task["name"], "wall", wall, [(r["name"], r["rc"], (r.get("manifest") or {}).get("summary", {})
                                            .get("real_tokens_per_s")) for r in runs], flush=True)
    return res


@app.function(gpu="L40S", timeout=3600, retries=0, volumes=VOLS)
def train_l40s(tasks: list[dict]) -> dict:
    return _run_tasks(tasks, "l40s")


@app.function(gpu="H100", timeout=3600, retries=0, volumes=VOLS)
def train_h100(tasks: list[dict]) -> dict:
    return _run_tasks(tasks, "h100")


@app.function(gpu="L4", timeout=3600, retries=0, volumes=VOLS)
def train_l4(tasks: list[dict]) -> dict:
    return _run_tasks(tasks, "l4")


def _run_evals(blocks: list[dict], gpu_tag: str) -> dict:
    res = {"gpu": _gpu(), "blocks": []}
    for b in blocks:
        tag = f"{gpu_tag}-{b['tag']}"
        Path("/rg/eval/_specs").mkdir(parents=True, exist_ok=True)
        configs = [json.loads(json.dumps(c).replace("{tag}", tag)) for c in b["configs"]]
        sp = f"/rg/eval/_specs/{tag}.json"
        Path(sp).write_text(json.dumps({"base": b["base"], "configs": configs}))
        outp = f"/rg/eval/_specs/{tag}.result.json"
        cmd = [sys.executable, "-m", "vi_math_verified_aug.bench_gpu.eval_bench", "--spec", sp, "--out", outp]
        if b.get("quant"):
            cmd += ["--quant", b["quant"]]
        t0 = time.time()
        p = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH="/root"), cwd="/tmp", capture_output=True, text=True,
                           timeout=b.get("timeout", 1500))
        r = json.loads(Path(outp).read_text()) if Path(outp).exists() else {}
        r.update(tag=tag, rc=p.returncode, wall_s=round(time.time() - t0, 1),
                 tail=(p.stdout + p.stderr)[-3000:] if p.returncode else (p.stdout + p.stderr)[-600:])
        res["blocks"].append(r)
        Path("/rg/results").mkdir(parents=True, exist_ok=True)
        Path(f"/rg/results/eval_{gpu_tag}_{blocks[0]['tag']}.json").write_text(json.dumps(res, indent=1))
        rg.commit()
        print(tag, p.returncode, r.get("engine_load_s"), {k: (v.get("throughput"), v.get("s")) for k, v in
                                                          r.get("runs", {}).items()}, flush=True)
    return res


@app.function(gpu="L4", timeout=3600, retries=0, volumes=VOLS)
def eval_l4(blocks: list[dict]) -> dict:
    return _run_evals(blocks, "l4")


@app.function(gpu="L40S", timeout=3600, retries=0, volumes=VOLS)
def eval_l40s(blocks: list[dict]) -> dict:
    return _run_evals(blocks, "l40s")


FN = {"train_l40s": train_l40s, "train_h100": train_h100, "train_l4": train_l4, "eval_l4": eval_l4,
      "eval_l40s": eval_l40s}


@app.local_entrypoint()
def run_main(spec: str):
    """spec = {"jobs": [{"fn": "train_l40s", "args": [...]}, ...]}: all jobs spawned in parallel."""
    s = json.loads(Path(spec).read_text(encoding="utf-8"))
    for j in s["jobs"]:
        for a in j["args"]:
            a["spawn_t"] = time.time()
    calls = [(f"{i}:{j['fn']}", FN[j["fn"]].spawn(j["args"])) for i, j in enumerate(s["jobs"])]
    print("spawned", [(f, c.object_id) for f, c in calls], flush=True)
    out = {}
    for f, c in calls:
        try:
            out[f] = c.get(timeout=s.get("wait_s", 3900))
        except Exception as e:  # noqa: BLE001
            out[f] = {"error": repr(e)[:2000]}
        Path(spec).with_suffix(".result.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        print(f, "done", flush=True)
