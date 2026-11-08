import json as _json
JOB = _json.loads('{"expect_gpu": "T4", "fail_fast": true, "deadline_s": 9000, "grace_s": 300, "env_fix": ["pip uninstall -y -q torchao"], "install": "pip install -q \'vllm==0.33.0\'", "wheels_dir_glob": null, "unsloth_install": "pip install -q --no-deps --target /tmp/uns_site unsloth==2026.11.3 unsloth_zoo==2026.11.3 trl==1.17.1 cut_cross_entropy==25.1.1 bitsandbytes==0.50.2", "drop_ckpt_on_done": true, "python": "3.13", "pins": {"torch": "2.14.0", "transformers": "5.20.2", "peft": "0.21.2", "accelerate": "1.15.0", "vllm": "0.33.0", "triton": "3.8.0", "datasets": "4.8.5"}, "run_id": "t1r-runs-v1-20261107T150455Z", "lanes": [[{"cmd": "sft", "config": {"kind": "sft", "run_id": "t1r-gate-uns-0p6B-s1", "arm": "S0", "model": "Qwen/Qwen3-0.6B-Base", "size": "0.6B", "seed": 1, "precision": "fp16", "engine": "unsloth", "budget": {"n_train": null, "epochs": 2, "max_steps": 300}, "data": {"train": "{code}/data/s0_train.jsonl"}, "sft": {"lr": 0.0002, "micro_batch": 8, "max_length": 1024, "packing": true, "dev_path": "{code}/data/dev300.jsonl", "dev_every_steps": 100, "act_monitor_every": 0, "ckpt_every_min": 30, "keep_ckpts": 1}, "notes": "T1r first real runs (S0 = clean Vi-GSM8K train minus R1\'s 16 rows minus dev300)"}, "budget_s": 1800}, {"cmd": "sft", "config": {"kind": "sft", "run_id": "t1r-S0-uns-0p6B-s1", "arm": "S0", "model": "Qwen/Qwen3-0.6B-Base", "size": "0.6B", "seed": 1, "precision": "fp16", "engine": "unsloth", "budget": {"n_train": null, "epochs": 2, "max_steps": null}, "data": {"train": "{code}/data/s0_train.jsonl"}, "sft": {"lr": 0.0002, "micro_batch": 8, "max_length": 1024, "packing": true, "dev_path": "{code}/data/dev300.jsonl", "dev_every_steps": 150, "act_monitor_every": 0, "ckpt_every_min": 30, "keep_ckpts": 1}, "notes": "T1r first real runs (S0 = clean Vi-GSM8K train minus R1\'s 16 rows minus dev300)"}, "budget_s": 3300, "requires": "t1r-gate-uns-0p6B-s1", "fallback_config": {"kind": "sft", "run_id": "t1r-S0-hf-0p6B-s1", "arm": "S0", "model": "Qwen/Qwen3-0.6B-Base", "size": "0.6B", "seed": 1, "precision": "fp16", "engine": "hf", "budget": {"n_train": null, "epochs": 2, "max_steps": null}, "data": {"train": "{code}/data/s0_train.jsonl"}, "sft": {"lr": 0.0002, "micro_batch": 8, "max_length": 1024, "packing": false, "dev_path": "{code}/data/dev300.jsonl", "dev_every_steps": 150, "act_monitor_every": 0, "ckpt_every_min": 30, "keep_ckpts": 1}, "notes": "T1r first real runs (S0 = clean Vi-GSM8K train minus R1\'s 16 rows minus dev300)"}}, {"cmd": "sft", "config": {"kind": "sft", "run_id": "t1r-lfm2.5-2.6B-smoke", "arm": "S0", "model": "LiquidAI/LFM2.5-2.6B-Base", "size": "2.6B", "seed": 1, "precision": "fp16", "engine": "hf", "budget": {"n_train": 3200, "epochs": 1, "max_steps": 100}, "data": {"train": "{code}/data/s0_train.jsonl"}, "sft": {"lr": 0.0002, "micro_batch": 8, "max_length": 1024, "packing": false, "dev_path": null, "dev_every_steps": null, "act_monitor_every": 4, "ckpt_every_min": 30, "keep_ckpts": 1}, "notes": "T1r first real runs (S0 = clean Vi-GSM8K train minus R1\'s 16 rows minus dev300)"}, "budget_s": 2400}], [{"cmd": "sft", "config": {"kind": "sft", "run_id": "t1r-gate-hf-0p6B-s1", "arm": "S0", "model": "Qwen/Qwen3-0.6B-Base", "size": "0.6B", "seed": 1, "precision": "fp16", "engine": "hf", "budget": {"n_train": null, "epochs": 2, "max_steps": 300}, "data": {"train": "{code}/data/s0_train.jsonl"}, "sft": {"lr": 0.0002, "micro_batch": 8, "max_length": 1024, "packing": false, "dev_path": "{code}/data/dev300.jsonl", "dev_every_steps": 100, "act_monitor_every": 0, "ckpt_every_min": 30, "keep_ckpts": 1}, "notes": "T1r first real runs (S0 = clean Vi-GSM8K train minus R1\'s 16 rows minus dev300)"}, "budget_s": 2400}, {"cmd": "sft", "config": {"kind": "sft", "run_id": "t1r-S0-uns-0p6B-s2", "arm": "S0", "model": "Qwen/Qwen3-0.6B-Base", "size": "0.6B", "seed": 2, "precision": "fp16", "engine": "unsloth", "budget": {"n_train": null, "epochs": 2, "max_steps": null}, "data": {"train": "{code}/data/s0_train.jsonl"}, "sft": {"lr": 0.0002, "micro_batch": 8, "max_length": 1024, "packing": true, "dev_path": "{code}/data/dev300.jsonl", "dev_every_steps": 150, "act_monitor_every": 0, "ckpt_every_min": 30, "keep_ckpts": 1}, "notes": "T1r first real runs (S0 = clean Vi-GSM8K train minus R1\'s 16 rows minus dev300)"}, "budget_s": 3300, "requires": "t1r-gate-uns-0p6B-s1", "wait_max_s": 1800, "fallback_config": {"kind": "sft", "run_id": "t1r-S0-hf-0p6B-s2", "arm": "S0", "model": "Qwen/Qwen3-0.6B-Base", "size": "0.6B", "seed": 2, "precision": "fp16", "engine": "hf", "budget": {"n_train": null, "epochs": 2, "max_steps": null}, "data": {"train": "{code}/data/s0_train.jsonl"}, "sft": {"lr": 0.0002, "micro_batch": 8, "max_length": 1024, "packing": false, "dev_path": "{code}/data/dev300.jsonl", "dev_every_steps": 150, "act_monitor_every": 0, "ckpt_every_min": 30, "keep_ckpts": 1}, "notes": "T1r first real runs (S0 = clean Vi-GSM8K train minus R1\'s 16 rows minus dev300)"}}]]}')
"""Y1 Kaggle job template -- runs ON a private Kaggle T4x2 kernel.  Stdlib only in this (parent) process.

The laptop (``train/kaggle/render.py``) prepends one line ``JOB = {...}``, exactly like S2's
``infra/pilot/submit_pilot.py``; outputs follow the same layout, so ``infra.kaggle_runner.KaggleRunner.output`` +
``verify_output_run_id`` work unchanged.

JOB schema::

    {"run_id": "<unique per push; echoed in results/meta.json>",
     "expect_gpu": "T4", "python": "3.13" (optional), "pins": {"peft": "0.21.2", ...},
     "install": "pip install -q ..." | null, "wheels_dir_glob": "/kaggle/input/*/wheels" | null,
     "env_fix": ["pip uninstall -y -q torchao"], "fail_fast": true,
     "unsloth_install": "pip install -q --no-deps --target /tmp/uns_site unsloth==... ..." | null,
     "deadline_s": 41400, "grace_s": 300,
     "lanes": [[TASK, ...], [TASK, ...]]}           # lane i runs on GPU i, tasks in order
    TASK = {"cmd": "sft"|"grpo"|"eval", "config": {<raw YAML dict>}, "budget_s": 20000,
            "requires": "<run_id whose DONE must exist>" (optional)}
         | {"cmd": "merge", "base": ..., "adapter": ..., "out": ..., "requires": ...}

String values may use ``{results}`` (= /kaggle/working/results), ``{input}`` (= /kaggle/input) and ``{code}`` (= the
mounted dataset dir that holds ``vi_math_verified_aug/``; put data files next to it).

Unsloth (T1r): installed with ``--no-deps`` into an isolated ``/tmp/uns_site`` (S2 §5: torch/vLLM stay untouched)
and put on ``PYTHONPATH`` only for tasks whose config has ``engine: unsloth``.  ENVCHECK records an import smoke of
it but does not fail the job on it (HF lanes still run; the Unsloth task fails on its own).  Tasks run with
cwd=/tmp so ``unsloth_compiled_cache/`` never lands in the kernel output.

Writes (incrementally): results/meta.json (job + envcheck), results/env_check.json, results/tasks.jsonl (one line per
finished task: rc, status, seconds), results/<run_id>/... (the run's own manifest, train_log.jsonl, preds.jsonl,
ckpt/, adapters/), results/logs/<run_id>.log.  Exit codes of a task: 0 COMPLETE, 75 NEEDS_RESUME, 4 ABORTED (fp16
overflow), "timeout" (hard kill at budget + grace), other = FAILED.
"""
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback

try:
    JOB  # noqa: F821  (injected by render.py)
except NameError:
    JOB = {"run_id": "local-default", "lanes": [], "pins": {}}

WORK = "/kaggle/working"
RES = f"{WORK}/results"
CFG_DIR = f"{WORK}/configs"
T0 = time.time()
STATUS = {0: "COMPLETE", 75: "NEEDS_RESUME", 4: "ABORTED_FP16_OVERFLOW", 3: "FAILED", 6: "GATE_FAILED"}
_LOCK = threading.Lock()


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


CODE_ROOT = None
UNS_SITE = "/tmp/uns_site"


def subst(x):
    if isinstance(x, str):
        x = x.replace("{results}", RES).replace("{input}", "/kaggle/input")
        return x.replace("{code}", CODE_ROOT) if CODE_ROOT else x
    if isinstance(x, list):
        return [subst(v) for v in x]
    if isinstance(x, dict):
        return {k: subst(v) for k, v in x.items()}
    return x


def find_code_root():
    """Directory that contains the vi_math_verified_aug package (the code dataset), else the script's dir."""
    hits = sorted(glob.glob("/kaggle/input/**/vi_math_verified_aug/__init__.py", recursive=True))
    if hits:
        return os.path.dirname(os.path.dirname(hits[0]))
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "vi_math_verified_aug", "__init__.py")):
        return here
    raise FileNotFoundError("vi_math_verified_aug not found under /kaggle/input (attach the code dataset)")


def versions(pkgs):
    import importlib.metadata as md
    out = {}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except Exception:  # noqa: BLE001
            out[p] = None
    return out


SMOKE = r"""
import json, torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16)
m = Qwen3ForCausalLM(cfg).half().cuda()
m = get_peft_model(m, LoraConfig(r=8, lora_alpha=8, target_modules="all-linear"))
x = torch.randint(0, 512, (2, 16), device="cuda")
s = torch.amp.GradScaler("cuda")
with torch.autocast("cuda", dtype=torch.float16):
    loss = m(input_ids=x, labels=x).loss
s.scale(loss).backward()
print(json.dumps({"ok": True, "loss": float(loss),
                  "lora_dtypes": sorted({str(p.dtype) for p in m.parameters() if p.requires_grad})}))
"""


def envcheck():
    """ENVCHECK: GPU, python, version pins, imports, and a fp16 LoRA fwd/bwd smoke on GPU 0."""
    out = {"t": round(time.time() - T0, 1)}
    smi = subprocess.run("nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader",
                         shell=True, capture_output=True, text=True).stdout.strip().splitlines()
    out["gpus"] = smi
    out["python"] = sys.version.split()[0]
    pk = ["torch", "transformers", "peft", "trl", "accelerate", "vllm", "datasets", "PyYAML", "math-verify",
          "torchao", "triton"]
    out["versions"] = versions(sorted(set(pk) | set(JOB.get("pins") or {})))
    cmds = {t.get("cmd") for lane in JOB.get("lanes", []) for t in lane}
    p = subprocess.run([sys.executable, "-c", SMOKE], capture_output=True, text=True, timeout=600)
    try:
        out["lora_fp16_smoke"] = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        out["lora_fp16_smoke"] = {"ok": False, "rc": p.returncode, "err": (p.stdout + p.stderr)[-2500:]}
    bad = []
    want = JOB.get("expect_gpu", "T4")
    if want and not any(want in g for g in smi):
        bad.append(f"expected GPU {want!r}, nvidia-smi: {smi}")
    if JOB.get("python") and not out["python"].startswith(JOB["python"]):
        bad.append(f"python {out['python']} != expected {JOB['python']} (Kaggle image changed)")
    for pkg, ver in (JOB.get("pins") or {}).items():
        got = out["versions"].get(pkg)
        if got is None or not (got == ver or got.startswith(ver + "+")):
            bad.append(f"pin violated: {pkg}=={ver}, found {got}")
    if not out["lora_fp16_smoke"].get("ok"):
        bad.append("fp16 LoRA smoke failed")
    if JOB.get("unsloth_install"):
        env = dict(os.environ, PYTHONPATH=UNS_SITE + os.pathsep + os.environ.get("PYTHONPATH", ""))
        q = subprocess.run([sys.executable, "-c", "import unsloth, importlib.metadata as m; "
                            "from unsloth import FastLanguageModel; import trl; "
                            "print('UNSLOTH_OK', m.version('unsloth'), m.version('unsloth_zoo'), trl.__version__)"],
                           capture_output=True, text=True, timeout=600, env=env, cwd="/tmp")
        out["unsloth_smoke"] = {"rc": q.returncode, "tail": (q.stdout + q.stderr)[-1500:]}
        out["unsloth_ok"] = q.returncode == 0 and "UNSLOTH_OK" in q.stdout
    if "grpo" in cmds and out["versions"].get("trl") is None:
        bad.append("trl missing but a grpo task is queued")
    out["violations"] = bad
    dump(f"{RES}/env_check.json", out)
    print("ENVCHECK", json.dumps({"ok": not bad, "violations": bad, "gpus": smi, "python": out["python"]}),
          flush=True)
    return out


def _finished(run_id):
    try:
        with open(f"{RES}/tasks.jsonl", encoding="utf-8") as f:
            return any(json.loads(line).get("run_id") == run_id for line in f if line.strip())
    except FileNotFoundError:
        return False


def ledger(rec):
    with _LOCK, open(f"{RES}/tasks.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_task(gpu, task, code_root):
    task = subst(task)
    cmd = task["cmd"]
    t = time.time()
    left = JOB.get("deadline_s", 41400) - (t - T0)
    if left < 120:
        rec = {"lane": gpu, "cmd": cmd, "status": "SKIPPED_DEADLINE"}
        ledger(rec)
        return rec
    req = task.get("requires")
    if req and task.get("wait_max_s"):  # the producer may still be running on the other lane
        t_w = time.time()
        while (time.time() - t_w < task["wait_max_s"] and not os.path.exists(f"{RES}/{req}/DONE")
               and not _finished(req)):
            time.sleep(20)
    if req and not os.path.exists(f"{RES}/{req}/DONE"):
        if task.get("fallback_config"):  # e.g. Unsloth gate failed -> the same run on the HF engine
            task = dict(task, config=task["fallback_config"])
            ledger({"lane": gpu, "cmd": cmd, "status": "FALLBACK", "requires": req,
                    "run_id": task["config"].get("run_id")})
        else:
            rec = {"lane": gpu, "cmd": cmd, "status": "SKIPPED_DEPENDENCY", "requires": req}
            ledger(rec)
            return rec
    t = time.time()
    left = JOB.get("deadline_s", 41400) - (t - T0)
    budget = int(min(task.get("budget_s", left), left))
    pp = [code_root]
    if cmd == "sft" and (task.get("config") or {}).get("engine") == "unsloth":
        pp.append(UNS_SITE)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               PYTHONPATH=os.pathsep.join(pp + [os.environ.get("PYTHONPATH", "")]),
               UNSLOTH_RETURN_LOGITS="0", HF_HUB_DISABLE_PROGRESS_BARS="1")
    if cmd == "merge":
        run_id = "merge-" + os.path.basename(task["out"].rstrip("/"))
        argv = [sys.executable, "-m", "vi_math_verified_aug.train.cli", "merge", "--base", task["base"],
                "--adapter", task["adapter"], "--out", task["out"], "--precision", task.get("precision", "fp16")]
    else:
        conf = dict(task["config"])
        run_id = conf.get("run_id") or f"{cmd}-{conf.get('arm')}-{conf.get('size')}-s{conf.get('seed')}"
        conf["run_id"] = run_id
        conf["out_dir"] = f"{RES}/{run_id}"
        os.makedirs(CFG_DIR, exist_ok=True)
        cpath = f"{CFG_DIR}/{run_id}.json"  # JSON is valid YAML
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(conf, f, ensure_ascii=False, indent=1)
        argv = [sys.executable, "-m", "vi_math_verified_aug.train.cli", cmd, "--config", cpath,
                "--budget-s", str(budget)]
    os.makedirs(f"{RES}/logs", exist_ok=True)
    logf = f"{RES}/logs/{run_id}.log"
    with open(logf, "a", encoding="utf-8") as lf:
        lf.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(argv)}\n")
        lf.flush()
        p = subprocess.Popen(argv, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
                             cwd="/tmp")
        try:
            rc = p.wait(timeout=budget + JOB.get("grace_s", 300))
        except subprocess.TimeoutExpired:  # hard kill: vLLM's engine is a grandchild, kill the whole group
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                p.kill()
            p.wait()
            rc = "timeout"
    rec = {"lane": gpu, "cmd": cmd, "run_id": run_id, "rc": rc,
           "status": "TIMEOUT_KILLED" if rc == "timeout" else STATUS.get(rc, "FAILED"),
           "start_min": round((t - T0) / 60, 2), "s": round(time.time() - t, 1), "budget_s": budget}
    if rc == 0 and JOB.get("drop_ckpt_on_done") and cmd != "merge":  # keep outputs small: adapters stay
        import shutil
        shutil.rmtree(f"{RES}/{run_id}/ckpt", ignore_errors=True)
    ledger(rec)
    print(f"[lane {gpu}] {cmd} {run_id} -> {rec['status']} ({rec['s']} s)", flush=True)
    return rec


def lane(gpu, tasks, code_root):
    for task in tasks:
        try:
            run_task(gpu, task, code_root)
        except Exception:  # noqa: BLE001
            ledger({"lane": gpu, "cmd": task.get("cmd"), "status": "LAUNCH_ERROR", "error": traceback.format_exc()})


def main():
    os.makedirs(RES, exist_ok=True)
    meta = {"job": JOB, "t0": T0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    dump(f"{RES}/meta.json", meta)  # run_id first, so a stale `kernels output` is detectable even on a crash
    if JOB.get("wheels_dir_glob"):
        hits = sorted(glob.glob(JOB["wheels_dir_glob"]))
        if hits:
            reqs = [f"{k}=={v}" for k, v in (JOB.get("pins") or {}).items() if k != "torch"]
            p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", hits[0]]
                               + reqs, capture_output=True, text=True)
            meta["wheels_install"] = {"dir": hits[0], "rc": p.returncode, "tail": (p.stdout + p.stderr)[-1500:]}
    if JOB.get("unsloth_install"):
        t = time.time()
        p = subprocess.run(JOB["unsloth_install"], shell=True, capture_output=True, text=True)
        meta["unsloth_install"] = {"cmd": JOB["unsloth_install"], "rc": p.returncode, "s": round(time.time() - t, 1),
                                   "tail": (p.stdout + p.stderr)[-1500:]}
    if JOB.get("install"):
        t = time.time()
        p = subprocess.run(JOB["install"], shell=True, capture_output=True, text=True)
        meta["install"] = {"cmd": JOB["install"], "rc": p.returncode, "s": round(time.time() - t, 1),
                           "tail": (p.stdout + p.stderr)[-1500:]}
    meta["env_fix"] = []
    for c in JOB.get("env_fix", ["pip uninstall -y -q torchao"]):  # S2/W2: peft refuses torchao<0.16
        p = subprocess.run(c, shell=True, capture_output=True, text=True)
        meta["env_fix"].append({"cmd": c, "rc": p.returncode})
    meta["envcheck"] = envcheck()
    dump(f"{RES}/meta.json", meta)
    if meta["envcheck"]["violations"] and JOB.get("fail_fast", True):
        meta["fatal"] = meta["envcheck"]["violations"]
        dump(f"{RES}/meta.json", meta)
        print("FATAL: ENVCHECK failed, no task run", flush=True)
        sys.exit(5)
    global CODE_ROOT
    code_root = find_code_root()
    CODE_ROOT = code_root
    meta["code_root"] = code_root
    dump(f"{RES}/meta.json", meta)
    th = [threading.Thread(target=lane, args=(g, tasks, code_root)) for g, tasks in enumerate(JOB["lanes"])]
    for t in th:
        t.start()
    for t in th:
        t.join()
    meta["wall_s"] = round(time.time() - T0, 1)
    dump(f"{RES}/meta.json", meta)
    print("ALL DONE", meta["wall_s"], flush=True)


if __name__ == "__main__":
    main()
