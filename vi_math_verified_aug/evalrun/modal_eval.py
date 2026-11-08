"""Modal runner for reported evaluation (O1 §3.2; T1r) + a GPU smoke of the training code paths.

Eval platform: **L4 bf16** with a GPU fallback list (``["L4", "A10G"]``; S2 saw "L4" served as A10G -- the real
GPU is recorded in every manifest), ``retries=0``, a hard ``timeout``, vLLM 0.33.0 with
``VLLM_USE_FLASHINFER_SAMPLER=0``.  One container = one vLLM engine = one base model; every eval config of a block
(4-shot base row, LoRA adapters) runs on that engine through ``evalrun.runner.run(cfg, backend=...)``.  Item-level
``preds.jsonl`` is fsynced per chunk and the Volume is committed after every chunk, so a killed container keeps
everything finished so far.

    modal billing summary                                   # log spend before
    modal volume put t1r-eval <local> /<remote>             # adapters / eval data (see T1r_first_runs.md §6)
    modal run vi_math_verified_aug/evalrun/modal_eval.py::eval_main --spec jobs/t1r/eval_<x>.json
    modal volume get t1r-eval /eval/<run_id> <local>        # pull preds.jsonl / metrics.json / manifest.json
    modal billing summary                                   # log spend after

``smoke_main`` runs the SFT engines (HF padded, Unsloth packed/padded, LFM2.5 HF) for a few steps on a T4 plus a tiny
vLLM eval of the produced adapters, on TRAIN/dev items only -- a code-path check before any Kaggle quota is spent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

if modal.is_local():  # never resolve repo paths inside the container (S2 Modal lesson 1)
    REPO = Path(__file__).resolve().parents[2]
    PKG = REPO / "vi_math_verified_aug"
else:
    REPO = PKG = None

PINS = ["vllm==0.33.0", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0", "math-verify==0.9.0",
        "pyyaml", "datasets==4.8.5"]   # datasets: Kaggle-image package that unsloth_zoo imports
UNS = ("pip install -q --no-deps --target /uns_site unsloth==2026.11.3 unsloth_zoo==2026.11.3 trl==1.17.1 "
       "cut_cross_entropy==25.1.1 bitsandbytes==0.50.2")
ENV = {"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_FLASHINFER_SAMPLER": "0",
       "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1",
       # QA-T E-det: batch-invariant kernels.  Measured (A10, Qwen3-0.6B, 128 dev items, greedy, chunk 128 vs 16):
       # default 87/128 identical responses, VLLM_BATCH_INVARIANT=1 128/128; cost +13 % at chunk 128
       # (jobs/logs/T2/det-*.log, jobs/outputs/t2-det/)
       "VLLM_BATCH_INVARIANT": "1", "UNSLOTH_RETURN_LOGITS": "0"}
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(*PINS)
         .run_commands("pip uninstall -y torchao || true", UNS)
         .env(ENV))
_SHIP = {"__init__.py", "train", "evalrun", "eval"}   # only the stack that runs; other workstreams edit the rest of
                                                    # the package concurrently ("modified during build" races)


def _ignore(p) -> bool:
    parts = Path(p).parts
    rel = parts[parts.index("vi_math_verified_aug") + 1:] if "vi_math_verified_aug" in parts else parts
    return (not rel or rel[0] not in _SHIP or "__pycache__" in rel or str(p).endswith(".pyc"))


if modal.is_local():
    image = image.add_local_dir(str(PKG), "/root/vi_math_verified_aug", ignore=_ignore)
vol = modal.Volume.from_name("t1r-eval", create_if_missing=True)
hf_cache = modal.Volume.from_name("t1r-hf-cache", create_if_missing=True)
app = modal.App("t1r-eval", image=image)
VOLS = {"/vol": vol, "/root/.cache/huggingface": hf_cache}


def _gpu() -> str:
    return subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                          capture_output=True, text=True).stdout.strip()


def _eval_block(spec: dict) -> dict:
    """spec = {"base": HF id, "precision": "bf16", "configs": [raw eval config dicts], "tag": str}."""
    sys.path.insert(0, "/root")
    from vi_math_verified_aug.evalrun.runner import VLLMBackend
    from vi_math_verified_aug.train import config as C
    from vi_math_verified_aug.evalrun import runner as RN

    t0 = time.time()
    gpu = _gpu()
    for k, v in (spec.get("env") or {}).items():   # e.g. VLLM_BATCH_INVARIANT=1 (determinism check)
        os.environ[k] = str(v)
    cfgs = []
    for raw in spec["configs"]:
        raw = json.loads(json.dumps(raw).replace("{vol}", "/vol"))
        raw.setdefault("out_dir", "/vol/eval/{run_id}")
        cfgs.append(C.build(raw))
    any_lora = any(c["eval"]["adapter_path"] for c in cfgs)
    t_load = time.time()
    backend = VLLMBackend(cfgs[0], spec["base"], None, enable_lora=any_lora)
    load_s = round(time.time() - t_load, 1)
    out = {"gpu": gpu, "engine_load_s": load_s, "env": spec.get("env"), "runs": {}}
    for cfg in cfgs:
        t = time.time()
        try:
            rc = RN.run(cfg, backend=backend, on_chunk=lambda _o: vol.commit())
        except Exception as e:  # noqa: BLE001  (one bad config must not lose the rest of the block)
            out["runs"][cfg["run_id"]] = {"rc": "error", "error": repr(e)[:1500]}
            vol.commit()
            continue
        man = json.loads(Path(cfg["out_dir"], "manifest.json").read_text(encoding="utf-8"))
        out["runs"][cfg["run_id"]] = {"rc": rc, "s": round(time.time() - t, 1), "out_dir": cfg["out_dir"],
                                      "metrics": man.get("metrics"), "throughput": man.get("throughput")}
        man["platform"] = {"gpu": gpu, "engine_load_s": load_s, "shared_engine": True}
        Path(cfg["out_dir"], "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
        vol.commit()
    out["wall_s"] = round(time.time() - t0, 1)
    Path("/vol/eval/_blocks").mkdir(parents=True, exist_ok=True)
    Path(f"/vol/eval/_blocks/{spec.get('tag', 'block')}.json").write_text(json.dumps(out, indent=1))
    vol.commit()
    return out


@app.function(gpu=["L4", "A10G"], timeout=3 * 3600, retries=0, volumes=VOLS)
def eval_l4(spec: dict) -> dict:
    return _eval_block(spec)


@app.function(gpu="L40S", timeout=3 * 3600, retries=0, volumes=VOLS)
def eval_l40s(spec: dict) -> dict:
    """THE primary-eval GPU (lead 2026-11-08, R-GPU): L40S bf16, pinned, NO fallback (same seed, L40S vs H100 flips
    13.7 % of dev items).  If L40S is unavailable the call waits."""
    return _eval_block(spec)


@app.function(gpu="A10G", timeout=3 * 3600, retries=0, volumes=VOLS)
def eval_a10g(spec: dict) -> dict:
    """THE primary-eval GPU (QA-T E-det, lead 2026-11-08): one pinned type, no fallback.  A10G because every Modal
    request for "L4" so far was served by A10G (S2 x2, T1r x3) and the base rows already ran there."""
    return _eval_block(spec)


def _run_train_task(t: dict, root: str = "/vol/train") -> dict:
    """One train.cli task as a subprocess (cwd=/tmp; Unsloth on PYTHONPATH only for engine=unsloth), committing the
    Volume every 60 s so a killed container keeps its train_log / dev_log / checkpoints."""
    if t.get("cmd") == "merge":  # fold an adapter into its base (GRPO starts from the merged SFT model)
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "vi_math_verified_aug.train.cli", "merge", "--base", t["base"],
                            "--adapter", t["adapter"].replace("{vol}", "/vol"),
                            "--out", t["out"].replace("{vol}", "/vol"), "--precision", t.get("precision", "fp16")],  # GRPO: fp16 merge
                           env=dict(os.environ, PYTHONPATH="/root"), cwd="/tmp", capture_output=True, text=True,
                           timeout=1800)
        vol.commit()
        return {"run_id": "merge:" + t["out"], "rc": p.returncode, "s": round(time.time() - t0, 1), "gpu": _gpu(),
                "tail": (p.stdout + p.stderr)[-2500:]}
    if t.get("cmd") == "sft-cell":   # all arms of one cell in one process (base loaded once)
        return _run_cell_task(t, root)
    conf = json.loads(json.dumps(t["config"]).replace("{vol}", "/vol"))
    conf.setdefault("out_dir", f"{root}/{conf['run_id']}")
    os.makedirs(f"{root}/_configs", exist_ok=True)
    cpath = f"{root}/_configs/{conf['run_id']}.json"
    Path(cpath).write_text(json.dumps(conf), encoding="utf-8")
    cmd = t.get("cmd", conf.get("kind", "sft"))
    # trl 1.17.1 lives in the isolated Unsloth dir (S2 lock): on the path for Unsloth SFT and for GRPO
    pp = ["/root"] + (["/uns_site"] if (conf.get("engine") == "unsloth" or cmd == "grpo") else [])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(pp), UNSLOTH_RETURN_LOGITS="0")
    budget = int(t.get("budget_s", 3600))
    Path(conf["out_dir"]).mkdir(parents=True, exist_ok=True)
    logf = open(Path(conf["out_dir"], "stdout.log"), "a", encoding="utf-8")
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-m", "vi_math_verified_aug.train.cli", cmd, "--config", cpath,
                          "--budget-s", str(budget)], env=env, cwd="/tmp", stdout=logf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    rc = None
    while rc is None:
        try:
            rc = p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            if time.time() - t0 > budget + 300:
                import signal
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                rc = "timeout"
            vol.commit()
    logf.close()
    vol.commit()
    tail = Path(conf["out_dir"], "stdout.log").read_text(encoding="utf-8", errors="replace")[-2500:]
    return {"run_id": conf["run_id"], "rc": rc, "s": round(time.time() - t0, 1), "gpu": _gpu(), "tail": tail}


def _flat(xs):
    out = []
    for x in xs:
        out += x if isinstance(x, list) else [x]
    return out


def _run_cell_task(t: dict, root: str) -> list[dict]:
    os.makedirs(f"{root}/_configs", exist_ok=True)
    paths, ids = [], []
    for c in t["configs"]:
        conf = json.loads(json.dumps(c).replace("{vol}", "/vol"))
        conf.setdefault("out_dir", f"{root}/{conf['run_id']}")
        Path(conf["out_dir"]).mkdir(parents=True, exist_ok=True)
        p = f"{root}/_configs/{conf['run_id']}.json"
        Path(p).write_text(json.dumps(conf), encoding="utf-8")
        paths.append(p)
        ids.append(conf["run_id"])
    res_path = f"{root}/_configs/cell-{ids[0]}.result.json"
    logp = Path(root, f"cell-{ids[0]}.stdout.log")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(["/root", "/uns_site"]), UNSLOTH_RETURN_LOGITS="0")
    budget = int(t.get("budget_s", 3600))
    t0 = time.time()
    with open(logp, "a", encoding="utf-8") as lf:
        p = subprocess.Popen([sys.executable, "-m", "vi_math_verified_aug.train.cli", "sft-cell", "--configs", *paths,
                              "--results", res_path] + (["--stop-on-fail"] if t.get("stop_on_fail") else []), env=env, cwd="/tmp", stdout=lf, stderr=subprocess.STDOUT,
                             start_new_session=True)
        rc = None
        while rc is None:
            try:
                rc = p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                if time.time() - t0 > budget + 300:
                    import signal
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                    rc = "timeout"
                vol.commit()
    vol.commit()
    per = json.loads(Path(res_path).read_text()) if Path(res_path).exists() else {}
    tail = logp.read_text(encoding="utf-8", errors="replace")[-2500:]
    return [{"run_id": i, "rc": per.get(i, rc if rc != 0 else "missing"), "s": round(time.time() - t0, 1),
             "gpu": _gpu(), "tail": tail if i == ids[-1] else ""} for i in ids]


@app.function(gpu=["L4", "A10G", "L40S"], timeout=4 * 3600, retries=0, volumes=VOLS)
def train_gpu(tasks: list[dict]) -> list[dict]:
    """bf16 training on the Ampere/Ada fallback list (one container runs its tasks in order)."""
    return _flat([_run_train_task(t) for t in tasks])


@app.function(gpu="L40S", timeout=4 * 3600, retries=0, volumes=VOLS)
def train_l40s(tasks: list[dict]) -> list[dict]:
    """THE training platform of every comparison block (lead, FINAL 2026-11-08): L40S bf16, no GPU fallback (wait
    instead), one lane per container (3 processes per L40S crashed with CUBLAS_STATUS_ALLOC_FAILED)."""
    return _flat([_run_train_task(t) for t in tasks])


@app.function(gpu="T4", timeout=4 * 3600, retries=0, volumes=VOLS)
def train_t4(tasks: list[dict]) -> list[dict]:
    """fp16 training on T4 (Kaggle-equivalent precision: fp16 overflow checks, LFM2.5 smoke)."""
    return _flat([_run_train_task(t) for t in tasks])


@app.local_entrypoint()
def train_main(spec: str):
    """spec = {"groups": [{"gpu": "ampere"|"t4", "tasks": [...]}, ...]}; one container per group, all in parallel."""
    s = json.loads(Path(spec).read_text(encoding="utf-8"))
    fns = {"t4": train_t4, "l40s": train_l40s, "ampere": train_gpu}
    calls = [fns[g["gpu"]].spawn(g["tasks"]) for g in s["groups"]]
    print("spawned", [c.object_id for c in calls], flush=True)
    res = []
    for c in calls:
        try:
            r = c.get(timeout=s.get("wait_s", 4 * 3600 + 1200))
        except Exception as e:  # noqa: BLE001  (timeout or remote failure: keep collecting the others)
            r = [{"error": repr(e)[:1500]}]
            try:
                c.cancel()
            except Exception:  # noqa: BLE001
                pass
        res.append(r)
        for x in r:
            print(x.get("run_id"), x.get("rc"), x.get("s"), x.get("gpu"), flush=True)
            print((x.get("tail") or x.get("error") or "")[-800:], flush=True)
    Path(spec).with_suffix(".result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


@app.function(gpu="T4", timeout=3600, retries=0, volumes=VOLS)
def smoke_t4(tasks: list[dict], eval_spec: dict | None) -> dict:
    """Run train.cli tasks exactly like the Kaggle template (subprocess, cwd=/tmp, Unsloth on PYTHONPATH only for
    engine=unsloth), then an optional tiny eval block on fp16 vLLM."""
    gpu = _gpu()
    res = {"gpu": gpu, "tasks": []}
    os.makedirs("/vol/smoke/configs", exist_ok=True)
    for t in tasks:
        conf = json.loads(json.dumps(t["config"]).replace("{vol}", "/vol"))
        cpath = f"/vol/smoke/configs/{conf['run_id']}.json"
        Path(cpath).write_text(json.dumps(conf), encoding="utf-8")
        pp = ["/root"] + (["/uns_site"] if conf.get("engine") == "unsloth" else [])
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(pp), UNSLOTH_RETURN_LOGITS="0")
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "vi_math_verified_aug.train.cli", "sft", "--config", cpath,
                            "--budget-s", str(t.get("budget_s", 900))], env=env, cwd="/tmp", capture_output=True,
                           text=True, timeout=t.get("budget_s", 900) + 300)
        log = (p.stdout + p.stderr)
        Path(conf["out_dir"]).mkdir(parents=True, exist_ok=True)
        Path(conf["out_dir"], "stdout.log").write_text(log, encoding="utf-8")
        res["tasks"].append({"run_id": conf["run_id"], "rc": p.returncode, "s": round(time.time() - t0, 1),
                             "tail": log[-2500:]})
        vol.commit()
    if eval_spec:
        try:
            res["eval"] = _eval_block(eval_spec)
        except Exception as e:  # noqa: BLE001
            import traceback
            res["eval"] = {"error": traceback.format_exc()[-3000:]}
    Path("/vol/smoke/result.json").write_text(json.dumps(res, indent=1))
    vol.commit()
    return res


@app.function(gpu="L40S", timeout=3600, retries=0, volumes=VOLS)
def parity_peft(spec: dict) -> dict:
    """QA-T: vLLM-adapter vs PEFT parity.  Greedy-decode ``n`` items with HF transformers + PEFT (bf16, batched
    generate, left padding) and compare with the greedy rows (sample_k = -1) of an existing vLLM preds.jsonl of the
    same adapter: grader-verdict agreement, extracted-span agreement, identical first 200 characters."""
    sys.path.insert(0, "/root")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vi_math_verified_aug.train import reward as R
    from vi_math_verified_aug.train import template as T
    from vi_math_verified_aug.train.modeling import bos_prefix
    preds = [json.loads(x) for x in Path(spec["preds"].replace("{vol}", "/vol")).read_text().splitlines() if x.strip()]
    greedy = [r for r in preds if r["sample_k"] == -1][:spec.get("n", 100)]
    items = {}
    for line in Path(spec["items"].replace("{vol}", "/vol")).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        items[str(r.get("id"))] = r
    tok = AutoTokenizer.from_pretrained(spec["base"])
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(spec["base"], dtype=torch.bfloat16).cuda()
    m = PeftModel.from_pretrained(m, spec["adapter"].replace("{vol}", "/vol")).eval()
    bos = bos_prefix(tok)
    out = []
    for i in range(0, len(greedy), 16):
        ch = greedy[i:i + 16]
        ids = [bos + tok(T.prompt(items[r["item_id"]]["question"], spec.get("lang", "vi")),
                         add_special_tokens=False)["input_ids"] for r in ch]
        L = max(map(len, ids))
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        x = torch.tensor([[pad] * (L - len(a)) + a for a in ids]).cuda()
        att = torch.tensor([[0] * (L - len(a)) + [1] * len(a) for a in ids]).cuda()
        with torch.no_grad():
            g = m.generate(input_ids=x, attention_mask=att, do_sample=False, max_new_tokens=1024, pad_token_id=pad,
                           eos_token_id=tok.eos_token_id)
        for r, row in zip(ch, g[:, L:].tolist()):
            if tok.eos_token_id in row:
                row = row[:row.index(tok.eos_token_id)]
            text = T.cut_at_stop(tok.decode(row, skip_special_tokens=True))
            v = R.score(text, items[r["item_id"]]["gold"])
            out.append({"item_id": r["item_id"], "same_verdict": int(v["correct"]) == r["correct"],
                        "same_span": v["span"] == r.get("pred_span"), "same_200": text[:200] == r["response"][:200],
                        "peft_correct": int(v["correct"]), "vllm_correct": r["correct"]})
    n = len(out)
    res = {"n": n, "gpu": _gpu(), **{k: round(sum(o[k] for o in out) / n, 4) for k in
                                     ("same_verdict", "same_span", "same_200", "peft_correct", "vllm_correct")}}
    Path("/vol/eval/_parity").mkdir(parents=True, exist_ok=True)
    Path(f"/vol/eval/_parity/{spec.get('tag', 'parity')}.json").write_text(json.dumps({"summary": res, "rows": out}))
    vol.commit()
    return res


@app.local_entrypoint()
def parity_main(spec: str):
    s = json.loads(Path(spec).read_text(encoding="utf-8"))
    print(json.dumps(parity_peft.remote(s)), flush=True)


@app.local_entrypoint()
def eval_main(spec: str, gpu: str = "l40s"):
    s = json.loads(Path(spec).read_text(encoding="utf-8"))
    call = {"l40s": eval_l40s, "a10g": eval_a10g, "l4": eval_l4}[gpu].spawn(s)
    print("spawned", call.object_id, flush=True)
    try:
        out = call.get(timeout=s.get("wait_s", 3 * 3600 + 1200))
    except TimeoutError:
        call.cancel()
        raise
    dest = Path(spec).with_suffix(".result.json")
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}), flush=True)
    for rid, r in out["runs"].items():
        print(rid, r["rc"], r.get("s"), json.dumps(r.get("metrics") or r.get("error"))[:600], flush=True)


@app.local_entrypoint()
def smoke_main(spec: str):
    s = json.loads(Path(spec).read_text(encoding="utf-8"))
    call = smoke_t4.spawn(s["tasks"], s.get("eval"))
    print("spawned", call.object_id, flush=True)
    try:
        out = call.get(timeout=3600 + 1200)
    except TimeoutError:
        call.cancel()
        raise
    Path(spec).with_suffix(".result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for t in out["tasks"]:
        print(t["run_id"], "rc", t["rc"], t["s"], "\n", t["tail"][-1200:], flush=True)
    print(json.dumps(out.get("eval"), indent=1)[:3000], flush=True)
