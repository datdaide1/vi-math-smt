"""T2 launcher: turn the costed queue (``train/grid.py``) into Kaggle kernels (2 runs per T4x2 kernel, one per GPU) or
Modal containers as the training files land, pull the adapters, and evaluate them on Modal L4 with the frozen T1r
eval config.  Laptop-light: stdlib + the kaggle / modal CLIs only (no torch).

    python -m vi_math_verified_aug.train.launcher status
    python -m vi_math_verified_aug.train.launcher tick [--dry-run] [--max-prio 5] [--backend kaggle|modal]
                                                     [--data-root DIR] [--n-train-cap N] [--max-kernels 2]
    python -m vi_math_verified_aug.train.launcher loop  [--every 300] ...          # tick until nothing is left

One ``tick``:
  1. poll launched kernels / Modal calls; pull finished outputs (adapters only, ckpt dropped in-kernel);
  2. upload new adapters to the Modal Volume and evaluate them (one eval block per base model);
  3. pick READY rows (data file present + valid, dependencies DONE) in priority order, pack them 2 per kernel and
     launch, unless a cap says stop:
       * Kaggle quota used this week - baseline >= ``--kaggle-cap-h`` (20)        -> no new kernels
       * Modal spend today (t1r-eval app) >= ``--modal-cap-usd`` (18)            -> no new Modal work
       * local time >= ``--stop-at`` (17:00)                                    -> no new runs
State: ``jobs/queue/t2_state.json`` (per run: status, kernel, timestamps, eval).  Logs: ``jobs/logs/T2/``.
``--dry-run`` renders kernels / datasets / configs and validates every data file, but pushes nothing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QUEUE = REPO / "jobs" / "queue" / "t2_queue.jsonl"
STATE = REPO / "jobs" / "queue" / "t2_state.json"
LOGS = REPO / "jobs" / "logs" / "T2"
BUILD = REPO / "jobs" / "build"
KAGGLE = os.environ.get("KAGGLE_BIN", r"E:\venv_tools\Scripts\kaggle.exe")
MODAL = os.environ.get("MODAL_BIN", r"E:\venv_tools\Scripts\modal.exe")
OWNER = "datuni"
EVAL_SETS = {"vi": {"name": "vi_primary_1000", "path": "{vol}/eval_data/vi_primary_1000.jsonl", "lang": "vi"},
             "en": {"name": "en_primary_1000", "path": "{vol}/eval_data/en_primary_1000.jsonl", "lang": "en"}}
# QA-T HIGH 1: arm runs use a dev set from the DEV seed bucket (seed_id % 20 == 0, excluded from generation),
# built by train/build_dev.py from S3's data/seeds/dev_v1.jsonl, EN mirror on the same ids.  (dev300 overlaps ~4 %
# of the N/Y generation seeds; it stays only for the S0-only G6 sweep.)
DEV = {"vi": "data/t1r/dev_seed_vi.jsonl", "en": "data/t1r/dev_seed_en.jsonl"}
EVAL_HASHES = "data/t1r/eval_qhashes.json"   # sha1 of normalised eval/benchmark questions (no text)
# step-0 train-loss band per base (guards.check_step0).  Qwen3-0.6B on S0-VI measured 0.733 (T1r smoke, HF and
# Unsloth agree to 4 decimals); generated arms / EN / 1.7B differ, so the band is wide and the canary records the
# real value per config.
STEP0_BAND = {"0.6B": [0.3, 1.6], "1.7B": [0.25, 1.5], "4B": [0.2, 1.4], "2.6B": [0.3, 2.0]}
CANARY_STEPS = 80
EVAL_GPU = "modal_l40s"   # ONE pinned GPU type for every primary eval (evalrun/modal_eval.py::eval_l40s), no fallback
# Platform of the comparison blocks (lead, after R-GPU): Modal L40S bf16, Unsloth packed, micro-batch 32 (one
# micro-batch per 32-sequence step, no accumulation); Kaggle T4 fp16 is the fallback.  Per-platform settings:
PLATFORMS = {
    "modal_l40s": {"precision": "bf16", "micro_batch": 32, "gpu": "l40s", "grad_ckpt": False,
                   "tok_per_s": {"0.6B": 36300}},                      # R-GPU [M]; 1.7B: not measured -> no guard
    "kaggle_t4": {"precision": "fp16", "micro_batch": 8, "gpu": None,
                  "tok_per_s": {"0.6B": 4226, "1.7B": 1865}},         # T1r smoke / O1 [M]
}
MAX_KERNEL_S = 9000                           # lead: a Kaggle kernel stays <= ~2.5 h


# ------------------------------------------------------------------------------------------------ helpers
_LIVE = {"st": None, "dry": True}
_CODE: dict = {}


def log(msg: str) -> None:
    """Every state transition is logged -> the state file is rewritten atomically right then (user rule: a restarted
    launcher resumes exactly where it was)."""
    if _LIVE["st"] is not None and not _LIVE["dry"]:
        save_state(_LIVE["st"])
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGS / "launcher.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sh(args: list[str], timeout: int = 1800, check: bool = False) -> str:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MSYS_NO_PATHCONV="1")
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                       timeout=timeout, cwd=str(REPO))
    out = (p.stdout or "") + (p.stderr or "")
    with open(LOGS / "cli.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "args": args[1:], "rc": p.returncode, "out": out[-3000:]}) + "\n")
    if check and p.returncode:
        raise RuntimeError(f"{args[:3]} rc={p.returncode}: {out[-800:]}")
    return out


def load_queue() -> list[dict]:
    return [json.loads(x) for x in QUEUE.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"runs": {}, "kernels": {},
                                                                                 "baseline_quota_h": None}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE)


def kaggle_quota_used() -> float | None:
    m = re.search(r"^GPU\s+([\d.]+)h", sh([KAGGLE, "quota"], timeout=120), re.M)
    return float(m.group(1)) if m else None


def modal_month_total() -> float:
    """Modal month-to-date for the WHOLE workspace (all apps) -- the hard guard (lead 2026-11-08)."""
    out = sh([MODAL, "billing", "report", "--for", "this month", "--json"], timeout=300)
    try:
        rows = json.loads(out[out.index("["):])
    except ValueError:
        return float("nan")
    return round(sum(float(r["cost"]) for r in rows), 4)


def modal_spend_today(app_prefixes=("t1r-eval",), period: str = "this month") -> float:
    """Billed Modal spend of our app(s) over ``period`` (month-to-date: a UTC day boundary must not reset it)."""
    out = sh([MODAL, "billing", "report", "--for", period, "--json"], timeout=300)
    try:
        rows = json.loads(out[out.index("["):])
    except ValueError:
        return float("nan")
    return round(sum(float(r["cost"]) for r in rows if str(r.get("description", "")).startswith(app_prefixes)), 4)


# ------------------------------------------------------------------------------------------------ traceability
CODE_PATHS = ["vi_math_verified_aug/train", "vi_math_verified_aug/evalrun", "vi_math_verified_aug/eval/math_verify.py",
              "vi_math_verified_aug/__init__.py"]


def code_stamp() -> dict:
    """git sha + dirty flag of the code that runs (the training/eval stack).  Other workstreams edit other parts of
    the package concurrently, so the dirty check is scoped to CODE_PATHS."""
    git = ["git", "-C", str(REPO)]
    sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    st = subprocess.run(git + ["status", "--porcelain", "--"] + CODE_PATHS, capture_output=True, text=True).stdout
    dirty = [ln[3:] for ln in st.splitlines() if ln.strip()]
    return {"git_sha": sha, "dirty": bool(dirty), "dirty_files": dirty[:20], "paths": CODE_PATHS}


def require_clean(args) -> dict:
    c = code_stamp()
    if c["dirty"] and not args.dry_run and not args.allow_dirty:
        raise SystemExit(f"refusing to launch from a dirty tree (commit first): {c['dirty_files']}")
    return c


# ------------------------------------------------------------------------------------------------ data check
class _CharTok:
    """Stand-in tokenizer for the laptop-side data check (field mapping, answer extraction, malformed rate)."""
    eos_token_id, bos_token_id = 1, None

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [2] * max(1, len(text) // 3)}


def check_data(path: Path) -> dict:
    from . import data as D
    try:
        rows = D.read_jsonl(path)
        items, st = D.prepare_pool(rows, _CharTok(), 10 ** 9)
        return {"ok": True, "n": st["n_eligible"], "malformed": st["n_malformed"], "approx_tokens_p50":
                st["pool_len_p50"]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)[:400]}


# ------------------------------------------------------------------------------------------------ configs
def data_identity(path: Path) -> dict:
    """sha256 from the producer's SHA256SUMS next to the file when present (else computed), and the row count."""
    from . import guards as G
    sha = None
    sums = path.parent / "SHA256SUMS"
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[-1].lstrip("*").endswith(path.name):
                sha = parts[0]
    real = G.file_sha256(path)
    if sha and sha != real:
        raise RuntimeError(f"{path}: SHA256SUMS says {sha}, file is {real}")
    with open(path, encoding="utf-8") as f:
        rows = sum(1 for line in f if line.strip())
    return {"sha256": real, "rows": rows, "from_sums": bool(sha)}


def sft_config(r: dict, data_rel: str, *, precision: str, engine: str, n_train_cap: int | None = None,
               ident: dict | None = None, platform: str = "kaggle_t4", micro_batch: int | None = None) -> dict:
    n = r["n_train"] if not n_train_cap else min(r["n_train"], n_train_cap)
    lang = r["lang"]
    raw = {"kind": "sft", "run_id": r["run_id"], "arm": r["arm"], "model": r["model"], "size": r["size"],
           "seed": r["seed"], "lang": lang, "precision": precision, "engine": engine,
           "budget": {"n_train": n, "epochs": r["epochs"]},
           "data": {"train": "{code}/" + data_rel},
           "sft": {"lr": r["lr"], "micro_batch": micro_batch or PLATFORMS.get(platform, {}).get("micro_batch", 8),
                   "max_length": 1024, "packing": engine == "unsloth",
                   "dev_path": "{code}/" + DEV[lang], "dev_every_steps": 200, "ckpt_every_min": 15,
                   "keep_ckpts": 1, "grad_ckpt": PLATFORMS.get(platform, {}).get("grad_ckpt", "auto"),
                   # lead: epoch hedge -- decay branches at every earlier epoch (with G6's 3-epoch main run: epochs
                   # 1 and 2; epoch 2 is G6's near-tie alternative); primary row = the final (fresh main-run) adapter
                   "branch_epochs": list(range(1, r["epochs"]))},
           "platform": platform, "code": _CODE.get("stamp"),
           "guards": {"enabled": True, "expected_sha256": (ident or {}).get("sha256"),
                      "expected_rows": (ident or {}).get("rows"), "eval_hashes_path": "{code}/" + EVAL_HASHES,
                      "step0_loss_band": STEP0_BAND.get(r["size"]),
                      "expected_tok_per_s": PLATFORMS.get(platform, {}).get("tok_per_s", {}).get(r["size"])},
           "notes": f"T2 {r['block']} | n_train={n} (arm-matched per language for Y+-/N+-)"}
    return raw


EXAM_SET = {"name": "vi_exam_core_verified", "path": "{vol}/eval_data/vi_exam_core_verified.jsonl", "lang": "vi"}


def base_eval_configs(model: str, size: str = "0.6B") -> list[dict]:
    """The base-model rows on THE eval platform (L40S, batch-invariant): 4-shot in the training template; VI primary
    K4+g + Vi-ExamMath greedy-only, EN primary K4+g.  run_id base-<model tag> is the GLMM base row."""
    tag = model.rstrip("/").split("/")[-1].lower().replace("-base", "")
    out = []
    for lang in ("vi", "en"):
        r = {"run_id": f"base-{tag}-4shot-{lang}", "arm": "base", "model": model, "size": size, "seed": 0,
             "lang": lang}
        for c in eval_configs(r, None):
            c["eval"]["shots"] = 4
            c["template"] = {"fewshot_path": f"{{vol}}/eval_data/fewshot4_{lang}.jsonl"}
            out.append(c)
    return out


def eval_configs(r: dict, adapter_vol: str) -> list[dict]:
    """Frozen eval protocol (lead 2026-11-08): primary_1000 (VI or EN) with K = 4 + greedy; Vi-ExamMath VERIFIED
    (1,367, VI runs only) GREEDY-ONLY -- a recorded deviation from T5's K = 4 (cost $0.357 vs $0.09 per adapter);
    the GLMM on Vi-ExamMath then uses one binary per item x run over 5 seeds.  Cap 1,024 (99 % of capped exam samples
    are repetition loops, T2 §5.1)."""
    def one(tag, sets, k):
        return {"kind": "eval", "code": _CODE.get("stamp"), "platform": EVAL_GPU,
                "run_id": f"ev-{r['run_id']}-{tag}", "arm": r["arm"], "model": r["model"], "size": r["size"],
                "seed": r["seed"], "lang": r["lang"], "precision": "bf16",
                "out_dir": "{vol}/eval/ev-" + r["run_id"] + "-" + tag,
                "eval": {"source_run_id": r["run_id"], "adapter_path": adapter_vol, "shots": 0, "max_new_tokens": 1024,
                         "k_samples": k, "greedy": True, "sets": sets}}
    out = [one("primary", [EVAL_SETS[r["lang"]]], 4)]
    if r["lang"] == "vi":
        out.append(one("exam-greedy", [EXAM_SET], 0))
    return out


# ------------------------------------------------------------------------------------------------ kaggle
def build_kernel(slug: str, runs: list[dict], *, data_root: Path, n_train_cap: int | None, engine_of) -> tuple:
    sys.path.insert(0, str(REPO))
    from .kaggle import render as RD
    data_dir = BUILD / f"{slug}-data"
    shutil.rmtree(data_dir, ignore_errors=True)
    RD.bundle_code(data_dir)
    lanes = []
    for r in runs:
        src = data_root / r["data"] if not Path(r["data"]).is_absolute() else Path(r["data"])
        rel = r["data"] if not str(r["data"]).startswith("data/") else r["data"]
        (data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, data_dir / rel)
        dev = DEV[r["lang"]]
        (data_dir / dev).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / dev, data_dir / dev)
        shutil.copy2(REPO / EVAL_HASHES, data_dir / EVAL_HASHES)
        ident = data_identity(data_dir / rel)
        raw = sft_config(r, rel, precision="fp16", engine=engine_of(r), n_train_cap=n_train_cap, ident=ident)
        from . import config as C
        C.build(json.loads(json.dumps(raw).replace("{code}", "/x")))
        b = int(3600 * (r["train_gpu_h"] * 1.8 + 0.25))
        t = {"cmd": "sft", "config": raw, "budget_s": b}
        if raw["engine"] == "unsloth":  # in-kernel fallback: same run id on the HF padded engine
            fb = sft_config(r, rel, precision="fp16", engine="hf", n_train_cap=n_train_cap, ident=ident)
            fb["guards"]["expected_tok_per_s"] = None  # HF is slower by design
            fb["notes"] += " | HF fallback after an Unsloth failure"
            fb["run_id"] = r["run_id"] + "-hf"  # own out dir: never resume the other engine's checkpoints
            t["on_fail"] = fb
        lanes.append([t])
    deadline = min(MAX_KERNEL_S, max(t["budget_s"] * (2 if t.get("on_fail") else 1) for l in lanes for t in l) + 900)
    job = RD.build_job(lanes, slug, env={"deadline_s": deadline, "drop_ckpt_on_done": True, "python": "3.13"})
    run_py = RD.render(job, slug, BUILD)
    return run_py, data_dir, job


def launch_kaggle(st: dict, slug: str, runs: list[dict], args, engine_of) -> None:
    run_py, data_dir, job = build_kernel(slug, runs, data_root=Path(args.data_root), n_train_cap=args.n_train_cap,
                                         engine_of=engine_of)
    rec = {"slug": slug, "job_run_id": job["run_id"], "runs": [r["run_id"] for r in runs], "backend": "kaggle",
           "built": str(run_py), "status": "BUILT"}
    if args.dry_run:
        log(f"DRY kernel {slug}: {rec['runs']} -> {run_py}")
        rec["status"] = "DRY"
        st["kernels"][slug] = rec
        return
    sys.path.insert(0, str(REPO))
    from ..infra.kaggle_runner import KaggleRunner
    kr = KaggleRunner(owner=OWNER, kaggle_bin=KAGGLE)
    ds = f"{slug}-data"
    kr.dataset_upsert(data_dir, ds, ds, message=job["run_id"])
    if not kr.dataset_ready(f"{OWNER}/{ds}"):
        log(f"dataset {ds} not ready; kernel {slug} not pushed")
        return
    kdir = kr.render_kernel(slug, run_py, dataset_sources=[f"{OWNER}/{ds}"])
    out = kr.push(kdir, timeout_s=int(job["deadline_s"]) + 1200)
    rec.update(status="PUSHED", pushed=time.time(), push_out=out.strip()[-300:])
    st["kernels"][slug] = rec
    for r in runs:
        st["runs"].setdefault(r["run_id"], {}).update(status="LAUNCHED", kernel=slug, launched=time.time())
    log(f"pushed {slug}: {rec['runs']}")


def poll_kaggle(st: dict, args) -> None:
    sys.path.insert(0, str(REPO))
    from ..infra.kaggle_runner import KaggleRunner
    kr = KaggleRunner(owner=OWNER, kaggle_bin=KAGGLE)
    for slug, k in st["kernels"].items():
        if k.get("backend") != "kaggle" or k["status"] not in ("PUSHED", "QUEUED", "RUNNING"):
            continue
        s = kr.status(f"{OWNER}/{slug}")
        if s != k["status"]:
            log(f"kernel {slug}: {k['status']} -> {s}")
            k["status"] = s
            k[f"t_{s.lower()}"] = time.time()
        if s in ("COMPLETE", "ERROR"):
            dest = REPO / "jobs" / "outputs" / slug
            kr.output(f"{OWNER}/{slug}", dest)
            if not kr.verify_output_run_id(dest, k["job_run_id"]):
                k["status"] = "STALE_OUTPUT"
                log(f"kernel {slug}: stale output, will retry pull")
                continue
            for rid in k["runs"]:
                d = dest / "results" / rid
                if not (d / "DONE").exists() and (dest / "results" / (rid + "-hf") / "DONE").exists():
                    d = dest / "results" / (rid + "-hf")  # the in-kernel HF fallback ran
                done = (d / "DONE").exists()
                ad = sorted((d / "adapters").glob("*")) if (d / "adapters").exists() else []
                st["runs"].setdefault(rid, {}).update(status="TRAINED" if done and ad else "FAILED",
                                                      out=str(d), adapter=str(ad[-1]) if ad else None,
                                                      finished=time.time())
                log(f"run {rid}: {st['runs'][rid]['status']}")
                post_check(st, rid, d)


# ------------------------------------------------------------------------------------------------ modal train
BASE_ROWS_USD = 0.22


def _spent_now(st: dict) -> float:
    """Modal spend for the cap: billed since the first live tick + the estimates of cells still in flight (their train
    and eval are not billed yet).  If billing is unavailable, the committed estimates of everything launched."""
    billed = st.get("modal_spent_usd")
    if billed is None:
        return st.get("committed_usd", 0.0)
    def unbilled(k):
        """Estimate of what this cell will still cost: Modal's per-app billing shows up within minutes (measured
        2026-11-08), so a finished training container is billed; an evaluated cell is billed."""
        if k.get("backend") != "modal":
            return 0.0
        if k.get("status") == "RUNNING":
            return k.get("est_usd", 0.0)
        spec = k.get("eval_spec")
        res = Path(spec).with_suffix(".result.json") if spec else None
        if res and res.exists():
            return 0.0
        return k.get("est_eval_usd", k.get("est_usd", 0.0))
    inflight = sum(unbilled(k) for k in st.get("kernels", {}).values())
    return round(billed + inflight, 3)


def cell_of(r: dict) -> str:
    """Balanced allocation unit (lead): every arm of one (size, language, seed) runs on ONE platform."""
    return f"{r['size']}|{r['lang']}|s{r['seed']}"


def launch_modal_train(st: dict, runs: list[dict], args, engine_of) -> None:
    """One L40S container per cell; its runs execute one after another inside it (``train_main`` group).  Each run
    writes checkpoints (<= 15 min, every epoch) and its adapters straight to the Volume (/vol/train/<run_id>), and
    the Volume is committed every 60 s, so a killed container loses at most a few minutes."""
    pf = PLATFORMS[args.platform]
    cells: dict[str, list] = {}
    for r in runs:
        cells.setdefault(cell_of(r), []).append(r)
    groups = []
    for cell, rs in cells.items():
        tasks = []
        for r in rs:
            ident = data_identity(Path(args.data_root) / r["data"])
            raw = sft_config(r, r["data"], precision=pf["precision"], engine=engine_of(r),
                             n_train_cap=args.n_train_cap, ident=ident, platform=args.platform)
            raw = json.loads(json.dumps(raw).replace("{code}/data/", "{vol}/train_data/"))
            tasks.append({"cmd": "sft", "config": raw, "budget_s": int(3600 * (r["train_gpu_h"] + 0.3))})
        # ONE process per cell (lead-approved after the GPU parity test, T2 §1): base loaded once, fresh seeded LoRA,
        # optimizer, scaler and schedule per arm
        seed0 = cell.endswith("|s0")   # the canary cell of its config keys: stop the cell at the first failure
        groups.append({"gpu": pf["gpu"], "cell": cell, "tasks": [{"cmd": "sft-cell", "configs": [t["config"] for t in tasks],
                                                                 "budget_s": sum(t["budget_s"] for t in tasks),
                                                                 "stop_on_fail": seed0}]})
    tag = f"t2-modal-{dt.datetime.now().strftime('%H%M%S%f')}-{sorted(cells)[0].replace('|', '-')}"
    spec = REPO / "jobs" / "t2" / f"{tag}.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(json.dumps({"groups": groups}, indent=1), encoding="utf-8")
    if args.dry_run:
        log(f"DRY modal train {tag}: {[r['run_id'] for r in runs]}")
        return
    for r in runs:  # data + dev + eval-hash files to the Volume (same relative layout under /train_data)
        _put_train_data(r, args)
    logf = open(LOGS / f"{tag}.log", "w", encoding="utf-8")
    p = subprocess.Popen([MODAL, "run", "vi_math_verified_aug/evalrun/modal_eval.py::train_main", "--spec",
                          str(spec)], cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT,
                         env=dict(os.environ, PYTHONIOENCODING="utf-8", MSYS_NO_PATHCONV="1"))
    st["kernels"][tag] = {"backend": "modal", "platform": args.platform, "pid": p.pid, "spec": str(spec),
                          "runs": [r["run_id"] for r in runs], "cells": sorted(cells), "status": "RUNNING",
                          "launched": time.time()}
    for r in runs:
        st.setdefault("cells", {})[cell_of(r)] = args.platform
        st["runs"].setdefault(r["run_id"], {}).update(
            status="LAUNCHED", kernel=tag, platform=args.platform, launched=time.time(),
            adapter_vol=f"{{vol}}/train/{r['run_id']}/adapters/epoch{r['epochs']}")
    log(f"modal train {tag}: {[r['run_id'] for r in runs]}")
    return tag


def poll_modal_train(st: dict) -> None:
    for tag, k in st["kernels"].items():
        if k.get("backend") != "modal" or k["status"] != "RUNNING":
            continue
        res = Path(k["spec"]).with_suffix(".result.json")
        if not res.exists():
            continue
        out = json.loads(res.read_text(encoding="utf-8"))
        k["status"] = "COMPLETE"
        for grp in out:
            for x in grp:
                rid = x.get("run_id")
                if rid:
                    st["runs"][rid].update(status="TRAINED" if x.get("rc") == 0 else "FAILED", finished=time.time(),
                                           rc=x.get("rc"))
                    log(f"run {rid}: {st['runs'][rid]['status']}")
                    post_check(st, rid, pull_modal_run(rid, REPO / "jobs" / "outputs" / "t2-modal" / rid))
                    r = st["runs"][rid]
                    if x.get("rc") != 0 and re.search(r"-s0(-|$)", rid):   # seed-0 cell failure blocks the language
                        lang = "vi" if "-vi-" in rid else "en"
                        st.setdefault("blocking", {})[f"lang:{lang}"] = {
                            "run": rid, "rc": x.get("rc"), "rule": "a guard failure in the seed-0 cell blocks the language"}
                        log(f"BLOCKING language {lang}: {rid} rc={x.get('rc')}")


# ------------------------------------------------------------------------------------------------ eval
def launch_evals(st: dict, rows: dict, args) -> None:
    todo = [rid for rid, s in st["runs"].items() if s.get("status") == "TRAINED" and not s.get("eval")]
    if not todo:
        return
    by_model = {}
    for rid in todo:
        r, s = rows[rid], st["runs"][rid]
        if s.get("adapter") and not s.get("adapter_vol"):
            if not args.dry_run:
                sh([MODAL, "volume", "put", "--force", "t1r-eval", s["adapter"], f"/adapters/{rid}"], check=True)
            s["adapter_vol"] = f"{{vol}}/adapters/{rid}"
        by_model.setdefault(r["model"], []).extend(eval_configs(r, s["adapter_vol"]))
    for model, cfgs in by_model.items():
        if not st.setdefault("base_rows", {}).get(model):   # base rows ride in the first eval container (lead: $0.22)
            cfgs = base_eval_configs(model) + cfgs
            st["base_rows"][model] = [c["run_id"] for c in base_eval_configs(model)]
            st["_pending_base_spec"] = True
        tag = f"t2-eval-{dt.datetime.now().strftime('%H%M%S%f')}"
        spec = REPO / "jobs" / "t2" / f"{tag}.json"
        spec.parent.mkdir(parents=True, exist_ok=True)
        spec.write_text(json.dumps({"base": model, "tag": tag, "configs": cfgs}, indent=1), encoding="utf-8")
        if st.pop("_pending_base_spec", False):
            st["base_eval_spec"] = str(spec)
        if args.dry_run:
            log(f"DRY eval {tag}: {[c['run_id'] for c in cfgs]}")
            continue
        logf = open(LOGS / f"{tag}.log", "w", encoding="utf-8")
        subprocess.Popen([MODAL, "run", "vi_math_verified_aug/evalrun/modal_eval.py::eval_main", "--spec",
                          str(spec)], cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT,
                         env=dict(os.environ, PYTHONIOENCODING="utf-8", MSYS_NO_PATHCONV="1"))
        for c in cfgs:
            if c["eval"]["source_run_id"] not in st["runs"]:
                continue   # base rows
            k = st["runs"][c["eval"]["source_run_id"]].get("kernel")
            if k in st.get("kernels", {}):
                st["kernels"][k]["eval_launched"] = time.time()
                st["kernels"][k]["eval_spec"] = str(spec)
            st["runs"][c["eval"]["source_run_id"]].setdefault("evals", []).append(c["run_id"])
            st["runs"][c["eval"]["source_run_id"]]["eval"] = {"spec": str(spec), "eval_run_id": c["run_id"],
                                                             "launched": time.time()}
        log(f"eval {tag}: {[c['run_id'] for c in cfgs]}")


# ------------------------------------------------------------------------------------------------ guards
def config_key(r: dict) -> str:
    """A NEW config = a new (arm, size, language, recipe) -> it gets a Modal canary before any Kaggle run."""
    return f"{r['arm']}|{r['size']}|{r['lang']}|lr{r['lr']:g}|ep{r['epochs']}"


def post_check(st: dict, rid: str, run_dir: Path) -> dict:
    """check_runs on a finished run; flags are recorded and BLOCK new launches until `launcher ack RID`."""
    from .check_runs import check
    res = check(run_dir, REPO / "jobs" / "outputs" / "T2_checks")
    st["runs"].setdefault(rid, {})["check"] = {"clean": res["clean"], "flags": res["flags"]}
    if not res["clean"]:
        st.setdefault("blocking", {})[rid] = res["flags"]
        log(f"BLOCKING: {rid} flags {res['flags']} (inspect jobs/outputs/T2_checks/{rid}; `launcher ack {rid}`)")
    return res


def pull_modal_run(rid: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("manifest.json", "train_log.jsonl", "dev_log.jsonl", "preflight.json", "config.yaml",
              "data_stats.json", "pack_gate.json", "stdout.log"):
        sh([MODAL, "volume", "get", "--force", "t1r-eval", f"/train/{rid}/{f}", str(dest / f)], timeout=300)
    return dest


def launch_canaries(st: dict, rows: list[dict], args, engine_of) -> None:
    """60-100-step canary of every NEW config on Modal (fp16 T4 = the Kaggle precision), logs live on the Volume
    (`launcher tail canary-...`).  One container per config; a config is released for Kaggle only when its canary
    finished rc 0 and check_runs found nothing."""
    todo, seen = [], set()
    for r in rows:
        k = config_key(r)
        if k in seen or k in st.setdefault("canary", {}):
            continue
        seen.add(k)
        todo.append(r)
    if not todo:
        return
    groups = []
    for r in todo:
        ident = data_identity(Path(args.data_root) / r["data"])
        pf = PLATFORMS[args.platform]   # the canary runs on THE platform of the block (L40S bf16 mb32)
        raw = sft_config(r, r["data"], precision=pf["precision"], engine=engine_of(r), n_train_cap=args.n_train_cap,
                         ident=ident, platform=args.platform)
        raw = json.loads(json.dumps(raw).replace("{code}/data/", "{vol}/train_data/"))
        raw["run_id"] = "canary-" + r["run_id"]
        raw["budget"]["max_steps"] = CANARY_STEPS
        raw["sft"]["ckpt_every_min"] = 0
        raw["notes"] += " | CANARY (80 steps; not a result)"
        groups.append({"gpu": pf["gpu"], "tasks": [{"cmd": "sft", "config": raw, "budget_s": 1800}]})
    tag = f"t2-canary-{dt.datetime.now().strftime('%H%M%S%f')}"
    spec = REPO / "jobs" / "t2" / f"{tag}.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(json.dumps({"groups": groups, "wait_s": 3600}, indent=1), encoding="utf-8")
    if args.dry_run:
        log(f"DRY canary {tag}: {[config_key(r) for r in todo]}")
        return
    for r in todo:
        _put_train_data(r, args)
    logf = open(LOGS / f"{tag}.log", "w", encoding="utf-8")
    subprocess.Popen([MODAL, "run", "vi_math_verified_aug/evalrun/modal_eval.py::train_main", "--spec", str(spec)],
                     cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT,
                     env=dict(os.environ, PYTHONIOENCODING="utf-8", MSYS_NO_PATHCONV="1"))
    for r in todo:
        st["canary"][config_key(r)] = {"status": "RUNNING", "run_id": "canary-" + r["run_id"], "spec": str(spec),
                                       "launched": time.time()}
    log(f"canary {tag}: {[config_key(r) for r in todo]}")


def _put_train_data(r: dict, args) -> None:
    sh([MODAL, "volume", "put", "--force", "t1r-eval", str(Path(args.data_root) / r["data"]),
        "/train_data/" + r["data"][len("data/"):]], check=True)
    for f in (DEV[r["lang"]], EVAL_HASHES):
        sh([MODAL, "volume", "put", "--force", "t1r-eval", str(REPO / f), "/train_data/" + f[len("data/"):]],
           check=True)


def poll_canaries(st: dict) -> None:
    for k, c in st.get("canary", {}).items():
        if c["status"] != "RUNNING":
            continue
        res = Path(c["spec"]).with_suffix(".result.json")
        if not res.exists():
            continue
        d = pull_modal_run(c["run_id"], REPO / "jobs" / "outputs" / "t2-canary" / c["run_id"])
        chk = post_check(st, c["run_id"], d)
        c.update(status="OK" if chk["clean"] else "FLAGGED", flags=chk["flags"], finished=time.time())
        log(f"canary {k}: {c['status']} {chk['flags']}")


# ------------------------------------------------------------------------------------------------ eval results
def _boot(vals: list, B: int = 2000, seed: int = 0) -> list:
    import random as _r
    rng = _r.Random(seed)
    n = len(vals)
    if not n:
        return [None, None]
    m = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(m[int(0.025 * B)], 4), round(m[int(0.975 * B)], 4)]


def eval_metrics(preds_path: Path) -> dict:
    """Per dataset: mean pass@1 over the K samples and greedy accuracy, each with a 95 % item-bootstrap CI."""
    from collections import defaultdict
    smp, grd = defaultdict(lambda: defaultdict(list)), defaultdict(dict)
    for line in preds_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["sample_k"] == -1:
            grd[r["dataset"]][r["item_id"]] = r["correct"]
        else:
            smp[r["dataset"]][r["item_id"]].append(r["correct"])
    out = {}
    for ds in sorted(set(smp) | set(grd)):
        p1 = [sum(v) / len(v) for v in smp[ds].values()]
        g = list(grd[ds].values())
        out[ds] = {"n": max(len(p1), len(g)),
                   "pass1": round(sum(p1) / len(p1), 4) if p1 else None, "pass1_ci": _boot(p1) if p1 else None,
                   "greedy": round(sum(g) / len(g), 4) if g else None, "greedy_ci": _boot(g) if g else None}
    return out


def poll_evals(st: dict) -> None:
    """Pull finished eval preds from the Volume and attach per-dataset metrics to the run (base rows too)."""
    want = [(rid, ev) for rid, s in st["runs"].items() for ev in s.get("evals", [])
            if ev not in s.get("eval_metrics", {})]
    want += [("_base", ev) for evs in st.get("base_rows", {}).values() for ev in evs
             if ev not in st.get("base_metrics", {})]
    for rid, ev in want:
        spec = st.get("base_eval_spec") if rid == "_base" else (st["runs"].get(rid, {}).get("eval") or {}).get("spec")
        if not spec or not Path(spec).with_suffix(".result.json").exists():
            continue
        dest = REPO / "jobs" / "outputs" / "t2-eval" / f"{ev}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        sh([MODAL, "volume", "get", "--force", "t1r-eval", f"/eval/{ev}/preds.jsonl", str(dest)], timeout=600)
        if not dest.exists():
            continue
        m = eval_metrics(dest)
        if rid == "_base":
            st.setdefault("base_metrics", {})[ev] = m
        else:
            st["runs"][rid].setdefault("eval_metrics", {})[ev] = m
        log(f"eval metrics {ev}: " + "; ".join(f"{d} p1 {v['pass1']} g {v['greedy']}" for d, v in m.items()))


def _fmt(d: dict, f: str) -> str:
    return "; ".join(f"{ds}: {v[f]} {v[f + '_ci']}" for ds, v in d.items() if v.get(f) is not None)


def write_cell_summary(st: dict, rows: dict, lang: str, seed: int, path: Path) -> bool:
    """The lead's review sheet for a finished cell (VI seed 0): per arm the loss-curve files, the pre-flight verdict,
    dev_v1 quick metrics, primary pass@1 / greedy with CIs per dataset, Vi-ExamMath greedy; base rows as reference."""
    rids = [r for r in rows if rows[r]["lang"] == lang and rows[r]["seed"] == seed and rows[r]["size"] == "0.6B"
            and rows[r]["kind"] == "sft" and rows[r]["priority"] <= 3]
    for r in rids:
        s = st["runs"].get(r, {})
        if not s.get("evals") or len(s.get("eval_metrics", {})) < len(s["evals"]):
            return False
    L = [f"# {lang.upper()} seed {seed} cell: review sheet (auto, launcher {time.strftime('%Y-%m-%d %H:%M')})", "",
         "Platform L40S bf16, Unsloth packed mb32, LR 1e-4, 3 epochs; eval L40S, VLLM_BATCH_INVARIANT=1, cap 1,024; "
         "primary = K=4 + greedy; Vi-ExamMath VERIFIED = greedy only. CIs: 95 % item bootstrap.", "",
         "| Arm | status / check | pre-flight | dev_v1 loss start -> end | quick-eval (end) | primary pass@1 [CI] | "
         "primary greedy [CI] | Vi-ExamMath greedy [CI] | loss curve |",
         "|---|---|---|---|---|---|---|---|---|"]
    for rid in rids:
        s = st["runs"][rid]
        out = REPO / "jobs" / "outputs" / "t2-modal" / rid
        pre = out / "preflight.json"
        pf = "missing"
        if pre.exists():
            pj = json.loads(pre.read_text(encoding="utf-8"))
            pf = f"OK ({len(pj['labels']['decoded'])} decoded, label frac {pj['labels']['label_frac_mean']})"
        dev = []
        if (out / "dev_log.jsonl").exists():
            dev = [json.loads(x) for x in (out / "dev_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        mid = [d for d in dev if d.get("kind") == "midrun"]
        dl = f"{mid[0]['dev_loss']:.4f} -> {mid[-1]['dev_loss']:.4f}" if len(mid) >= 2 else "?"
        q = (mid[-1].get("quick") or {}) if mid else {}
        qs = f"acc {q.get('correct')}, rep {q.get('repetition')}, no-ans {q.get('no_answer')}" if q else "?"
        prim = next((v for k, v in s["eval_metrics"].items() if k.endswith("-primary")), {})
        exam = next((v for k, v in s["eval_metrics"].items() if k.endswith("-exam-greedy")), {})
        chk = s.get("check") or {}
        L.append(f"| {rows[rid]['arm']} | {s.get('status')} / {'clean' if chk.get('clean') else chk.get('flags')} "
                 f"| {pf} | {dl} | {qs} | {_fmt(prim, 'pass1')} | {_fmt(prim, 'greedy')} | {_fmt(exam, 'greedy')} "
                 f"| jobs/outputs/T2_checks/{rid}/loss.svg |")
    if st.get("base_metrics"):
        L += ["", "Base rows (4-shot): " + "; ".join(
            f"{ev}: " + ", ".join(f"{d} p1 {v['pass1']} g {v['greedy']}" for d, v in m.items())
            for ev, m in st["base_metrics"].items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(L) + chr(10), encoding="utf-8")
    log(f"wrote {path}")
    return True


# ------------------------------------------------------------------------------------------------ tick
PM_ARMS = ("Y+", "Y-", "N+", "N-")
N_MATCH_MIN = 1500   # lead 2026-11-08: D5 projects final n ~1,600-1,800 (VI wave 2 stopped); below 1,500 block


def _file_check(path: Path, st: dict) -> dict:
    """Cached laptop-side validity check of one data file (keyed by size + mtime)."""
    cache = st.setdefault("files", {})
    sig = [path.stat().st_size, path.stat().st_mtime]
    c = cache.get(str(path))
    if not c or c.get("sig") != sig:
        c = {"sig": sig, **check_data(path)}
        cache[str(path)] = c
    return c


def final_set_status(root: Path, lang: str, rows_of=None) -> tuple[int | None, str, dict]:
    """READY test for one language (lead 2026-11-08), against D5's final writer (analysis/select_v1.py):
    data/train/final/manifest_<lang>.json MUST exist and is the ONLY source of n (``n_effective``; never a
    selection_* file -- part 1 lists S- only and the final n may be below 3,000); the 4 files
    {Y,N}/<lang>/train_{plus,minus}.jsonl and each directory's SHA256SUMS must exist; every file's sha256 must equal
    both its SHA256SUMS entry and the manifest's ``files`` entry; every file must hold EXACTLY n rows.
    Returns (n or None, reason, per-file details)."""
    from . import guards as Gd
    man_p = root / f"manifest_{lang}.json"
    if not man_p.exists():
        return None, f"{man_p} missing", {}
    try:
        man = json.loads(man_p.read_text(encoding="utf-8"))
    except ValueError as e:
        return None, f"{man_p} unreadable: {e}", {}
    n = man.get("n_effective")
    if not isinstance(n, int) or n <= 0:
        return None, f"{man_p}: n_effective={n!r}", {}
    mfiles = {k.replace("\\", "/"): v for k, v in (man.get("files") or {}).items()}
    det = {}
    for x in ("Y", "N"):
        sums_p = root / x / lang / "SHA256SUMS"
        if not sums_p.exists():
            return None, f"{sums_p} missing", det
        sums = {}
        for line in sums_p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                sums[parts[-1].lstrip("*")] = parts[0]
        for sign in ("plus", "minus"):
            f = root / x / lang / f"train_{sign}.jsonl"
            if not f.exists():
                return None, f"{f} missing", det
            sha = Gd.file_sha256(f)
            if sums.get(f.name) != sha:
                return None, f"{f}: sha256 {sha[:12]} != SHA256SUMS {str(sums.get(f.name))[:12]}", det
            mk = next((k for k in mfiles if k.endswith(f"/{x}/{lang}/train_{sign}.jsonl")), None)
            if mk is None or mfiles[mk].get("sha256") != sha:
                return None, f"{f}: not in manifest_{lang}.json files or sha differs", det
            with open(f, encoding="utf-8") as fh:
                nrows = sum(1 for line in fh if line.strip())
            if nrows != n or mfiles[mk].get("rows") != n:
                return None, f"{f}: {nrows} rows (manifest {mfiles[mk].get('rows')}) != n_effective {n}", det
            det[f"{x}{'+' if sign == 'plus' else '-'}"] = {"rows": nrows, "sha256": sha}
    return n, "", det


def matched_n(rows: list[dict], lang: str, size: str, st: dict, args) -> tuple[int | None, str]:
    """n_train of every arm AND S0 of a language = D5's ``n_effective`` -- only once ``final_set_status`` says the
    language is READY.  Each file is also field-validated (``check_data``)."""
    root = Path(args.data_root) / "data" / "train" / "final"
    n, why, det = final_set_status(root, lang)
    if n is None:
        return None, why
    for arm in PM_ARMS:
        c = _file_check(root / arm[0] / lang / f"train_{'plus' if arm[1] == '+' else 'minus'}.jsonl", st)
        if not c.get("ok") or c["n"] != n:
            return None, f"{arm}/{lang}: field check {c}"
    if args.n_train_cap:
        n = min(n, args.n_train_cap)
    st.setdefault("n_match", {})[f"{lang}|{size}"] = {"n_train": n, "per_arm": {k: v["rows"] for k, v in det.items()}}
    return n, ""


def ready_rows(rows: list[dict], st: dict, args) -> list[dict]:
    out = []
    for r in rows:
        if r["priority"] > args.max_prio or r["kind"] != "sft" or r.get("status") in ("hold", "done"):
            continue
        s = st["runs"].get(r["run_id"], {})
        if s.get("status") in ("LAUNCHED", "TRAINED", "FAILED_FINAL"):
            continue
        p = Path(args.data_root) / r["data"]
        if not p.exists():
            continue
        chk = _file_check(p, st)
        if r["arm"] in PM_ARMS or r["arm"] == "S0":   # S0 trains on the same n as the arms of its language
            n, why = matched_n(rows, r["lang"], r["size"], st, args)
            if n is None:
                log(f"{r['run_id']}: waiting ({why})")
                continue
            if n < N_MATCH_MIN and not args.n_train_cap:
                st.setdefault("blocking", {})[f"n_match:{r['lang']}|{r['size']}"] = {
                    "n_train": n, "per_arm": st["n_match"][f"{r['lang']}|{r['size']}"]["per_arm"],
                    "rule": f"matched n_train < {N_MATCH_MIN}: tell the lead"}
                log(f"BLOCKING {r['lang']}|{r['size']}: matched n_train {n} < {N_MATCH_MIN} "
                    f"{st['n_match'][r['lang'] + '|' + r['size']]['per_arm']}")
                continue
            r = dict(r, n_train=n)          # every arm of the language trains on the same item count
        else:
            need = min(r["n_train"], args.n_train_cap or r["n_train"])
            if not chk.get("ok") or chk["n"] < need:
                log(f"{r['run_id']}: data not usable yet ({chk}); need {need}")
                continue
        st["runs"].setdefault(r["run_id"], {})["n_train"] = r["n_train"]
        out.append(r)
    return out


def tick(args) -> dict:
    rows = load_queue()
    by_id = {r["run_id"]: r for r in rows}
    st = load_state()
    _LIVE.update(st=st, dry=args.dry_run)
    _CODE["stamp"] = require_clean(args)
    if st.get("baseline_quota_h") is None and not args.dry_run:
        st["baseline_quota_h"] = kaggle_quota_used()
    if not args.dry_run:
        poll_kaggle(st, args)
        poll_modal_train(st)
        poll_canaries(st)
    launch_evals(st, by_id, args)
    if not args.dry_run:
        poll_evals(st)
        if not st.get("vi_s0_summary") and write_cell_summary(st, by_id, "vi", 0,
                                                               REPO / "jobs" / "logs" / "T2" / "vi_s0_summary.md"):
            st["vi_s0_summary"] = time.time()
    stop_reason = None
    if not args.dry_run and "s0-review" not in st.setdefault("approved", []):
        stop_reason = "waiting for the lead's S0 review (`launcher approve s0-review`)"
    glob_block = sorted(k for k in st.get("blocking", {}) if not k.startswith(("lang:", "n_match:")))
    if glob_block:
        stop_reason = f"blocked by post-run flags on {glob_block} (`launcher ack RUN_ID`)"
    now = dt.datetime.now().strftime("%H:%M")
    if now >= args.stop_at and now < "23:59" and not args.ignore_clock:
        stop_reason = f"past stop time {args.stop_at}"
    if not args.dry_run and not stop_reason:
        q = kaggle_quota_used()
        res = 1 - args.reserve  # keep a 15 % reserve on both caps (lead)
        if q is not None and st["baseline_quota_h"] is not None and \
                q - st["baseline_quota_h"] >= args.kaggle_cap_h * res:
            stop_reason = f"kaggle quota {q - st['baseline_quota_h']:.2f} h >= {args.kaggle_cap_h * res:.1f}"
        usd = modal_spend_today()
        if usd == usd:
            if st.get("baseline_modal_usd") is None:
                st["baseline_modal_usd"] = usd          # the cap counts spend from the launcher's first live tick
            st["modal_spent_usd"] = round(usd - st["baseline_modal_usd"], 4)
        spent = _spent_now(st)
        if spent >= args.modal_cap_usd:
            stop_reason = f"modal spend ${spent:.2f} >= cap ${args.modal_cap_usd:.2f}"
    ready = ready_rows(rows, st, args)
    lang_block = {k.split(":", 1)[1].split("|")[0] for k in st.get("blocking", {}) if k.startswith(("lang:", "n_match:"))}
    if lang_block:   # a failed seed-0 cell / too-small n blocks only its language (`launcher ack lang:<xx>`)
        log(f"languages blocked: {sorted(lang_block)}")
        ready = [r for r in ready if r["lang"] not in lang_block]
    if args.canary_mode == "separate":
        if not stop_reason:
            launch_canaries(st, ready, args, (lambda r: args.engine) if args.engine != "auto" else
                            (lambda r: "unsloth" if r["engine"] == "unsloth" else "hf"))
        ready = [r for r in ready if (st.get("canary", {}).get(config_key(r)) or {}).get("status") == "OK"
                 or args.dry_run]
    else:
        # canary = the seed-0 cell itself (cost: a separate 80-step canary is ~60 % of a 3-minute L40S run): seed 0 of
        # a config runs first with every guard live and its logs on the Volume; any other seed of that config waits
        # until the seed-0 run is TRAINED and its post-run check is clean
        def seed0_ok(r):
            if r["seed"] == 0:
                return True
            s0 = st["runs"].get(re.sub(r"-s\d+(?=-|$)", "-s0", r["run_id"], count=1), {})
            return s0.get("status") == "TRAINED" and (s0.get("check") or {}).get("clean") is True
        ready = [r for r in ready if seed0_ok(r) or args.dry_run]
    active = [k for k in st["kernels"].values() if k.get("backend") == args.backend and k["status"] in
              ("PUSHED", "QUEUED", "RUNNING")]
    engine_of = (lambda r: args.engine) if args.engine != "auto" else (
        lambda r: "unsloth" if r["engine"] == "unsloth" else "hf")
    if stop_reason:
        log(f"not launching: {stop_reason}; {len(ready)} ready rows wait")
    elif ready:
        slots = max(0, args.max_kernels - len(active))
        n = 0
        while ready and n < slots:
            if args.backend == "modal":   # a whole cell per container (balanced allocation)
                c0 = cell_of(ready[0])
                take = [r for r in ready if cell_of(r) == c0]
                ready = [r for r in ready if cell_of(r) != c0]
                # never launch a partial cell: every not-yet-trained member of the cell must be ready (seed0 mode
                # gates seeds 1-3 per arm, so S0 can be ready before the arms' seed-0 runs are clean)
                members = [r for r in rows if cell_of(r) == c0 and r["kind"] == "sft" and r["priority"] <= args.max_prio
                           and r.get("status") not in ("hold", "done")
                           and st["runs"].get(r["run_id"], {}).get("status") not in ("TRAINED", "LAUNCHED")]
                if {r["run_id"] for r in members} - {r["run_id"] for r in take}:
                    continue
            else:
                take, ready = ready[:args.per_kernel], ready[args.per_kernel:]
            if any(st.get("cells", {}).get(cell_of(r)) not in (None, args.platform) for r in take):
                log(f"skip {[r['run_id'] for r in take]}: cell already on {st['cells'].get(cell_of(take[0]))}")
                n += 1
                continue
            est = round(sum(r.get("train_usd", 0.0) + r.get("eval_usd", 0.0) for r in take), 3)
            if not st.get("base_rows_committed"):
                est += BASE_ROWS_USD      # the base rows ride in the first eval container (counted once)
            spent = _spent_now(st)
            if args.backend == "modal" and not args.dry_run:
                mtd = modal_month_total()
                # billing lags: add this tranche's in-flight unbilled estimates (conservative)
                if mtd == mtd and st.get("modal_spent_usd") is not None:
                    mtd = round(mtd + max(0.0, spent - st["modal_spent_usd"]), 3)
                st["workspace_mtd_usd"] = mtd
                if mtd != mtd or mtd + est > args.workspace_cap_usd:   # NaN = billing unreadable -> refuse too
                    log(f"HARD STOP: workspace month-to-date ${mtd} + next cell ${est:.2f} > ${args.workspace_cap_usd:.2f};"
                        f" not launching {[r['run_id'] for r in take]}; the loop stops")
                    st["hard_stop"] = f"workspace MTD {mtd} + {est} > {args.workspace_cap_usd}"
                    break
                st.pop("hard_stop", None)   # re-checked every tick; exact once in-flight work has billed
            if args.backend == "modal" and not args.dry_run and spent + est > args.modal_cap_usd:
                log(f"STOP at the Modal cap: spent/committed ${spent:.2f} + next cell ${est:.2f} > ${args.modal_cap_usd:.2f}"
                    f"; not launching {[r['run_id'] for r in take]}")
                break
            if args.backend == "kaggle":
                slug = f"t2-k{len([k for k in st['kernels'] if k.startswith('t2-k')]) + 1:02d}"
                launch_kaggle(st, slug, take, args, engine_of)
            else:
                tag = launch_modal_train(st, take, args, engine_of)
                if tag in st["kernels"]:
                    st["kernels"][tag]["est_usd"] = est
                    st["kernels"][tag]["est_eval_usd"] = round(sum(r.get("eval_usd", 0.0) for r in take), 3)
                st["committed_usd"] = round(st.get("committed_usd", 0.0) + est, 3)
                st["base_rows_committed"] = True
                log(f"committed ${est:.2f} for {cell_of(take[0])}; total committed ${st['committed_usd']:.2f}")
            n += 1
    save_state(st) if not args.dry_run else None
    summary = {}
    for s in st["runs"].values():
        summary[s.get("status", "?")] = summary.get(s.get("status", "?"), 0) + 1
    log(f"tick done: {summary}; ready-not-launched {len(ready)}")
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(prog="launcher")
    ap.add_argument("cmd", choices=["tick", "loop", "status", "approve", "ack", "tail"])
    ap.add_argument("what", nargs="?")
    ap.add_argument("--reserve", type=float, default=0.15)
    ap.add_argument("--canary-mode", default="seed0", choices=["seed0", "separate"])
    ap.add_argument("--allow-dirty", action="store_true", help="dry runs / emergencies only; recorded in manifests")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backend", default="modal", choices=["kaggle", "modal"])
    ap.add_argument("--platform", default="modal_l40s", choices=sorted(PLATFORMS))
    ap.add_argument("--engine", default="auto", choices=["auto", "unsloth", "hf"])
    ap.add_argument("--max-prio", type=int, default=5)
    ap.add_argument("--data-root", default=str(REPO))
    ap.add_argument("--n-train-cap", type=int)
    ap.add_argument("--max-kernels", type=int, default=2)
    ap.add_argument("--per-kernel", type=int, default=2)
    ap.add_argument("--kaggle-cap-h", type=float, default=20.0)
    ap.add_argument("--modal-cap-usd", type=float, default=6.9)   # lead 2026-11-08 08:10: VI s3 + EN s1 must fit
    ap.add_argument("--workspace-cap-usd", type=float, default=29.5)  # HARD: Modal month-to-date, ALL apps
    ap.add_argument("--stop-at", default="17:00")
    ap.add_argument("--ignore-clock", action="store_true")
    ap.add_argument("--every", type=int, default=300)
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(json.dumps(load_state(), indent=1)[:20000])
        return 0
    if a.cmd in ("approve", "ack"):
        st = load_state()
        if a.cmd == "approve":
            st.setdefault("approved", []).append(a.what)
        else:
            st.setdefault("acked", {})[a.what] = st.get("blocking", {}).pop(a.what, None)
        save_state(st)
        log(f"{a.cmd} {a.what}")
        return 0
    if a.cmd == "tail":  # live train/dev log of a Modal run (canary) from the Volume
        d = pull_modal_run(a.what, REPO / "jobs" / "outputs" / "t2-live" / a.what)
        for f in ("train_log.jsonl", "dev_log.jsonl"):
            p = d / f
            if p.exists():
                print(f"== {f}")
                print("\n".join(p.read_text(encoding="utf-8").splitlines()[-15:]))
        return 0
    if a.cmd == "tick":
        tick(a)
        return 0
    while True:
        st = tick(a)
        open_runs = [r for r in st["runs"].values() if r.get("status") in ("LAUNCHED",)]
        if a.dry_run:
            return 0
        # in flight = training cells OR evals whose cost is not yet billed (pending result)
        unsettled = st.get("modal_spent_usd") is not None and _spent_now(st) - st["modal_spent_usd"] > 1e-6
        if st.get("hard_stop") and not open_runs and not unsettled:
            log(f"loop: HARD STOP ({st['hard_stop']}) and nothing in flight; exiting")
            return 0
        if not open_runs and dt.datetime.now().strftime("%H:%M") >= a.stop_at and not a.ignore_clock:
            log("loop: past stop time and nothing in flight; exiting")
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
