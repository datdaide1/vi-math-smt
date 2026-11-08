"""T1r: build the Kaggle job (T4x2) for the parity gate + S0 runs + LFM2.5 smoke, and its private dataset.

    .venv/Scripts/python.exe jobs/t1r/make_kaggle_job.py SLUG [--push] [--steps-scale 1.0]

Dataset folder jobs/build/<slug>-data/ = vi_math_verified_aug/ (code) + data/{s0_train,dev300}.jsonl (train split
only).  Lanes (one GPU each; T4x2 bills 1 quota-h per wall-h, S2 §7):
  lane 0: gate-uns (Unsloth packed, 300 steps)  -> S0 seed 1 (Unsloth packed if gate-uns DONE, else HF padded)
          -> LFM2.5-2.6B HF 100-step smoke
  lane 1: gate-hf  (HF padded, 300 steps)       -> S0 seed 2 (same engine rule; waits for gate-uns to finish)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vi_math_verified_aug.train import config as C  # noqa: E402
from vi_math_verified_aug.train.kaggle import render as RD  # noqa: E402

Q06 = "Qwen/Qwen3-0.6B-Base"
LFM = "LiquidAI/LFM2.5-2.6B-Base"


def sft(run_id, *, model=Q06, size="0.6B", engine, packing, seed=1, max_steps=None, epochs=2, n_train=None,
        dev_every=None, act=0, micro=8, dev=True):
    raw = {"kind": "sft", "run_id": run_id, "arm": "S0", "model": model, "size": size, "seed": seed,
           "precision": "fp16", "engine": engine,
           "budget": {"n_train": n_train, "epochs": epochs, "max_steps": max_steps},
           "data": {"train": "{code}/data/s0_train.jsonl"},
           "sft": {"lr": 2.0e-4, "micro_batch": micro, "max_length": 1024, "packing": packing,
                   "dev_path": "{code}/data/dev300.jsonl" if dev else None, "dev_every_steps": dev_every,
                   "act_monitor_every": act, "ckpt_every_min": 30, "keep_ckpts": 1},
           "notes": "T1r first real runs (S0 = clean Vi-GSM8K train minus R1's 16 rows minus dev300)"}
    C.build(json.loads(json.dumps(raw).replace("{code}", "/x")))  # validate on the laptop
    return raw


def main(slug: str, push: bool = False):
    gate_uns = sft("t1r-gate-uns-0p6B-s1", engine="unsloth", packing=True, max_steps=300, dev_every=100)
    gate_hf = sft("t1r-gate-hf-0p6B-s1", engine="hf", packing=False, max_steps=300, dev_every=100)
    s0 = {s: sft(f"t1r-S0-uns-0p6B-s{s}", engine="unsloth", packing=True, seed=s, dev_every=150) for s in (1, 2)}
    s0_hf = {s: sft(f"t1r-S0-hf-0p6B-s{s}", engine="hf", packing=False, seed=s, dev_every=150) for s in (1, 2)}
    lfm = sft("t1r-lfm2.5-2.6B-smoke", model=LFM, size="2.6B", engine="hf", packing=False, max_steps=100,
              epochs=1, n_train=3200, act=4, dev=False)
    lane0 = [{"cmd": "sft", "config": gate_uns, "budget_s": 1800},
             {"cmd": "sft", "config": s0[1], "budget_s": 3300, "requires": gate_uns["run_id"],
              "fallback_config": s0_hf[1]},
             {"cmd": "sft", "config": lfm, "budget_s": 2400}]
    lane1 = [{"cmd": "sft", "config": gate_hf, "budget_s": 2400},
             {"cmd": "sft", "config": s0[2], "budget_s": 3300, "requires": gate_uns["run_id"], "wait_max_s": 1800,
              "fallback_config": s0_hf[2]}]
    job = RD.build_job([lane0, lane1], slug, env={"deadline_s": 9000, "grace_s": 300, "drop_ckpt_on_done": True,
                                                   "python": "3.13"})
    run_py = RD.render(job, slug, REPO / "jobs" / "build")
    data = REPO / "jobs" / "build" / f"{slug}-data"
    shutil.rmtree(data, ignore_errors=True)
    RD.bundle_code(data)
    (data / "data").mkdir(parents=True)
    for f in ("s0_train.jsonl", "dev300.jsonl", "MANIFEST.json"):
        shutil.copy2(REPO / "data" / "t1r" / f, data / "data" / f)
    print(run_py, data, job["run_id"])
    if push:
        from vi_math_verified_aug.infra.kaggle_runner import KaggleRunner
        r = KaggleRunner()
        ds = f"{slug}-data"
        print(r.dataset_upsert(data, ds, ds, message=job["run_id"], dir_mode="zip"))
        print("dataset ready:", r.dataset_ready(f"datuni/{ds}"))
        kdir = r.render_kernel(slug, run_py, dataset_sources=[f"datuni/{ds}"])
        shutil.copy2(run_py.parent / "job.json", kdir / "job.json")
        q0 = r.gpu_quota_used()
        out = r.push(kdir, timeout_s=int(job["deadline_s"]) + 1200)
        rec = {"run_id": job["run_id"], "push_out": out.strip()[-400:], "gpu_quota_used_h_at_push": q0}
        (kdir / "last_push.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(rec)


if __name__ == "__main__":
    main(sys.argv[1], push="--push" in sys.argv)
