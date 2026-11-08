"""T1r on Modal (Kaggle kernel stuck in QUEUED): same runs as make_kaggle_job.py, one container per run.

    .venv/Scripts/python.exe jobs/t1r/make_modal_train.py   -> jobs/t1r/train_modal_v1.json
    modal run vi_math_verified_aug/evalrun/modal_eval.py::train_main --spec jobs/t1r/train_modal_v1.json

Qwen3-0.6B runs: bf16 on the ["L4", "A10G", "L40S"] fallback list (gate-uns, gate-hf, S0 seeds 1-2).
LFM2.5-2.6B smoke: fp16 on T4 (the point of the smoke is fp16 overflow on the Kaggle GPU class).
Data on the Volume: /vol/data/{s0_train,dev300}.jsonl (= data/t1r/).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_kaggle_job import LFM, sft  # noqa: E402


def vol(c, precision):
    c = json.loads(json.dumps(c).replace("{code}/data", "{vol}/data"))
    c["precision"] = precision
    c["sft"]["ckpt_every_min"] = 20
    return c


def main():
    g = lambda c, prec="bf16", gpu="ampere", b=3600: {"gpu": gpu, "tasks": [{"cmd": "sft", "config": vol(c, prec),
                                                                              "budget_s": b}]}
    groups = [
        g(sft("t1r-gate-uns-0p6B-s1", engine="unsloth", packing=True, max_steps=300, dev_every=100)),
        g(sft("t1r-gate-hf-0p6B-s1", engine="hf", packing=False, max_steps=300, dev_every=100)),
        g(sft("t1r-S0-uns-0p6B-s1", engine="unsloth", packing=True, seed=1, dev_every=150)),
        g(sft("t1r-S0-uns-0p6B-s2", engine="unsloth", packing=True, seed=2, dev_every=150)),
        g(sft("t1r-lfm2.5-2.6B-smoke", model=LFM, size="2.6B", engine="hf", packing=False, max_steps=100, epochs=1,
              n_train=3200, act=4, dev=False), prec="fp16", gpu="t4", b=3000),
    ]
    out = HERE / "train_modal_v1.json"
    out.write_text(json.dumps({"groups": groups, "wait_s": 4 * 3600}, indent=1), encoding="utf-8")
    print(out, [x["tasks"][0]["config"]["run_id"] for x in groups])


if __name__ == "__main__":
    main()
