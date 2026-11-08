"""CROSS-PLATFORM, INDICATIVE ONLY: base Qwen3-0.6B 0-shot VI eval on Kaggle T4 (vLLM fp16).

Lead 2026-11-08: the Modal base 0-shot eval was refused by the $29.5 workspace guard, so this runs on Kaggle's free
quota.  It answers ONE question -- is the arms-vs-4-shot-base gap a prompting-format artefact? -- and is NEVER part
of the confirmatory analysis (eval numerics differ across GPUs: ~11-15 % of items flip, T2 §3).

Same configs as jobs/t2/run_base0shot.py (shots 0, the arms' template, cap 1,024; vi_primary_1000 K=4 + greedy,
Vi-ExamMath VERIFIED greedy) except precision fp16, platform kaggle_t4, and no VLLM_BATCH_INVARIANT.
Two lanes on T4x2: lane 0 = primary, lane 1 = exam.  Private kernel + private dataset.

    .venv/Scripts/python.exe jobs/t2/make_base0shot_kaggle.py SLUG [--push]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vi_math_verified_aug.train import config as C  # noqa: E402
from vi_math_verified_aug.train import launcher as L  # noqa: E402
from vi_math_verified_aug.train.kaggle import render as RD  # noqa: E402

LABEL = "CROSS-PLATFORM, INDICATIVE ONLY (Kaggle T4 fp16; not the L40S eval platform; never confirmatory)"


def configs() -> list[dict]:
    c = L.code_stamp()
    if c["dirty"]:
        raise SystemExit(f"refusing to build from a dirty tree: {c['dirty_files']}")
    L._CODE["stamp"] = c
    r = {"run_id": "base-qwen3-0.6b-0shot-vi", "arm": "base", "model": "Qwen/Qwen3-0.6B-Base", "size": "0.6B",
         "seed": 0, "lang": "vi"}
    out = []
    for raw in L.eval_configs(r, None):
        raw = json.loads(json.dumps(raw).replace("{vol}/eval_data", "{code}/eval_data"))
        raw["run_id"] = raw["run_id"].replace("ev-base-", "ev-kgl-base-")   # never collides with an L40S row
        raw.pop("out_dir", None)                                             # the kernel sets {RES}/<run_id>
        raw["platform"], raw["precision"] = "kaggle_t4", "fp16"
        raw["notes"] = LABEL
        C.build(json.loads(json.dumps(raw).replace("{code}", "/x")))       # validate on the laptop
        out.append(raw)
    return out


def main(slug: str, push: bool = False):
    cfgs = configs()
    lanes = [[{"cmd": "eval", "config": cfgs[0], "budget_s": 7200}],
             [{"cmd": "eval", "config": cfgs[1], "budget_s": 5400}]]
    job = RD.build_job(lanes, slug, env={"deadline_s": 8400, "grace_s": 300, "python": "3.13",
                                         "install": "pip install -q 'vllm==0.33.0' 'math-verify==0.9.0'",
                                         "label": LABEL})
    run_py = RD.render(job, slug, REPO / "jobs" / "build")
    data = REPO / "jobs" / "build" / f"{slug}-data"
    shutil.rmtree(data, ignore_errors=True)
    RD.bundle_code(data)
    (data / "eval_data").mkdir(parents=True)
    for f in ("vi_primary_1000.jsonl", "vi_exam_core_verified.jsonl"):
        shutil.copy2(REPO / "jobs" / "t1r" / "eval_data" / f, data / "eval_data" / f)
    print(run_py, data, job["run_id"], [c["run_id"] for c in cfgs])
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
        rec = {"run_id": job["run_id"], "push_out": out.strip()[-400:], "gpu_quota_used_h_at_push": q0,
               "label": LABEL}
        (kdir / "last_push.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(rec)


if __name__ == "__main__":
    main(sys.argv[1], push="--push" in sys.argv)
