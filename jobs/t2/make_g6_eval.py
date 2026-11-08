"""G6 selection eval: every sweep adapter (4 LRs x epochs 1/2/3 + the fresh 2-epoch G6 run) on dev_v1 (VI, 512 DEV-bucket
seeds) with the frozen protocol (K=4 + greedy, cap 1,024), Modal L40S pinned, VLLM_BATCH_INVARIANT=1.  All configs share
the sampling-seed key "g6-sweep" (common random numbers across the candidates).  Never a test set.

    .venv/Scripts/python.exe jobs/t2/make_g6_eval.py && modal run vi_math_verified_aug/evalrun/modal_eval.py::eval_main \
        --spec jobs/t2/g6_eval_l40s.json
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vi_math_verified_aug.train import launcher as L  # noqa: E402

DEVV1 = {"name": "dev_v1_vi", "path": "{vol}/train_data/t1r/dev_seed_vi.jsonl", "lang": "vi"}


def main():
    stamp = L.code_stamp()
    cands = [(f"t2-g6-S0-vi-0p6B-n3k-lr{lr:g}-ep3", e) for lr in (1e-4, 2e-4, 4e-4, 8e-4) for e in (1, 2, 3)]
    cands.append(("t2-g6-S0-vi-0p6B-n3k-lr0.0002-ep2fresh", 2))
    cfgs = []
    for rid, e in cands:
        tag = f"{rid}@ep{e}"
        cfgs.append({"kind": "eval", "run_id": f"ev-g6-{rid}-e{e}", "arm": "S0", "model": "Qwen/Qwen3-0.6B-Base",
                     "size": "0.6B", "seed": 1, "precision": "bf16", "platform": "modal_l40s", "code": stamp,
                     "out_dir": f"{{vol}}/eval/g6/{rid}-e{e}", "notes": tag,
                     "eval": {"source_run_id": "g6-sweep", "adapter_path": f"{{vol}}/train/{rid}/adapters/epoch{e}",
                              "max_new_tokens": 1024, "sets": [DEVV1], "model_tag": "qwen3-0.6b"}})
    # references on the same dev set: the 4-shot base row and the two Kaggle-T4 S0 pilot adapters (review only)
    cfgs.append({"kind": "eval", "run_id": "ev-g6-base-4shot", "arm": "base", "model": "Qwen/Qwen3-0.6B-Base",
                 "size": "0.6B", "seed": 0, "precision": "bf16", "platform": "modal_l40s", "code": stamp,
                 "out_dir": "{vol}/eval/g6/base-4shot", "template": {"fewshot_path": "{vol}/eval_data/fewshot4_vi.jsonl"},
                 "eval": {"source_run_id": "g6-sweep", "shots": 4, "max_new_tokens": 1024, "sets": [DEVV1],
                          "model_tag": "qwen3-0.6b"}})
    for sd in (1, 2):
        rid = f"t1r-S0-uns-0p6B-s{sd}"
        cfgs.append({"kind": "eval", "run_id": f"ev-g6-{rid}", "arm": "S0", "model": "Qwen/Qwen3-0.6B-Base",
                     "size": "0.6B", "seed": sd, "precision": "bf16", "platform": "modal_l40s", "code": stamp,
                     "out_dir": f"{{vol}}/eval/g6/{rid}", "notes": "Kaggle T4 fp16 pilot (n=7,157, includes dev_v1 GSM8K seeds)",
                     "eval": {"source_run_id": "g6-sweep", "adapter_path": f"{{vol}}/adapters/{rid}",
                              "max_new_tokens": 1024, "sets": [DEVV1], "model_tag": "qwen3-0.6b"}})
    out = REPO / "jobs" / "t2" / "g6_eval_l40s.json"
    out.write_text(json.dumps({"base": "Qwen/Qwen3-0.6B-Base", "tag": "g6-eval", "configs": cfgs}, indent=1),
                   encoding="utf-8")
    print(out, len(cfgs), stamp["git_sha"][:8], stamp["dirty"])


if __name__ == "__main__":
    main()
