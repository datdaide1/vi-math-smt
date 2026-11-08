"""G6 quality gate on THE platform (lead, FINAL 2026-11-08): Modal L40S bf16, Unsloth packed, micro-batch 32, no
gradient checkpointing, n_train = 3,000 (seeded subsample of S0-VI, seed 1, the arms' procedure), dev = dev_v1 (VI).

    .venv/Scripts/python.exe jobs/t2/make_sweep_modal.py      -> jobs/t2/g6_sweep_l40s.json
    modal run vi_math_verified_aug/evalrun/modal_eval.py::train_main --spec jobs/t2/g6_sweep_l40s.json

One container (one lane), runs in order:
  LR in {1e-4, 2e-4, 4e-4, 8e-4}: 3-epoch WSD main run with decay branches at epochs 1 and 2 (T3 §3.1; 8e-4 added
  because at n = 3,000 a 2-epoch run is only 186 steps -- short runs favour a higher LoRA LR, QA-T's LoRA-Without-
  Regret note); + the G6 fidelity run: a fresh 2-epoch WSD run at 2e-4 (compare with the 2e-4 epoch-2 branch).
Selection: dev_v1 accuracy (mean pass@1 over K=4, then greedy) on Modal L40S eval; dev NLL only as a tie-break.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vi_math_verified_aug.train import config as C  # noqa: E402
from vi_math_verified_aug.train import grid as G  # noqa: E402
from vi_math_verified_aug.train import launcher as L  # noqa: E402

N = 3000


def cfg(run_id, lr, epochs, branches, ident, stamp):
    r = dict(G.row(0, "G6 sweep", "S0", "vi", "0.6B", 1), run_id=run_id, lr=lr, epochs=epochs, n_train=N)
    L._CODE["stamp"] = stamp
    raw = L.sft_config(r, "data/t1r/s0_train.jsonl", precision="bf16", engine="unsloth", ident=ident,
                       platform="modal_l40s")
    raw["sft"].update(branch_epochs=branches, dev_every_steps=50, dev_path="{code}/data/t1r/dev_seed_vi.jsonl",
                      ckpt_every_min=15)
    raw["notes"] = f"T2 G6 LR x epoch sweep (T3 §3.1) on L40S bf16 mb32, n={N}, dev=dev_v1"
    raw = json.loads(json.dumps(raw).replace("{code}/data/", "{vol}/train_data/"))
    C.build(json.loads(json.dumps(raw).replace("{vol}", "/x")))
    return raw


def main():
    stamp = L.code_stamp()
    ident = L.data_identity(REPO / "data" / "t1r" / "s0_train.jsonl")
    tasks = []
    for lr in (1e-4, 2e-4, 4e-4, 8e-4):
        tasks.append({"cmd": "sft", "config": cfg(f"t2-g6-S0-vi-0p6B-n3k-lr{lr:g}-ep3", lr, 3, [1, 2], ident, stamp),
                      "budget_s": 2400})
    tasks.append({"cmd": "sft", "config": cfg("t2-g6-S0-vi-0p6B-n3k-lr0.0002-ep2fresh", 2e-4, 2, [], ident, stamp),
                  "budget_s": 2400})
    out = REPO / "jobs" / "t2" / "g6_sweep_l40s.json"
    out.write_text(json.dumps({"groups": [{"gpu": "l40s", "cell": "G6 S0 0.6B vi s1", "tasks": tasks}],
                               "wait_s": 4 * 3600}, indent=1), encoding="utf-8")
    print(out, [t["config"]["run_id"] for t in tasks], "code", stamp["git_sha"][:10], "dirty", stamp["dirty"])


if __name__ == "__main__":
    main()
