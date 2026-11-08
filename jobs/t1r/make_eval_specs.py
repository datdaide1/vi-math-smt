"""T1r: eval specs for evalrun/modal_eval.py (Modal L4, bf16, K=4 + greedy, cap 1,024, per-request seeds).

    .venv/Scripts/python.exe jobs/t1r/make_eval_specs.py base      -> jobs/t1r/eval_base.json
    .venv/Scripts/python.exe jobs/t1r/make_eval_specs.py sft RUNS   -> jobs/t1r/eval_sft.json
       RUNS = comma list of training run_ids whose adapters are on the Volume at /adapters/<run_id>

Volume layout (t1r-eval): /eval_data/{vi_primary_1000,vi_primary_spot200,fewshot4_vi}.jsonl, /data/dev300.jsonl,
/adapters/<run_id>/ (PEFT dir), outputs /eval/<eval run_id>/{preds.jsonl,metrics.json,manifest.json}.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q06 = "Qwen/Qwen3-0.6B-Base"
PRIMARY = {"name": "vi_primary_1000", "path": "{vol}/eval_data/vi_primary_1000.jsonl", "lang": "vi"}
SPOT = {"name": "vi_primary_spot200", "path": "{vol}/eval_data/vi_primary_spot200.jsonl", "lang": "vi"}
DEV = {"name": "dev300", "path": "{vol}/data/dev300.jsonl", "lang": "vi"}


def ev(run_id, *, arm, seed, source, sets, adapter=None, shots=0, cap=1024):
    c = {"kind": "eval", "run_id": run_id, "arm": arm, "model": Q06, "size": "0.6B", "seed": seed,
         "precision": "bf16", "out_dir": "{vol}/eval/" + run_id,
         "eval": {"source_run_id": source, "adapter_path": adapter, "shots": shots, "max_new_tokens": cap,
                  "sets": sets, "model_tag": "qwen3-0.6b"}}
    if shots:
        c["template"] = {"fewshot_path": "{vol}/eval_data/fewshot4_vi.jsonl"}
    return c


def base():
    src = "base-qwen3-0.6b"
    return {"base": Q06, "tag": "base", "configs": [
        ev("ev-base-qwen3-0.6b-4shot-primary", arm="base", seed=0, source=src, sets=[PRIMARY], shots=4),
        ev("ev-base-qwen3-0.6b-4shot-spot200-cap2048", arm="base", seed=0, source=src, sets=[SPOT], shots=4,
           cap=2048)]}


def sft(runs):
    cfgs = []
    for rid in runs:
        seed = int(rid.rsplit("-s", 1)[1])
        ad = "{vol}/adapters/" + rid
        if "gate" in rid:
            cfgs.append(ev(f"ev-{rid}-dev300", arm="S0", seed=seed, source=rid, sets=[DEV], adapter=ad))
        else:
            cfgs.append(ev(f"ev-{rid}-primary", arm="S0", seed=seed, source=rid, sets=[PRIMARY], adapter=ad))
            if seed == 1:
                cfgs.append(ev(f"ev-{rid}-spot200-cap2048", arm="S0", seed=seed, source=rid, sets=[SPOT],
                               adapter=ad, cap=2048))
    return {"base": Q06, "tag": "sft", "configs": cfgs}


if __name__ == "__main__":
    which = sys.argv[1]
    spec = base() if which == "base" else sft(sys.argv[2].split(","))
    out = HERE / f"eval_{which}.json"
    out.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(out, [c["run_id"] for c in spec["configs"]])
