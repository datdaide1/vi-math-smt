"""T1r: summarise Kaggle training outputs + Modal eval preds into jobs/t1r/summary.json (numbers for the report).

    .venv/Scripts/python.exe jobs/t1r/summarize.py --train jobs/outputs/t1r-runs-v1/results --eval jobs/outputs/t1r-eval
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def jl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def train_summary(d: Path) -> dict:
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    log = [r for r in jl(d / "train_log.jsonl") if r.get("tag") == "main"]
    s = man.get("summary") or {}
    out = {"status": man.get("status"), "engine": man.get("engine"), "packed": man.get("packed"),
           "micro_batch": man.get("micro_batch"), "grad_ckpt": man.get("grad_ckpt"), "n_params": man.get("n_params"),
           "steps": len(log), "env": {k: man.get("env", {}).get(k) for k in ("torch", "transformers", "peft", "gpu")},
           "data": {k: (man.get("data_stats") or {}).get(k) for k in ("n_pool", "n_eligible", "n_dropped_too_long",
                                                                      "pool_len_p99", "n_items")}}
    for k in ("skipped_steps", "skipped_first10", "nonfinite_losses", "pad_fraction", "real_tokens_per_s", "train_s",
              "dev_loss_final", "max_abs_act", "peak_mem_gb", "final_scale"):
        out[k] = s.get(k)
    if log:
        un = [r for r in log[5:] if "max_abs_act" not in r]
        if un:
            out["tok_per_s_steady"] = round(sum(r["real_tokens"] for r in un) / sum(r["s"] for r in un), 1)
        out["loss_first5"] = [round(r["loss"], 4) for r in log[:5]]
        out["loss_last5"] = [round(r["loss"], 4) for r in log[-5:]]
        out["loss_by_50"] = [round(sum(r["loss"] for r in log[i:i + 50]) / len(log[i:i + 50]), 4)
                             for i in range(0, len(log), 50)]
        out["max_abs_act_log"] = max((r.get("max_abs_act", 0) for r in log), default=None)
        out["wall_s_train_steps"] = round(sum(r["s"] for r in log), 1)
    out["dev"] = jl(d / "dev_log.jsonl")
    if (d / "pack_gate.json").exists():
        g = json.loads((d / "pack_gate.json").read_text(encoding="utf-8"))
        out["pack_gate"] = {"ok": g["ok"], "max_dnll": max(r["dnll_per_tok"] for r in g["items"]),
                            "min_argmax": min(r["argmax_agree"] for r in g["items"]),
                            "fused_vs_logits_per_tok": g["fused_vs_logits_per_tok"]}
    return out


def item_table(rows):
    """item -> {'p1': mean over sampled, 'g': greedy, 'trunc': any sampled truncated, 'ds': dataset}."""
    by = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["item_id"])].append(r)
    t = {}
    for k, rs in by.items():
        sm = [r for r in rs if r["sample_k"] != -1]
        gr = [r for r in rs if r["sample_k"] == -1]
        t[k] = {"p1": sum(r["correct"] for r in sm) / len(sm) if sm else None,
                "g": gr[0]["correct"] if gr else None, "ds": k[0],
                "trunc_s": sum(r["truncated"] for r in sm) / len(sm) if sm else None,
                "tok_s": sum(r["tokens"] for r in sm) / len(sm) if sm else None}
    return t


def boot_ci(vals, B=4000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    ms = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(ms[int(0.025 * B)], 4), round(ms[int(0.975 * B)], 4)]


def acc_summary(t, keys=None):
    keys = keys or list(t)
    out = {}
    for name, ks in [("all", keys)] + [(ds, [k for k in keys if k[0] == ds]) for ds in sorted({k[0] for k in keys})]:
        p1 = [t[k]["p1"] for k in ks if t[k]["p1"] is not None]
        g = [t[k]["g"] for k in ks if t[k]["g"] is not None]
        out[name] = {"n": len(ks), "pass1": round(sum(p1) / len(p1), 4) if p1 else None,
                     "pass1_ci95": boot_ci(p1) if p1 else None,
                     "greedy": round(sum(g) / len(g), 4) if g else None,
                     "trunc_sampled": round(sum(t[k]["trunc_s"] for k in ks) / len(ks), 4),
                     "tokens_mean": round(sum(t[k]["tok_s"] for k in ks) / len(ks), 1)}
    return out


def paired(ta, tb, field="p1"):
    ks = [k for k in ta if k in tb and ta[k][field] is not None and tb[k][field] is not None]
    d = [ta[k][field] - tb[k][field] for k in ks]
    return {"n": len(ks), "diff": round(sum(d) / len(d), 4), "ci95": boot_ci(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(REPO / "jobs/outputs/t1r-runs-v1/results"))
    ap.add_argument("--eval", default=str(REPO / "jobs/outputs/t1r-eval"))
    a = ap.parse_args()
    S = {"train": {}, "eval": {}, "paired": {}}
    for d in sorted(Path(a.train).glob("*/manifest.json")):
        S["train"][d.parent.name] = train_summary(d.parent)
    T = {}
    for d in sorted(Path(a.eval).glob("*/preds.jsonl")):
        rows = jl(d)
        man = json.loads((d.parent / "manifest.json").read_text(encoding="utf-8"))
        T[d.parent.name] = item_table(rows)
        S["eval"][d.parent.name] = {"acc": acc_summary(T[d.parent.name]), "gpu": (man.get("platform") or {}).get("gpu"),
                                    "throughput": man.get("throughput"), "engine_load_s":
                                        (man.get("platform") or {}).get("engine_load_s"), "rows": len(rows)}
    names = list(T)

    def find(*subs):
        return next((n for n in names if all(s in n for s in subs)), None)
    # parity on dev300
    u, h = find("gate-uns", "dev300"), find("gate-hf", "dev300")
    if u and h:
        S["paired"]["gate_uns_minus_hf_dev300_pass1"] = paired(T[u], T[h])
        S["paired"]["gate_uns_minus_hf_dev300_greedy"] = paired(T[u], T[h], "g")
    base = find("ev-base", "primary")
    s1, s2 = find("S0", "s1-primary"), find("S0", "s2-primary")
    S["paired"]["_names"] = {"base": base, "s1": s1, "s2": s2}
    for nm, x in (("S0s1", s1), ("S0s2", s2)):
        if base and x:
            S["paired"][f"{nm}_minus_base_primary_pass1"] = paired(T[x], T[base])
    if s1 and s2:
        S["paired"]["S0s1_minus_S0s2_primary_pass1"] = paired(T[s1], T[s2])
    # cap check
    for src, cap in ((base, find("ev-base", "cap2048")), (s1, find("S0", "s1-spot200"))):
        if src and cap:
            ks = list(T[cap])
            S["paired"][f"cap2048_minus_cap1024__{cap}"] = {
                **paired(T[cap], {k: T[src][k] for k in ks if k in T[src]}),
                "acc1024": acc_summary(T[src], [k for k in ks if k in T[src]])["all"],
                "acc2048": acc_summary(T[cap])["all"]}
    # Y2 seed variance (2 seeds; primary_1000 is all "parallel" -> sigma_run(mirror) only)
    if s1 and s2:
        try:
            from vi_math_verified_aug.analysis import schema as Y
            from vi_math_verified_aug.analysis.variance import seed_variance
            d = Y.load_predictions([str(Path(a.eval) / s1 / "preds.jsonl"), str(Path(a.eval) / s2 / "preds.jsonl")])
            S["y2_schema_report"] = {k: v for k, v in d.report.items() if not isinstance(v, (list, dict))
                                     or len(str(v)) < 400}
            S["seed_variance"] = seed_variance(d)
        except Exception as e:  # noqa: BLE001
            S["seed_variance"] = {"error": repr(e)}
    out = REPO / "jobs/t1r/summary.json"
    out.write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
    print(json.dumps(S, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
