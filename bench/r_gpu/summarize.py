"""Summarise the R-GPU benchmark (laptop; stdlib only).

    python bench/r_gpu/summarize.py train <dir with train_v2_*.json>          -> table + train_summary.json
    python bench/r_gpu/summarize.py pair <preds_a.jsonl> <preds_b.jsonl>      -> paired greedy / pass@1 deltas

Throughput = real (non-pad) tokens / step seconds over steps >= 5 (steady state; excludes model load, compile warm-up,
dev-loss passes).  $/run = 9M tokens at that rate x the Modal list price (no cold start).
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

RATE = {"l4": 0.80, "l40s": 1.95, "h100": 3.95, "a10g": 1.10, "t4": 0.59, "a100": 2.10}
RUN_TOK = 9e6


def steady(rows, skip=5):
    rr = [r for r in rows if r["step"] >= skip]
    if not rr:
        return None
    return sum(r["real_tokens"] for r in rr) / max(1e-9, sum(r["s"] for r in rr))


def bucket(rows, w=50):
    b = defaultdict(list)
    for r in rows:
        b[r["step"] // w].append(r["loss"])
    return [round(sum(v) / len(v), 4) for _, v in sorted(b.items())]


def train(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "train_v*_*.json"))):
        gpu_tag = os.path.basename(f).split("_")[2]
        res = json.load(open(f, encoding="utf-8"))
        for t in res["tasks"]:
            runs = t["runs"]
            tps = [steady(r.get("train_log") or []) for r in runs]
            mem = [max([x.get("mem_gb", 0) for x in (r.get("train_log") or [])] or [0]) for r in runs]
            ok = [r["rc"] == 0 for r in runs]
            agg = sum(x for x in tps if x) if all(tps) else None
            tot_real = sum(sum(x["real_tokens"] for x in (r.get("train_log") or [])) for r in runs)
            row = {"gpu_tag": gpu_tag, "gpu": res["gpu"], "name": t["task"]["name"], "k": len(runs),
                   "variant": t["task"].get("variant", "none"), "engine": t["task"].get("engine", "unsloth"),
                   "mb": t["task"].get("mb", 8), "steps": t["task"]["steps"], "rc": [r["rc"] for r in runs],
                   "tok_s": round(agg, 0) if agg else None, "tok_s_per_proc": [round(x) if x else None for x in tps],
                   "tok_s_wall": round(tot_real / t["wall_s"], 0) if t["wall_s"] else None, "wall_s": t["wall_s"],
                   "peak_gb": mem, "usd_per_run": round(RUN_TOK / agg / 3600 * RATE[gpu_tag], 3) if agg else None,
                   "h_per_run": round(RUN_TOK / agg / 3600, 3) if agg else None}
            r0 = runs[0]
            tl = r0.get("train_log") or []
            if tl:
                row["loss_by50"] = bucket(tl)
                row["loss_last10"] = round(sum(x["loss"] for x in tl[-10:]) / len(tl[-10:]), 4)
            if r0.get("dev_log"):
                row["dev_loss"] = [(x.get("step"), x.get("dev_loss")) for x in r0["dev_log"]]
            v = r0.get("rgpu_variant") or {}
            if v.get("error"):
                row["error"] = v["error"].strip().splitlines()[-1][:300]
            elif not all(ok):
                row["error"] = (r0.get("tail") or "")[-300:]
            row["dtype_hist"] = v.get("dtype_hist")
            row["float8_modules"] = v.get("float8_modules")
            row["versions"] = v.get("versions")
            out.append(row)
    for r in out:
        print(f"{r['gpu_tag']:5} {r['name']:22} k={r['k']} rc={r['rc']} tok/s={r['tok_s']} wall_tok/s={r['tok_s_wall']}"
              f" $/run={r['usd_per_run']} h/run={r['h_per_run']} peak={r['peak_gb']} last10={r.get('loss_last10')}"
              f" dev={r.get('dev_loss')} {('ERR ' + r['error']) if r.get('error') else ''}")
    json.dump(out, open(os.path.join(d, "train_summary.json"), "w"), indent=1)


def load_preds(p):
    g, s = {}, defaultdict(list)
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        key = (r["dataset"], r["item_id"])
        if r.get("decoding") == "greedy":
            g[key] = r["correct"]
        else:
            s[key].append(r["correct"])
    return g, {k: sum(v) / len(v) for k, v in s.items()}


def paired(a, b):
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0
    return n, m, (m - 1.96 * sd / math.sqrt(n), m + 1.96 * sd / math.sqrt(n)), keys


def mcnemar(a, b, keys):
    b01 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
    b10 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return b01, b10, min(1.0, 2 * p)


def pair(pa, pb):
    ga, sa = load_preds(pa)
    gb, sb = load_preds(pb)
    n, m, ci, keys = paired(ga, gb)
    b01, b10, p = mcnemar(ga, gb, keys)
    agree = sum(1 for k in keys if ga[k] == gb[k]) / n
    out = {"a": pa, "b": pb, "n": n, "greedy_a": round(sum(ga[k] for k in keys) / n, 4),
           "greedy_b": round(sum(gb[k] for k in keys) / n, 4), "greedy_delta": round(m, 4),
           "greedy_ci95": [round(ci[0], 4), round(ci[1], 4)], "a_only": b01, "b_only": b10, "mcnemar_p": round(p, 4),
           "greedy_correctness_agreement": round(agree, 4)}
    if sa and sb:
        n2, m2, ci2, k2 = paired(sa, sb)
        out.update(pass1_a=round(sum(sa[k] for k in k2) / n2, 4), pass1_b=round(sum(sb[k] for k in k2) / n2, 4),
                   pass1_delta=round(m2, 4), pass1_ci95=[round(ci2[0], 4), round(ci2[1], 4)])
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2])
    elif sys.argv[1] == "pair":
        pair(sys.argv[2], sys.argv[3])
