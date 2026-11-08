"""G6 selection: pull every sweep eval from the Volume, rank candidates by dev_v1 mean pass@1 (K=4), then greedy,
then dev NLL; paired item-bootstrap CIs vs the winner; per-dataset (GSM8K / MATH seeds) breakdown.

    .venv/Scripts/python.exe jobs/t2/g6_select.py      -> jobs/t2/g6_selection.json (+ printed table)
"""
import json
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "jobs" / "outputs" / "t2-g6-eval"
MODAL = r"E:\venv_tools\Scripts\modal.exe"


def pull(name):
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{name}.jsonl"
    if not dest.exists():
        subprocess.run([MODAL, "volume", "get", "--force", "t1r-eval", f"/eval/g6/{name}/preds.jsonl", str(dest)],
                       capture_output=True, env=dict(os.environ, PYTHONIOENCODING="utf-8", MSYS_NO_PATHCONV="1"))
    return [json.loads(x) for x in dest.read_text(encoding="utf-8").splitlines() if x.strip()] if dest.exists() else []


def per_item(rows):
    s, g = defaultdict(list), {}
    for r in rows:
        k = (r["dataset"], r["item_id"])
        if r["sample_k"] == -1:
            g[k] = r["correct"]
        else:
            s[k].append(r["correct"])
    return {k: sum(v) / len(v) for k, v in s.items()}, g


def boot(d, B=4000, seed=0):
    rng = random.Random(seed)
    n = len(d)
    m = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(m[int(0.025 * B)], 4), round(m[int(0.975 * B)], 4)]


def main():
    cands = [f"t2-g6-S0-vi-0p6B-n3k-lr{lr:g}-ep3-e{e}" for lr in (1e-4, 2e-4, 4e-4, 8e-4) for e in (1, 2, 3)]
    cands += ["t2-g6-S0-vi-0p6B-n3k-lr0.0002-ep2fresh-e2", "base-4shot", "t1r-S0-uns-0p6B-s1", "t1r-S0-uns-0p6B-s2"]
    T = {}
    for c in cands:
        rows = pull(c)
        if rows:
            T[c] = per_item(rows)
    res = {}
    for c, (p, g) in T.items():
        ks = sorted(p)
        res[c] = {"n": len(ks), "pass1": round(sum(p.values()) / len(p), 4), "pass1_ci": boot([p[k] for k in ks]),
                  "greedy": round(sum(g.values()) / len(g), 4)}
        for ds in sorted({k[0] for k in ks}):
            kk = [k for k in ks if k[0] == ds]
            res[c][ds] = {"n": len(kk), "pass1": round(sum(p[k] for k in kk) / len(kk), 4),
                          "greedy": round(sum(g[k] for k in kk) / len(kk), 4)}
    sweep = {c: v for c, v in res.items() if c.startswith("t2-g6")}
    best = max(sweep, key=lambda c: (sweep[c]["pass1"], sweep[c]["greedy"]))
    pb, _ = T[best]
    for c in res:
        p, _ = T[c]
        d = [p[k] - pb[k] for k in sorted(pb) if k in p]
        res[c]["minus_best"] = {"diff": round(sum(d) / len(d), 4), "ci": boot(d)}
    fid = None
    a, b = "t2-g6-S0-vi-0p6B-n3k-lr0.0002-ep3-e2", "t2-g6-S0-vi-0p6B-n3k-lr0.0002-ep2fresh-e2"
    if a in T and b in T:
        pa, ga = T[a]
        pf, gf = T[b]
        d = [pa[k] - pf[k] for k in sorted(pa)]
        fid = {"branch_minus_fresh_pass1": round(sum(d) / len(d), 4), "ci": boot(d),
               "branch_greedy": round(sum(ga.values()) / len(ga), 4), "fresh_greedy": round(sum(gf.values()) / len(gf), 4),
               "rule": "accept the epoch-branch mechanism if |diff| <= 1 pt"}
    # confirmation (2 extra seeds of the winner and the runner-up): mean over seeds 1-3 per (LR, epochs)
    conf = {}
    for lr in ("0.0004", "0.0001"):
        for e in (2, 3):
            names = [f"t2-g6-S0-vi-0p6B-n3k-lr{lr}-ep3-e{e}"] + [f"t2-g6c-S0-vi-0p6B-n3k-lr{lr}-ep3-s{sd}-e{e}"
                                                                 for sd in (2, 3)]
            got = [n for n in names if (n in T) or pull(n)]
            for n in got:
                if n not in T:
                    T[n] = per_item(pull(n))
            if len(got) == 3:
                ks = sorted(T[got[0]][0])
                per = {k: sum(T[n][0][k] for n in got) / 3 for k in ks}
                gr = {k: sum(T[n][1][k] for n in got) / 3 for k in ks}
                comp = {k: (per[k] + gr[k]) / 2 for k in ks}   # lead: composite (pass@1 + greedy) / 2
                row = {"composite_3seeds": round(sum(comp.values()) / len(comp), 4), "composite_ci": boot(list(comp.values())),
                       "pass1_3seeds": round(sum(per.values()) / len(per), 4),
                       "greedy_3seeds": round(sum(gr.values()) / len(gr), 4),
                       "per_seed_pass1": [round(sum(T[n][0].values()) / len(T[n][0]), 4) for n in got],
                       "per_seed_greedy": [round(sum(T[n][1].values()) / len(T[n][1]), 4) for n in got],
                       "_comp": comp}
                for ds in sorted({k[0] for k in ks}):
                    kk = [k for k in ks if k[0] == ds]
                    row[ds] = {"n": len(kk), "pass1": round(sum(per[k] for k in kk) / len(kk), 4),
                               "greedy": round(sum(gr[k] for k in kk) / len(kk), 4),
                               "composite": round(sum(comp[k] for k in kk) / len(kk), 4)}
                conf[f"lr{lr}-ep{e}"] = row
    names = [n for n in conf if "_comp" in conf[n]]
    if names:
        top = max(names, key=lambda n: conf[n]["composite_3seeds"])
        for n in names:
            d = [conf[n]["_comp"][k] - conf[top]["_comp"][k] for k in sorted(conf[top]["_comp"])]
            conf[n]["composite_minus_top"] = {"diff": round(sum(d) / len(d), 4), "ci": boot(d)}
        conf["_top_by_composite"] = top
    for v in conf.values():
        if isinstance(v, dict):
            v.pop("_comp", None)
    out = {"winner": best, "table": res, "g6_fidelity": fid, "confirmation": conf}
    (REPO / "jobs" / "t2" / "g6_selection.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for c in sorted(res, key=lambda c: -res[c]["pass1"]):
        v = res[c]
        print(f"{c:48s} pass1 {v['pass1']:.4f} {v['pass1_ci']} greedy {v['greedy']:.4f} "
              f"gsm {v.get('dev_seed_gsm8k', {}).get('pass1')} math {v.get('dev_seed_math', {}).get('pass1')} "
              f"vs best {v['minus_best']}")
    print("winner", best, "fidelity", fid)
    print("confirmation", json.dumps(conf, indent=1))


if __name__ == "__main__":
    main()
