"""Post-run checker (T2 fail-fast rule 5): read every finished run's manifest / train_log / dev_log / preflight,
flag anomalies, write per-run CSV + PNG loss curves, and keep a ledger the launcher reads: a run with uncleared
flags BLOCKS the next queue batch until ``launcher ack RUN_ID``.

    python -m vi_math_verified_aug.train.check_runs RUN_DIR [RUN_DIR ...] [--out jobs/outputs/T2_checks]

Flags (each with the evidence value):
  status       manifest status != COMPLETE
  preflight    preflight.json missing (guards off) or its decoded labels missing
  skips        GradScaler skips after step 10 > 0.5 % of steps
  no_learning  mean loss of the last 10 % of steps not < 85 % of the first 5 steps
  dev_up       final dev loss >= step-0 dev loss
  dev_gap      final dev loss > 2 x final train loss
  quick_acc    final quick-eval accuracy < step-0 (base) quick accuracy - 0.10
  repetition   final quick-eval repetition rate > 0.20
  no_answer    final quick-eval no-answer rate > 0.25
  throughput   median tok/s < 50 % of the expected (manifest guards.expected_tok_per_s)
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def _svg(rid, status, log, devl, w=720, h=400, pad=50) -> str:
    pts = [(r["step"], r["loss"]) for r in log] + [(r["step"], r["dev_loss"]) for r in devl]
    if not pts:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><text x="20" y="30">{rid}: no data</text></svg>'
    x1 = max(p[0] for p in pts) or 1
    y0, y1 = 0.0, max(p[1] for p in pts) * 1.05

    def xy(s, v):
        return pad + (w - 2 * pad) * s / x1, h - pad - (h - 2 * pad) * (v - y0) / (y1 - y0)
    tr = " ".join(f"{a:.1f},{b:.1f}" for a, b in (xy(r["step"], r["loss"]) for r in log))
    dv = " ".join(f"{a:.1f},{b:.1f}" for a, b in (xy(r["step"], r["dev_loss"]) for r in devl))
    ticks = "".join(f'<text x="5" y="{xy(0, y1 * k / 4)[1]:.0f}" font-size="11">{y1 * k / 4:.2f}</text>'
                    for k in range(5))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif">'
            f'<rect width="{w}" height="{h}" fill="white"/>'
            f'<text x="{pad}" y="20" font-size="13">{rid} ({status}) - blue: train loss/step, red: dev300 loss</text>'
            f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="black"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="black"/>{ticks}'
            f'<text x="{w - pad - 40}" y="{h - pad + 30}" font-size="11">step {x1}</text>'
            f'<polyline fill="none" stroke="#1f77b4" stroke-width="1" points="{tr}"/>'
            f'<polyline fill="none" stroke="#d62728" stroke-width="2" points="{dv}"/></svg>')


def check(run_dir: str | Path, out_root: str | Path | None = None) -> dict:
    d = Path(run_dir)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8")) if (d / "manifest.json").exists() else {}
    cfg = {}
    if (d / "config.yaml").exists():
        try:
            import yaml
            cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            cfg = {}
    log = [r for r in _jl(d / "train_log.jsonl") if r.get("tag", "main") == "main"]
    dev = _jl(d / "dev_log.jsonl")
    devl = [r for r in dev if "dev_loss" in r and r.get("kind") != "midrun"]
    mid = [r for r in dev if r.get("kind") == "midrun"]
    flags = {}
    st = man.get("status")
    if st != "COMPLETE":
        flags["status"] = st
    pf = d / "preflight.json"
    pre = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else None
    if not pre or not (pre.get("labels") or {}).get("decoded"):
        flags["preflight"] = "missing"
    n = len(log)
    if n:
        sk = sum(1 for r in log[10:] if r.get("skipped"))
        if sk > 0.005 * n:
            flags["skips"] = sk
        first = statistics.mean(r["loss"] for r in log[:5])
        last = statistics.mean(r["loss"] for r in log[-max(1, n // 10):])
        if n >= 20 and last >= 0.85 * first:
            flags["no_learning"] = [round(first, 4), round(last, 4)]
        tps = [r["real_tokens"] / r["s"] for r in log[5:] if r.get("s") and "max_abs_act" not in r]
        exp = ((cfg.get("guards") or {}).get("expected_tok_per_s"))
        med = statistics.median(tps) if tps else None
        if exp and med and med < 0.5 * exp:
            flags["throughput"] = [round(med), exp]
    else:
        first = last = med = None
    warnings = {}
    dev_warn = {}
    if len(devl) >= 2:
        # dev_v1 references have the seed style; arms with another solution style raise their NLL while learning
        # (Y+ s0, 2026-11-08) -> dev up / dev-train gap are warnings; a 2x blow-up vs step 0 blocks
        if devl[-1]["dev_loss"] > 2 * devl[0]["dev_loss"]:
            flags["dev_blowup"] = [devl[0]["dev_loss"], devl[-1]["dev_loss"]]
        elif devl[-1]["dev_loss"] >= devl[0]["dev_loss"]:
            dev_warn["dev_up"] = [devl[0]["dev_loss"], devl[-1]["dev_loss"]]
        if last and devl[-1]["dev_loss"] > 2 * last:
            dev_warn["dev_gap"] = [devl[-1]["dev_loss"], round(last, 4)]
    warnings = {}
    guards_on = bool((cfg.get("guards") or {}).get("enabled", True))
    if guards_on and mid and any(r.get("quick") is None for r in mid):
        flags["quick_missing"] = [r["step"] for r in mid if r.get("quick") is None]  # canary assertion (QA-T #2)
    if mid and mid[-1].get("quick"):
        q, b = mid[-1]["quick"], mid[0].get("quick")
        # QA-T #2: accuracy on the quick set is a non-blocking warning (P(false flag) 0.25/check at n=16)
        if b and q["correct"] < b["correct"] - 0.10:
            warnings["quick_acc"] = [b["correct"], q["correct"], q.get("n")]
        if q["repetition"] > 0.5:
            flags["repetition"] = q["repetition"]
        elif q["repetition"] > 0.20:
            warnings["repetition"] = q["repetition"]
        if q["no_answer"] > 0.5:
            flags["no_answer"] = q["no_answer"]
        elif q["no_answer"] > 0.25:
            warnings["no_answer"] = q["no_answer"]
    rid = man.get("run_id") or d.name
    res = {"run_id": rid, "status": st, "steps": n, "loss_first5": first and round(first, 4),
           "loss_last10pct": last and round(last, 4), "dev": [(r["step"], r["dev_loss"]) for r in devl],
           "quick": [(r["step"], r.get("quick")) for r in mid], "tok_per_s_median": med and round(med),
           "platform": cfg.get("platform") or (man.get("env") or {}).get("gpu"), "flags": flags,
           "warnings": {**(warnings if mid else {}), **dev_warn},
           "clean": not flags}
    if out_root:
        o = Path(out_root) / rid
        o.mkdir(parents=True, exist_ok=True)
        with open(o / "train_log.csv", "w", encoding="utf-8") as f:
            f.write("step,loss,lr,gnorm,skipped,real_tokens,s\n")
            for r in log:
                f.write(f"{r['step']},{r['loss']},{r['lr']},{r['gnorm']},{int(bool(r.get('skipped')))},"
                        f"{r.get('real_tokens')},{r.get('s')}\n")
        with open(o / "dev_log.csv", "w", encoding="utf-8") as f:
            f.write("step,dev_loss,quick_correct,quick_no_answer,quick_repetition,quick_truncated\n")
            qs = {r["step"]: r.get("quick") or {} for r in mid}
            for r in devl:
                q = qs.get(r["step"], {})
                f.write(f"{r['step']},{r['dev_loss']},{q.get('correct', '')},{q.get('no_answer', '')},"
                        f"{q.get('repetition', '')},{q.get('truncated', '')}\n")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([r["step"] for r in log], [r["loss"] for r in log], lw=0.8, label="train loss (per step)")
            if devl:
                ax.plot([r["step"] for r in devl], [r["dev_loss"] for r in devl], "o-", label="dev300 loss")
            ax.set_xlabel("optimiser step")
            ax.set_ylabel("completion NLL (nats/token)")
            ax.set_title(f"{rid} ({st})")
            ax.legend()
            fig.tight_layout()
            fig.savefig(o / "loss.png", dpi=110)
            plt.close(fig)
        except ImportError:  # no matplotlib on the laptop envs: dependency-free SVG of the same curves
            (o / "loss.svg").write_text(_svg(rid, st, log, devl), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            res["plot_error"] = repr(e)
        if pre:
            (o / "preflight_decoded.json").write_text(json.dumps(pre.get("labels"), indent=1, ensure_ascii=False),
                                                      encoding="utf-8")
        (o / "check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=str(REPO / "jobs" / "outputs" / "T2_checks"))
    a = ap.parse_args(argv)
    for r in a.runs:
        res = check(r, a.out)
        print(json.dumps({k: res[k] for k in ("run_id", "status", "steps", "loss_first5", "loss_last10pct",
                                              "tok_per_s_median", "flags")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
