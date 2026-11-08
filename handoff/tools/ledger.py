"""Build handoff/LEDGER.md: API spend (from logs/api), Modal spend (saved billing reports), Kaggle quota (timelines).

Streams every ledger line (constant memory; the biggest file is ~25 MB). No network, no API calls, no keys read.

    .venv/Scripts/python.exe handoff/tools/ledger.py            # writes handoff/LEDGER.md
    .venv/Scripts/python.exe handoff/tools/ledger.py --json x   # also dumps the numbers as JSON

Cost rules per ledger format (formats differ by workstream; see LEDGER.md "Method"):
  * common/apilog.py records (G2, G3, G4, Q1, S3, ...): field `cost_usd` (0 for cache hits); Q1 `batch_submit`
    events are bookkeeping and skipped.
  * G1 records (pilot_gen/clients.py): field `cost`.
  * X2: only `cost_ledger.jsonl` (one row per collected batch); request/result dumps are not costed again.
  * X3: `*_results.jsonl` rows carry `model` + `usage`; costed at list price x 0.5 (Message Batches).
  * NIM / Gemini: $0 (free tiers), counted as calls.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
# Mirrors vi_math_verified_aug/common/costs.py DEFAULT_PRICES (list $/MTok, recorded by G1). Re-check before reuse.

CAPS = {  # final Claude caps (DELIVER_BRIEF "Changes 03:30" + later lead approvals recorded in the WS reports)
    "D1": "$21", "G4": "G4+G5 $27", "G5": "G4+G5 $27", "D5": "$4.8", "Q1": "$15.5", "S3": "$4",
    "G1": "$10", "G2": "$8", "G3": "$6", "X2": "$8.84 approved", "X3": "–",
}


def _provider(rec: dict, fname: str) -> str:
    p = rec.get("provider")
    if p:
        return p
    m = str(rec.get("model") or rec.get("model_requested") or "")
    if m.startswith("claude"):
        return "claude"
    for k in ("claude", "gemini", "nim"):
        if k in fname:
            return k
    return "claude" if "custom_id" in rec else "?"


def scan_api(root: Path):
    agg = defaultdict(lambda: {"calls": 0, "errors": 0, "cache_hits": 0, "in": 0, "out": 0, "usd": 0.0,
                               "first": None, "last": None})
    models_returned = defaultdict(set)
    files = sorted((root / "logs" / "api").glob("*/*.jsonl"))
    for f in files:
        ws = f.parent.name
        name = f.name
        if ws == "X2" and name != "cost_ledger.jsonl":
            continue
        if ws == "X3" and not name.endswith("_results.jsonl"):
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "batch_submit":
                    continue
                if ws == "X2":  # batch-level cost rows
                    key = (ws, "claude", "claude-opus-5-5")  # X2 keying = Opus 5.5 Batch (X2 report §1)
                    models_returned[ws].add("claude-opus-5-5 (X2 report; results files carry no model field)")
                    a = agg[key]
                    a["calls"] += int(r.get("succeeded", 0))
                    a["in"] += int(r.get("input_tokens", 0))
                    a["out"] += int(r.get("output_tokens", 0))
                    a["usd"] += float(r.get("cost_usd", 0))
                    ts = r.get("collected_at")
                    a["first"] = min(filter(None, [a["first"], ts])) if ts else a["first"]
                    a["last"] = max(filter(None, [a["last"], ts])) if ts else a["last"]
                    continue
                prov = _provider(r, name)
                model = r.get("model_requested") or r.get("model") or "?"
                key = (ws, prov, model)
                a = agg[key]
                a["calls"] += 1
                u = r.get("usage") or {}
                i = int(u.get("input_tokens", u.get("in", 0)) or 0)
                o = int(u.get("output_tokens", u.get("out", 0)) or 0)
                a["in"] += i
                a["out"] += o
                ok = r.get("ok", r.get("ok_response", r.get("type") != "errored"))
                if ok is False or (isinstance(r.get("status"), int) and r["status"] not in (200,)):
                    a["errors"] += 1
                if r.get("cache_hit"):
                    a["cache_hits"] += 1
                if "cost_usd" in r:
                    a["usd"] += float(r.get("cost_usd") or 0.0)
                elif "cost" in r:
                    a["usd"] += float(r.get("cost") or 0.0)
                elif prov == "claude" and model in PRICES:  # X3 batch results
                    pi, po = PRICES[model]
                    a["usd"] += (i * pi + o * po) / 1e6 * 0.5
                ts = r.get("ts_utc") or r.get("ts")
                if isinstance(ts, str):
                    a["first"] = ts if a["first"] is None or ts < a["first"] else a["first"]
                    a["last"] = ts if a["last"] is None or ts > a["last"] else a["last"]
                mr = r.get("model_returned")
                if not mr and isinstance(r.get("raw_response"), dict):
                    mr = r["raw_response"].get("model")
                if not mr and ws in ("G1", "X3"):
                    mr = r.get("model")  # G1/X3 record the model id of the call
                if prov == "claude" and mr:
                    models_returned[ws].add(mr)
    return agg, models_returned


def spend_db(root: Path):
    p = root / ".cache" / "vimath" / "spend.sqlite"
    if not p.exists():
        return {}
    con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
    try:
        return {s: (n, u) for s, n, u in con.execute(
            "select scope, count(*), sum(usd) from charges where provider='claude' group by scope")}
    finally:
        con.close()


def modal(root: Path):
    reps = []
    mtd = [f for f in glob.glob(str(root / "jobs" / "logs" / "**" / "*billing_mtd*.json"), recursive=True)]
    for f in mtd or glob.glob(str(root / "**" / "*billing_report*.json"), recursive=True):
        if ".venv" in f:
            continue
        reps.append((os.path.getmtime(f), f))
    if not reps:
        return None
    mt, latest = max(reps)
    data = json.load(open(latest, encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("rows", data.get("items", []))
    per = defaultdict(float)
    for r in rows:
        per[r.get("description", "?")] += float(r.get("cost", 0) or 0)
    txts = []
    for f in glob.glob(str(root / "**" / "*billing*.txt"), recursive=True):
        if ".venv" in f:
            continue
        t = open(f, encoding="utf-8", errors="replace").read()
        m = re.search(r"Metered Cost:\s*([0-9.]+)", t)
        if m:
            txts.append((os.path.getmtime(f), f, float(m.group(1)),
                         (re.search(r"Billed Cost:\s*\$?([0-9.]+)", t) or [None, None])[1]))
    return {"report": Path(latest).resolve().relative_to(root).as_posix(), "report_mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(mt)),
            "per_app": dict(sorted(per.items(), key=lambda x: -x[1])), "total": sum(per.values()),
            "summary": max(txts) if txts else None}


def kaggle(root: Path):
    out = {}
    for f in sorted((root / "jobs" / "logs").glob("*_timeline.jsonl")):
        last = None
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "gpu_quota_used_h" in r:
                last = (r["ts"], r["gpu_quota_used_h"], r.get("status"))
        if last:
            out[f.name] = last
    return out


def fmt_usd(x):
    return f"${x:,.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "handoff" / "LEDGER.md"))
    ap.add_argument("--json")
    ap.add_argument("--console-usd", type=float, default=None,
                    help="Claude spend shown in the Anthropic console (source of truth), as read by the user")
    ap.add_argument("--console-when", default="")
    a = ap.parse_args()
    agg, mret = scan_api(ROOT)
    db = spend_db(ROOT)
    md = modal(ROOT)
    kg = kaggle(ROOT)

    claude_ws = defaultdict(float)
    for (ws, prov, model), v in agg.items():
        if prov == "claude":
            claude_ws[ws] += v["usd"]
    total_claude = sum(claude_ws.values())

    L = ["# LEDGER — compute and API spend", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M')} (UTC+7) by `handoff/tools/ledger.py` from files on disk. "
         "Re-run it after the last job finishes; do not hand-edit the tables._", "",
         "## 1. Claude API (Anthropic) by workstream", "",
         "**Source of truth for Claude spend is the Anthropic console** (workspace usage/cost page). The ledger "
         "numbers below are a reconstruction from our own logs, used to attribute spend to workstreams.", "",
         "Source: `logs/api/<WS>/*.jsonl` (append-only ledgers, keys never logged). Cross-check: "
         "`.cache/vimath/spend.sqlite` (the `CostTracker` hard-cap database used by `common/llm.py`).", "",
         "| WS | Claude USD (ledger) | spend.sqlite USD | Cap (DELIVER_BRIEF) | Models returned |",
         "|---|---|---|---|---|"]
    for ws in sorted(claude_ws):
        dbv = db.get(ws)
        L.append(f"| {ws} | {fmt_usd(claude_ws[ws])} | {fmt_usd(dbv[1]) if dbv else '– (not via common/llm.py)'} | "
                 f"{CAPS.get(ws, '–')} | {', '.join(sorted(mret.get(ws, []))) or '–'} |")
    for ws in sorted(set(db) - set(claude_ws)):
        L.append(f"| {ws} | – (no ledger file) | {fmt_usd(db[ws][1])} | {CAPS.get(ws, '–')} | – |")
    L += [f"| **Total** | **{fmt_usd(total_claude)}** | {fmt_usd(sum(v[1] for v in db.values()))} | | |", "",
          "**Reconciliation with the console:** 2026-11-07 ~22:40 the user read ≈ $43 in the console while this "
          "ledger said $43.22 (+0.5 %). "
          + (f"Latest console reading ≈ {fmt_usd(a.console_usd)} ({a.console_when}) vs ledger {fmt_usd(total_claude)}: "
             f"difference {fmt_usd(total_claude - a.console_usd)} "
             f"({(total_claude - a.console_usd) / a.console_usd * 100:+.1f} %)."
             if a.console_usd else
             f"**Final console reading: [the user reads the console at the freeze and the lead re-runs this script with "
             f"`--console-usd X --console-when ...`]**; ledger now {fmt_usd(total_claude)}.")
          + " A gap above ~$1 needs investigation (most likely: a Batch whose results were never collected, or "
          "calls made outside `common/llm.py`).", "",
          "Rule: where a client logged `cost_usd`/`cost`, that value is used (it carries the Batch −50 % discount). "
          "Cost is recomputed from `usage` only where nothing was logged (X3 Batch results, at Batch rate). "
          "Recomputing G4/Q1 from usage at list price would overstate them, because part of their calls ran on Batch.", "",
          "Claude budget: $100 (+ workspace header). Prices: list $/MTok as in `common/costs.py` "
          "(Haiku 5.5 0.10/0.50, Sonnet 5.5 2/10, Opus 5.5 4/20; Batch ×0.5).", "",
          "## 2. All providers, per workstream × model", "",
          "| WS | Provider | Model requested | Calls | Errors | Cache hits | In tok | Out tok | USD | First | Last |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (ws, prov, model), v in sorted(agg.items()):
        L.append(f"| {ws} | {prov} | `{model}` | {v['calls']:,} | {v['errors']:,} | {v['cache_hits']:,} | "
                 f"{v['in']:,} | {v['out']:,} | {fmt_usd(v['usd'])} | {(v['first'] or '')[:16]} | "
                 f"{(v['last'] or '')[:16]} |")
    L += ["", "Gemini (`gemini-*`, free tier, 6 keys/projects) and NIM (free trial) are $0. Timestamps are UTC as logged.", ""]

    L += ["## 3. Modal", ""]
    if md:
        L += [f"Latest saved per-app report: `{md['report']}` (written {md['report_mtime']}). "
              "Totals are metered (pre-credit) USD, summed over every object and interval of each app in that report "
              "(a `*billing_mtd*` report = month to date, `modal billing report --for \"this month\" --json`). "
              "Modal credit: $30/month (November 2026).", "",
              "| Modal app | Metered USD |", "|---|---|"]
        for k, v in md["per_app"].items():
            L.append(f"| `{k}` | {v:.3f} |")
        L.append(f"| **Total (apps in this report)** | **{md['total']:.2f}** |")
        if md["summary"] and "mtd" not in md["report"]:
            _, f, met, billed = md["summary"]
            L += ["", f"Latest saved `modal billing` summary: `{Path(f).resolve().relative_to(ROOT).as_posix()}` — metered ${met:.2f}, "
                  f"billed ${billed or '?'} (the monthly $30 credit absorbs metered cost)."]
    else:
        L.append("No saved Modal billing report found.")
    L += ["", "Modal notes (2026-11-08): the launcher's workspace guard read **$29.95 month to date at ~08:20** "
          "(billed apps + its estimate of in-flight unbilled work) and hard-stopped (`jobs/queue/t2_state.json` "
          "`hard_stop`; guard limit $29.5 of the $30 credit). Billing lags by minutes, so the per-app sum above can "
          "trail the guard. By workstream: T1r/T2 = `t1r-eval` (tranche training + eval, G6 sweep, pilots); "
          "G5 = `g5-vllm-cycle*` + `g5-decontam` ($9.50 of its $9.5 cap; one timed-out call cost $2.89); "
          "R-GPU = `r-gpu-bench`; S2, O1, Y2, X5, S3, G3, G4, D1, QA-T as named.",
          "", "To refresh (read-only, no spend): `E:\\venv_tools\\Scripts\\modal.exe billing report --for \"this month\" "
          "--json` (save the output as `jobs/logs/D4/modal_billing_mtd_<timestamp>.json`, then re-run this script).", ""]

    L += ["## 4. Kaggle GPU quota", "",
          "Weekly quota 30 GPU-h (T4×2 bills 1 quota-hour per wall-hour, measured by S2). Last reading per kernel timeline:", "",
          "| Timeline | Last reading (UTC) | gpu_quota_used_h (week) | Status |", "|---|---|---|---|"]
    for k, (ts, q, st) in kg.items():
        L.append(f"| `jobs/logs/{k}` | {ts} | {q} | {st} |")
    L += ["", "The quota counter is weekly and cumulative across kernels; the largest recent value is the week's usage. "
          "Kernels launched without the timeline writer (e.g. D5's `datuni/d5-select-embed`, ≈3 min T4) are not in "
          "these files; Kaggle shows the exact weekly figure on the account's settings page.", ""]

    L += ["## 5. Method notes and known gaps", "",
          "- G1 and X2/X3 predate `common/llm.py`, so they are absent from `spend.sqlite`; their ledger costs are summed here.",
          "- X2's request/result dumps are not re-costed; `logs/api/X2/cost_ledger.jsonl` is the per-batch cost of record.",
          "- G2's three one-shot Gemini quota probes were not logged (G2 report); they were free-tier.",
          "- W3/W5 are early (pre-brief) NIM/Gemini probes; $0. No Claude calls are in them.",
          "- STATUS.md's \"≈ $32.5\" (22:35) omits X2 ($8.84) and X3 ($1.73); this table includes them.",
          "- Kaggle: no API exposes remaining quota; the timeline value is read from the Kaggle kernel status at poll time.",
          ""]
    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps({
            "claude_by_ws": claude_ws, "claude_total": total_claude,
            "spend_db": {k: v[1] for k, v in db.items()},
            "modal": md, "kaggle": kg,
            "detail": {"|".join(k): v for k, v in agg.items()}}, indent=1, default=str), encoding="utf-8")
    print(f"Claude total {total_claude:.2f}; modal {md['total'] if md else None}; -> {a.out}")


if __name__ == "__main__":
    main()
