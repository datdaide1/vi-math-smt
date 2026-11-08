"""Markdown tables for the G1 report from metrics.json + items_scored.jsonl (+ throughput projections).

python -m vi_math_verified_aug.pilot_gen.report_tables -> data/pilot/generator/report_tables.md
"""
from __future__ import annotations

import json
import math

from . import clients as C
from . import seeds as S

OUT = S.OUT
CAND = ["haiku", "sonnet", "gemini", "gemini31"]
NIM = ["deepseek", "nemotron", "glm"]
NIM_UNCONTENDED = {"deepseek": 260.0, "nemotron": 48.0, "glm": 96.0}  # p50 native s, 5-seed v1 run, low load
NAME = {"haiku": "Claude Haiku 5.5", "sonnet": "Claude Sonnet 5.5", "gemini": "gemini-3.5-flash-lite",
        "gemini31": "gemini-3.1-flash-lite", "deepseek": "deepseek-v4.1-flash", "nemotron": "nemotron-3-ultra",
        "glm": "glm-5.3-flash"}


def t(m, k):
    v = m
    for p in k.split("."):
        v = (v or {}).get(p)
    return v["txt"] if isinstance(v, dict) and "txt" in v else ("–" if v is None else v)


def f(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


def main():
    R = json.load(open(OUT / "metrics.json", encoding="utf-8"))
    G = R["generators"]
    L = []
    L.append("### A. Native generation (arm N), Opus subsample (n = 30 seeds, paired), primary judge per policy\n")
    L.append("| Generator | primary judge | yield (valid ∧ Opus-agrees ∧ no leak) | strict yield (+teacher, new, no leak) | "
             "teacher-acceptable | new (M2–M6) | leak | absurd | tail | near-dup | answer-correct (Opus) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    pj = {"haiku": "Gemini-lite", "sonnet": "Gemini-lite", "gemini": "Opus", "gemini31": "Opus"}
    for g in CAND:
        m = G[g]
        L.append(f"| {NAME[g]} | {pj[g]} | {t(m, 'S_yield')} | {t(m, 'S_yield_strict')} | {t(m, 'S_primary.teacher')} | "
                 f"{t(m, 'S_primary.new_M2_M6')} | {t(m, 'S_primary.leak')} | {t(m, 'S_primary.absurd')} | "
                 f"{t(m, 'S_primary.tail')} | {t(m, 'S_primary.neardup')} | {t(m, 'S_answer_correct')} |")
    L.append("\n### B. Same subsample, one judge for everyone (Opus 5.5, effort medium) — rank cross-check\n")
    L.append("| Generator | valid | teacher-acceptable | new (M2–M6) | leak | absurd | tail | near-dup | natural (mean) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for g in CAND:
        m = G[g]["S_opus"]
        L.append(f"| {NAME[g]} | {t(m, 'valid')} | {t(m, 'teacher')} | {t(m, 'new_M2_M6')} | {t(m, 'leak')} | "
                 f"{t(m, 'absurd')} | {t(m, 'tail')} | {t(m, 'neardup')} | {f(m.get('natural_mean'), 2)} |")
    L.append("\n### C. All 100 seeds, Gemini-lite judge for everyone (same-family for the two Gemini generators)\n")
    L.append("| Generator | parsed | valid | teacher-acceptable | new (M2–M6) | leak (regex ∨ judge) | absurd | tail | "
             "near-dup | answer value in text | chrF to seed (median) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in CAND + NIM:
        m = G[g]
        a = m["all_gemini_judge"]
        L.append(f"| {NAME[g]}{' (partial)' if g in NIM else ''} | {t(m, 'parsed')} | {t(a, 'valid')} | "
                 f"{t(a, 'teacher')} | {t(a, 'new_M2_M6')} | {t(a, 'leak')} | {t(a, 'absurd')} | {t(a, 'tail')} | "
                 f"{t(a, 'neardup')} | {t(m, 'answer_in_text')} | {f(m.get('chrf_median'), 2)} |")
    L.append("\n### D. Formalizer (b), editor (c), solution writer (d)\n")
    L.append("| Model | formalize pass@1 | formalize pass (≤3 attempts) | passes with a secondary flag | "
             "render pass (tags exact, no extra numerals) | solve seeds (gold) | solve common accepted set |")
    L.append("|---|---|---|---|---|---|---|")
    for g in CAND + NIM:
        m = G[g]
        L.append(f"| {NAME[g]}{' (partial)' if g in NIM else ''} | {t(m, 'formalize_pass1')} | "
                 f"{t(m, 'formalize_pass3')} | {t(m, 'formalize_flagged')} | {t(m, 'render_pass')} | "
                 f"{t(m, 'solve_seeds')} | {t(m, 'solve_common')} |")
    L.append("\nFormalize failure reasons (final attempt): " + "; ".join(
        f"{NAME[g]}: {G[g]['formalize_fail_reasons']}" for g in CAND))
    L.append("\nRender failure reasons: " + "; ".join(f"{NAME[g]}: {G[g]['render_fail_reasons']}" for g in CAND))
    # editor output quality (Opus judge on rendered texts, subsample)
    from .analyze import lab
    from .util import fmt_ci
    L.append("\n### D2. Editor output quality: Opus 5.5 judge on tag-passing rendered texts (Opus subsample seeds)\n")
    L.append("| Editor | n | valid | teacher-acceptable | new vs seed | op followed | absurd | fluency issue |")
    L.append("|---|---|---|---|---|---|---|---|")
    for g in CAND:
        p = OUT / "judge_render" / f"{g}.jsonl"
        if not p.exists():
            continue
        V = [lab(json.loads(l)["verdict"]) for l in open(p, encoding="utf-8") if json.loads(l)["verdict"]]
        n = len(V)
        L.append(f"| {NAME[g]} | {n} | " + " | ".join(fmt_ci(sum(bool(v[k]) for v in V), n) for k in
                                                    ("valid", "teacher", "new", "op", "absurd", "fluency")) + " |")
    lb = OUT / "latbench.json"
    if lb.exists():
        B = json.load(open(lb))
        L.append("\n### E0. Clean API latency (no queueing; 12 calls per cell, 3 in flight)\n")
        L.append("| Model | native p50 s | native p90 s | native out tok | solve p50 s | solve p90 s | solve out tok |")
        L.append("|---|---|---|---|---|---|---|")
        for g, d in B.items():
            L.append(f"| {NAME[g]} | {f(d['native']['p50'])} | {f(d['native']['p90'])} | {f(d['native']['out_mean'], 0)} | "
                     f"{f(d['solve']['p50'])} | {f(d['solve']['p90'])} | {f(d['solve']['out_mean'], 0)} |")
    # latency / tokens / cost
    L.append("\n### E. Latency, tokens, cost per call (this run, uncached successful calls)\n")
    L.append("| Model | task | calls | errors (by HTTP status) | wall p50 s (incl. queue) | wall p90 s | "
             "HTTP p50 s (n) | HTTP p90 s | in tok | out tok (incl. thinking) | $ / call |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in CAND + NIM:
        for task, d in sorted(G[g]["tasks"].items()):
            L.append(f"| {NAME[g]} | {task} | {d['calls']} | {d['errors'] or ''} | {f(d['lat_med'])} | "
                     f"{f(d['lat_p90'])} | {f(d.get('http_med'))} ({d.get('http_n', 0)}) | {f(d.get('http_p90'))} | "
                     f"{f(d['in_mean'], 0)} | {f(d['out_mean'], 0)} | {d['cost_per_call']:.5f} |")
    for k, d in sorted(R["judge_calls"].items()):
        L.append(f"| {k.split(':')[0]} | {k.split(':')[1]} | {d['calls']} | {d['errors'] or ''} | {f(d['lat_med'])} | "
                 f"{f(d['lat_p90'])} | {f(d['in_mean'], 0)} | {f(d['out_mean'], 0)} | "
                 f"{(d['cost_per_call'] or 0):.5f} |")
    # judge agreement
    L.append("\n### F. Judge agreement: Opus 5.5 vs gemini-3.5-flash-lite (items judged by both)\n")
    L.append("| Field | n | Cohen's κ | raw agreement | Opus positive rate | Gemini positive rate |")
    L.append("|---|---|---|---|---|---|")
    for k, v in R["judge_agreement"].items():
        if "@" in k:
            continue
        L.append(f"| {k} | {v['n']} | {f(v['kappa'], 2)} | {100 * v['agree']:.0f}% | {100 * v['opus_pos']:.0f}% | "
                 f"{100 * v['gem_pos']:.0f}% |")
    # throughput projection
    L.append("\n### G. Throughput and projected wall-clock / cost for 10k accepted items per arm\n")
    rows = []
    for g in CAND + NIM:
        m = G[g]
        nat = m["tasks"].get("native", {})
        y = m["S_yield"]["p"] if m["S_yield"]["n"] else None
        rp = m["render_pass"]["p"] if m["render_pass"]["n"] else None
        B = json.load(open(OUT / "latbench.json")) if (OUT / "latbench.json").exists() else {}
        lat = (B.get(g, {}).get("native", {}).get("p50") or nat.get("http_med") or NIM_UNCONTENDED.get(g)
               or nat.get("lat_med"))
        prov = C.MODELS[g][0]
        if prov == "nim":
            cap_rpm = min(C.NIM_RPM, C.NIM_INFLIGHT * 60 / lat) if lat else None
            how = f"NIM 1 key: min(38 RPM, 35 in flight / p50 {f(lat, 0)} s)"
        elif prov == "claude":
            cap_rpm = 24 * 60 / lat if lat else None
            how = f"24 in flight / p50 {f(lat, 0)} s (Batch: no RPM limit, ≤24 h turnaround)"
        else:
            obs = R.get("lane_obs", {}).get(g)
            cap_rpm = 31.0 if lat else None  # sustained, measured over the 25-min backlog (see section H)
            how = ("6 independent projects (1 key each); sustained 31 OK calls/min/model measured "
                   "(bursts of ~70/min ≈ 12k tok/min/project trigger 429s after 2–3 min)")
        raw_needed = 10000 / y if y else None
        hours_n = raw_needed / (cap_rpm * 60) if (raw_needed and cap_rpm) else None
        cpc = nat.get("cost_per_call", 0) or 0
        usd_n = raw_needed * cpc if raw_needed else None
        ren = m["tasks"].get("render", {})
        lat_r = ren.get("lat_med")
        cap_r = (min(C.NIM_RPM, C.NIM_INFLIGHT * 60 / lat_r) if prov == "nim" else
                 (24 * 60 / lat_r if prov == "claude" else 6 * 12 * 0.9)) if lat_r else None
        ren_needed = 10000 / rp if rp else None
        hours_y = ren_needed / (cap_r * 60) if (ren_needed and cap_r) else None
        usd_y = ren_needed * (ren.get("cost_per_call", 0) or 0) if ren_needed else None
        rows.append((g, how, cap_rpm, y, raw_needed, hours_n, usd_n, rp, hours_y, usd_y))
    vj = R["judge_calls"].get("opus:judge", {}).get("cost_per_call") or 0
    vs = R["judge_calls"].get("opus:opus_solve", {}).get("cost_per_call") or 0
    L.append("| Generator | throughput basis | gen calls/min | N yield (S) | raw N items for 10k | "
             "N generation wall-clock h | N generator $ direct (Batch −50%) | + Opus judge & blind-solve $ per 10k "
             f"(${vj:.4f}+${vs:.4f} per raw item, direct) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for g, how, cap, y, raw, h, usd, rp, hy, uy in rows:
        L.append(f"| {NAME[g]}{' (partial)' if g in NIM else ''} | {how} | {f(cap)} | "
                 f"{'–' if y is None else f'{100 * y:.0f}%'} | {f(raw, 0)} | {f(h)} | "
                 f"{'–' if usd is None else f'${usd:.2f}'} | {'–' if raw is None else f'${raw * (vj + vs):.0f}'} |")
    L.append(f"\nObserved provider throughput during the run (successful calls / active minute): "
             f"{json.dumps({k: round(v['per_min'], 1) for k, v in R['throughput'].items()})}")
    L.append(f"\nErrors by model and HTTP status (this run): {json.dumps(R['errors_by_status'])}")
    L.append(f"\nClaude spend (all G1 logs): ${R['claude_spent_total']:.2f}")
    (OUT / "report_tables.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
