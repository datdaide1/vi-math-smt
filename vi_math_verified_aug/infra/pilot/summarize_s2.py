"""Summarize S2 student-pilot outputs (s2_pilot.py) into markdown tables + a grid T4-hour re-estimate.

Usage: python -m vi_math_verified_aug.infra.pilot.summarize_s2 RESULTS_DIR [V1_RESULTS_DIR] > summary.md

Grid run counts are T3_training.md sec.5.2 (a run = 9M training tokens): 0.6B x30, 1.7B x92, second family x78, 4B x30.
"""
import json
import re
import sys
from pathlib import Path

GRID = {"Qwen3-0.6B-Base": 30, "Qwen3-1.7B-Base": 92, "Qwen3-4B-Base": 30}
SECOND_FAMILY_RUNS = 78
RUN_TOKENS = 9e6


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def first_line_error(d, name):
    e = load(d / f"{name}.error.json")
    if not e:
        return None
    lines = [l for l in e["error"].strip().splitlines() if l.strip()]
    return lines[-1][:220] if lines else "error"


def vllm_backend(d):
    for p in sorted(d.glob("eval_vllm*.log")):
        txt = p.read_text(encoding="utf-8", errors="replace")
        m = re.findall(r"Using (?:AttentionBackendEnum\.)?([A-Z_]+) (?:attention )?backend", txt)
        if m:
            return m[-1]
    return None


def main(res_dir, v1_dir=None):
    res_dir = Path(res_dir)
    meta = load(res_dir / "meta.json") or {}
    env = load(res_dir / "env_check.json") or {}
    print("## Environment\n")
    print("```")
    print(meta.get("pip", "").strip())
    print("```")
    print(f"env_fix: {[(x['cmd'], x['rc']) for x in meta.get('env_fix', [])]}; envcheck rc={meta.get('envcheck_rc')}; "
          f"peft smoke={env.get('peft_lora_fp16_smoke', {}).get('ok')}; wall_s={meta.get('wall_s')}")
    if meta.get("unsloth_install"):
        print(f"unsloth_install rc={meta['unsloth_install'].get('rc')}")
    models = sorted(p.name for p in res_dir.iterdir() if p.is_dir() and p.name not in ("check",))
    v1 = Path(v1_dir) if v1_dir else None
    print("\n## Per model\n")
    print("| model | 4-shot acc (300) | 95% CI | engine / attn | eval gen tok/s | SFT real tok/s | T4-h / 9M tok | "
          "micro×accum, GC | peak GB | pad frac (grouped / naive) | loss first5→last5 | skipped (after step 10) | "
          "act absmax (headroom×) | top act key |")
    print("|" + "---|" * 14)
    rows = {}
    for m in models:
        d = res_dir / m
        ev = load(d / "eval.json")
        src = "S2"
        if ev is None and v1 and (v1 / m / "eval.json").exists():
            ev, src = load(v1 / m / "eval.json"), "W2-v1"
        sft = load(d / "sft.json")
        rows[m] = {"eval": ev, "sft": sft}
        sk = {p.stem.replace(".skipped", ""): load(p)["why"][:120] for p in d.glob("*.skipped.json")}
        f = lambda x, k, dflt="—": (x or {}).get(k, dflt)  # noqa
        acc = f(ev, "accuracy")
        eng = f(ev, "engine")
        if ev and eng == "vllm":
            eng += f"/{vllm_backend(d) or 'TRITON_ATTN'}"
        if ev and src != "S2":
            eng += f" ({src})"
        if not ev:
            eng = sk.get("eval") or first_line_error(d, "eval_hf") or "—"
        s_cell = "—"
        if sft:
            s_cell = (f"| {f(sft,'real_tokens_per_s')} | {f(sft,'t4_hours_per_9M_tokens')} | "
                      f"{f(sft,'micro_batch')}×{f(sft,'accum')}, {'on' if sft.get('grad_ckpt') else 'off'} | "
                      f"{f(sft,'peak_mem_gb')} | {f(sft,'pad_fraction')} / {f(sft,'pad_fraction_naive_random_order')} | "
                      f"{f(sft,'loss_first5_mean')}→{f(sft,'loss_last5_mean')} | "
                      f"{f(sft,'skipped_steps')} ({f(sft,'skipped_steps_after_10')}) | "
                      f"{round(sft['act_absmax_max'],1) if sft.get('act_absmax_max') else 'nonfinite'} "
                      f"({f(sft,'act_fp16_headroom_x')}) | {(sft.get('act_top_keys') or [['—']])[0][0]} |")
        else:
            why = sk.get("sft") or first_line_error(d, "sft") or "not run"
            s_cell = f"| {why} |" + " |" * 8
        ci = f(ev, "ci95_wilson")
        print(f"| {m} | {acc} | {ci} | {eng} | {f(ev,'gen_tok_per_s')} {s_cell}")
        for name in ("eval_vllm0", "eval_hf", "sft", "packcheck", "sft_packed_hf", "unsloth"):
            e = first_line_error(d, name)
            if e:
                print(f"<!-- {m} {name}: {e} -->")
    # packcheck
    for m in models:
        pc = load(res_dir / m / "packcheck.json")
        if not pc:
            continue
        print(f"\n## Packing check — {m} (max |Δlogit| vs separate, all 4 items; items 2–4 in brackets)\n")
        print("| variant | fp32 | fp16 | NLL item2 sep→variant (fp16) |")
        print("|---|---|---|---|")
        for v in pc["by_dtype"]["fp32"]:
            a, b = pc["by_dtype"]["fp32"][v], pc["by_dtype"]["fp16"][v]
            it = b["per_item"][1]
            print(f"| {v} | {a['max_abs_all']} [{a['max_abs_items_2_4']}] {'PASS' if a['pass_1e-2'] else 'fail'} | "
                  f"{b['max_abs_all']} [{b['max_abs_items_2_4']}] | {it['nll_sep']}→{it['nll_var']} |")
    for m in models:
        u = load(res_dir / m / "unsloth.json")
        if not u:
            continue
        print(f"\n## Unsloth — {m}\n")
        print(f"libs: { {k: v for k, v in (u.get('libs') or {}).items() if v} }")
        print(f"auto_installed: {u.get('auto_installed')}; accepted: {u.get('padding_free_accepted')}")
        for dn, c in (u.get("checks") or {}).items():
            if "error" in c and "max_abs_all" not in c:
                print(f"- check {dn}: ERROR {c['error'].strip().splitlines()[-1][:300]}")
            else:
                print(f"- check {dn}: param_dtype={c.get('param_dtype')} segments={c.get('n_segments')} "
                      f"max_abs={c.get('max_abs_all')} pass={c.get('pass_1e-2')} keys={list(c.get('batch_keys', {}))}")
        for k, s in (u.get("sft") or {}).items():
            if "error" in s:
                print(f"- sft {k}: ERROR {s['error'].strip().splitlines()[-1][:300]}")
            else:
                print(f"- sft {k}: tok/s={s.get('real_tokens_per_s_steps3plus')} T4-h/9M={s.get('t4_hours_per_9M_tokens')} "
                      f"skipped={s.get('skipped_steps')} peak={s.get('peak_mem_gb')} loss10={s.get('loss_by_10')}")
    # grid
    print("\n## Grid re-estimate (runs × 9M tokens / measured padded tok/s)\n")
    print("| block | runs | measured tok/s | T4-h/run | T4-h |")
    print("|---|---|---|---|---|")
    tot = 0.0
    for m, n in GRID.items():
        s = (rows.get(m) or {}).get("sft") or {}
        tps = s.get("real_tokens_per_s")
        if tps:
            h = RUN_TOKENS / tps / 3600
            tot += n * h
            print(f"| {m} | {n} | {tps} | {h:.2f} | {n*h:.0f} |")
    print(f"| **Qwen3 subtotal** | | | | **{tot:.0f}** |")
    for m, r in rows.items():
        s = r.get("sft") or {}
        if m in GRID or not s.get("real_tokens_per_s"):
            continue
        h = RUN_TOKENS / s["real_tokens_per_s"] / 3600
        print(f"| 2nd family = {m} | {SECOND_FAMILY_RUNS} | {s['real_tokens_per_s']} | {h:.2f} | "
              f"{SECOND_FAMILY_RUNS*h:.0f} (total {tot + SECOND_FAMILY_RUNS*h:.0f}) |")


if __name__ == "__main__":
    main(*sys.argv[1:])
