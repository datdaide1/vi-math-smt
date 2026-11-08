"""Summarize W2 pilot outputs + re-estimate grid GPU-hours from MEASURED T4 throughput.

Usage: python -m vi_math_verified_aug.infra.pilot.summarize_pilot RESULTS_DIR [FERTILITY_JSON] > summary.md
"""
import json
import sys
from pathlib import Path

MODELS = ["Qwen3.5-0.8B-Base", "Qwen3-0.6B-Base", "Qwen3.5-2B-Base", "Qwen3-1.7B-Base", "gemma-4-E2B"]


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def main(res_dir, fert=None):
    res_dir = Path(res_dir)
    rows = {}
    print("| model | engine | 4-shot acc (300) | 95% CI | boxed | gen tok/s | SFT tok/s off | SFT tok/s on | "
          "s/step off | peak GB off | pad frac off | loss first5→last5 (off) | loss last5 (on) | skipped off/on | "
          "act absmax | packcheck max diff (items 2-4) |")
    print("|" + "---|" * 16)
    for m in MODELS:
        d = res_dir / m
        ev, off, on, pc = (load(d / f) for f in ("eval.json", "sft_off.json", "sft_on.json", "packcheck.json"))
        rows[m] = {"eval": ev, "off": off, "on": on, "pc": pc}
        f = lambda x, k, default="ERR": (x or {}).get(k, default)  # noqa
        print(f"| {m} | {f(ev,'engine')} | {f(ev,'accuracy')} | {f(ev,'ci95_wilson')} | {f(ev,'boxed_rate')} | "
              f"{f(ev,'gen_tok_per_s')} | {f(off,'real_tokens_per_s_steps3plus')} | {f(on,'real_tokens_per_s_steps3plus')} | "
              f"{f(off,'s_per_step_steps3plus')} | {f(off,'peak_mem_gb')} | {f(off,'pad_fraction')} | "
              f"{f(off,'loss_first5_mean')}→{f(off,'loss_last5_mean')} | {f(on,'loss_last5_mean')} | "
              f"{f(off,'skipped_steps')}/{f(on,'skipped_steps')} | {f(off,'act_absmax_max')} | {f(pc,'max_diff_items_2_4')} |")
        for name, x in (("eval", ev), ("sft_off", off), ("sft_on", on), ("packcheck", pc)):
            if x is None:
                e = load(d / f"{name}.error.json")
                if e:
                    print(f"<!-- {m} {name} ERROR: {e['error'][-600:]!r} -->")
    # ---------------- grid re-estimate
    fe = load(fert) if fert else None
    tok_item = {}
    if fe:
        for k, v in fe.items():
            if isinstance(v, dict) and "train_item_tokens_mean" in v:
                tok_item[k.split("/")[-1]] = v["train_item_tokens_mean"]
    print("\n## Grid re-estimate (T4-h), 10k items x 2 epochs per run\n")
    print("| base | runs | measured SFT tok/s (best mode) | tok/item measured | T4-h/run @measured len | "
          "T4-h/run @450 tok | block T4-h @measured | block T4-h @450 |")
    print("|---|---|---|---|---|---|---|---|")
    grid = [("Qwen3.5-0.8B-Base", 30), ("Qwen3.5-2B-Base", 92), ("gemma-4-E2B", 78),
            ("Qwen3-0.6B-Base", 30), ("Qwen3-1.7B-Base", 92)]
    tot = {"m": 0, "450": 0}
    for m, runs in grid:
        r = rows.get(m, {})
        cands = [x.get("real_tokens_per_s_steps3plus") for x in (r.get("off"), r.get("on")) if x]
        cands = [c for c in cands if c]
        if not cands:
            print(f"| {m} | {runs} | n/a | | | | | |")
            continue
        tps = cands[0] if len(cands) == 1 else min(cands) if m.startswith("Qwen3.5") else max(cands)
        ti = tok_item.get(m, 190)
        h_m = 10000 * 2 * ti / tps / 3600
        h_450 = 10000 * 2 * 450 / tps / 3600
        print(f"| {m} | {runs} | {tps} | {ti} | {h_m:.2f} | {h_450:.2f} | {h_m*runs:.0f} | {h_450*runs:.0f} |")
    return rows


if __name__ == "__main__":
    main(*sys.argv[1:3])
