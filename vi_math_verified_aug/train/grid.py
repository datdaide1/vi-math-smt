"""The whole remaining SFT/GRPO grid as an ordered, costed queue (T2, 2026-11-08).

    python -m vi_math_verified_aug.train.grid [--out jobs/queue/t2_queue.jsonl]

Blocks (design: T5 D3 = 8 arms x 4 bases x 3 seeds + 2 extra seeds for the H1-H3 arms at the ~2B size;
E1 O3 = EN mirror Y+-/N+- x {0.6B, 1.7B} x 5 seeds + S0-EN anchor x 3 + recipe check x 2 + VI top-ups):

  P1  0.6B VI  S0 + Y+ Y- N+ N-, seed 0                      (tomorrow, tier 1)
  P2  0.6B VI  same arms, seeds 1-2                           (tier 2; S0 s1/s2 = T1r runs)
  P3  0.6B EN  S0-EN + Y+ Y- N+ N-, seed 0                    (tier 3)
  P4  GRPO 0.6B smoke: math_verify reward + random-reward control, from the S0 seed-0 SFT model (tier 4)
  P5  1.7B VI  S0 + Y+ Y- N+ N-, seed 0                       (tier 5, if quota is left)
  P6+ everything else of T5 D3 and E1 O3 (successor's queue), in E1/T5 priority order.

Every row carries its data path(s), engine/precision/platform, and the measured cost estimate (T4-h on Kaggle,
quota-h with 2 runs per T4x2 kernel, Modal $ for evaluation).  Rows whose data file does not exist yet stay
``pending_data``; the launcher (``train/launcher.py``) picks them up when the file lands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = {"0.6B": "Qwen/Qwen3-0.6B-Base", "1.7B": "Qwen/Qwen3-1.7B-Base", "4B": "Qwen/Qwen3-4B-Base",
          "2.6B": "LiquidAI/LFM2.5-2.6B-Base"}
ARMS_VI = ["S0", "T", "T-off", "M", "Y+", "Y-", "N+", "N-"]
ARMS_PM = ["Y+", "Y-", "N+", "N-"]

# ---- measured costs (T1r_first_runs.md §3; S2 §1/§6; O1 §1).  tok/s = real (non-pad) training tokens/s.
TOKS_PER_ITEM = 195          # mean prompt+completion Qwen3 tokens per S0 item (T1r smoke: 6,200 real tok per 32-seq step)
TOKS = {  # (platform, engine, size) -> tok/s
    ("kaggle_t4", "unsloth", "0.6B"): 4226, ("kaggle_t4", "hf", "0.6B"): 2630,   # T1r smoke (Modal T4 fp16)
    ("kaggle_t4", "unsloth", "1.7B"): 1865, ("kaggle_t4", "hf", "1.7B"): 1358,   # O1 [M], S2 [M]
    ("kaggle_t4", "hf", "4B"): 426, ("modal_l40s", "hf", "4B"): 3969,            # S2 [M]
    ("kaggle_t4", "hf", "2.6B"): 692,                                            # S2 [M] (LFM2.5, HF loop)
    ("modal_l40s", "unsloth", "0.6B"): 36300,   # R-GPU [M]: bf16, Unsloth packed, micro-batch 32
    ("modal_l40s", "unsloth", "1.7B"): 14000,   # [EST] = 0.6B rate x (0.6/1.7 FLOPs) x 1.15; re-measure in P5's canary
}
L40S_USD_H = 1.95
L40S_RUN_OVERHEAD_H = 0.06   # per run inside a cell container: model load + pack gate + probe + 4 dev/quick-eval checks
L40S_CONTAINER_H = 0.05      # per container: image start + volume reads
SETUP_H = 0.05               # model download + probe + dev loss per run (T1r)
KERNEL_SETUP_H = 0.15        # vLLM install 7.6 min + Unsloth install + ENVCHECK (S2 §7, T1r)
EVAL_USD_06B = 0.10          # Modal A10/L4: primary_1000 x (K=4 + greedy), cap 1024 (T1r §4)
CONTAINER_USD_S = 0.00056    # [M] L40S container incl. CPU/RAM (modal billing report per app)
# eval $ per adapter (L40S, batch-invariant, cap 1,024) + 1/5 of a $0.10 engine load:
#   VI = primary_1000 K4+g $0.071 [M] + Vi-ExamMath VERIFIED greedy-only $0.076 (probe: greedy = 21.3 % of the
#        K4+g tokens x $0.357) + 0.02 = 0.167
#   EN = en_primary_1000 K4+g $0.06 [EST] + 0.02 = 0.08
EVAL_USD = {"0.6B": 0.167, "1.7B": 0.30, "4B": 0.60, "2.6B": 0.45}
EVAL_USD_LANG = {"vi": 1.0, "en": 0.08 / 0.167}


def data_path(arm: str, lang: str) -> str:
    if arm == "S0":
        return "data/t1r/s0_train.jsonl" if lang == "vi" else "data/t1r/s0_train_en.jsonl"   # 7,157 rows
    if arm in ARMS_PM:  # D5's selector output (4,000 rows each, SHA256SUMS + manifest); never the provisional pools
        return f"data/train/final/{arm[0]}/{lang}/train_{'plus' if arm[1] == '+' else 'minus'}.jsonl"
    return f"data/train/{arm}/{lang}/train_10k.jsonl"


N_TRAIN = 1700      # planning value (lead 2026-11-08: D5 projects n ~1,600-1,800); the run uses D5's manifest n_effective


def cost(size: str, n_items: int = N_TRAIN, epochs: int = 3) -> dict:
    """Comparison blocks (0.6B, 1.7B; VI and EN) on Modal L40S bf16 (lead, after R-GPU); 4B on L40S HF; LFM2.5 on
    Kaggle T4 (HF loop, the only measured path).  +5 % compute for the epoch-1 branch hedge."""
    if size in ("0.6B", "1.7B"):
        platform, engine = "modal_l40s", "unsloth"
    elif size == "4B":
        platform, engine = "modal_l40s", "hf"
    else:
        platform, engine = "kaggle_t4", "hf"
    tps = TOKS[(platform, engine, size)]
    work_h = n_items * (epochs + 0.2) * TOKS_PER_ITEM / tps / 3600   # + epoch-1/-2 decay branches (hedges)
    if platform == "modal_l40s" and engine == "unsloth":
        # [M] per-run seconds inside a 5-arm cell process (T2 §5.2/5.3): setup 95 s for the first arm, 22 s for each
        # further arm (avg 36.6 s), training 55 s per 3,000 x 3 epochs at 0.6B, mid-run checks 30 s, saves 10 s;
        # container $0.00056/s (billing).  1.7B: training scaled by the tok/s ratio [EST].
        train_s = 55.0 * (n_items / 3000) * (TOKS[(platform, engine, "0.6B")] / tps) * (epochs / 3)
        sec = 36.6 + train_s + 40.0
        out = {"platform": platform, "engine": engine, "tok_per_s": tps, "train_gpu_h": round(sec / 3600, 4),
               "train_usd": round(sec * CONTAINER_USD_S, 4), "eval_usd": EVAL_USD[size]}
    elif platform == "modal_l40s":
        h = work_h + L40S_RUN_OVERHEAD_H
        out = {"platform": platform, "engine": engine, "tok_per_s": tps, "train_gpu_h": round(h, 3),
               "train_usd": round(h * L40S_USD_H, 3), "eval_usd": EVAL_USD[size]}
    else:
        h = work_h + SETUP_H
        out = {"platform": platform, "engine": engine, "tok_per_s": tps, "train_gpu_h": round(h, 3),
               "eval_usd": EVAL_USD[size], "quota_h_2_per_kernel": round(h / 2 + KERNEL_SETUP_H / 2, 3)}
    return out


# G6 (2026-11-08, exec/T2_training_tranche.md §2): LR 1e-4, 3 epochs (fresh main run), chosen on dev_v1 composite
# (pass@1 + greedy)/2 over 3 seeds: 38.7 [35.7, 41.8] vs 4e-4/ep3 -1.9 [-3.5, -0.3], 1e-4/ep2 -0.5 [-1.9, +0.9].
LR, EPOCHS = 1.0e-4, 3


def row(prio, block, arm, lang, size, seed, *, kind="sft", lr=LR, extra=None):
    n_items = N_TRAIN   # every arm AND S0 (S0: seeded subsample of the 7,157 rows)
    r = {"run_id": f"t2-{kind}-{arm.replace('+', 'p').replace('-', 'm')}-{lang}-{size.replace('.', 'p')}-s{seed}"
                   + (f"-lr{lr:g}" if lr != LR else ""),
         "priority": prio, "block": block, "kind": kind, "arm": arm, "lang": lang, "size": size,
         "model": MODELS[size], "seed": seed, "lr": lr, "epochs": EPOCHS, "n_train": n_items,
         "data": data_path(arm, lang), **cost(size, n_items), "status": "queued"}
    r["eval_usd"] = round(r["eval_usd"] * EVAL_USD_LANG[lang], 4)
    r.update(extra or {})
    if kind == "grpo":  # fp16 everywhere (T3 §4.1, lead); Modal L4 (O1 §3.3: ~5.5 L4-h ~ $4.4 per 1.7B run [EST])
        r["run_id"] += "-" + ("rand" if r.get("reward") == "random" else "mv")
        r.update(platform="modal_l4_fp16", engine="hf", train_usd=round(r["train_gpu_h"] * 0.80, 2))
        r.pop("quota_h_2_per_kernel", None)
    return r


def build() -> list[dict]:
    rows, seen = [], set()

    def add(r):
        if r["run_id"] not in seen:
            seen.add(r["run_id"])
            rows.append(r)
    # lead 2026-11-08 (Modal $6.5 for ALL training + eval): VI 0.6B seeds 0-3 (primary K4+g, ExamMath greedy), EN 0.6B
    # seeds 0-1 (primary only).  Launch order: VI s0 -> EN s0 -> VI s1-s3 -> EN s1.  One cell = S0 + Y+ Y- N+ N-.
    order = [("vi", 0, 1, "P1 0.6B VI seed 0"), ("en", 0, 2, "P2 0.6B EN seed 0")] +             [("vi", s, 3, "P3 0.6B VI seeds 1-3") for s in (1, 2, 3)]
    for lang, sd, prio, blk in order:
        for a in ["S0"] + ARMS_PM:
            add(row(prio, blk, a, lang, "0.6B", sd))
    # lead 2026-11-08 08:19: EN s1 outranks VI s3, but VI s3 was already training -> EN s1 is the FIRST successor item
    # (refused anyway by the $29.5 workspace guard at 08:14); cost = the row estimates (cell ~$0.55 measured, $0.70 est)
    for a in ["S0"] + ARMS_PM:
        add(row(11, "P11 HOLD 0.6B EN seed 1 (first successor; H1 needs >= 2 EN seeds)", a, "en", "0.6B", 1,
                extra={"status": "hold"}))
    # HOLD (successor; costs in the queue rows): VI seed 4, EN seeds 2-4, the S0-EN LR x0.5/x2 recipe check
    for a in ["S0"] + ARMS_PM:
        add(row(12, "P12 HOLD 0.6B VI seed 4", a, "vi", "0.6B", 4, extra={"status": "hold"}))
    for sd in (2, 3, 4):
        for a in ["S0"] + ARMS_PM:
            add(row(12, "P12 HOLD 0.6B EN seeds 2-4", a, "en", "0.6B", sd, extra={"status": "hold"}))
    for lr in (LR / 2, LR * 2):
        add(row(12, "P12 HOLD 0.6B EN recipe check (S0-EN, LR x0.5 / x2)", "S0", "en", "0.6B", 0, lr=lr,
                extra={"status": "hold"}))
    for rw in ("math_verify", "random"):                                     # P4 (DONE 2026-11-08, T2 §GRPO)
        add(row(4, "P4 GRPO 0.6B smoke", "S0", "vi", "0.6B", 0, kind="grpo",
                extra={"reward": rw, "max_steps": 10, "sft_from": "t1r-S0-uns-0p6B-s1",
                       "data": "data/t1r/s0_train.jsonl", "train_gpu_h": 0.1, "status": "done",
                       "note": "smoke done on Modal T4 fp16 (t2-grpo-smoke-mv / -rand)"}))
    for a in ["S0"] + ARMS_PM:                                               # P5
        add(row(5, "P5 1.7B VI seed 0", a, "vi", "1.7B", 0))
    # ---- P6+: the rest of T5 D3 and E1 O3 (successor), most valuable first
    for s in (1, 2):                                                         # 1.7B VI core seeds 1-2
        for a in ["S0"] + ARMS_PM:
            add(row(6, "P6 1.7B VI seeds 1-2 (Y+-N+-, S0)", a, "vi", "1.7B", s))
    for size in ("0.6B", "1.7B"):                                            # E1 O3 EN mirror, 5 seeds
        for s in range(5):
            for a in ARMS_PM:
                add(row(7, "P7 E1-O3 EN mirror Y+-N+- x5 seeds", a, "en", size, s))
    for size in ("0.6B", "1.7B"):                                            # E1 S0-EN anchor x3
        for s in range(3):
            add(row(7, "P7 E1 S0-EN anchor", "S0", "en", size, s))
    for lr in (LR / 2, LR * 2):                                              # E1 recipe-transfer check
        add(row(8, "P8 E1 EN recipe check (LR x0.5, x2)", "S0", "en", "1.7B", 0, lr=lr))
    for s in (3, 4):                                                         # E1 VI top-ups
        for a in ARMS_PM:
            add(row(8, "P8 E1 VI top-up 0.6B seeds 3-4", a, "vi", "0.6B", s))
        for a in ARMS_PM:                                                    # T5 D3: 5 seeds at ~2B for H1-H3
            add(row(8, "P8 VI 1.7B seeds 3-4 (E1 + T5 D3)", a, "vi", "1.7B", s))
    for s in range(3):                                                       # T5 D3 remaining arms
        for size in ("0.6B", "1.7B"):
            for a in ("T", "T-off", "M"):
                add(row(9, "P9 T5-D3 T/T-off/M arms", a, "vi", size, s))
    for s in (3, 4):
        for a in ("T", "M", "S0"):
            add(row(9, "P9 T5-D3 extra 2B seeds (H1-H3 arms)", a, "vi", "1.7B", s))
    for size in ("4B", "2.6B"):                                              # T5 D3 other bases
        for s in range(3):
            for a in ARMS_VI:
                add(row(10, f"P10 T5-D3 {size} all arms x3", a, "vi", size, s))
    for s in range(3):                                                       # T3 RQ-F GRPO (successor)
        for src in ("S0", "Y+", "N+"):
            add(row(11, "P11 GRPO 1.7B from SFT ckpts", src, "vi", "1.7B", s, kind="grpo",
                    extra={"reward": "math_verify", "train_gpu_h": 5.5, "note": "O1 §3.3 [EST]"}))
        add(row(11, "P11 GRPO 1.7B random-reward control", "S0", "vi", "1.7B", s, kind="grpo",
                extra={"reward": "random", "train_gpu_h": 5.5}))
    return rows


def totals(rows) -> dict:
    by = {}
    for r in rows:
        b = by.setdefault(r["block"], {"runs": 0, "train_gpu_h": 0.0, "eval_usd": 0.0, "quota_h": 0.0})
        b["runs"] += 1
        b["train_gpu_h"] = round(b["train_gpu_h"] + r["train_gpu_h"], 2)
        b["eval_usd"] = round(b["eval_usd"] + r["eval_usd"], 2)
        b["quota_h"] = round(b["quota_h"] + r.get("quota_h_2_per_kernel", 0.0), 2)
        b["train_usd"] = round(b.get("train_usd", 0.0) + r.get("train_usd", 0.0), 2)
    return by


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "jobs" / "queue" / "t2_queue.jsonl"))
    a = ap.parse_args(argv)
    rows = build()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    t = totals(rows)
    (out.with_suffix(".totals.json")).write_text(json.dumps(t, indent=1), encoding="utf-8")
    print(out, len(rows))
    for k, v in t.items():
        print(f"{k:45s} {v}")


if __name__ == "__main__":
    main()
