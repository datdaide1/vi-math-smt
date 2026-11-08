"""Command line for one run.  Exit codes: 0 done, 3 error, 4 fp16-overflow abort, 75 needs resume.

    python -m vi_math_verified_aug.train.cli check  --config RUN.yaml           # validate + print run_id
    python -m vi_math_verified_aug.train.cli sft    --config RUN.yaml [--budget-s N] [--out DIR]
    python -m vi_math_verified_aug.train.cli grpo   --config RUN.yaml [--budget-s N] [--out DIR]
    python -m vi_math_verified_aug.train.cli eval   --config RUN.yaml [--budget-s N] [--out DIR]
    python -m vi_math_verified_aug.train.cli merge  --base BASE --adapter DIR --out DIR [--precision fp16]

``--set key.sub=value`` (repeatable, YAML-parsed value) overrides a config field, e.g. ``--set budget.max_steps=5``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

from . import EXIT_ERROR
from . import config as C


def _overrides(pairs: list[str]) -> dict:
    import yaml
    out: dict = {}
    for p in pairs:
        k, _, v = p.partition("=")
        cur = out
        parts = k.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = yaml.safe_load(v)
    return out


def main(argv=None) -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser(prog="vi_math_verified_aug.train.cli")
    ap.add_argument("cmd", choices=["check", "sft", "sft-cell", "grpo", "eval", "merge"])
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--results")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--budget-s", type=float)
    ap.add_argument("--out")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--base")
    ap.add_argument("--adapter")
    ap.add_argument("--precision", default="fp16")
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        from .modeling import merge_adapter
        print(merge_adapter(a.base, a.adapter, a.out, a.precision))
        return 0
    if a.cmd == "sft-cell":   # every arm of one cell in this process (sft.run_cell)
        cfgs = [C.load(p) for p in a.configs]
        if any(c["engine"] == "unsloth" for c in cfgs):
            import os
            os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")
            import unsloth  # noqa: F401
        from .sft import run_cell
        rcs = run_cell(cfgs, stop_on_fail=a.stop_on_fail)
        if a.results:
            from pathlib import Path
            Path(a.results).write_text(json.dumps({c["run_id"]: rc for c, rc in zip(cfgs, rcs)}), encoding="utf-8")
        print("CELL", json.dumps({c["run_id"]: rc for c, rc in zip(cfgs, rcs)}), flush=True)
        return max(rcs) if rcs else 0
    ov = _overrides(a.set)
    if a.budget_s:
        ov.setdefault("budget", {})["wall_s"] = a.budget_s
    if a.out:
        ov["out_dir"] = a.out
    cfg = C.load(a.config, ov or None)
    if a.cmd == "check":
        print(json.dumps({**C.identity(cfg), "out_dir": cfg["out_dir"]}, ensure_ascii=False))
        return 0
    if cfg["kind"] != a.cmd:
        print(f"config kind {cfg['kind']!r} != command {a.cmd!r}", file=sys.stderr)
        return EXIT_ERROR
    if cfg.get("engine") == "unsloth" and a.cmd == "sft":
        import os
        os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")
        import unsloth  # noqa: F401  (Unsloth must be imported before transformers / peft / trl)
    try:
        if a.cmd == "sft":
            from .sft import run
        elif a.cmd == "grpo":
            from .grpo import run
        else:
            from ..evalrun.runner import run
        return run(cfg, t0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
