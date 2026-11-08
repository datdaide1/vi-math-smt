"""``python -m vi_math_verified_aug.evalrun.cli --config EVAL.yaml [--budget-s N] [--out DIR] [--set k=v]``

``python -m vi_math_verified_aug.evalrun.cli rescore PREDS.jsonl --config EVAL.yaml --out RESCORED.jsonl``
"""
from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "rescore":
        import argparse
        from ..train import config as C
        from .runner import rescore
        ap = argparse.ArgumentParser()
        ap.add_argument("preds")
        ap.add_argument("--config", required=True)
        ap.add_argument("--out", required=True)
        a = ap.parse_args(argv[1:])
        cfg = C.load(a.config)
        print("flips:", rescore(a.preds, cfg["eval"]["sets"], a.out))
        return 0
    from ..train.cli import main as train_main
    return train_main(["eval"] + argv)


if __name__ == "__main__":
    sys.exit(main())
