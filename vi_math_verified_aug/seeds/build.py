"""S3 seed-pool CLI: ``python -m vi_math_verified_aug.seeds.build <stage> [options]`` (see the package docstring)."""
from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="seeds.build")
    ap.add_argument("stage", choices=["align", "detect", "solvecheck", "mta", "gold", "decontam", "embed-inputs",
                                      "translationese", "freeze", "spend"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="seed ids or source#index")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--version", default=None, help="freeze: new version tag (v1 is frozen, write-once)")
    ap.add_argument("--deadline", default=None, help="local HH:MM; solvecheck stops starting new seeds after it")
    a = ap.parse_args(argv)
    if a.stage == "align":
        from . import align
        align.run()
    elif a.stage == "detect":
        from . import detect
        detect.run()
    elif a.stage == "solvecheck":
        from . import solvecheck
        dl = None
        if a.deadline:
            import datetime as _dt
            h, m = map(int, a.deadline.split(":"))
            now = _dt.datetime.now()
            t = now.replace(hour=h, minute=m, second=0, microsecond=0)
            dl = (t if t > now else t + _dt.timedelta(days=1)).timestamp()
        solvecheck.run(limit=a.limit, concurrency=a.concurrency or 24, deadline=dl)
    elif a.stage == "mta":
        from . import mta
        mta.run(limit=a.limit, only=a.only)
    elif a.stage == "gold":
        from . import gold
        gold.run()
    elif a.stage == "embed-inputs":
        from . import decontam
        decontam.write_embed_inputs()
    elif a.stage == "decontam":
        from . import decontam
        decontam.run()
    elif a.stage == "translationese":
        from . import translationese
        translationese.run()
    elif a.stage == "freeze":
        from . import freeze
        freeze.run(a.version)
    elif a.stage == "spend":
        from .common import client
        print(json.dumps(client().costs.summary(), indent=1))


if __name__ == "__main__":
    main()
