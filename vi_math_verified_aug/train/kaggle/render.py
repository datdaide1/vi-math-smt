"""Laptop side: fill the Kaggle job template with run configs.  Never pushes (S2 owns Kaggle dispatch).

    python -m vi_math_verified_aug.train.kaggle.render SLUG --lane a.yaml,b.yaml --lane c.yaml \
        [--budget-s 20000] [--out-root jobs/build] [--bundle-code DIR]

* Every config is validated locally with ``train.config.build`` and its ``run_id`` is pinned into the job, so the
  kernel and the laptop agree on ids.  Per-task ``budget_s`` defaults to the config's ``budget.wall_s``.
* ``pins`` come from ``train/requirements-kaggle.txt`` (ENVCHECK fails fast on a mismatch).
* Output: ``<out-root>/<slug>/run.py`` (= ``JOB = {...}`` header + job_template.py) and ``job.json``.  Pushing is
  ``infra.kaggle_runner.KaggleRunner.render_kernel(slug, run.py, dataset_sources=[code, data, wheels, ckpt])`` +
  ``push`` -- S2's interface, used unchanged.
* ``--bundle-code DIR`` copies the package (no __pycache__) into DIR/vi_math_verified_aug for the code dataset.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

from .. import config as C

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
REQ = HERE.parent / "requirements-kaggle.txt"
DEFAULT_ENV = {"expect_gpu": "T4", "fail_fast": True, "deadline_s": 41400, "grace_s": 300,
               "env_fix": ["pip uninstall -y -q torchao"], "install": "pip install -q 'vllm==0.33.0'",
               "wheels_dir_glob": None, "unsloth_install": None}


def read_pins(path: Path = REQ) -> dict:
    """``pkg==version`` lines; comments, TODO lines and ``torch`` (preinstalled CUDA build) are not enforced
    as pip pins but torch is still checked by ENVCHECK."""
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.+\-]+)", line)
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def unsloth_install(path: Path = REQ, target: str = "/tmp/uns_site") -> str | None:
    """``pip install --no-deps --target`` command from the ``# uns: pkg==ver`` lines (isolated Unsloth stack)."""
    pk = re.findall(r"^#\s*uns:\s*([A-Za-z0-9_.\-]+==[A-Za-z0-9_.+\-]+)", Path(path).read_text(encoding="utf-8"), re.M)
    return f"pip install -q --no-deps --target {target} " + " ".join(pk) if pk else None


def task_from_config(path: str | Path, budget_s: float | None = None, requires: str | None = None) -> dict:
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    built = C.build(raw)  # validate now, on the laptop
    raw = dict(raw, run_id=built["run_id"])
    t = {"cmd": built["kind"], "config": raw, "budget_s": budget_s or built["budget"]["wall_s"] or 3600}
    if requires:
        t["requires"] = requires
    return t


def build_job(lanes: list[list[dict]], slug: str, env: dict | None = None, pins: dict | None = None) -> dict:
    job = {**DEFAULT_ENV, **(env or {})}
    job["pins"] = read_pins() if pins is None else pins
    if job.get("unsloth_install") is None and any((t.get("config") or {}).get("engine") == "unsloth"
                                                  for lane in lanes for t in lane):
        job["unsloth_install"] = unsloth_install()
    job["run_id"] = f"{slug}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    job["lanes"] = lanes
    if len(lanes) > 2:
        raise ValueError("Kaggle T4x2 has 2 GPUs -> at most 2 lanes")
    return job


def render(job: dict, slug: str, out_root: str | Path) -> Path:
    out = Path(out_root) / slug
    out.mkdir(parents=True, exist_ok=True)
    header = "import json as _json\nJOB = _json.loads(" + repr(json.dumps(job, ensure_ascii=False)) + ")\n"
    (out / "run.py").write_text(header + (HERE / "job_template.py").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "job.json").write_text(json.dumps(job, indent=1, ensure_ascii=False), encoding="utf-8")
    return out / "run.py"


def bundle_code(dest: str | Path) -> Path:
    dest = Path(dest) / "vi_math_verified_aug"
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(PKG, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dest


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="render")
    ap.add_argument("slug")
    ap.add_argument("--lane", action="append", default=[], help="comma-separated config paths, run in order")
    ap.add_argument("--budget-s", type=float)
    ap.add_argument("--out-root", default="jobs/build")
    ap.add_argument("--bundle-code")
    a = ap.parse_args(argv)
    lanes = [[task_from_config(p, a.budget_s) for p in lane.split(",") if p] for lane in a.lane]
    print(render(build_job(lanes, a.slug), a.slug, a.out_root))
    if a.bundle_code:
        print(bundle_code(a.bundle_code))


if __name__ == "__main__":
    _main()
