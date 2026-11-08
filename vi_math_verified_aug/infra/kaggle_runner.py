"""Kaggle runner (W2 / T6 §1): private GPU kernels driven from the laptop via the `kaggle` CLI.

Everything goes through the official CLI (kaggle==2.2.5, needs `six`); no secrets ever leave the laptop.
The CLI binary is taken from $KAGGLE_BIN (default: `kaggle` on PATH).

Library use::

    r = KaggleRunner(owner="datuni")
    kdir = r.render_kernel("w2-pilot-eval", code_file="path/run.py", dataset_sources=["datuni/vimath-w2-pilot"])
    r.push(kdir)                       # private, GPU T4 x2, internet on
    st = r.wait("datuni/w2-pilot-eval")  # QUEUED/RUNNING -> COMPLETE|ERROR|CANCEL*
    r.output("datuni/w2-pilot-eval", "jobs/runs/w2-pilot-eval")   # outputs + <slug>.log
    r.dataset_upsert("build/data", "vimath-w2-pilot", "vimath w2 pilot")  # create private or new version

CLI::

    python -m vi_math_verified_aug.infra.kaggle_runner quota
    python -m vi_math_verified_aug.infra.kaggle_runner render SLUG CODE_FILE [--dataset owner/slug ...]
    python -m vi_math_verified_aug.infra.kaggle_runner push KDIR [--timeout S]
    python -m vi_math_verified_aug.infra.kaggle_runner status OWNER/SLUG
    python -m vi_math_verified_aug.infra.kaggle_runner wait OWNER/SLUG [--poll 60]
    python -m vi_math_verified_aug.infra.kaggle_runner output OWNER/SLUG DEST
    python -m vi_math_verified_aug.infra.kaggle_runner dataset DIR SLUG TITLE [-m MSG]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNELS_DIR = REPO / "jobs" / "kernels"
TERMINAL = ("COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED")


class KaggleError(RuntimeError):
    pass


class KaggleRunner:
    def __init__(self, owner: str = "datuni", kaggle_bin: str | None = None, log_dir: Path | None = None):
        self.owner = owner
        self.bin = kaggle_bin or os.environ.get("KAGGLE_BIN", "kaggle")
        self.log_dir = Path(log_dir or REPO / "jobs" / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ low level
    def _run(self, *args: str, check: bool = True, timeout: int = 1800) -> str:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        p = subprocess.run([self.bin, *args], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        with open(self.log_dir / "kaggle_cli.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "args": list(args), "rc": p.returncode,
                                "out": out[-4000:]}, ensure_ascii=False) + "\n")
        if check and p.returncode != 0:
            raise KaggleError(f"kaggle {' '.join(args)} -> rc={p.returncode}\n{out}")
        return out

    # ------------------------------------------------------------------ kernels
    def render_kernel(self, slug: str, code_file: str | Path, *, title: str | None = None,
                      dataset_sources: list[str] | None = None, kernel_sources: list[str] | None = None,
                      model_sources: list[str] | None = None, gpu: bool = True, internet: bool = True,
                      machine_shape: str = "NvidiaTeslaT4", extra_files: list[str | Path] | None = None,
                      out_root: Path = KERNELS_DIR) -> Path:
        """Create jobs/kernels/<slug>/ with kernel-metadata.json + the script. Always private."""
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{3,48}[a-z0-9]", slug):
            raise ValueError(f"bad slug {slug!r}: lowercase letters, digits, '-' (5-50 chars)")
        kdir = Path(out_root) / slug
        kdir.mkdir(parents=True, exist_ok=True)
        code_file = Path(code_file)
        shutil.copy2(code_file, kdir / code_file.name)
        for f in extra_files or []:
            shutil.copy2(f, kdir / Path(f).name)
        meta = {
            "id": f"{self.owner}/{slug}",
            "title": title or slug,           # title must slugify to the id
            "code_file": code_file.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,               # project rule: private only
            "enable_gpu": gpu,
            "enable_tpu": False,
            "enable_internet": internet,      # set explicitly (source default != docs default)
            "machine_shape": machine_shape if gpu else None,
            "dataset_sources": dataset_sources or [],
            "kernel_sources": kernel_sources or [],
            "model_sources": model_sources or [],
            "competition_sources": [],
        }
        meta = {k: v for k, v in meta.items() if v is not None}
        (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return kdir

    def push(self, kdir: str | Path, *, accelerator: str | None = "NvidiaTeslaT4",
             timeout_s: int | None = None) -> str:
        meta = json.loads((Path(kdir) / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta.get("is_private", True) is True, "refusing to push a public kernel"
        args = ["kernels", "push", "-p", str(kdir)]
        if accelerator and meta.get("enable_gpu"):
            args += ["--accelerator", accelerator]
        if timeout_s:
            args += ["-t", str(int(timeout_s))]
        return self._run(*args)

    def status(self, ref: str) -> str:
        out = self._run("kernels", "status", ref, check=False)
        m = re.search(r'has status "?([A-Za-z_.]+)"?', out)
        if not m:
            m = re.search(r"(QUEUED|RUNNING|COMPLETE|ERROR|CANCEL_REQUESTED|CANCEL_ACKNOWLEDGED|NEW_SCRIPT)",
                          out, re.I)
        if not m:
            return "UNKNOWN:" + out.strip()[-200:]
        s = m.group(1).upper().replace("KERNELWORKERSTATUS.", "")
        return s

    def wait(self, ref: str, *, poll_s: int = 60, max_wait_s: int = 13 * 3600, verbose: bool = True,
             max_queued_s: int | None = 3600, timeline: str | Path | None = None) -> str:
        """Poll (never faster than every 30 s, never slower than every 90 s) until a terminal status.

        Kaggle is flaky: a kernel can sit in QUEUED for a long time. We stop waiting (we do NOT cancel) after
        `max_queued_s` in QUEUED or `max_wait_s` overall and return "QUEUED_TOO_LONG:<s>" / "TIMEOUT_WAITING:<s>".
        Every status transition and a GPU-quota snapshot at start/RUNNING/end go to a JSONL timeline
        (default jobs/logs/<slug>_timeline.jsonl): this is how the T4x2 billing rate is measured.
        """
        poll_s = min(max(int(poll_s), 30), 90)
        tl = Path(timeline) if timeline else self.log_dir / f"{ref.split('/')[-1]}_timeline.jsonl"

        def mark(status, **kw):
            rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "epoch": round(time.time(), 1),
                   "ref": ref, "status": status, **kw}
            with open(tl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        t0 = time.time()
        last, t_queued0 = None, None
        mark("WAIT_START", gpu_quota_used_h=self.gpu_quota_used())
        while True:
            s = self.status(ref)
            if s != last:
                extra = {"gpu_quota_used_h": self.gpu_quota_used()} if s in ("RUNNING",) + TERMINAL else {}
                mark(s, waited_min=round((time.time() - t0) / 60, 1), **extra)
                if verbose:
                    print(f"[{time.strftime('%H:%M:%S')}] {ref}: {s} ({(time.time()-t0)/60:.1f} min)", flush=True)
                last = s
                t_queued0 = time.time() if s == "QUEUED" else None
            if s in TERMINAL:
                return s
            if max_queued_s and t_queued0 and time.time() - t_queued0 > max_queued_s:
                mark("QUEUED_TOO_LONG", queued_min=round((time.time() - t_queued0) / 60, 1))
                return "QUEUED_TOO_LONG:" + s
            if time.time() - t0 > max_wait_s:
                mark("TIMEOUT_WAITING")
                return "TIMEOUT_WAITING:" + s
            time.sleep(poll_s)

    def gpu_quota_used(self) -> float | None:
        """Hours of GPU quota used this week, parsed from `kaggle quota` (None if unparsable)."""
        try:
            m = re.search(r"^GPU\s+([\d.]+)h", self._run("quota", check=False), re.M)
            return float(m.group(1)) if m else None
        except Exception:  # noqa
            return None

    def verify_output_run_id(self, dest: str | Path, run_id: str) -> bool:
        """`kernels output` returns the latest *completed* version, which can be a stale one. Each JOB carries a
        unique run_id that the kernel writes to results/meta.json; check it after every pull."""
        meta = Path(dest) / "results" / "meta.json"
        try:
            got = json.loads(meta.read_text(encoding="utf-8")).get("job", {}).get("run_id")
        except Exception:  # noqa
            got = None
        return got == run_id

    def output(self, ref: str, dest: str | Path, *, pattern: str | None = None) -> Path:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        args = ["kernels", "output", ref, "-p", str(dest), "-o"]
        if pattern:
            args += ["--file-pattern", pattern]
        self._run(*args, timeout=3600)
        return dest

    def logs(self, ref: str) -> str:
        return self._run("kernels", "logs", ref, check=False)

    # ------------------------------------------------------------------ datasets
    def dataset_upsert(self, folder: str | Path, slug: str, title: str, *, message: str = "update",
                       dir_mode: str = "zip") -> str:
        """Create a PRIVATE dataset from `folder`, or add a new version if it already exists."""
        folder = Path(folder)
        meta = {"title": title, "id": f"{self.owner}/{slug}", "licenses": [{"name": "other"}]}
        (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        exists = "ready" in self._run("datasets", "status", f"{self.owner}/{slug}", check=False).lower()
        if exists:
            return self._run("datasets", "version", "-p", str(folder), "-m", message, "-r", dir_mode, timeout=3600)
        return self._run("datasets", "create", "-p", str(folder), "-r", dir_mode, timeout=3600)  # private default

    def dataset_ready(self, ref: str, *, poll_s: int = 15, max_wait_s: int = 900) -> bool:
        t0 = time.time()
        while time.time() - t0 < max_wait_s:
            if "ready" in self._run("datasets", "status", ref, check=False).lower():
                return True
            time.sleep(poll_s)
        return False

    def quota(self) -> str:
        return self._run("quota")


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="kaggle_runner")
    ap.add_argument("--owner", default=os.environ.get("KAGGLE_OWNER", "datuni"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("quota")
    p = sub.add_parser("render"); p.add_argument("slug"); p.add_argument("code_file")
    p.add_argument("--dataset", action="append", default=[]); p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--no-internet", action="store_true")
    p = sub.add_parser("push"); p.add_argument("kdir"); p.add_argument("--timeout", type=int)
    p = sub.add_parser("status"); p.add_argument("ref")
    p = sub.add_parser("wait"); p.add_argument("ref"); p.add_argument("--poll", type=int, default=60)
    p.add_argument("--max-queued", type=int, default=3600); p.add_argument("--max-wait", type=int, default=13 * 3600)
    p = sub.add_parser("output"); p.add_argument("ref"); p.add_argument("dest")
    p = sub.add_parser("dataset"); p.add_argument("dir"); p.add_argument("slug"); p.add_argument("title")
    p.add_argument("-m", "--message", default="update")
    a = ap.parse_args(argv)
    r = KaggleRunner(owner=a.owner)
    if a.cmd == "quota":
        print(r.quota())
    elif a.cmd == "render":
        print(r.render_kernel(a.slug, a.code_file, dataset_sources=a.dataset, gpu=not a.no_gpu,
                              internet=not a.no_internet))
    elif a.cmd == "push":
        print(r.push(a.kdir, timeout_s=a.timeout))
    elif a.cmd == "status":
        print(r.status(a.ref))
    elif a.cmd == "wait":
        print(r.wait(a.ref, poll_s=a.poll, max_queued_s=a.max_queued, max_wait_s=a.max_wait))
    elif a.cmd == "output":
        print(r.output(a.ref, a.dest))
    elif a.cmd == "dataset":
        print(r.dataset_upsert(a.dir, a.slug, a.title, message=a.message))


if __name__ == "__main__":
    sys.exit(_main())
