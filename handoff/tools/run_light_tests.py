"""Run the light (laptop) test suite, one pytest process per test file, under a hard per-process memory cap.

Windows: each pytest process (and any child it spawns) is placed in a Job Object with
JOB_OBJECT_LIMIT_PROCESS_MEMORY, so an allocation past the cap fails (MemoryError) instead of eating the laptop.
The job's PeakProcessMemoryUsed (committed bytes, stricter than RSS) is reported per file.
Elsewhere: falls back to resource.RLIMIT_AS in the child.

Usage (repo root):
    .venv/Scripts/python.exe handoff/tools/run_light_tests.py [--cap-mb 300] [--out handoff/test_report.md]

Heavy tests are excluded by name (HEAVY below); they need torch/JAX or simulation-scale data and run on Modal.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEAVY = {"test_analysis_heavy.py"}  # simulation-scale; runs on Modal (VIMATH_HEAVY=1)


def _win_job(cap_bytes: int):
    import ctypes
    import ctypes.wintypes as wt

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

    class EXT(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32.CreateJobObjectW.restype = wt.HANDLE
    job = k32.CreateJobObjectW(None, None)
    info = EXT()
    info.Basic.LimitFlags = 0x100 | 0x2000  # PROCESS_MEMORY | KILL_ON_JOB_CLOSE
    info.ProcessMemoryLimit = cap_bytes
    if not k32.SetInformationJobObject(wt.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

    def assign(proc: subprocess.Popen) -> None:
        if not k32.AssignProcessToJobObject(wt.HANDLE(job), wt.HANDLE(int(proc._handle))):  # noqa: SLF001
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def peak() -> int:
        q = EXT()
        k32.QueryInformationJobObject(wt.HANDLE(job), 9, ctypes.byref(q), ctypes.sizeof(q), None)
        return int(q.PeakProcessMemoryUsed)

    def close() -> None:
        k32.CloseHandle(wt.HANDLE(job))

    return assign, peak, close


def run_file(py: str, test: Path, cap_mb: int, timeout: int) -> dict:
    cmd = [py, "-m", "pytest", str(test.relative_to(ROOT)), "-o", "addopts=", "-q", "-p", "no:cacheprovider", "--no-header", "-rfE"]
    # One BLAS thread: OpenBLAS commits ~250 MB of per-thread buffers at `import numpy` on this 8-thread laptop,
    # which alone breaks a 300 MB commit cap (measured 2026-11-07: test_analysis/test_decontam 290 MB -> 95/43 MB).
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    t0 = time.time()
    if os.name == "nt":
        assign, peak, close = _win_job(cap_mb * 2**20)
        # Start suspended is not exposed by subprocess; assigning right after spawn is enough here because the
        # interpreter needs far less than the cap before it imports anything.
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assign(proc)
    else:
        import resource

        def lim():
            resource.setrlimit(resource.RLIMIT_AS, (cap_mb * 2**20, cap_mb * 2**20))
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                preexec_fn=lim)
        peak = close = None
    try:
        out, _ = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        rc = "timeout"
    peak_mb = round(peak() / 2**20, 1) if peak else None
    if close:
        close()
    text = out.decode("utf-8", "replace")
    tail = [ln for ln in text.strip().splitlines() if ln.strip()][-1:] or [""]
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed|deselected)",
                                                tail[0])}
    return {"file": test.name, "rc": rc, "secs": round(time.time() - t0, 1), "peak_mb": peak_mb,
            "summary": tail[0].strip(" ="), "counts": counts, "output": text}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-mb", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--python", default=str(ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")))
    ap.add_argument("--out", default=str(ROOT / "handoff" / "test_report.md"))
    ap.add_argument("--alt", action="append", default=[], metavar="PYTHON=file1,file2",
                    help="run these test files with another interpreter (e.g. one that has pyyaml)")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    if a.files:
        tests = [t for t in tests if t.name in set(a.files)]
    alt = {}
    for spec in a.alt:
        py, _, names = spec.rpartition("=")
        alt.update({n: py for n in names.split(",") if n})
    rows, tot = [], {}
    for t in tests:
        if t.name in HEAVY:
            rows.append({"file": t.name, "rc": "excluded", "secs": 0, "peak_mb": None,
                         "summary": "excluded (heavy: Modal only)", "counts": {}, "output": ""})
            continue
        r = run_file(alt.get(t.name, a.python), t, a.cap_mb, a.timeout)
        r["python"] = "alt: " + alt[t.name] if t.name in alt else ".venv"
        rows.append(r)
        for k, v in r["counts"].items():
            tot[k] = tot.get(k, 0) + v
        print(f"{r['file']:<36} rc={r['rc']!s:<8} peak={r['peak_mb']} MB  {r['secs']:>6}s  {r['summary']}", flush=True)
    lines = [f"# Light test-suite report", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} local by `handoff/tools/run_light_tests.py` "
             f"(one pytest process per file, per-process memory cap {a.cap_mb} MB via a Windows Job Object; "
             f"peak = committed bytes).", "",
             "**Totals:** " + ", ".join(f"{v} {k}" for k, v in sorted(tot.items())), "",
             "| File | Interpreter | Exit | Peak MB | Seconds | pytest summary |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['file']}` | {r.get('python', '–')} | {r['rc']} | {r['peak_mb'] if r['peak_mb'] is not None else '–'} | "
                     f"{r['secs']} | {r['summary']} |")
    bad = [r for r in rows if r["rc"] not in (0, "excluded")]
    if bad:
        lines += ["", "## Output of files that did not exit 0", ""]
        for r in bad:
            lines += [f"### {r['file']} (exit {r['rc']})", "", "```", r["output"][-6000:].rstrip(), "```", ""]
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("TOTALS", tot, "->", a.out)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
