"""Laptop side of the W2 pilot: inject a JOB into kaggle_pilot.py, push a private T4 kernel, wait, pull outputs.

Usage:
  python -m vi_math_verified_aug.infra.pilot.submit_pilot SLUG JOB_JSON [--no-wait] [--src FILE.py] [--dest DIR]
    --src   kernel source next to this file (default kaggle_pilot.py; S2 uses s2_pilot.py)
    --dest  where outputs are pulled (default jobs/runs/<SLUG>)
Outputs land in jobs/runs/<SLUG>/ (results/ tree + <slug>.log). Each push is recorded in jobs/queue.sqlite.
"""
import json
import re
import sys
import time
from pathlib import Path

from vi_math_verified_aug.infra.jobqueue import JobQueue
from vi_math_verified_aug.infra.kaggle_runner import REPO, KaggleRunner

HERE = Path(__file__).resolve().parent


def main(slug, job_json, wait=True, src="kaggle_pilot.py", dest=None):
    job = json.loads(Path(job_json).read_text(encoding="utf-8")) if Path(job_json).exists() else json.loads(job_json)
    # unique id per push: the kernel echoes it in results/meta.json, so a stale `kernels output` is detected
    job["run_id"] = f"{slug}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    build = REPO / "jobs" / "build" / slug
    build.mkdir(parents=True, exist_ok=True)
    src = (HERE / src).read_text(encoding="utf-8")
    script = build / "run.py"
    header = "import json as _json\nJOB = _json.loads(" + repr(json.dumps(job, ensure_ascii=False)) + ")\n"
    script.write_text(header + src, encoding="utf-8")
    r = KaggleRunner()
    q = JobQueue()
    spec = {"kind": "pilot", "base": None, "id_extra": f"{slug}:{json.dumps(job, sort_keys=True)}"}
    jid, created = q.submit(spec)
    kdir = r.render_kernel(slug, script, dataset_sources=["datuni/vimath-w2-pilot"])
    quota0 = r.gpu_quota_used()
    pushed = r.push(kdir, timeout_s=job.get("kernel_timeout_s", 3 * 3600))
    ver = re.search(r"version (\d+)", pushed)
    rec = {"run_id": job["run_id"], "kernel_version": int(ver.group(1)) if ver else None,
           "gpu_quota_used_h_at_push": quota0, "pushed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (kdir / "last_push.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(pushed.strip(), rec)
    q.set_status(jid, "SUBMITTED", ref=f"datuni/{slug}")
    if not wait:
        return
    st = r.wait(f"datuni/{slug}", poll_s=90, max_queued_s=job.get("max_queued_s", 3600),
                max_wait_s=job.get("kernel_timeout_s", 3 * 3600) + 2 * 3600)
    if st not in ("COMPLETE", "ERROR"):
        q.set_status(jid, "FAILED", error=st)
        print("status", st, "- not pulling (kernel did not reach a terminal state)")
        return
    out = Path(dest) if dest else REPO / "jobs" / "runs" / slug
    for attempt in range(4):  # ERROR kernels still have partial outputs; retry if the pull is stale
        r.output(f"datuni/{slug}", out)
        if r.verify_output_run_id(out, job["run_id"]):
            break
        print(f"output is stale (run_id mismatch), retry {attempt + 1}/4 in 60 s", flush=True)
        time.sleep(60)
    else:
        st = "STALE_OUTPUT:" + st
    q.set_status(jid, "COMPLETE" if st == "COMPLETE" else "FAILED", error=None if st == "COMPLETE" else st)
    print("status", st, "->", out, "quota used h:", r.gpu_quota_used())


def _opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], wait="--no-wait" not in sys.argv, src=_opt("--src", "kaggle_pilot.py"),
         dest=_opt("--dest"))
