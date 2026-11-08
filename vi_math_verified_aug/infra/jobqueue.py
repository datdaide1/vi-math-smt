"""Minimal SQLite job queue (T6 §6.2) — skeleton.

* job_id = sha1(canonical JSON of the identity fields of the spec) -> the same spec always gets the same id.
* submit() is idempotent: re-submitting an existing id is a no-op (returns the existing row).
* Status machine: PENDING -> READY -> SUBMITTED -> RUNNING -> COMPLETE | FAILED | NEEDS_RESUME | BLOCKED.
* Every status change is appended to `events` (audit log).
* platform is derived from (kind, base) by PLATFORM, never chosen freely (determinism rule, T6 §6.4).

CLI::

    python -m vi_math_verified_aug.infra.jobqueue init
    python -m vi_math_verified_aug.infra.jobqueue submit spec.json
    python -m vi_math_verified_aug.infra.jobqueue list [--status READY]
    python -m vi_math_verified_aug.infra.jobqueue set JOB_ID STATUS [--ref REF] [--error MSG]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO / "jobs" / "queue.sqlite"

STATUSES = ("PENDING", "READY", "SUBMITTED", "RUNNING", "NEEDS_RESUME", "COMPLETE", "FAILED", "BLOCKED")
KINDS = ("gen", "verify", "sft", "grpo", "eval", "upload", "pilot")
ID_FIELDS = ("kind", "base", "arm", "n_train", "seed", "recipe_ver", "data_ver", "prompt_ver")

# Fixed hardware map (T6 §5.4). Edit only by a recorded decision; a job never switches platform on failure.
PLATFORM = {
    ("sft", "Qwen/Qwen3.5-0.8B-Base"): "kaggle_t4",
    ("sft", "Qwen/Qwen3.5-2B-Base"): "kaggle_t4",
    ("sft", "Qwen/Qwen3-0.6B-Base"): "kaggle_t4",
    ("sft", "Qwen/Qwen3-1.7B-Base"): "kaggle_t4",
    ("sft", "google/gemma-4-E2B"): "kaggle_t4",
    ("sft", "Qwen/Qwen3.5-4B-Base"): "modal_l40s",
    ("eval", "*"): "kaggle_t4",
    ("pilot", "*"): "kaggle_t4",
    ("gen", "*"): "nim",
    ("verify", "*"): "laptop",
    ("upload", "*"): "laptop",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, spec_json TEXT NOT NULL, deps TEXT NOT NULL DEFAULT '[]',
  platform TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0, external_ref TEXT, ckpt_uri TEXT,
  est_cost REAL, actual_cost REAL, created REAL, started REAL, finished REAL, last_error TEXT
);
CREATE TABLE IF NOT EXISTS events (
  ts REAL NOT NULL, job_id TEXT NOT NULL, old TEXT, new TEXT NOT NULL, detail TEXT
);
CREATE TABLE IF NOT EXISTS budget (
  ts REAL NOT NULL, resource TEXT NOT NULL, amount REAL NOT NULL, unit TEXT NOT NULL, note TEXT
);
"""


def job_id(spec: dict) -> str:
    ident = {k: spec.get(k) for k in ID_FIELDS}
    ident["extra"] = spec.get("id_extra")  # optional free-form identity (e.g. pilot name)
    return hashlib.sha1(json.dumps(ident, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def platform_for(spec: dict) -> str:
    k = (spec["kind"], spec.get("base"))
    if k in PLATFORM:
        return PLATFORM[k]
    if (spec["kind"], "*") in PLATFORM:
        return PLATFORM[(spec["kind"], "*")]
    raise KeyError(f"no fixed platform for {k}; add it to PLATFORM by decision")


class JobQueue:
    def __init__(self, db: str | Path = DEFAULT_DB):
        Path(db).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(db))
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(SCHEMA)

    def _event(self, jid, old, new, detail=None):
        self.con.execute("INSERT INTO events VALUES (?,?,?,?,?)", (time.time(), jid, old, new, detail))

    def submit(self, spec: dict, deps: list[str] | None = None, priority: int = 0,
               est_cost: float | None = None) -> tuple[str, bool]:
        """Returns (job_id, created). Idempotent."""
        if spec.get("kind") not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        jid = job_id(spec)
        if self.get(jid) is not None:
            return jid, False
        status = "PENDING" if deps else "READY"
        with self.con:
            self.con.execute(
                "INSERT INTO jobs (id, kind, spec_json, deps, platform, priority, status, est_cost, created)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (jid, spec["kind"], json.dumps(spec, sort_keys=True, ensure_ascii=False), json.dumps(deps or []),
                 platform_for(spec), priority, status, est_cost, time.time()))
            self._event(jid, None, status, "submit")
        return jid, True

    def get(self, jid: str):
        return self.con.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()

    def list(self, status: str | None = None, platform: str | None = None):
        q, a = "SELECT * FROM jobs WHERE 1=1", []
        if status:
            q += " AND status=?"; a.append(status)
        if platform:
            q += " AND platform=?"; a.append(platform)
        return self.con.execute(q + " ORDER BY priority DESC, created", a).fetchall()

    def set_status(self, jid: str, new: str, *, ref: str | None = None, error: str | None = None,
                   ckpt_uri: str | None = None, actual_cost: float | None = None):
        if new not in STATUSES:
            raise ValueError(new)
        row = self.get(jid)
        if row is None:
            raise KeyError(jid)
        now = time.time()
        with self.con:
            self.con.execute(
                "UPDATE jobs SET status=?, external_ref=COALESCE(?, external_ref), last_error=COALESCE(?, last_error),"
                " ckpt_uri=COALESCE(?, ckpt_uri), actual_cost=COALESCE(?, actual_cost),"
                " attempts=attempts + (CASE WHEN ?='SUBMITTED' THEN 1 ELSE 0 END),"
                " started=CASE WHEN ?='RUNNING' AND started IS NULL THEN ? ELSE started END,"
                " finished=CASE WHEN ? IN ('COMPLETE','FAILED') THEN ? ELSE finished END WHERE id=?",
                (new, ref, error, ckpt_uri, actual_cost, new, new, now, new, now, jid))
            self._event(jid, row["status"], new, error or ref)
            if new == "COMPLETE":
                self._release_dependents(jid)

    def _release_dependents(self, done_id: str):
        for r in self.con.execute("SELECT id, deps FROM jobs WHERE status='PENDING'").fetchall():
            deps = json.loads(r["deps"])
            if done_id in deps and all((self.get(d) or {"status": None})["status"] == "COMPLETE" for d in deps):
                self.con.execute("UPDATE jobs SET status='READY' WHERE id=?", (r["id"],))
                self._event(r["id"], "PENDING", "READY", f"deps done ({done_id})")

    def next_ready(self, platform: str):
        return self.con.execute("SELECT * FROM jobs WHERE status='READY' AND platform=?"
                                " ORDER BY priority DESC, created LIMIT 1", (platform,)).fetchone()

    def record_budget(self, resource: str, amount: float, unit: str, note: str = ""):
        with self.con:
            self.con.execute("INSERT INTO budget VALUES (?,?,?,?,?)", (time.time(), resource, amount, unit, note))


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="jobqueue")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("submit"); p.add_argument("spec"); p.add_argument("--dep", action="append", default=[])
    p = sub.add_parser("list"); p.add_argument("--status")
    p = sub.add_parser("set"); p.add_argument("job_id"); p.add_argument("status")
    p.add_argument("--ref"); p.add_argument("--error")
    a = ap.parse_args(argv)
    q = JobQueue(a.db)
    if a.cmd == "init":
        print("ok", a.db)
    elif a.cmd == "submit":
        spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
        jid, created = q.submit(spec, deps=a.dep)
        print(jid, "created" if created else "exists (no-op)")
    elif a.cmd == "list":
        for r in q.list(a.status):
            print(r["id"], r["kind"], r["platform"], r["status"], r["external_ref"] or "", sep="\t")
    elif a.cmd == "set":
        q.set_status(a.job_id, a.status, ref=a.ref, error=a.error)
        print(a.job_id, a.status)


if __name__ == "__main__":
    sys.exit(_main())
