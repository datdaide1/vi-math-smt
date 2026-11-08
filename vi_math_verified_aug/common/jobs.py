"""Minimal job ledger (T6 §6.2): SQLite (WAL), idempotent job ids, checked status transitions.

Pure library — no scheduling, no platform calls. The orchestrator (later) owns the loop.

* ``job_id = sha1(canonical_json([kind, base, arm, n_train, seed, recipe_ver, data_ver, prompt_ver]))``
  (+ ``id_extra`` as a 9th element when the spec has one). Same spec -> same id; ``submit`` of an existing id is
  a no-op (``created=False``), whatever its status. ``RunConfig.to_spec()`` produces matching specs, so
  ``RunConfig.job_id() == job_id(kind, cfg.to_spec())``.
* ``jobs`` columns: id, kind, spec_json, deps (JSON list), platform, priority, status, attempts, external_ref,
  ckpt_uri, est_cost, actual_cost, started, finished, last_error (+ created, updated).
* ``platform`` is derived from the spec by a ``platform_for(kind, spec)`` function given to the ledger
  (T6: "derived from the base, never chosen freely"); without one, the caller passes it explicitly.
* Status machine (``TRANSITIONS``); every change is a compare-and-set inside ``BEGIN IMMEDIATE`` and appends an
  ``events`` row plus a line in the JSONL mirror ``<db stem>.events.jsonl``.
* ``budget`` table: named counters (Modal USD, Kaggle hours, NIM calls/day ...).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

from .apilog import append_line, utc_now
from .cache import connect
from .config import JOB_KINDS
from .hashing import canonical_json, stable_hash
from .paths import state_dir

STATUSES = ("PENDING", "READY", "SUBMITTED", "RUNNING", "NEEDS_RESUME", "COMPLETE", "FAILED", "BLOCKED")
TRANSITIONS: dict[str, frozenset[str]] = {
    "PENDING": frozenset({"READY", "BLOCKED", "FAILED"}),
    "READY": frozenset({"SUBMITTED", "BLOCKED", "PENDING", "FAILED"}),
    "SUBMITTED": frozenset({"RUNNING", "COMPLETE", "FAILED", "NEEDS_RESUME", "READY"}),  # READY = push rolled back
    "RUNNING": frozenset({"COMPLETE", "FAILED", "NEEDS_RESUME"}),
    "NEEDS_RESUME": frozenset({"READY", "SUBMITTED", "FAILED"}),
    "FAILED": frozenset({"READY", "PENDING"}),                                          # manual retry
    "BLOCKED": frozenset({"READY", "PENDING", "FAILED"}),
    "COMPLETE": frozenset(),                                                            # terminal
}
ID_FIELDS = ("base", "arm", "n_train", "seed", "recipe_ver", "data_ver", "prompt_ver")
DEP_FAILED = "dep_failed:"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    spec_json    TEXT NOT NULL,
    deps         TEXT NOT NULL DEFAULT '[]',
    platform     TEXT,
    priority     INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    external_ref TEXT,
    ckpt_uri     TEXT,
    est_cost     REAL,
    actual_cost  REAL,
    created      TEXT NOT NULL,
    updated      TEXT NOT NULL,
    started      TEXT,
    finished     TEXT,
    last_error   TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, priority);
CREATE TABLE IF NOT EXISTS events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      TEXT NOT NULL,
    job_id      TEXT NOT NULL,
    event       TEXT NOT NULL,
    from_status TEXT,
    to_status   TEXT,
    data_json   TEXT
);
CREATE INDEX IF NOT EXISTS events_job ON events(job_id);
CREATE TABLE IF NOT EXISTS budget (
    key     TEXT PRIMARY KEY,
    value   REAL NOT NULL,
    updated TEXT NOT NULL
);
"""


class LedgerError(RuntimeError):
    pass


class IllegalTransition(LedgerError):
    pass


class StaleTransition(LedgerError):
    """``expect=`` did not match the current status (another actor moved the job first)."""


def job_id(kind: str, spec: dict[str, Any]) -> str:
    t: list[Any] = [kind] + [spec.get(k) for k in ID_FIELDS]
    if spec.get("id_extra"):
        t.append(spec["id_extra"])
    return stable_hash(t, "sha1")


class JobLedger:
    def __init__(self, path: str | os.PathLike | None = None, *,
                 platform_for: Callable[[str, dict], str | None] | None = None, mirror_events: bool = True):
        self.path = Path(path) if path else state_dir() / "ledger.sqlite"
        self._con = connect(self.path)
        self._con.executescript(_SCHEMA)
        self._lock = threading.Lock()
        self.platform_for = platform_for
        self.events_path = self.path.with_name(self.path.stem + ".events.jsonl") if mirror_events else None

    # ---------------------------------------------------------------- internals
    def _tx(self):
        ledger = self

        class _T:
            def __enter__(self_):
                ledger._lock.acquire()
                ledger._con.execute("BEGIN IMMEDIATE")
                return ledger._con

            def __exit__(self_, et, e, tb):
                try:
                    ledger._con.execute("COMMIT" if et is None else "ROLLBACK")
                finally:
                    ledger._lock.release()
        return _T()

    def _event(self, con, jid: str, event: str, frm: str | None, to: str | None, data: dict | None) -> dict:
        ev = {"ts_utc": utc_now(), "job_id": jid, "event": event, "from_status": frm, "to_status": to,
              "data": data or {}}
        con.execute("INSERT INTO events(ts_utc, job_id, event, from_status, to_status, data_json) "
                    "VALUES (?,?,?,?,?,?)", (ev["ts_utc"], jid, event, frm, to,
                                             json.dumps(data or {}, ensure_ascii=False, default=str)))
        return ev

    def _mirror(self, evs: Iterable[dict]) -> None:
        if self.events_path is None:
            return
        for ev in evs:
            append_line(self.events_path, json.dumps(ev, ensure_ascii=False, default=str))

    @staticmethod
    def _row(r: sqlite3.Row | tuple, cols: list[str]) -> dict[str, Any]:
        d = dict(zip(cols, r))
        d["spec"] = json.loads(d.pop("spec_json"))
        d["deps"] = json.loads(d["deps"])
        return d

    def _select(self, where: str = "", args: tuple = (), order: str = "") -> list[dict]:
        with self._lock:
            cur = self._con.execute(f"SELECT * FROM jobs {where} {order}", args)
            cols = [c[0] for c in cur.description]
            return [self._row(r, cols) for r in cur.fetchall()]

    # ---------------------------------------------------------------- submit / read
    def submit(self, kind: str, spec: dict[str, Any], *, deps: Iterable[str] = (), priority: int = 0,
               platform: str | None = None, est_cost: float | None = None) -> tuple[str, bool]:
        """Insert a job; returns ``(job_id, created)``. Idempotent: an existing id is left untouched."""
        if kind not in JOB_KINDS:
            raise LedgerError(f"kind {kind!r} not in {JOB_KINDS}")
        if spec.get("kind", kind) != kind:
            raise LedgerError(f"spec.kind {spec.get('kind')!r} != kind {kind!r}")
        if self.platform_for is not None:
            if platform is not None:
                raise LedgerError("platform is derived by platform_for(); do not pass it explicitly")
            platform = self.platform_for(kind, spec)
        jid = job_id(kind, spec)
        deps = list(dict.fromkeys(deps))
        if jid in deps:
            raise LedgerError("a job cannot depend on itself")
        spec_json = canonical_json(spec)
        evs = []
        with self._tx() as con:
            if con.execute("SELECT 1 FROM jobs WHERE id=?", (jid,)).fetchone():
                return jid, False
            st = {}
            for d in deps:
                row = con.execute("SELECT status FROM jobs WHERE id=?", (d,)).fetchone()
                if row is None:
                    raise LedgerError(f"unknown dependency {d}")
                st[d] = row[0]
            status = "READY" if all(s == "COMPLETE" for s in st.values()) else "PENDING"
            now = utc_now()
            con.execute("INSERT INTO jobs(id, kind, spec_json, deps, platform, priority, status, est_cost, created,"
                        " updated) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (jid, kind, spec_json, json.dumps(deps), platform, int(priority), status, est_cost, now,
                         now))
            evs.append(self._event(con, jid, "submit", None, status, {"kind": kind, "platform": platform}))
        self._mirror(evs)
        return jid, True

    def get(self, jid: str) -> dict[str, Any] | None:
        rows = self._select("WHERE id=?", (jid,))
        return rows[0] if rows else None

    def list(self, *, status: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        conds, args = [], []
        if status:
            conds.append("status=?")
            args.append(status)
        if kind:
            conds.append("kind=?")
            args.append(kind)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        return self._select(where, tuple(args), "ORDER BY created, id")

    def ready(self, *, kind: str | None = None, platform: str | None = None, limit: int | None = None
              ) -> list[dict[str, Any]]:
        conds, args = ["status='READY'"], []
        if kind:
            conds.append("kind=?")
            args.append(kind)
        if platform:
            conds.append("platform=?")
            args.append(platform)
        order = "ORDER BY priority DESC, created, id" + (f" LIMIT {int(limit)}" if limit else "")
        return self._select("WHERE " + " AND ".join(conds), tuple(args), order)

    def counts(self) -> dict[str, int]:
        with self._lock:
            got = dict(self._con.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())
        return {s: got.get(s, 0) for s in STATUSES}

    def events(self, jid: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            q = "SELECT ts_utc, job_id, event, from_status, to_status, data_json FROM events"
            rows = self._con.execute(q + (" WHERE job_id=? ORDER BY seq" if jid else " ORDER BY seq"),
                                     (jid,) if jid else ()).fetchall()
        return [{"ts_utc": r[0], "job_id": r[1], "event": r[2], "from_status": r[3], "to_status": r[4],
                 "data": json.loads(r[5] or "{}")} for r in rows]

    # ---------------------------------------------------------------- transitions
    def transition(self, jid: str, to: str, *, expect: str | None = None, external_ref: str | None = None,
                   ckpt_uri: str | None = None, actual_cost: float | None = None, error: str | None = None,
                   data: dict | None = None) -> dict[str, Any]:
        if to not in STATUSES:
            raise LedgerError(f"unknown status {to!r}")
        with self._tx() as con:
            row = con.execute("SELECT status FROM jobs WHERE id=?", (jid,)).fetchone()
            if row is None:
                raise LedgerError(f"unknown job {jid}")
            frm = row[0]
            if expect is not None and frm != expect:
                raise StaleTransition(f"{jid}: expected {expect}, found {frm}")
            if to not in TRANSITIONS[frm]:
                raise IllegalTransition(f"{jid}: {frm} -> {to} not allowed")
            now = utc_now()
            sets = {"status": to, "updated": now}
            if to == "SUBMITTED":
                con.execute("UPDATE jobs SET attempts = attempts + 1 WHERE id=?", (jid,))
            if to == "RUNNING":
                con.execute("UPDATE jobs SET started = COALESCE(started, ?) WHERE id=?", (now, jid))
            if to in ("COMPLETE", "FAILED"):
                sets["finished"] = now
            if to in ("READY", "PENDING") and frm in ("FAILED", "BLOCKED"):
                sets["finished"] = None
            if external_ref is not None:
                sets["external_ref"] = external_ref
            if ckpt_uri is not None:
                sets["ckpt_uri"] = ckpt_uri
            if actual_cost is not None:
                sets["actual_cost"] = actual_cost
            if error is not None:
                sets["last_error"] = error
            elif to == "COMPLETE":
                sets["last_error"] = None
            cols = ", ".join(f"{k}=?" for k in sets)
            con.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*sets.values(), jid))
            payload = {k: v for k, v in {"external_ref": external_ref, "ckpt_uri": ckpt_uri,
                                         "actual_cost": actual_cost, "error": error}.items() if v is not None}
            payload.update(data or {})
            ev = self._event(con, jid, "transition", frm, to, payload)
        self._mirror([ev])
        return self.get(jid)  # type: ignore[return-value]

    def retry(self, jid: str) -> dict[str, Any]:
        """FAILED / NEEDS_RESUME / BLOCKED -> READY (attempts keep counting)."""
        j = self.get(jid)
        if j is None:
            raise LedgerError(f"unknown job {jid}")
        return self.transition(jid, "READY", expect=j["status"])

    def refresh_ready(self) -> dict[str, list[str]]:
        """Promote PENDING jobs whose deps are all COMPLETE to READY; PENDING jobs with a FAILED dep go BLOCKED
        (``last_error = 'dep_failed:<id>'``); dep-BLOCKED jobs whose deps all completed (after a retry) go READY.
        Returns the ids moved, by target status."""
        moved: dict[str, list[str]] = {"READY": [], "BLOCKED": []}
        evs = []
        with self._tx() as con:
            status = dict(con.execute("SELECT id, status FROM jobs").fetchall())
            rows = con.execute("SELECT id, status, deps, last_error FROM jobs WHERE status IN ('PENDING','BLOCKED')"
                               ).fetchall()
            now = utc_now()
            for jid, st, deps_json, last_error in rows:
                deps = json.loads(deps_json)
                ds = [status.get(d) for d in deps]
                if st == "BLOCKED" and not (last_error or "").startswith(DEP_FAILED):
                    continue                                  # blocked for another reason (budget, quota)
                if all(s == "COMPLETE" for s in ds):
                    con.execute("UPDATE jobs SET status='READY', updated=?, last_error=NULL WHERE id=?", (now, jid))
                    evs.append(self._event(con, jid, "transition", st, "READY", {"reason": "deps_complete"}))
                    moved["READY"].append(jid)
                elif st == "PENDING":
                    bad = [d for d, s in zip(deps, ds) if s == "FAILED"]
                    if bad:
                        err = DEP_FAILED + ",".join(bad)
                        con.execute("UPDATE jobs SET status='BLOCKED', updated=?, last_error=? WHERE id=?",
                                    (now, err, jid))
                        evs.append(self._event(con, jid, "transition", st, "BLOCKED", {"reason": err}))
                        moved["BLOCKED"].append(jid)
        self._mirror(evs)
        return moved

    # ---------------------------------------------------------------- budget counters
    def add_budget(self, key: str, delta: float) -> float:
        with self._tx() as con:
            con.execute("INSERT INTO budget(key, value, updated) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET "
                        "value = value + excluded.value, updated = excluded.updated", (key, float(delta), utc_now()))
            return con.execute("SELECT value FROM budget WHERE key=?", (key,)).fetchone()[0]

    def set_budget(self, key: str, value: float) -> None:
        with self._tx() as con:
            con.execute("INSERT INTO budget(key, value, updated) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET "
                        "value = excluded.value, updated = excluded.updated", (key, float(value), utc_now()))

    def get_budget(self, key: str) -> float:
        with self._lock:
            row = self._con.execute("SELECT value FROM budget WHERE key=?", (key,)).fetchone()
        return row[0] if row else 0.0

    def close(self) -> None:
        with self._lock:
            self._con.close()
