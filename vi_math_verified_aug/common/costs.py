"""Prices, cost computation and a concurrency-safe spend tracker with a hard budget cap.

Correctness under concurrency (the G1 over-count bug): G1 rebuilt "spent so far" by summing ledger costs while
the same process was also adding live costs, so concurrent cache loads double-counted. Here:

* spend is **only** written when a live response is billed, exactly once per ``charge_id`` (``INSERT OR
  IGNORE`` on a primary key, so a replay or a duplicate settle cannot double-count); cache hits never touch it;
* the total is read incrementally (``rowid > last_seen``) from SQLite, so other processes' charges count too;
* the cap is **hard**: before a call, ``reserve()`` books the call's worst-case cost (input estimate +
  ``max_tokens`` at the output price) under a lock, and refuses if ``spent + outstanding reservations +
  estimate > cap``. N concurrent calls therefore cannot all pass the check and overshoot together.
  ``settle()`` replaces the reservation with the actual cost. Reservations are per process; the persisted
  spend is shared across processes.
"""
from __future__ import annotations

import itertools
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any

from .apilog import utc_now
from .cache import connect


class BudgetExceeded(RuntimeError):
    pass


class UnknownPrice(KeyError):
    pass


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""
    input: float
    output: float
    cache_write_mult: float = 1.25   # 5-minute ephemeral cache write
    cache_read_mult: float = 0.10
    batch_mult: float = 0.50         # Message Batches API discount


# Claude list prices as recorded by G1 (pilot_gen/clients.py, Nov 2026). Re-check the pricing page before a
# large run; override with LLMClient(prices=...) or VIMATH_PRICES_JSON='{"model": [in, out], ...}'.
DEFAULT_PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.50),
    "claude-sonnet-5-5": Price(2.0, 10.0),
    "claude-opus-5-5": Price(4.0, 20.0),
    # free tiers / trial endpoints: tracked as $0 (calls are still counted)
    "gemini-": Price(0.0, 0.0),
    "gemma-": Price(0.0, 0.0),
}
FREE_PROVIDERS = {"gemini", "nim"}


def load_prices(overrides: dict[str, Any] | None = None) -> dict[str, Price]:
    import json
    prices = dict(DEFAULT_PRICES)
    env = os.environ.get("VIMATH_PRICES_JSON")
    for src in (json.loads(env) if env else {}, overrides or {}):
        for k, v in src.items():
            prices[k] = v if isinstance(v, Price) else Price(*v)
    return prices


def price_for(model: str, provider: str, prices: dict[str, Price]) -> Price:
    if model in prices:
        return prices[model]
    best = max((k for k in prices if model.startswith(k)), key=len, default=None)
    if best is not None:
        return prices[best]
    if provider in FREE_PROVIDERS:
        return Price(0.0, 0.0)
    raise UnknownPrice(f"no price for model {model!r}; pass prices={{...}} to the client")


def cost_usd(usage: dict[str, Any], price: Price, *, batch: bool = False) -> float:
    """``usage`` uses the normalised keys input_tokens / output_tokens / cache_read / cache_write.

    ``input_tokens`` excludes cache reads and writes (Anthropic's convention); Gemini/NIM never report cache
    fields, so they reduce to input + output. Reasoning tokens are already inside ``output_tokens``.
    """
    i = usage.get("input_tokens") or 0
    o = usage.get("output_tokens") or 0
    cw = usage.get("cache_write") or 0
    cr = usage.get("cache_read") or 0
    usd = (i * price.input + cw * price.input * price.cache_write_mult + cr * price.input * price.cache_read_mult
           + o * price.output) / 1e6
    return usd * (price.batch_mult if batch else 1.0)


def worst_case_cost(est_input_tokens: int, max_tokens: int, price: Price, *, batch: bool = False) -> float:
    return cost_usd({"input_tokens": est_input_tokens, "output_tokens": max_tokens}, price, batch=batch)


_CHARGES = """
CREATE TABLE IF NOT EXISTS charges (
    charge_id TEXT PRIMARY KEY,
    scope     TEXT NOT NULL,
    provider  TEXT,
    model     TEXT,
    usd       REAL NOT NULL,
    ts_utc    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS charges_scope ON charges(scope);
"""


class CostTracker:
    """Spend for one ``scope`` (e.g. a workstream) with an optional hard cap in USD."""

    def __init__(self, cap_usd: float | None = None, *, db_path: str | os.PathLike | None = None,
                 scope: str = "default"):
        self.cap_usd = cap_usd
        self.scope = scope
        self._lock = threading.Lock()
        self._reserved: dict[int, float] = {}
        self._ids = itertools.count(1)
        self._con: sqlite3.Connection | None = None
        self._mem: dict[str, float] = {}
        self._spent = 0.0
        self._last_rowid = 0
        if db_path is not None:
            self._con = connect(db_path)
            self._con.executescript(_CHARGES)

    # ---------------------------------------------------------------- reads
    def _refresh_locked(self) -> None:
        if self._con is None:
            return
        row = self._con.execute(
            "SELECT COALESCE(SUM(usd),0), COALESCE(MAX(rowid),?) FROM charges WHERE scope=? AND rowid>?",
            (self._last_rowid, self.scope, self._last_rowid)).fetchone()
        self._spent += row[0]
        self._last_rowid = row[1]

    def spent(self) -> float:
        with self._lock:
            self._refresh_locked()
            return self._spent

    def reserved(self) -> float:
        with self._lock:
            return sum(self._reserved.values())

    def remaining(self) -> float | None:
        if self.cap_usd is None:
            return None
        with self._lock:
            self._refresh_locked()
            return self.cap_usd - self._spent - sum(self._reserved.values())

    # ---------------------------------------------------------------- writes
    def reserve(self, est_usd: float) -> int:
        """Book ``est_usd``; returns a reservation id. Raises ``BudgetExceeded`` if the cap would be crossed."""
        with self._lock:
            self._refresh_locked()
            out = sum(self._reserved.values())
            if self.cap_usd is not None and self._spent + out + est_usd > self.cap_usd:
                raise BudgetExceeded(
                    f"[{self.scope}] spent ${self._spent:.4f} + reserved ${out:.4f} + this call "
                    f"${est_usd:.4f} > cap ${self.cap_usd:.2f}")
            rid = next(self._ids)
            self._reserved[rid] = est_usd
            return rid

    def release(self, rid: int | None) -> None:
        if rid is None:
            return
        with self._lock:
            self._reserved.pop(rid, None)

    def charge(self, charge_id: str, usd: float, *, provider: str = "", model: str = "",
               rid: int | None = None) -> bool:
        """Record a billed amount once per ``charge_id`` (idempotent) and drop reservation ``rid``.
        Returns False if this ``charge_id`` was already recorded."""
        with self._lock:
            if rid is not None:
                self._reserved.pop(rid, None)
            if self._con is None:
                if charge_id in self._mem:
                    return False
                self._mem[charge_id] = usd
                self._spent += usd
                return True
            cur = self._con.execute("INSERT OR IGNORE INTO charges VALUES (?,?,?,?,?,?)",
                                    (charge_id, self.scope, provider, model, float(usd), utc_now()))
            new = cur.rowcount == 1
            self._refresh_locked()
            return new

    def summary(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            return {"scope": self.scope, "spent_usd": round(self._spent, 6),
                    "reserved_usd": round(sum(self._reserved.values()), 6), "cap_usd": self.cap_usd}

    def close(self) -> None:
        with self._lock:
            if self._con is not None:
                self._con.close()
                self._con = None
