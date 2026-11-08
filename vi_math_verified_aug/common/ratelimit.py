"""Per-(provider, key, model) lanes: token-bucket RPM, in-flight cap, cool-down and adaptive token rate.

A **lane** is one quota unit: one API key x one model. Each lane has

* a token bucket for request starts (``rpm`` refill, ``burst`` capacity; burst 1 = even spacing),
* an in-flight cap (``max_inflight``),
* a cool-down after a 429 (``Retry-After`` / Gemini ``RetryInfo.retryDelay`` if given, else exponential),
* an **adaptive token-rate limit** (AIMD on tokens per rolling 60 s). G1 found that Gemini 429s on long
  outputs behave like a per-key token-rate limit, not an RPM limit: a lane can sit well under its RPM and still
  be refused when several long generations land in the same minute. So on a 429 the lane sets
  ``tpm_limit = max(min_tpm, 0.7 x tokens seen in the last 60 s)`` (multiplicative decrease), and every success
  raises it by 5 % (additive-ish increase). Requests are admitted only if the rolling token use plus the
  request's estimate fits. ``tpm`` may also be set up front if the real quota is known.

A **pool** holds the lanes that can serve one (provider, model) — one per key — and hands a request to the
lane that can start soonest. Admission is a synchronous critical section (no ``await`` between check and take),
so it is race-free on one event loop. Clock and sleep are injectable for tests.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

WINDOW_S = 60.0


@dataclass
class LaneConfig:
    rpm: float = 12.0
    burst: float = 1.0
    max_inflight: int = 1
    tpm: float | None = None          # known token-rate limit; None = learn it from 429s
    min_tpm: float = 2_000.0
    decrease: float = 0.7
    increase: float = 1.05
    base_cooldown_s: float = 10.0
    max_cooldown_s: float = 120.0


@dataclass
class _Ticket:
    ts: float
    tokens: float
    done: bool = False


@dataclass
class Lane:
    name: str                          # e.g. "gemini:k2:gemini-3.5-flash-lite" (never the key itself)
    key_index: int
    cfg: LaneConfig
    clock: Callable[[], float] = time.monotonic
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    inflight: int = 0
    cooldown_until: float = 0.0
    consecutive_429: int = 0
    tpm_limit: float | None = field(init=False)
    out_ewma: float | None = None
    window: deque = field(default_factory=deque)
    n_started: int = 0
    n_429: int = 0

    def __post_init__(self):
        self.tokens = self.cfg.burst
        self.last_refill = self.clock()
        self.tpm_limit = self.cfg.tpm

    # ---------------------------------------------------------------- estimates
    def estimate(self, input_tokens: int, max_tokens: int) -> float:
        out = max_tokens / 2 if self.out_ewma is None else min(max_tokens, 1.25 * self.out_ewma)
        return float(input_tokens + out)

    def _refill(self, now: float) -> None:
        rate = self.cfg.rpm / 60.0
        self.tokens = min(self.cfg.burst, self.tokens + (now - self.last_refill) * rate)
        self.last_refill = now

    def _trim(self, now: float) -> None:
        while self.window and self.window[0].done and now - self.window[0].ts > WINDOW_S:
            self.window.popleft()

    def used_tokens(self, now: float) -> float:
        self._trim(now)
        return sum(t.tokens for t in self.window if now - t.ts <= WINDOW_S or not t.done)

    # ---------------------------------------------------------------- admission
    def wait_time(self, est_tokens: float, now: float | None = None) -> float:
        """Seconds until this lane could start a request of ``est_tokens`` (0 = now)."""
        now = self.clock() if now is None else now
        self._refill(now)
        waits = [0.0]
        if now < self.cooldown_until:
            waits.append(self.cooldown_until - now)
        if self.inflight >= self.cfg.max_inflight:
            waits.append(0.05)                       # freed by a finishing request; poll
        if self.tokens < 1.0:
            waits.append((1.0 - self.tokens) / (self.cfg.rpm / 60.0))
        if self.tpm_limit is not None:
            used = self.used_tokens(now)
            if used > 0 and used + est_tokens > self.tpm_limit:
                excess = used + est_tokens - self.tpm_limit
                w = 0.05
                for t in self.window:                # oldest first: wait until enough tokens age out
                    if not t.done or now - t.ts > WINDOW_S:
                        continue
                    excess -= t.tokens
                    w = max(w, t.ts + WINDOW_S - now)
                    if excess <= 0:
                        break
                waits.append(w)
        return max(waits)

    def take(self, est_tokens: float, now: float | None = None) -> _Ticket:
        now = self.clock() if now is None else now
        self._refill(now)
        self.tokens -= 1.0
        self.inflight += 1
        self.n_started += 1
        t = _Ticket(now, est_tokens)
        self.window.append(t)
        return t

    # ---------------------------------------------------------------- feedback
    def finish(self, ticket: _Ticket, *, input_tokens: int | None = None, output_tokens: int | None = None,
               ok: bool = True) -> None:
        self.inflight = max(0, self.inflight - 1)
        ticket.done = True
        if input_tokens is not None or output_tokens is not None:
            ticket.tokens = float((input_tokens or 0) + (output_tokens or 0))
        if ok and output_tokens is not None:
            self.out_ewma = output_tokens if self.out_ewma is None else 0.8 * self.out_ewma + 0.2 * output_tokens
        if ok:
            self.consecutive_429 = 0
            if self.tpm_limit is not None:
                ceiling = self.cfg.tpm if self.cfg.tpm is not None else float("inf")
                self.tpm_limit = min(ceiling, self.tpm_limit * self.cfg.increase)

    def rate_limited(self, retry_after_s: float | None = None) -> float:
        """Register a 429 on this lane. Returns the cool-down applied (seconds)."""
        now = self.clock()
        self.n_429 += 1
        self.consecutive_429 += 1
        observed = self.used_tokens(now)
        base = self.tpm_limit if self.tpm_limit is not None else observed
        if base and base > 0:
            self.tpm_limit = max(self.cfg.min_tpm, self.cfg.decrease * min(base, observed or base))
        cd = retry_after_s if retry_after_s is not None else min(
            self.cfg.max_cooldown_s, self.cfg.base_cooldown_s * 2 ** (self.consecutive_429 - 1))
        self.cooldown_until = max(self.cooldown_until, now + cd)
        return cd

    def snapshot(self) -> dict:
        now = self.clock()
        return {"lane": self.name, "inflight": self.inflight, "started": self.n_started, "n_429": self.n_429,
                "tpm_limit": None if self.tpm_limit is None else round(self.tpm_limit),
                "tokens_60s": round(self.used_tokens(now)),
                "cooldown_s": round(max(0.0, self.cooldown_until - now), 1)}


class LaneStall(RuntimeError):
    """No lane of a pool can start a request within ``max_block_s`` (e.g. every key is in a server-imposed
    multi-hour cool-down after a daily-quota 429). Raised instead of waiting forever (S3 hang, 2026-11-07)."""


class Watchdog:
    """Raise :class:`LaneStall` from :meth:`check` / :meth:`run` when :meth:`tick` was not called for
    ``timeout_s`` seconds (no completed request). Opt-in; nothing in the library uses it implicitly."""

    def __init__(self, timeout_s: float = 300.0, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.timeout_s, self.clock, self.sleep = timeout_s, clock, sleep
        self.last = clock()
        self.n = 0

    def tick(self) -> None:
        self.last = self.clock()
        self.n += 1

    def check(self) -> None:
        idle = self.clock() - self.last
        if idle > self.timeout_s:
            raise LaneStall(f"watchdog: no completion for {idle:.0f}s (> {self.timeout_s:.0f}s); {self.n} so far")

    async def run(self, poll_s: float = 10.0) -> None:
        """Await this next to the work (e.g. ``asyncio.wait(..., FIRST_EXCEPTION)``); it raises on a stall."""
        while True:
            await self.sleep(poll_s)
            self.check()


class LanePool:
    """The lanes serving one (provider, model).

    ``max_block_s`` (default 300 s): :meth:`acquire` raises :class:`LaneStall` instead of waiting when the
    soonest lane cannot start within that time, or after it has waited that long in total. Before this guard a
    server cool-down of hours (Gemini ``RetryInfo`` "16h40m" on a per-day quota) parked every lane and the
    caller hung with no error. ``max_block_s=None`` restores the old unbounded wait.
    """

    def __init__(self, lanes: list[Lane], *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, max_wait_step: float = 2.0,
                 max_block_s: float | None = 300.0):
        if not lanes:
            raise ValueError("a lane pool needs at least one lane (no API key configured?)")
        self.lanes = lanes
        self.clock = clock
        self.sleep = sleep
        self.max_wait_step = max_wait_step
        self.max_block_s = max_block_s
        self._rr = 0

    async def acquire(self, input_tokens: int, max_tokens: int, *, exclude: set[int] | None = None
                      ) -> tuple[Lane, _Ticket]:
        """Wait for, then take, the lane that can start soonest. ``exclude`` = key indices to avoid if any
        other lane exists (e.g. the lane that just returned a 429). Raises :class:`LaneStall` (see class doc)."""
        t0 = self.clock()
        while True:
            now = self.clock()
            cands = [ln for ln in self.lanes if not exclude or ln.key_index not in exclude] or self.lanes
            best, best_w = None, float("inf")
            n = len(cands)
            for j in range(n):                       # round-robin tie-break spreads load across keys
                ln = cands[(self._rr + j) % n]
                w = ln.wait_time(ln.estimate(input_tokens, max_tokens), now)
                if w < best_w:
                    best, best_w = ln, w
                    if w == 0:
                        break
            if best is not None and best_w <= 0:
                self._rr += 1
                return best, best.take(best.estimate(input_tokens, max_tokens), now)
            if self.max_block_s is not None and (best_w > self.max_block_s or now - t0 > self.max_block_s):
                raise LaneStall(f"no lane can start within {self.max_block_s:.0f}s (soonest in {best_w:.0f}s, "
                                f"waited {now - t0:.0f}s): {[ln.snapshot() for ln in cands]}")
            await self.sleep(min(best_w, self.max_wait_step))

    def snapshot(self) -> list[dict]:
        return [ln.snapshot() for ln in self.lanes]
