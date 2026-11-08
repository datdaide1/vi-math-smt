"""common/ratelimit.py — token bucket, in-flight cap, cool-down, adaptive token rate (fake clock)."""
from __future__ import annotations

import asyncio

import pytest

from vi_math_verified_aug.common.ratelimit import Lane, LaneConfig, LanePool


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    async def sleep(self, s):
        self.t += s
        await asyncio.sleep(0)


def test_token_bucket_spacing():
    ck = Clock()
    ln = Lane("l", 0, LaneConfig(rpm=60, burst=1, max_inflight=10), clock=ck)
    assert ln.wait_time(10) == 0
    t = ln.take(10)
    ln.finish(t, input_tokens=5, output_tokens=5)
    assert ln.wait_time(10) == pytest.approx(1.0)
    ck.t = 1.0
    assert ln.wait_time(10) == 0


def test_inflight_cap():
    ck = Clock()
    ln = Lane("l", 0, LaneConfig(rpm=6000, burst=100, max_inflight=1), clock=ck)
    t = ln.take(1)
    assert ln.wait_time(1) > 0
    ln.finish(t)
    assert ln.wait_time(1) == 0


def test_adaptive_token_rate_cut_wait_and_recover():
    ck = Clock()
    cfg = LaneConfig(rpm=6000, burst=100, max_inflight=10, min_tpm=1000, base_cooldown_s=5)
    ln = Lane("l", 0, cfg, clock=ck)
    assert ln.tpm_limit is None
    for _ in range(3):                                   # three long generations land in the same minute
        ln.finish(ln.take(4000), input_tokens=500, output_tokens=3500)
    t = ln.take(4000)
    cd = ln.rate_limited(None)                           # the 4th is refused (no Retry-After given)
    ln.finish(t, input_tokens=0, output_tokens=0, ok=False)
    assert cd == pytest.approx(5.0)
    assert ln.tpm_limit == pytest.approx(0.7 * 16000)    # 0.7 x tokens observed in the window at the 429
    ck.t = 6.0                                           # cool-down over, but the token window is still full
    assert ln.wait_time(4000) == pytest.approx(54.0)     # ... until the oldest generation ages out at t=60
    ck.t = 61.0
    assert ln.wait_time(4000) == 0
    before = ln.tpm_limit
    ln.finish(ln.take(4000), input_tokens=500, output_tokens=3500)
    assert ln.tpm_limit == pytest.approx(before * 1.05)  # additive-ish recovery
    ln2 = Lane("k", 1, LaneConfig(tpm=5000), clock=ck)
    ln2.finish(ln2.take(100), input_tokens=50, output_tokens=50)
    assert ln2.tpm_limit == 5000                         # never grows past a configured ceiling


def test_consecutive_429_exponential_cooldown_and_retry_after():
    ck = Clock()
    ln = Lane("l", 0, LaneConfig(base_cooldown_s=10, max_cooldown_s=25), clock=ck)
    assert [ln.rate_limited(None) for _ in range(3)] == [10, 20, 25]
    assert ln.rate_limited(3.0) == 3.0


def test_pool_prefers_ready_lane_and_waits_when_all_busy():
    ck = Clock()
    cfg = LaneConfig(rpm=60, burst=1, max_inflight=5)
    lanes = [Lane(f"k{i}", i, cfg, clock=ck) for i in range(2)]
    pool = LanePool(lanes, clock=ck, sleep=ck.sleep)
    lanes[0].rate_limited(30.0)

    async def go():
        a, ta = await pool.acquire(10, 10)
        b, tb = await pool.acquire(10, 10)            # k1 used, k0 cooling -> waits for k1's bucket (1 s)
        return a, b
    a, b = asyncio.run(go())
    assert a.key_index == 1 and b.key_index == 1 and ck.t == pytest.approx(1.0)


def test_pool_exclude_falls_back_when_single_lane():
    ck = Clock()
    pool = LanePool([Lane("k0", 0, LaneConfig(rpm=6000, burst=10), clock=ck)], clock=ck, sleep=ck.sleep)
    lane, _ = asyncio.run(pool.acquire(1, 1, exclude={0}))
    assert lane.key_index == 0


# ---- S3 2026-11-07: lanes recover after cool-down; a multi-hour server cool-down raises instead of hanging
def test_lane_recovers_after_cooldown():
    ck = Clock()
    ln = Lane("l", 0, LaneConfig(rpm=60, burst=1, max_inflight=1, base_cooldown_s=10), clock=ck)
    t = ln.take(10)
    ln.finish(t, ok=False)
    cd = ln.rate_limited(None)
    assert cd == 10 and ln.wait_time(10) > 0
    ck.t += 61.0                                   # cool-down over and the 60 s token window aged out
    assert ln.wait_time(10) == 0
    t = ln.take(10)
    ln.finish(t, input_tokens=5, output_tokens=5)
    assert ln.consecutive_429 == 0


def test_pool_recovers_after_cooldown_without_raising():
    ck = Clock()
    lanes = [Lane(f"l{i}", i, LaneConfig(rpm=600, burst=1, max_inflight=1, base_cooldown_s=30), clock=ck)
             for i in range(2)]
    for ln in lanes:
        ln.rate_limited(None)                      # every lane cooling down for 30 s
    pool = LanePool(lanes, clock=ck, sleep=ck.sleep, max_block_s=300)
    lane, _ = asyncio.run(pool.acquire(10, 10))
    assert ck.t >= 30 and lane in lanes


def test_pool_raises_on_multi_hour_cooldown():
    from vi_math_verified_aug.common.ratelimit import LaneStall
    ck = Clock()
    lanes = [Lane(f"l{i}", i, LaneConfig(), clock=ck) for i in range(3)]
    for ln in lanes:
        ln.rate_limited(60003.0)                   # Gemini RetryInfo "16h40m3s" (per-day quota)
    pool = LanePool(lanes, clock=ck, sleep=ck.sleep)
    with pytest.raises(LaneStall):
        asyncio.run(pool.acquire(10, 10))
    assert ck.t < 1                                # raised at once, did not wait


def test_pool_unbounded_wait_still_available():
    ck = Clock()
    ln = Lane("l", 0, LaneConfig(), clock=ck)
    ln.rate_limited(1000.0)
    pool = LanePool([ln], clock=ck, sleep=ck.sleep, max_block_s=None)
    asyncio.run(pool.acquire(10, 10))
    assert ck.t >= 1000


def test_watchdog_raises_without_completions():
    from vi_math_verified_aug.common.ratelimit import LaneStall, Watchdog
    ck = Clock()
    wd = Watchdog(300, clock=ck, sleep=ck.sleep)
    ck.t = 200
    wd.tick()
    ck.t = 450
    wd.check()                                     # 250 s idle: fine
    with pytest.raises(LaneStall):
        asyncio.run(wd.run(poll_s=10))
    assert 500 < ck.t <= 520
