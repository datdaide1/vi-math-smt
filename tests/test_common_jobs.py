"""common/jobs.py — idempotent ids, checked transitions, deps, events, budget counters."""
from __future__ import annotations

import json
import threading

import pytest

from vi_math_verified_aug.common.config import RunConfig
from vi_math_verified_aug.common.jobs import (IllegalTransition, JobLedger, LedgerError, StaleTransition,
                                              TRANSITIONS, job_id)


def spec(arm="S1", seed=1, **kw):
    d = {"base": "Qwen3-0.6B", "arm": arm, "n_train": 1000, "seed": seed, "recipe_ver": "v1", "data_ver": "v1",
         "prompt_ver": "v1"}
    d.update(kw)
    return d


@pytest.fixture
def led(tmp_path):
    lg = JobLedger(tmp_path / "ledger.sqlite")
    yield lg
    lg.close()


def test_idempotent_submit_and_id_matches_runconfig(led):
    jid, new = led.submit("sft", spec())
    assert new and len(jid) == 40
    assert led.submit("sft", spec()) == (jid, False)
    assert led.submit("sft", spec(extra_note="not identity")) == (jid, False)   # non-identity fields ignored
    assert led.submit("eval", spec())[0] != jid                                   # kind is identity
    cfg = RunConfig(arm="S1", lang="vi", model="Qwen3-0.6B", seed=1, size=1000, kind="sft")
    assert cfg.job_id() == jid == job_id("sft", cfg.to_spec())
    led.transition(jid, "SUBMITTED")
    led.transition(jid, "RUNNING")
    led.transition(jid, "COMPLETE")
    assert led.submit("sft", spec()) == (jid, False) and led.get(jid)["status"] == "COMPLETE"


def test_transitions_checked(led):
    jid, _ = led.submit("gen", spec())
    assert led.get(jid)["status"] == "READY"
    with pytest.raises(IllegalTransition):
        led.transition(jid, "COMPLETE")                    # READY -> COMPLETE skips submission
    j = led.transition(jid, "SUBMITTED", external_ref="slot-1/v3")
    assert j["attempts"] == 1 and j["external_ref"] == "slot-1/v3"
    with pytest.raises(StaleTransition):
        led.transition(jid, "RUNNING", expect="READY")
    j = led.transition(jid, "RUNNING", expect="SUBMITTED")
    assert j["started"]
    j = led.transition(jid, "NEEDS_RESUME", ckpt_uri="hf://ckpt/1", error="12h cap")
    assert j["ckpt_uri"] == "hf://ckpt/1" and j["last_error"] == "12h cap"
    led.transition(jid, "SUBMITTED")
    j = led.transition(jid, "COMPLETE", actual_cost=1.25)
    assert j["attempts"] == 2 and j["finished"] and j["actual_cost"] == 1.25 and j["last_error"] is None
    for to in TRANSITIONS:
        with pytest.raises((IllegalTransition, LedgerError)):
            led.transition(jid, to)                        # COMPLETE is terminal
    evs = led.events(jid)
    assert [e["to_status"] for e in evs] == ["READY", "SUBMITTED", "RUNNING", "NEEDS_RESUME", "SUBMITTED",
                                             "COMPLETE"]
    mirror = [json.loads(x) for x in led.events_path.read_text(encoding="utf-8").splitlines()]
    assert [e["to_status"] for e in mirror if e["job_id"] == jid] == [e["to_status"] for e in evs]


def test_deps_ready_blocked_retry(led):
    a, _ = led.submit("gen", spec("A"))
    b, _ = led.submit("sft", spec("B"), deps=[a])
    assert led.get(b)["status"] == "PENDING"
    with pytest.raises(LedgerError):
        led.submit("sft", spec("C"), deps=["nope"])
    led.transition(a, "SUBMITTED")
    led.transition(a, "FAILED", error="OOM at step 0")
    assert led.refresh_ready() == {"READY": [], "BLOCKED": [b]}
    assert led.get(b)["last_error"].startswith("dep_failed:")
    led.retry(a)
    led.transition(a, "SUBMITTED")
    led.transition(a, "COMPLETE")
    assert led.refresh_ready() == {"READY": [b], "BLOCKED": []}
    assert [j["id"] for j in led.ready()] == [b]


def test_budget_blocked_jobs_not_auto_released(led):
    a, _ = led.submit("sft", spec())
    led.transition(a, "BLOCKED", error="modal cap")
    assert led.refresh_ready() == {"READY": [], "BLOCKED": []}


def test_ready_order_and_filters(led):
    lo, _ = led.submit("eval", spec("lo"), priority=1, platform="kaggle")
    hi, _ = led.submit("eval", spec("hi"), priority=9, platform="modal")
    assert [j["id"] for j in led.ready()] == [hi, lo]
    assert [j["id"] for j in led.ready(platform="kaggle")] == [lo]
    assert led.counts()["READY"] == 2


def test_platform_is_derived_not_chosen(tmp_path):
    lg = JobLedger(tmp_path / "l.sqlite", platform_for=lambda kind, s: "modal" if "1.7B" in s["base"] else "kaggle")
    jid, _ = lg.submit("sft", spec(base="Qwen3-1.7B"))
    assert lg.get(jid)["platform"] == "modal"
    with pytest.raises(LedgerError):
        lg.submit("sft", spec("X"), platform="laptop")
    lg.close()


def test_budget_counters_and_concurrent_submits(led):
    assert led.add_budget("modal_usd", 1.5) == 1.5
    assert led.add_budget("modal_usd", 2.0) == 3.5
    led.set_budget("kaggle_h", 30)
    assert led.get_budget("kaggle_h") == 30 and led.get_budget("missing") == 0
    res = []

    def work():
        res.append(led.submit("verify", spec("same")))
    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(1 for _, new in res if new) == 1 and len({j for j, _ in res}) == 1
