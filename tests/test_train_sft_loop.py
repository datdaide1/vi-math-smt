"""Y1: SFT trainer control flow (WSD branches, resume, skip-abort, exit codes) with the model mocked.

The torch parts of ``SFTRun`` (setup, train_step, checkpoint tensors) are replaced by a fake whose "weights" are
one float updated deterministically from (step, lr, item ids).  Everything else -- the main loop, branch
fork/restore, budget handling, checkpoint discovery/pruning, resume, the >1% skipped-step rule and run()'s exit
codes -- is the real code.  No torch import (laptop RAM rule).
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from vi_math_verified_aug.train import EXIT_ABORT_OVERFLOW, EXIT_NEEDS_RESUME, EXIT_OK  # noqa: E402
from vi_math_verified_aug.train import config as C  # noqa: E402
from vi_math_verified_aug.train import data as D  # noqa: E402
from vi_math_verified_aug.train import schedule as S  # noqa: E402
from vi_math_verified_aug.train import sft as SFT  # noqa: E402


class FakeScaler:
    def is_enabled(self):
        return False


class FakeRun(SFT.SFTRun):
    skip_steps: set = set()
    budget_after: int | None = None  # executed steps after which the wall-clock budget "runs out"

    def setup(self):
        cfg, s = self.cfg, self.s
        self.items = [{"id": f"i{k}", "ids": [1] * (k % 5 + 2), "labels": [-100, 1] + [1] * (k % 5)}
                      for k in range(cfg["budget"]["n_train"])]
        self.fp = D.fingerprint(self.items)
        self.splan = D.StepPlan(len(self.items), s["seq_per_step"], cfg["seed"])
        self.plan = S.plan(self.splan.steps_per_epoch, cfg["budget"]["epochs"], s["branch_epochs"],
                           s["warmup_frac"], s["decay_frac"])
        self.dev_type, self.micro, self.scaler, self.w = "cpu", s["micro_batch"], FakeScaler(), 0.0
        self.lrs = []
        self.st = {"global_step": 0, "branches_done": [], "exec_steps": 0, "exec_skipped": 0,
                   "exec_skipped_first10": 0, "nonfinite_losses": 0, "real_tokens": 0, "padded_tokens": 0,
                   "completion_tokens": 0, "train_s": 0.0, "resumed_from": [], "outputs": {}}

    def train_step(self, step, mult, tag):
        ids = self.splan.step_indices(step)
        skipped = step in self.skip_steps and tag == "main"
        if not skipped:
            self.w = self.w * 0.9 + mult * (sum(ids) % 97 + 1)
        self.lrs.append((step, tag, mult))
        self.st["exec_steps"] += 1
        self.st["exec_skipped"] += int(skipped)
        self.st["real_tokens"] += 1
        self.st["padded_tokens"] += 1
        self._check_skips(final=False)
        return {"loss": 0.0, "lr": mult, "gnorm": 0.0}

    def _budget_left(self):
        if self.budget_after is not None and self.st["exec_steps"] >= self.budget_after:
            return -1.0
        return 1e9

    def save_ckpt(self, tag=""):
        self.ckpt_root.mkdir(parents=True, exist_ok=True)
        name = f"step_{self.st['global_step']:07d}_b{len(self.st['branches_done'])}{('_' + tag) if tag else ''}"
        d = self.ckpt_root / name
        d.mkdir(exist_ok=True)
        (d / "state.json").write_text(json.dumps({**self._ckpt_state(), "w": self.w}), encoding="utf-8")
        self._prune()
        self._last_ckpt_t = 0.0
        return d

    def load_ckpt(self, path):
        st = json.loads((Path(path) / "state.json").read_text(encoding="utf-8"))
        assert st["config_hash"] == self.cfg["config_hash"] and st["fingerprint"] == self.fp
        self.w = st["w"]
        for k in self.st:
            if k in st:
                self.st[k] = st[k]

    def _save_adapter(self, label, extra):
        self.st["outputs"][label] = self.w
        return label


@pytest.fixture(autouse=True)
def no_env_probe(monkeypatch):
    monkeypatch.setattr(SFT, "env_info", lambda: {"fake": True})


def cfg(tmp_path, **sft):
    raw = {"kind": "sft", "arm": "S0", "model": "fake", "size": "tiny", "seed": 3, "precision": "fp32",
           "out_dir": str(tmp_path / "{run_id}"), "data": {"train": "unused.jsonl"},
           "budget": {"n_train": 40, "epochs": 3},
           "sft": {"seq_per_step": 4, "micro_batch": 2, "warmup_frac": 0.1, "decay_frac": 0.1,
                   "ckpt_every_min": 0, "ckpt_every_steps": 4, "skip_check_after": 5, **sft},
           "guards": {"enabled": False}}  # control-flow tests; the guards have their own tests (test_train_guards.py)
    return C.build(raw)


def go(c, **attrs):
    r = FakeRun(c)
    for k, v in attrs.items():
        setattr(r, k, v)
    r.setup()
    return r, r.train()


def test_branches_do_not_perturb_the_main_trajectory(tmp_path):
    r0, s0 = go(cfg(tmp_path / "a"))
    r1, s1 = go(cfg(tmp_path / "b", branch_epochs=[1, 2]))
    assert set(s1["outputs"]) == {"epoch1", "epoch2", "epoch3"}
    assert s1["outputs"]["epoch3"] == s0["outputs"]["epoch3"]  # fork state restored exactly
    main0 = [x for x in r0.lrs]
    main1 = [x for x in r1.lrs if x[1] == "main"]
    assert main0 == main1
    # branch steps follow the decay of a fresh e-epoch run (10 steps/epoch, decay 1 step at epoch 1)
    b1 = [x for x in r1.lrs if x[1] == "branch1"]
    assert b1 == [(9, "branch1", S.lr_mult(9, 10, r1.plan["warmup"], 1))] and b1[0][2] < 1.0
    assert s1["exec_steps"] == 30 + 1 + 2


def test_epoch1_branch_equals_a_fresh_one_epoch_run(tmp_path):
    # with the same warm-up length the branch must equal a fresh 1-epoch WSD run (gate G6 in miniature)
    _, sb = go(cfg(tmp_path / "a", branch_epochs=[1], warmup_frac=0.0))
    c1 = cfg(tmp_path / "b", warmup_frac=0.0)
    c1["budget"]["epochs"] = 1
    _, s1 = go(c1)
    assert sb["outputs"]["epoch1"] == pytest.approx(s1["outputs"]["epoch1"])


def test_budget_stop_then_resume_gives_the_same_models(tmp_path):
    ref = go(cfg(tmp_path / "ref", branch_epochs=[1, 2]))[1]
    for cut in (7, 10, 19):  # before a fork, inside branch 1 region, inside branch 2
        c = cfg(tmp_path / f"cut{cut}", branch_epochs=[1, 2])
        r = FakeRun(c)
        r.budget_after = cut
        r.setup()
        with pytest.raises(SFT.BudgetReached):
            r.train()
        r2 = FakeRun(c)
        r2.setup()
        out = r2.train()
        assert out["resumed_from"], "second session must resume from a checkpoint"
        assert out["outputs"] == pytest.approx(ref["outputs"]), f"cut at {cut}"


def test_checkpoints_are_pruned_but_pending_forks_kept(tmp_path):
    r, _ = go(cfg(tmp_path, branch_epochs=[1, 2], keep_ckpts=2))
    names = sorted(p.name for p in r.ckpt_root.glob("step_*"))
    assert len(names) <= 3 and not any("fork" in n for n in names)  # forks deleted once their branch is done
    assert any(n.endswith("_final") for n in names)


def test_skip_rule_aborts_above_one_percent(tmp_path):
    c = cfg(tmp_path)
    FakeRun.skip_steps = {6}
    try:
        with pytest.raises(SFT.TooManySkippedSteps, match="skipped"):
            go(c)
    finally:
        FakeRun.skip_steps = set()


def test_skip_rule_tolerates_under_one_percent(tmp_path):
    c = cfg(tmp_path, max_skip_frac=0.05)
    FakeRun.skip_steps = {25}  # 1 skip in 26+ executed steps < 5%
    try:
        _, s = go(c)
    finally:
        FakeRun.skip_steps = set()
    assert s["skipped_steps"] == 1 and s["skipped_frac"] == pytest.approx(1 / 30, abs=1e-4)


def test_run_maps_outcomes_to_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(SFT, "SFTRun", FakeRun)
    assert SFT.run(cfg(tmp_path / "ok")) == EXIT_OK
    assert (Path(cfg(tmp_path / "ok")["out_dir"]) / "DONE").exists()
    assert SFT.run(cfg(tmp_path / "ok")) == EXIT_OK  # idempotent: DONE short-circuits
    monkeypatch.setattr(FakeRun, "budget_after", 5)
    assert SFT.run(cfg(tmp_path / "budget")) == EXIT_NEEDS_RESUME
    m = json.loads((Path(cfg(tmp_path / "budget")["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "NEEDS_RESUME"
    monkeypatch.setattr(FakeRun, "budget_after", None)
    monkeypatch.setattr(FakeRun, "skip_steps", {1, 2})
    assert SFT.run(cfg(tmp_path / "ovf")) == EXIT_ABORT_OVERFLOW


def test_fp16_on_cpu_is_refused(tmp_path):
    c = cfg(tmp_path)
    c["precision"] = "fp16"
    c["device"] = "cpu"
    r = SFT.SFTRun(c)
    with pytest.raises(C.ConfigError, match="CUDA"):
        r.setup()


def test_crash_mid_run_then_rerun_resumes_from_the_last_checkpoint(tmp_path):
    """User rule 2026-11-08: an interruption (kernel/container killed) keeps what is done.  A run that dies with an
    exception mid-way leaves its <= 15-min / epoch-boundary checkpoints; re-running the same config resumes from
    the newest one and ends with exactly the uninterrupted model."""
    ref = go(cfg(tmp_path / "ref", branch_epochs=[1]))[1]

    class Crash(RuntimeError):
        pass

    class Crashing(FakeRun):
        crash_at = 13

        def train_step(self, step, mult, tag):
            if tag == "main" and step == self.crash_at and not self.st["resumed_from"]:
                raise Crash("simulated kill")
            return super().train_step(step, mult, tag)

    c = cfg(tmp_path / "crash", branch_epochs=[1])
    r = Crashing(c)
    r.setup()
    with pytest.raises(Crash):
        r.train()
    saved = sorted(p.name for p in (Path(c["out_dir"]) / "ckpt").glob("step_*"))
    assert saved, "no checkpoint survived the crash"
    r2 = Crashing(c)                      # a fresh process: same config, same out_dir
    r2.setup()
    s2 = r2.train()
    assert r2.st["resumed_from"] and s2["outputs"] == ref["outputs"]
    # the epoch-boundary checkpoint (10 steps/epoch) was among those kept
    assert any(n.startswith("step_0000010") for n in saved) or any(n.startswith("step_0000012") for n in saved)
