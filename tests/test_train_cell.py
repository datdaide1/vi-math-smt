"""Single process per cell (sft.run_cell): arm k trained after arms 1..k-1 in the same process must equal a fresh
process run of the same config (mock model; the GPU parity check is jobs/t2/cell_parity.json)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from vi_math_verified_aug.train import sft as SFT  # noqa: E402
from test_train_sft_loop import FakeRun, cfg  # noqa: E402


def test_cell_arms_equal_fresh_process_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(SFT, "SFTRun", FakeRun)
    monkeypatch.setattr(SFT, "env_info", lambda: {"fake": True})
    arms = [cfg(tmp_path / f"cell{i}", lr=1e-4 * (i + 1), branch_epochs=[1]) for i in range(3)]
    fresh = [cfg(tmp_path / f"fresh{i}", lr=1e-4 * (i + 1), branch_epochs=[1]) for i in range(3)]
    rcs = SFT.run_cell(arms)
    assert rcs == [0, 0, 0]
    for a, f in zip(arms, fresh):
        assert SFT.run(f) == 0
        ma = json.loads((Path(a["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))["summary"]
        mf = json.loads((Path(f["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))["summary"]
        assert ma["outputs"].keys() == mf["outputs"].keys()
        assert ma["exec_steps"] == mf["exec_steps"] and ma["outputs"]["epoch3"] == mf["outputs"]["epoch3"]


def test_cell_continues_after_a_failed_arm(tmp_path, monkeypatch):
    class Boom(FakeRun):
        def train_step(self, step, mult, tag):
            if self.cfg["sft"]["lr"] > 2.5e-4 and step == 3:
                raise RuntimeError("arm crash")
            return super().train_step(step, mult, tag)
    monkeypatch.setattr(SFT, "SFTRun", Boom)
    monkeypatch.setattr(SFT, "env_info", lambda: {"fake": True})
    arms = [cfg(tmp_path / f"c{i}", lr=1e-4 * (i + 1)) for i in range(3)]
    assert SFT.run_cell(arms) == [0, 0, 3]


def test_seed0_cell_stops_at_the_first_guard_failure(tmp_path, monkeypatch):
    """Lead: in a seed-0 (canary) cell a guard failure stops the remaining arms (exit 9, manifest CELL_STOPPED)."""
    from vi_math_verified_aug.train import EXIT_CELL_STOPPED, EXIT_GUARD_ABORT
    from vi_math_verified_aug.train import guards as G

    class Abort(FakeRun):
        def train_step(self, step, mult, tag):
            if self.cfg["sft"]["lr"] > 1.5e-4 and step == 2:
                raise G.GuardAbort("LOSS_COLLAPSE", "test")
            return super().train_step(step, mult, tag)
    monkeypatch.setattr(SFT, "SFTRun", Abort)
    monkeypatch.setattr(SFT, "env_info", lambda: {"fake": True})
    arms = [cfg(tmp_path / f"s{i}", lr=1e-4 * (i + 1)) for i in range(4)]
    assert SFT.run_cell(arms, stop_on_fail=True) == [0, EXIT_GUARD_ABORT, EXIT_CELL_STOPPED, EXIT_CELL_STOPPED]
    m = json.loads((Path(arms[3]["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "CELL_STOPPED"
    # without stop_on_fail (seeds >= 1) the cell continues
    arms2 = [cfg(tmp_path / f"t{i}", lr=1e-4 * (i + 1)) for i in range(3)]
    assert SFT.run_cell(arms2) == [0, EXIT_GUARD_ABORT, EXIT_GUARD_ABORT]
