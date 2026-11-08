"""Y1: Kaggle job template + renderer.  Nothing is pushed; subprocesses are mocked."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from vi_math_verified_aug.train.kaggle import render as RD  # noqa: E402

CONFIGS = Path(RD.__file__).resolve().parents[1] / "configs"


def load_template(tmp_path, job):
    spec = importlib.util.spec_from_file_location("y1_job_template", RD.HERE / "job_template.py")
    mod = importlib.util.module_from_spec(spec)
    mod.JOB = job
    spec.loader.exec_module(mod)
    mod.RES, mod.CFG_DIR = str(tmp_path / "results"), str(tmp_path / "configs")
    Path(mod.RES).mkdir(parents=True)
    return mod


def test_pins_read_from_requirements():
    pins = RD.read_pins()
    assert pins["peft"] == "0.21.2" and pins["transformers"] == "5.20.2" and pins["vllm"] == "0.33.0"
    assert "trl" not in pins and all("TODO" not in v for v in pins.values())  # trl: isolated Unsloth stack
    assert "trl==1.17.1" in RD.unsloth_install()


def test_render_injects_job_and_pins_run_ids(tmp_path):
    t1 = RD.task_from_config(CONFIGS / "sft_S0_1p7B_lr2e-4_s1.yaml")
    t2 = RD.task_from_config(CONFIGS / "eval_S0_1p7B_s1.yaml", requires=t1["config"]["run_id"])
    assert t1["cmd"] == "sft" and t1["budget_s"] == 39600 and t1["config"]["run_id"].startswith("sft-S0-1p7B-s1-")
    job = RD.build_job([[t1, t2]], "y1-test-slug")
    run_py = RD.render(job, "y1-test-slug", tmp_path)
    src = run_py.read_text(encoding="utf-8")
    assert src.startswith("import json as _json\nJOB = _json.loads(")
    compile(src, str(run_py), "exec")
    assert json.loads((run_py.parent / "job.json").read_text(encoding="utf-8"))["run_id"].startswith("y1-test-slug-")
    with pytest.raises(ValueError, match="2 GPUs"):
        RD.build_job([[], [], []], "x")


def test_template_subst_and_skips(tmp_path):
    job = {"run_id": "r1", "deadline_s": 1000, "lanes": []}
    mod = load_template(tmp_path, job)
    assert mod.subst({"a": ["{results}/x", "{input}/d"]}) == {"a": [f"{mod.RES}/x", "/kaggle/input/d"]}
    rec = mod.run_task(0, {"cmd": "eval", "config": {}, "requires": "missing-run"}, str(tmp_path))
    assert rec["status"] == "SKIPPED_DEPENDENCY"
    mod.T0 -= 10_000  # past the deadline
    assert mod.run_task(0, {"cmd": "sft", "config": {}}, str(tmp_path))["status"] == "SKIPPED_DEADLINE"
    ledger = [json.loads(x) for x in Path(mod.RES, "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in ledger] == ["SKIPPED_DEPENDENCY", "SKIPPED_DEADLINE"]


@pytest.mark.parametrize("rc,status", [(0, "COMPLETE"), (75, "NEEDS_RESUME"), (4, "ABORTED_FP16_OVERFLOW"),
                                       (3, "FAILED"), ("timeout", "TIMEOUT_KILLED")])
def test_template_runs_task_with_budget_and_gpu(tmp_path, monkeypatch, rc, status):
    mod = load_template(tmp_path, {"run_id": "r1", "deadline_s": 40000, "grace_s": 60, "lanes": []})
    seen = {}

    class P:
        pid = 1

        def __init__(self, argv, env, stdout, stderr, start_new_session, cwd=None):
            seen.update(argv=argv, env=env)

        def wait(self, timeout=None):
            if rc == "timeout" and timeout is not None:
                raise mod.subprocess.TimeoutExpired("x", timeout)
            seen["timeout"] = seen.get("timeout", timeout)
            return rc

        def kill(self):
            seen["killed"] = True

    monkeypatch.setattr(mod.subprocess, "Popen", P)
    monkeypatch.setattr(mod.os, "killpg", lambda pid, sig: seen.setdefault("killed", True), raising=False)
    task = {"cmd": "sft", "budget_s": 3000,
            "config": {"kind": "sft", "run_id": "sft-S0-1p7B-s1-abc", "data": {"train": "{input}/d/S0.jsonl"}}}
    rec = mod.run_task(1, task, "/code")
    assert rec["status"] == status and rec["run_id"] == "sft-S0-1p7B-s1-abc"
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "1" and seen["env"]["PYTHONPATH"].startswith("/code")
    assert seen["argv"][2:5] == ["vi_math_verified_aug.train.cli", "sft", "--config"]
    assert seen["argv"][-2:] == ["--budget-s", "3000"]
    conf = json.loads(Path(seen["argv"][5]).read_text(encoding="utf-8"))
    assert conf["out_dir"] == f"{mod.RES}/sft-S0-1p7B-s1-abc" and conf["data"]["train"] == "/kaggle/input/d/S0.jsonl"
    if rc == "timeout":
        assert seen.get("killed")
    else:
        assert seen["timeout"] == 3060  # budget + grace: the child checkpoints and exits 75 before this


def test_bundle_code_excludes_caches(tmp_path):
    dest = RD.bundle_code(tmp_path)
    assert (dest / "train" / "sft.py").exists() and (dest / "evalrun" / "runner.py").exists()
    assert not list(dest.rglob("__pycache__"))


def test_unsloth_install_is_isolated_and_only_for_unsloth_jobs():
    from vi_math_verified_aug.train.kaggle import render as R
    cmd = R.unsloth_install()
    assert cmd.startswith("pip install -q --no-deps --target /tmp/uns_site ") and "unsloth==2026.11.3" in cmd
    assert "unsloth" not in R.read_pins() and R.read_pins()["transformers"] == "5.20.2"
    t_hf = {"cmd": "sft", "config": {"engine": "hf"}}
    t_un = {"cmd": "sft", "config": {"engine": "unsloth"}}
    assert R.build_job([[t_hf]], "x-hf")["unsloth_install"] is None
    assert R.build_job([[t_un]], "x-un")["unsloth_install"] == cmd
