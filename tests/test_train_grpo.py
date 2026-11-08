"""Y1: GRPO stage + rewards with TRL, datasets, the model and the grader mocked (no torch import)."""
import dataclasses
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from vi_math_verified_aug.train import EXIT_NEEDS_RESUME, EXIT_OK  # noqa: E402
from vi_math_verified_aug.train import config as C  # noqa: E402
from vi_math_verified_aug.train import grpo as G  # noqa: E402
from vi_math_verified_aug.train import reward as R  # noqa: E402
from vi_math_verified_aug.train import template as T  # noqa: E402


class CharTok:
    eos_token_id, pad_token_id, bos_token_id = 1, 0, None
    padding_side = "right"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [5] * len(text)}


def write_pool(path: Path, n=6, long_q=False):
    rows = [{"id": f"p{i}", "question": f"Tính {i}+{i}?", "gold": {"answer_canonical": str(2 * i),
                                                                       "answer_type": "int"}} for i in range(n)]
    rows.append({"id": "plain", "question": "1+1?", "ground_truth": "2"})
    rows.append({"id": "bad", "question": ""})
    if long_q:
        rows.append({"id": "long", "question": "x" * 2000, "ground_truth": "1"})
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def gcfg(tmp_path, **grpo):
    raw = {"kind": "grpo", "arm": "S0", "model": "fake", "size": "1.7B", "seed": 4, "precision": "fp16",
           "out_dir": str(tmp_path / "{run_id}"), "data": {"train": str(write_pool(tmp_path / "pool.jsonl",
                                                                                     long_q=True))},
           "grpo": grpo}
    return C.build(raw)


# ------------------------------------------------------------------------------------------------ args
def test_grpo_args_are_the_t3_recipe(tmp_path):
    c = gcfg(tmp_path)
    a = G.grpo_args(c, Path(c["out_dir"]), "cuda:0")
    assert a["loss_type"] == "dr_grpo" and a["beta"] == 0.0 and a["scale_rewards"] == "none"
    assert a["num_generations"] == 8 and a["max_steps"] == 200 and a["max_completion_length"] == 512
    assert a["per_device_train_batch_size"] * a["gradient_accumulation_steps"] == 32 * 8  # 32 prompts x G=8
    assert a["fp16"] is True and a["bf16"] is False and a["use_cpu"] is False
    assert (a["temperature"], a["top_p"], a["learning_rate"], a["warmup_steps"]) == (1.0, 1.0, 2e-5, 10)
    assert a["lr_scheduler_type"] == "constant_with_warmup"
    assert (a["epsilon"], a["epsilon_high"], a["num_iterations"], a["mask_truncated_completions"]) == (
        0.2, 0.28, 1, True)
    assert a["seed"] == a["data_seed"] == 4


def test_use_vllm_auto_without_vllm(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)  # import vllm -> ImportError
    c = gcfg(tmp_path)
    assert G.grpo_args(c, Path(c["out_dir"]), "cuda:0")["use_vllm"] is False


def test_make_config_refuses_unknown_trl_fields(monkeypatch):
    @dataclasses.dataclass
    class GRPOConfig:
        output_dir: str = ""
        beta: float = 0.0

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(GRPOConfig=GRPOConfig))
    assert G.make_config({"output_dir": "x", "beta": 0.0}).beta == 0.0
    with pytest.raises(RuntimeError, match="max_prompt_length"):
        G.make_config({"output_dir": "x", "max_prompt_length": 512})


def test_prompt_rows_use_the_template_and_drop_long_prompts(tmp_path):
    c = gcfg(tmp_path, max_prompt_length=100)
    rows, st = G.build_prompt_rows(c, CharTok())
    assert st == {"n_pool": 9, "n_malformed": 1, "n_dropped_prompt_too_long": 1, "n_prompts": 7}
    assert rows[0]["prompt"] == T.prompt("Tính 0+0?")
    assert json.loads(rows[0]["gold"])["answer_canonical"] == "0" and rows[-1]["gold"] == "2"


# ------------------------------------------------------------------------------------------------ rewards
@pytest.fixture
def fake_grader(monkeypatch):
    """A grader stub with X1's public API: grade(pred, gold, locale) -> {correct, span, flag}."""
    import re

    def grade(pred, gold, locale=None):
        g = gold["answer_canonical"] if isinstance(gold, dict) else str(gold)
        m = re.findall(r"\\boxed\{([^}]*)\}", pred)
        return {"correct": bool(m) and m[-1] == g, "span": m[-1] if m else None, "flag": None if m else "no-answer"}

    mod = types.SimpleNamespace(grade=grade, GRADER_VERSION="stub", BACKEND="stub")
    monkeypatch.setattr(R, "_grader", lambda: mod)
    return mod


def test_math_verify_reward_is_binary(fake_grader):
    rf = R.MathVerifyReward()
    logged = {}
    out = rf(["p"] * 3, ["Đáp số: \\boxed{4}", "\\boxed{5}", "no box"],
             [json.dumps({"answer_canonical": "4"}), "4", "4"], log_metric=lambda k, v: logged.setdefault(k, v))
    assert out == [1.0, 0.0, 0.0] and logged["graded_no_answer_frac"] == pytest.approx(1 / 3)
    conv = rf(["p"], [[{"role": "assistant", "content": "\\boxed{4}"}]], ["4"])
    assert conv == [1.0]


def test_score_never_raises(monkeypatch):
    def boom():
        raise ImportError("grader being rewritten")
    monkeypatch.setattr(R, "_grader", boom)
    v = R.score("\\boxed{1}", "1")
    assert v["correct"] is False and v["flag"].startswith("grader-error:ImportError")


def test_random_reward_is_seeded_balanced_and_ignores_correctness(fake_grader):
    rr = R.RandomReward(seed=1)
    st = types.SimpleNamespace(global_step=3)
    logged = {}
    comps = ["\\boxed{4}"] * 256
    a = rr(["p"] * 256, comps, ["4"] * 256, trainer_state=st, log_metric=lambda k, v: logged.setdefault(k, v))
    b = R.RandomReward(seed=1)(["p"] * 256, comps, ["4"] * 256, trainer_state=st)
    c = R.RandomReward(seed=2)(["p"] * 256, comps, ["4"] * 256, trainer_state=st)
    assert a == b != c  # resumable (keyed by seed, step, row) and seed-dependent
    assert 0.4 < sum(a) / len(a) < 0.6 and set(a) == {0.0, 1.0}
    assert logged["true_accuracy"] == 1.0  # every completion was correct, rewards still random


# ------------------------------------------------------------------------------------------------ run()
@pytest.fixture
def mocked_stack(monkeypatch, tmp_path):
    calls = {}

    class Trainer:
        def __init__(self, model, reward_funcs, args, train_dataset, processing_class, callbacks):
            calls.update(model=model, reward_funcs=reward_funcs, args=args, data=train_dataset, cbs=callbacks)
            self.model = model
            self.state = types.SimpleNamespace(global_step=0, log_history=[])

        def train(self, resume_from_checkpoint=None):
            calls["resume"] = resume_from_checkpoint
            stop = calls.get("stop_at")
            self.state.global_step = stop or self.args_max
            self.state.log_history = [{"step": 1, "reward": 0.5}]
            if stop:
                type(calls["cbs"][0]).hit = True

    @dataclasses.dataclass
    class Cfg:
        pass

    def make_config(args):
        calls["cfg_args"] = args
        Trainer.args_max = args["max_steps"]
        return args

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(GRPOTrainer=Trainer))
    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(Dataset=types.SimpleNamespace(from_list=lambda rows: rows)))
    monkeypatch.setattr(G, "make_config", make_config)
    monkeypatch.setattr(G, "env_info", lambda: {})
    monkeypatch.setattr(G, "set_seed", lambda s: calls.setdefault("seeds", []).append(s))

    class Budget:
        hit = False

        def __new__(cls, deadline):
            calls["deadline"] = deadline
            Budget.hit = False
            return object.__new__(Budget)

    monkeypatch.setattr(G, "_Budget", Budget)
    M = G.M
    monkeypatch.setattr(M, "pick_device", lambda d: "cuda:0")
    monkeypatch.setattr(M, "resolve_model", lambda name, cache_root: f"/models/{name}")
    monkeypatch.setattr(M, "load_tokenizer", lambda p: CharTok())
    monkeypatch.setattr(M, "load_model", lambda p, prec, dev, attn: ("base", p, prec))
    monkeypatch.setattr(M, "add_lora", lambda base, r, a, d, t: ("lora", base, r, a))
    monkeypatch.setattr(M, "save_adapter", lambda m, p, tok, meta: Path(p).mkdir(parents=True, exist_ok=True))
    return calls


def test_run_end_to_end_mocked(tmp_path, mocked_stack):
    c = gcfg(tmp_path, sft_model="/sft/merged-S0", reward="math_verify")
    assert G.run(c) == EXIT_OK
    out = Path(c["out_dir"])
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "COMPLETE" and m["global_step"] == 200 and (out / "DONE").exists()
    assert mocked_stack["model"] == ("lora", ("base", "/models//sft/merged-S0", "fp16"), 32, 64)
    assert isinstance(mocked_stack["reward_funcs"][0], R.MathVerifyReward)
    assert mocked_stack["seeds"] == [4, 4]  # global seed, then again right before the LoRA init
    assert (out / "train_log.jsonl").exists()


def test_run_random_reward_and_budget_resume(tmp_path, mocked_stack):
    c = gcfg(tmp_path, reward="random")
    c["budget"]["wall_s"] = 3600
    mocked_stack["stop_at"] = 120
    assert G.run(c) == EXIT_NEEDS_RESUME
    assert isinstance(mocked_stack["reward_funcs"][0], R.RandomReward)
    out = Path(c["out_dir"])
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["status"] == "NEEDS_RESUME"
    (out / "trainer" / "checkpoint-100").mkdir(parents=True)
    (out / "trainer" / "checkpoint-75").mkdir(parents=True)
    mocked_stack["stop_at"] = None
    assert G.run(c) == EXIT_OK
    assert mocked_stack["resume"].endswith("checkpoint-100")


def test_last_checkpoint(tmp_path):
    assert G.last_checkpoint(tmp_path) is None
    for n in ("checkpoint-25", "checkpoint-200", "checkpoint-50", "checkpoint-x"):
        (tmp_path / n).mkdir()
    assert G.last_checkpoint(tmp_path).endswith("checkpoint-200")
