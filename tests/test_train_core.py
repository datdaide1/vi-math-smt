"""Y1: config / template / data / schedule.  Pure Python: no torch, no model (laptop RAM rule)."""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from vi_math_verified_aug.train import config as C  # noqa: E402
from vi_math_verified_aug.train import data as D  # noqa: E402
from vi_math_verified_aug.train import schedule as S  # noqa: E402
from vi_math_verified_aug.train import template as T  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "vi_math_verified_aug" / "train" / "configs"


def base_sft(**kw):
    raw = {"kind": "sft", "arm": "S0", "model": "m", "size": "1.7B", "seed": 1, "data": {"train": "x.jsonl"},
           "budget": {"n_train": 64, "epochs": 3}}
    raw.update(kw)
    return raw


class CharTok:
    """Fake tokenizer: one id per character; eos=1, pad=0, no BOS."""
    eos_token_id, pad_token_id, bos_token_id, add_bos_token = 1, 0, None, False

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (ord(c) % 1000) for c in text]}


# ------------------------------------------------------------------------------------------------ config
def test_defaults_are_the_t3_recipe():
    cfg = C.build(base_sft())
    s, l, g, e = cfg["sft"], cfg["lora"], cfg["grpo"], cfg["eval"]
    assert (l["r"], l["alpha"], l["dropout"], l["targets"]) == (32, 32, 0.0, "all-linear")
    assert s["seq_per_step"] == 32 and s["max_skip_frac"] == 0.01 and s["packing"] is False
    assert (s["warmup_frac"], s["decay_frac"], s["grad_clip"], s["weight_decay"]) == (0.03, 0.10, 1.0, 0.0)
    assert cfg["precision"] == "fp16"
    assert (g["loss_type"], g["beta"], g["num_generations"], g["prompts_per_step"], g["max_steps"],
            g["max_completion_length"], g["lora_alpha"]) == ("dr_grpo", 0.0, 8, 32, 200, 512, 64)
    assert (e["temperature"], e["top_p"], e["k_samples"], e["greedy"], e["max_new_tokens"]) == (0.6, 0.95, 4, True,
                                                                                                 1024)


def test_run_id_is_deterministic_and_carries_identity():
    a, b = C.build(base_sft()), C.build(base_sft())
    assert a["run_id"] == b["run_id"] and a["run_id"].startswith("sft-S0-1p7B-s1-")
    c = C.build(base_sft(seed=2))
    assert c["run_id"] != a["run_id"]
    assert set(C.identity(a)) >= {"run_id", "arm", "model", "size", "seed"}
    # a dumped+reloaded config keeps its id
    assert C.build(yaml.safe_load(yaml.safe_dump(a)))["run_id"] == a["run_id"]


def test_session_budget_does_not_change_the_hash():
    a = C.build(base_sft())
    b = C.build(base_sft(), {"budget": {"wall_s": 3600}, "out_dir": "/elsewhere/{run_id}"})
    assert a["config_hash"] == b["config_hash"] and a["run_id"] == b["run_id"]
    assert C.build(base_sft(), {"sft": {"lr": 4e-4}})["config_hash"] != a["config_hash"]


def test_unknown_key_and_missing_identity_are_errors():
    with pytest.raises(C.ConfigError, match="unknown config key 'sft.lr_typo'"):
        C.build(base_sft(sft={"lr_typo": 1}))
    raw = base_sft()
    del raw["arm"]
    with pytest.raises(C.ConfigError, match="arm"):
        C.build(raw)


def test_packing_needs_the_unsloth_engine():
    with pytest.raises(C.ConfigError, match="needs engine: unsloth"):
        C.build(base_sft(sft={"packing": True}))
    cfg = C.build({**base_sft(sft={"packing": True}), "engine": "unsloth"})
    assert cfg["engine"] == "unsloth" and cfg["sft"]["packing"] is True
    with pytest.raises(C.ConfigError, match="engine"):
        C.build({**base_sft(), "engine": "deepspeed"})


def test_collate_packed_never_crosses_items():
    a = {"ids": [5, 6, 7], "labels": [-100, 6, 7], "n_completion": 2}
    b = {"ids": [8, 9], "labels": [-100, 9], "n_completion": 1}
    p = D.collate_packed([a, b])
    assert p["input_ids"] == [[5, 6, 7, 8, 9]] and p["position_ids"] == [[0, 1, 2, 0, 1]]
    assert p["labels"] == [[-100, 6, 7, -100, 9]] and p["packed_seq_lengths"] == [3, 2]
    # the shifted targets of the packed row = the items' own completion targets (no boundary target)
    assert sum(1 for x in p["labels"][0][1:] if x != -100) == D.n_completion_tokens([a, b]) == 3
    with pytest.raises(AssertionError):
        D.collate_packed([{"ids": [1, 2], "labels": [1, 2]}])


def test_bos_prefix_detects_tokenizer_bos():
    from vi_math_verified_aug.train import modeling as M

    class Tok:
        def __init__(self, bos):
            self.bos = bos

        def __call__(self, text, add_special_tokens=True):
            ids = [ord(c) for c in text]
            return {"input_ids": ([1] if (self.bos and add_special_tokens) else []) + ids}

    assert M.bos_prefix(Tok(True)) == [1] and M.bos_prefix(Tok(False)) == []
    assert M.family_of("LiquidAI/LFM2.5-2.6B-Base") == "lfm2" and M.family_of("Qwen/Qwen3-0.6B-Base") == "qwen3"


def test_branch_epochs_must_be_inside_the_run():
    with pytest.raises(C.ConfigError, match="branch_epochs"):
        C.build(base_sft(sft={"branch_epochs": [3]}))
    assert C.build(base_sft(sft={"branch_epochs": [1, 2]}))["sft"]["branch_epochs"] == [1, 2]


def test_grpo_micro_batch_must_divide_rollouts():
    raw = {"kind": "grpo", "arm": "S0", "model": "m", "size": "1.7B", "seed": 1, "data": {"train": "p.jsonl"},
           "grpo": {"micro_batch": 24}}
    with pytest.raises(C.ConfigError, match="micro_batch"):
        C.build(raw)


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_example_configs_validate(path):
    cfg = C.load(path)
    assert cfg["kind"] in ("sft", "grpo", "eval") and cfg["run_id"]


# ------------------------------------------------------------------------------------------------ template
def test_template_strings():
    assert T.prompt(" 1+1? ") == "Bài toán:\n1+1?\n\nLời giải:\n"
    assert T.completion("a\nb\n#### 18", "18") == "a\nb\nĐáp số: \\boxed{18}"
    assert T.completion("x\nĐáp số: 5", "5") == "x\nĐáp số: \\boxed{5}"
    assert T.completion("Vậy \\boxed{\\frac{1}{2}}.", None).endswith("Đáp số: \\boxed{\\frac{1}{2}}")
    with pytest.raises(ValueError):
        T.completion("no answer here", None)
    assert T.cut_at_stop("ans\n\nBài toán:\nnext") == "ans"


# ------------------------------------------------------------------------------------------------ data
def rows(n, long_every=0):
    out = []
    for i in range(n):
        sol = "y " * (300 if long_every and i % long_every == 0 else (i % 7 + 1))
        out.append({"id": f"r{i}", "question": f"q{i}", "solution": sol, "answer_canonical": str(i)})
    return out


def test_encode_masks_prompt_and_trains_eos():
    tok = CharTok()
    enc = D.encode(tok, "q", "s", "3", max_length=10 ** 6)
    n_prompt = len(T.prompt("q"))
    assert enc["labels"][:n_prompt] == [-100] * n_prompt
    assert enc["labels"][n_prompt:] == enc["ids"][n_prompt:]
    assert enc["ids"][-1] == tok.eos_token_id and enc["labels"][-1] == tok.eos_token_id


def test_prepare_pool_drops_never_truncates_and_logs_rate():
    tok = CharTok()
    items, st = D.prepare_pool(rows(40, long_every=4), tok, max_length=200)
    assert st["n_dropped_too_long"] == 10 and st["drop_rate"] == 0.25 and st["n_eligible"] == 30
    assert all(len(it["ids"]) <= 200 for it in items)
    full = {r["id"]: D.encode(tok, r["question"], r["solution"], r["answer_canonical"], 10 ** 6)
            for r in rows(40, long_every=4)}
    assert all(it["ids"] == full[it["id"]]["ids"] for it in items)  # kept items are untouched (no truncation)


def test_subsample_is_seeded():
    items, _ = D.prepare_pool(rows(100), CharTok(), 512)
    a, b, c = D.subsample(items, 40, 1), D.subsample(items, 40, 1), D.subsample(items, 40, 2)
    assert [x["id"] for x in a] == [x["id"] for x in b] != [x["id"] for x in c]
    assert D.fingerprint(a) == D.fingerprint(b) != D.fingerprint(c)
    assert len(D.subsample(items, 101, 1)) == 100  # QA-T B3: shortfall tolerated here, judged by SFTRun (> 1 % aborts)


def test_step_plan_32_per_step_seeded_and_resumable():
    p = D.StepPlan(100, 32, seed=5)
    assert p.steps_per_epoch == 3
    e0 = [p.step_indices(s) for s in range(3)]
    assert all(len(x) == 32 for x in e0) and len(set(sum(e0, []))) == 96
    assert D.StepPlan(100, 32, seed=5).step_indices(4) == p.step_indices(4)  # fresh object (resume) agrees
    assert p.step_indices(3) != p.step_indices(0)  # new permutation each epoch
    assert D.StepPlan(100, 32, seed=6).step_indices(0) != p.step_indices(0)


def test_micro_batches_length_grouped_and_padded_not_packed():
    items, _ = D.prepare_pool(rows(32), CharTok(), 512)
    mbs = D.micro_batches(items, 8)
    assert [len(m) for m in mbs] == [8, 8, 8, 8]
    lens = [len(it["ids"]) for m in mbs for it in m]
    assert lens == sorted(lens, reverse=True)
    b = D.collate(mbs[0], pad_id=0)
    L = max(len(it["ids"]) for it in mbs[0])
    assert all(len(r) == L for r in b["input_ids"])  # one sequence per row
    assert "position_ids" not in b
    for it, ids, att, lab in zip(mbs[0], b["input_ids"], b["attention_mask"], b["labels"]):
        n = len(it["ids"])
        assert ids[:n] == it["ids"] and att == [1] * n + [0] * (L - n) and lab[n:] == [-100] * (L - n)
    assert D.n_completion_tokens(items) == sum(it["n_completion"] for it in items)


def test_length_stats_and_max_length_suggestion():
    items, _ = D.prepare_pool(rows(50), CharTok(), 512)
    st = D.length_stats(items)
    assert st["n_items"] == 50 and st["completion_tokens_p99"] >= st["completion_tokens_mean"]
    assert D.suggest_max_length([100] * 99 + [1000]) == 128


# ------------------------------------------------------------------------------------------------ schedule
def test_wsd_shape():
    total, W, Dd = 100, 3, 10
    m = [S.lr_mult(s, total, W, Dd) for s in range(total)]
    assert m[0] == pytest.approx(1 / 3) and m[2] == 1.0 and m[50] == 1.0 and m[89] == 1.0
    assert m[90] < 1.0 and m[-1] == pytest.approx(1 / 11)
    assert all(a > b for a, b in zip(m[90:], m[91:]))


def test_branch_plan_matches_fresh_runs():
    p = S.plan(steps_per_epoch=100, epochs=3, branch_epochs=[1, 2], warmup_frac=0.03, decay_frac=0.1)
    assert p["total"] == 300 and p["warmup"] == 9 and p["decay"] == 30
    b1, b2 = p["branches"]
    assert (b1["fork_step"], b1["end_step"], b1["decay"]) == (90, 100, 10)  # = a fresh 1-epoch run's decay
    assert (b2["fork_step"], b2["end_step"], b2["decay"]) == (180, 200, 20)
    assert S.branch_mult(90, b1, 9) == S.lr_mult(90, 100, 9, 10)
    with pytest.raises(ValueError):
        S.plan(steps_per_epoch=2, epochs=3, branch_epochs=[1], warmup_frac=0.5, decay_frac=0.1)
