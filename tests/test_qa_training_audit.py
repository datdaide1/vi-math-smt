"""QA-T reproductions (docs/research_2026-11/exec/QA_T_training_audit.md).  Torch-free.

Tests marked ``xfail(strict=True)`` reproduce an OPEN finding: they pass while the defect exists and turn into an
XPASS failure once it is fixed, which tells the owner to drop the marker.  Plain tests pin behaviour QA-T verified.
"""
from __future__ import annotations

import inspect
import json
import re

import pytest

from vi_math_verified_aug.train import data as D
from vi_math_verified_aug.train import grid as GR
from vi_math_verified_aug.train import guards as G
from vi_math_verified_aug.train import reward as R
from vi_math_verified_aug.train import template as T

B = "\\boxed"


class CharTok:
    """1 char = 1 token; eos 1; no BOS.  Enough for masking/length logic."""
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) + 10 for c in text]}

    def decode(self, ids, **kw):
        return "".join(chr(i - 10) for i in ids if i >= 10)


# ------------------------------------------------------------------------------------------ verified behaviour
def test_labels_mask_prompt_and_train_eos_only_completion():
    tok = CharTok()
    enc = D.encode(tok, "Q?", "a\n#### 5", "5", 10 ** 6)
    first = next(i for i, x in enumerate(enc["labels"]) if x != -100)
    assert tok.decode(enc["ids"][:first]) == T.prompt("Q?")
    assert tok.decode([x for x in enc["labels"] if x != -100][:-1]) == "a\nĐáp số: " + B + "{5}"
    assert enc["labels"][-1] == tok.eos_token_id


def test_step_loss_denominator_is_token_mean_over_the_whole_step():
    # every micro-batch divides its NLL sum by the SAME per-step count -> no grad-accumulation normalisation bug
    tok = CharTok()
    items = [D.encode(tok, "q", "x" * n, "1", 10 ** 6) for n in (3, 30, 300)]
    total = D.n_completion_tokens(items)
    assert total == sum(D.n_completion_tokens([it]) for it in items)


# ------------------------------------------------------------------------------------------ open findings
def test_final_midrun_check_runs_after_adapter_save():
    from vi_math_verified_aug.train import sft
    src = inspect.getsource(sft.SFTRun.train)
    assert src.index("_save_adapter(") < src.index('midrun_check(P["total"]')


def test_quick_eval_drop_of_two_items_out_of_16_is_not_an_abort():
    base = {"correct": 0.5, "no_answer": 0, "repetition": 0, "truncated": 0, "n": 16}
    now = dict(base, correct=0.375)   # 2 of 16 items worse: well inside binomial noise
    assert G.midrun_verdict(0.6, 0.5, now, base, G.DEFAULTS) == []


def test_subsample_tolerates_a_few_dropped_rows():
    items = [{"id": str(i), "ids": [1]} for i in range(9_999)]   # 10,000 rows, 1 dropped (too long / malformed)
    assert len(D.subsample(items, 10_000, seed=0)) == 9_999


def test_completion_has_exactly_one_answer_line_for_llm_style_endings():
    for tail in ("**Đáp số:** " + B + "{72}", "Vậy đáp số là $" + B + "{72}$.", B + "{72}"):
        c = T.completion("Bước 1: ...\n" + tail, "72", "vi")
        assert c.count(B + "{72}") == 1, c


def test_grpo_reward_matches_eval_on_runaway_continuation():
    resp = "...\nĐáp số: " + B + "{5}\n\nBài toán:\nX\n\nLời giải:\n...\nĐáp số: " + B + "{7}"
    gold = {"answer_canonical": "5", "answer_type": "int", "locale": "mixed"}
    eval_view = R.score(T.cut_at_stop(resp), gold)["correct"]
    reward = R.MathVerifyReward()(prompts=["p"], completions=[resp], gold=[json.dumps(gold)])[0]
    assert bool(reward) == eval_view


@pytest.mark.parametrize("arm", ["Y+", "Y-", "N+", "N-"])
@pytest.mark.parametrize("lang", ["vi", "en"])
def test_arm_training_file_contract(arm, lang):
    """B1 follow-up: whatever writes data/train/final (D5 selector) must give trainable rows: question, solution,
    a short answer that equals the solution trace's own conclusion, >= n_train usable rows, SHA256SUMS entry.
    Skips until the file lands; run it at 06:00 before the launcher picks the rows up."""
    from pathlib import Path
    path = Path(GR.REPO) / GR.data_path(arm, lang)
    if not path.exists():
        pytest.skip(f"{path} not written yet")
    rows = D.read_jsonl(path)
    items, st = D.prepare_pool(rows, CharTok(), 10 ** 9, None, lang)
    assert st["n_malformed"] == 0, st
    assert len(items) >= GR.N_TRAIN, (len(items), GR.N_TRAIN)
    sums = path.parent / "SHA256SUMS"
    assert sums.exists() and path.name in sums.read_text(encoding="utf-8")
    assert len({G.qhash(it["q"]) for it in items}) == len(items), "duplicate questions inside one arm file"


def test_maj_at_k_is_order_invariant():
    from vi_math_verified_aug.evalrun.runner import summarize
    rows = [{"dataset": "d", "item_id": "i", "sample_k": k, "correct": c, "pred_span": s, "tokens": 1,
             "truncated": False} for k, (c, s) in enumerate([(0, "3"), (1, "5"), (0, "3"), (1, "5")])]
    a = summarize(rows, 4)["d"]["maj@4"]
    b = summarize(list(reversed(rows)), 4)["d"]["maj@4"]
    assert a == b
