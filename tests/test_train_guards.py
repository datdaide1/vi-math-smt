"""T2 fail-fast guards (train/guards.py + their wiring in train/sft.py), all on mocks -- no torch."""
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from vi_math_verified_aug.train import EXIT_GUARD_ABORT  # noqa: E402
from vi_math_verified_aug.train import config as C  # noqa: E402
from vi_math_verified_aug.train import data as D  # noqa: E402
from vi_math_verified_aug.train import guards as G  # noqa: E402
from vi_math_verified_aug.train import template as T  # noqa: E402

GD = dict(G.DEFAULTS)
NL = chr(10)


class CharTok:
    """Exact char-level tokenizer: decode(encode(x)) == x; eos = 1, pad = 0."""
    eos_token_id, pad_token_id, bos_token_id = 1, 0, None

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) + 10 for c in text]}

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(i - 10) for i in ids if i >= 10)


ROWS = [{"id": f"r{i}", "question": f"Lan có {i} quả táo, mua thêm 3 quả. Hỏi có bao nhiêu quả?",
         "solution": f"Lan có {i} + 3 = {i + 3} quả.\n#### {i + 3}", "answer_canonical": str(i + 3)} for i in range(8)]


def items(rows=ROWS, lang="vi"):
    its, _ = D.prepare_pool(rows, CharTok(), 4096, lang=lang)
    return its


def pre(its, lang="vi", g=GD):
    tok = CharTok()
    return G.preflight_items(its, tok.decode, prompt_fn=lambda q: T.prompt(q, lang),
                             completion_fn=lambda s, a: T.completion(s, a, lang), eos_id=1, pad_id=0, g=g)


def test_preflight_passes_and_decodes_exactly_the_completion():
    for lang in ("vi", "en"):
        r = pre(items(lang=lang), lang)
        assert len(r["decoded"]) == 3
        e = r["decoded"][0]
        assert e["trained_text"] == T.completion(ROWS[0]["solution"], "3", lang)
        assert "\\boxed{3}" in e["trained_text"] and "Bài toán" not in e["trained_text"]
        assert 0.15 <= r["label_frac_min"] <= r["label_frac_max"] <= 0.95


def test_preflight_catches_prompt_in_labels():
    its = items()
    for it in its:
        it["labels"] = list(it["ids"])          # the classic bug: nothing masked
    with pytest.raises(G.GuardAbort) as e:
        pre(its)
    assert e.value.reason in ("PREFLIGHT_LABEL_FRAC", "PREFLIGHT_LABEL_TEXT")


def test_preflight_catches_missing_eos_and_wrong_answer():
    its = items()
    its[0]["ids"], its[0]["labels"] = its[0]["ids"][:-1], its[0]["labels"][:-1]
    with pytest.raises(G.GuardAbort, match="EOS"):
        pre(its)
    its = items()
    its[0]["ans"] = "999"                         # gold answer not what the solution trains
    with pytest.raises(G.GuardAbort) as e:
        pre(its)
    assert e.value.reason in ("PREFLIGHT_LABEL_TEXT", "PREFLIGHT_ANSWER")


def test_dataset_identity_and_eval_overlap(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ROWS), encoding="utf-8")
    sha = G.file_sha256(p)
    assert G.check_dataset(p, 8, {"expected_sha256": sha, "expected_rows": 8})["sha256"] == sha
    with pytest.raises(G.GuardAbort, match="PREFLIGHT_DATA_SHA"):
        G.check_dataset(p, 8, {"expected_sha256": "0" * 64})
    with pytest.raises(G.GuardAbort, match="PREFLIGHT_DATA_ROWS"):
        G.check_dataset(p, 8, {"expected_rows": 10})
    ev = {G.qhash("LAN có 2 quả táo,  mua thêm 3 quả. Hỏi có bao nhiêu quả?")}  # case/space/punct-insensitive
    with pytest.raises(G.GuardAbort, match="EVAL_OVERLAP"):
        G.overlap_with_eval([r["question"] for r in ROWS], ev)
    assert G.overlap_with_eval([r["question"] for r in ROWS], {G.qhash("other")})["n_overlap"] == 0


def test_step0_band():
    G.check_step0(0.75, {"step0_loss_band": [0.5, 1.1]})
    with pytest.raises(G.GuardAbort, match="STEP0"):
        G.check_step0(3.2, {"step0_loss_band": [0.5, 1.1]})


def run_guard(losses, g=None, gn=None, skipped=None, tps=None, scaler=False, total=400):
    lg = G.LossGuard({**GD, **(g or {})}, total, scaler)
    for i, l in enumerate(losses):
        lg.update(i, l, (gn or {}).get(i, 0.5), (skipped or {}).get(i, False), None if tps is None else tps[i])
    return lg


def healthy(n=120):
    return [0.75 * (0.6 + 0.4 * math.exp(-i / 30)) + 0.02 * ((i * 7) % 5 - 2) / 2 for i in range(n)]


def test_loss_guard_healthy_run_passes():
    run_guard(healthy(), g={"expected_tok_per_s": 4000}, tps=[4200] * 120)
    run_guard(healthy(), gn={7: float("inf")}, skipped={7: True}, scaler=True)  # fp16 skipped step is fine


@pytest.mark.parametrize("losses,kw,reason", [
    (healthy(30) + [float("nan")], {}, "NONFINITE_LOSS"),
    (healthy(30), {"gn": {12: float("inf")}}, "NONFINITE_GNORM"),
    (healthy(40) + [3.0], {}, "LOSS_SPIKE"),
    ([0.7, 0.5, 0.2, 0.08, 0.04, 0.03], {}, "LOSS_COLLAPSE"),
    ([0.7] * 100, {}, "NO_PROGRESS"),
])
def test_loss_guard_aborts(losses, kw, reason):
    with pytest.raises(G.GuardAbort) as e:
        run_guard(losses, **kw)
    assert e.value.reason == reason


def test_throughput_guard():
    with pytest.raises(G.GuardAbort, match="THROUGHPUT"):
        run_guard(healthy(60), g={"expected_tok_per_s": 4000}, tps=[4000] * 20 + [1500] * 40)


def test_midrun_verdict_quick_acc_is_only_a_warning():
    """QA-T #2: accuracy drops on the quick set warn; only format collapse aborts."""
    base = {"correct": 0.25, "repetition": 0.0, "no_answer": 0.1, "truncated": 0.0, "n": 64}
    ok = {"correct": 0.375, "repetition": 0.0625, "no_answer": 0.0, "truncated": 0.0, "n": 64}
    assert G.midrun_verdict(0.62, 0.55, ok, base, GD) == [] and G.midrun_warnings(0.62, 0.55, ok, base, GD) == []
    # dev vs train gap = warning only (style shift, Y+ s0 2026-11-08); a 2x blow-up vs step-0 dev aborts
    assert G.midrun_verdict(0.8956, 0.3155, ok, base, GD, dev0=0.7915) == []
    assert G.midrun_warnings(0.8956, 0.3155, ok, base, GD, dev0=0.7915)[0].startswith("dev ")
    assert G.midrun_verdict(1.7, 0.3, ok, base, GD, dev0=0.79)[0].startswith("DEV_BLOWUP")
    assert G.midrun_verdict(float("nan"), 0.3, ok, base, GD, dev0=0.79)[0] == "DEV_NONFINITE"
    ab, w = G._midrun(0.6, 0.55, {**ok, "correct": 0.0625}, base, GD)             # -19 pt: warning only
    assert ab == [] and w[0].startswith("quick_acc")
    ab, w = G._midrun(0.6, 0.55, {**ok, "repetition": 0.3}, base, GD)             # 0.2 < rep <= 0.5: warning
    assert ab == [] and w[0].startswith("repetition")
    assert G.midrun_verdict(0.6, 0.55, {**ok, "repetition": 0.55}, base, GD)[0].startswith("FORMAT_REPETITION")
    assert G.midrun_verdict(0.6, 0.55, {**ok, "no_answer": 0.6}, base, GD)[0].startswith("FORMAT_NO_ANSWER")
    assert G.midrun_verdict(0.6, 0.55, {**ok, "correct": 0.0}, base, GD)[0].startswith("FORMAT_ZERO_ACC")
    assert GD["quick_n"] == 64


def test_final_check_never_aborts_and_runs_after_the_adapter_is_saved(tmp_path, monkeypatch):
    """QA-T #2 item 2: the end-of-run midrun_check comes after _save_adapter + save_ckpt('final') and records only."""
    from vi_math_verified_aug.train import sft as SFT
    from test_train_sft_loop import FakeRun
    order = []

    class R(FakeRun):
        def train_step(self, step, mult, tag):
            row = super().train_step(step, mult, tag)
            row.update(loss=0.7 * (0.97 ** step) + 0.2, gnorm=0.4, skipped=False, real_tokens=1000, s=0.1)
            return row

        def _save_adapter(self, label, extra):
            order.append("adapter")
            return str(self.out / "adapters" / label)

        def dev_loss(self, step):
            self.last_dev = {"step": step, "dev_loss": 0.6}
            return self.last_dev

        def quick_eval(self, step):
            order.append(f"quick{step}")
            bad = step == self.plan["total"]   # collapse exactly at the end
            return {"correct": 0.0 if bad else 0.3, "repetition": 0.9 if bad else 0.0, "no_answer": 0.0,
                    "truncated": 0.0, "n": 64}

    monkeypatch.setattr(SFT, "SFTRun", R)
    monkeypatch.setattr(SFT, "env_info", lambda: {"fake": True})
    raw = {"kind": "sft", "arm": "S0", "model": "fake", "size": "tiny", "seed": 3, "precision": "fp32",
           "out_dir": str(tmp_path / "{run_id}"), "data": {"train": "unused.jsonl"},
           "budget": {"n_train": 40, "epochs": 3},
           "sft": {"seq_per_step": 4, "micro_batch": 2, "ckpt_every_min": 0, "ckpt_every_steps": 4},
           "guards": {"spike_after": 1000}}
    c = C.build(raw)
    rc = SFT.run(c)
    assert rc == 0
    assert order.index("adapter") < order.index(f"quick{30}")
    last = [json.loads(x) for x in (Path(c["out_dir"]) / "dev_log.jsonl").read_text().splitlines()][-1]
    assert last["final"] and last["abort"] == [] and any("FORMAT_REPETITION" in w for w in last["warnings"])


def test_check_runs_quick_acc_is_non_blocking_and_missing_quick_blocks(tmp_path):
    from vi_math_verified_aug.train.check_runs import check
    d = tmp_path / "run"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"run_id": "r", "status": "COMPLETE"}))
    (d / "preflight.json").write_text(json.dumps({"labels": {"decoded": [{"id": "x"}]}}))
    (d / "train_log.jsonl").write_text("".join(json.dumps({"step": i, "loss": 0.8 * 0.98 ** i, "lr": 1e-4,
                                                           "gnorm": 0.3, "real_tokens": 1000, "s": 0.2}) + NL
                                               for i in range(60)))
    q0 = {"correct": 0.4, "repetition": 0.0, "no_answer": 0.0, "truncated": 0.0, "n": 64}
    q1 = {"correct": 0.2, "repetition": 0.0, "no_answer": 0.0, "truncated": 0.0, "n": 64}
    rows = [{"step": 0, "dev_loss": 0.8}, {"step": 0, "kind": "midrun", "quick": q0},
            {"step": 60, "dev_loss": 0.45}, {"step": 60, "kind": "midrun", "quick": q1}]
    (d / "dev_log.jsonl").write_text("".join(json.dumps(r) + NL for r in rows))
    res = check(d)
    assert res["clean"] and "quick_acc" in res["warnings"], (res["flags"], res["warnings"])
    rows[3]["quick"] = None
    (d / "dev_log.jsonl").write_text("".join(json.dumps(r) + NL for r in rows))
    res = check(d)
    assert not res["clean"] and "quick_missing" in res["flags"]


# ---------------------------------------------------------------- wiring in SFTRun (FakeRun from the loop tests)
def test_sftrun_aborts_fast_with_manifest_reason(tmp_path, monkeypatch):
    from vi_math_verified_aug.train import sft as SFT
    from test_train_sft_loop import FakeRun

    class CollapsingRun(FakeRun):
        def train_step(self, step, mult, tag):
            row = super().train_step(step, mult, tag)
            row.update(loss=max(0.01, 0.7 - 0.2 * step), gnorm=0.4, skipped=False, real_tokens=1000, s=0.1)
            return row

        def midrun_check(self, step, final=False):
            self.last_dev = None

    monkeypatch.setattr(SFT, "SFTRun", CollapsingRun)
    monkeypatch.setattr(SFT, "env_info", lambda: {"fake": True})
    raw = {"kind": "sft", "arm": "S0", "model": "fake", "size": "tiny", "seed": 3, "precision": "fp32",
           "out_dir": str(tmp_path / "{run_id}"), "data": {"train": "unused.jsonl"},
           "budget": {"n_train": 40, "epochs": 3},
           "sft": {"seq_per_step": 4, "micro_batch": 2, "ckpt_every_min": 0, "ckpt_every_steps": 4}}
    c = C.build(raw)
    rc = SFT.run(c)
    m = json.loads((Path(c["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert rc == EXIT_GUARD_ABORT and m["status"] == "ABORTED_LOSS_COLLAPSE"
    steps = [json.loads(x)["step"] for x in (Path(c["out_dir"]) / "train_log.jsonl").read_text().splitlines()] \
        if (Path(c["out_dir"]) / "train_log.jsonl").exists() else []
    assert len(steps) <= 6  # stopped within a few steps, not at the end of 30


def test_quick_accuracy_collapse_needs_two_consecutive_checks():
    base = {"correct": 0.36, "repetition": 0.0, "no_answer": 0.0, "truncated": 0.0, "n": 64}
    low = {"correct": 0.15, "repetition": 0.0, "no_answer": 0.0, "truncated": 0.0, "n": 64}
    ok = {"correct": 0.40, "repetition": 0.0, "no_answer": 0.0, "truncated": 0.0, "n": 64}
    assert G.quick_collapse([low], base, GD) is None                      # one low check: not yet
    assert G.quick_collapse([ok, low], base, GD) is None
    assert G.quick_collapse([low, low], base, GD).startswith("QUICK_ACC_COLLAPSE")
    assert G.quick_collapse([low, low], {"correct": 0.0}, GD) is None      # no step-0 signal: never fires
