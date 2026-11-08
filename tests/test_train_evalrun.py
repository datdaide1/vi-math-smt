"""Y1: evaluation runner with a fake generation backend and a stub grader (no torch / vLLM)."""
import json
import re
import types
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from vi_math_verified_aug.evalrun import GREEDY_K, PRED_FIELDS  # noqa: E402
from vi_math_verified_aug.evalrun import runner as RN  # noqa: E402
from vi_math_verified_aug.train import EXIT_NEEDS_RESUME, EXIT_OK  # noqa: E402
from vi_math_verified_aug.train import config as C  # noqa: E402
from vi_math_verified_aug.train import reward as R  # noqa: E402
from vi_math_verified_aug.train import template as T  # noqa: E402


class FakeBackend:
    """Item i answers 2i on greedy; sample k answers 2i iff k < (i % 5)."""
    name = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, prompts, *, n, greedy, temperature, top_p, top_k, min_p, max_new_tokens, seeds):
        self.calls.append({"n": n, "greedy": greedy, "T": temperature, "top_p": top_p, "max": max_new_tokens,
                           "prompts": list(prompts), "seeds": list(seeds)})
        out = []
        for p in prompts:
            i = int(re.search(r"Tính (\d+)", p.rsplit("Bài toán:", 1)[1]).group(1))
            if greedy:
                out.append([{"text": f"... Đáp số: \\boxed{{{2 * i}}}", "tokens": 12, "finish_reason": "stop"}])
            else:
                out.append([{"text": f"\\boxed{{{2 * i if k < i % 5 else -1}}}" if i != 3 else "lặp " * 100,
                             "tokens": 2048 if i == 3 else 10, "finish_reason": "length" if i == 3 else "stop"}
                            for k in range(n)])
        return out


@pytest.fixture
def stub(monkeypatch):
    def grade(pred, gold, locale=None):
        g = gold["answer_canonical"] if isinstance(gold, dict) else str(gold)
        m = re.findall(r"\\boxed\{([^}]*)\}", pred)
        return {"correct": bool(m) and m[-1] == g, "span": m[-1] if m else None, "flag": None if m else "no-answer"}

    monkeypatch.setattr(R, "_grader", lambda: types.SimpleNamespace(grade=grade, GRADER_VERSION="stub",
                                                                    BACKEND="stub"))
    be = FakeBackend()
    monkeypatch.setattr(RN, "make_backend", lambda cfg, path, adapter: be)
    monkeypatch.setattr(RN.M, "resolve_model", lambda name, cache_root: name)
    return be


def write_set(path: Path, n: int, start=0):
    rows = [{"id": f"x{i}", "question": f"Tính {i} cộng {i}?",
             "gold": {"answer_canonical": str(2 * i), "answer_type": "int", "locale": "vi"}}
            for i in range(start, start + n)]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def ecfg(tmp_path, n=10, **ev):
    sets = [{"name": "setA", "path": str(write_set(tmp_path / "a.jsonl", n)), "lang": "vi"},
            {"name": "setB", "path": str(write_set(tmp_path / "b.jsonl", 4, start=100)), "lang": "en"}]
    raw = {"kind": "eval", "arm": "S0", "model": "Qwen/Qwen3-1.7B-Base", "size": "1.7B", "seed": 2,
           "out_dir": str(tmp_path / "out"), "eval": {"sets": sets, "source_run_id": "sft-S0-1p7B-s2-abc",
                                                       "chunk_items": 3, **ev}}
    return C.build(raw)


def test_rows_carry_every_glmm_field(tmp_path, stub):
    c = ecfg(tmp_path)
    assert RN.run(c) == EXIT_OK
    rows = RN.read_preds(Path(c["out_dir"]) / "preds.jsonl")
    assert len(rows) == (10 + 4) * (4 + 1)
    for r in rows:
        assert set(PRED_FIELDS) <= set(r)
        assert r["run_id"] == "sft-S0-1p7B-s2-abc" and r["arm"] == "S0" and r["size"] == "1.7B" and r["seed"] == 2
        assert r["correct"] in (0, 1) and isinstance(r["tokens"], int)
    assert {r["lang"] for r in rows if r["dataset"] == "setB"} == {"en"}
    ks = sorted({r["sample_k"] for r in rows})
    assert ks == [GREEDY_K, 0, 1, 2, 3]
    g = [r for r in rows if r["sample_k"] == GREEDY_K]
    assert all(r["decoding"] == "greedy" and r["temperature"] == 0.0 and r["correct"] == 1 for r in g)


def test_decoding_protocol_and_template(tmp_path, stub):
    c = ecfg(tmp_path, n=3)
    RN.run(c)
    sampled = [x for x in stub.calls if not x["greedy"]]
    assert all(x["n"] == 4 and x["T"] == 0.6 and x["top_p"] == 0.95 and x["max"] == 1024 for x in sampled)
    assert stub.calls[0]["prompts"][0] == T.prompt("Tính 0 cộng 0?")  # zero-shot, the SFT template exactly
    assert stub.calls[0]["seeds"][0] == RN.sampling_seed("sft-S0-1p7B-s2-abc", "setA", "x0")


def test_metrics(tmp_path, stub):
    c = ecfg(tmp_path)
    RN.run(c)
    m = json.loads((Path(c["out_dir"]) / "metrics.json").read_text(encoding="utf-8"))["per_dataset"]["setA"]
    # items 0..9: sample correct count = min(i%5, 4) except item 3 (looping, never correct)
    per_item = [min(i % 5, 4) / 4 for i in range(10)]
    per_item[3] = 0.0
    assert m["n_items"] == 10 and m["mean_pass1"] == pytest.approx(sum(per_item) / 10, abs=1e-5)
    assert m["greedy_acc"] == 1.0
    assert m["truncation_rate_sampled"] == pytest.approx(4 / 40)
    assert m["repetition_loop_rate"] == pytest.approx(4 / 50)
    # maj@4: majority span correct iff more than 2 of 4 samples correct (ties: Counter keeps first seen)
    assert 0 < m["maj@4"] <= 1


def test_incremental_and_resumable(tmp_path, stub):
    c = ecfg(tmp_path)
    preds = Path(c["out_dir"]) / "preds.jsonl"
    c["budget"]["wall_s"] = 1  # deadline already passed after setup -> stops before the first chunk
    c["budget"]["reserve_s"] = 0
    assert RN.run(c, t_start=0.0) == EXIT_NEEDS_RESUME
    assert not preds.exists() or preds.read_text() == ""
    c["budget"]["wall_s"] = None
    assert RN.run(c) == EXIT_OK
    n_full = len(RN.read_preds(preds))
    # simulate a killed kernel: drop the last 7 rows and leave a torn line
    lines = preds.read_text(encoding="utf-8").splitlines()
    preds.write_text("\n".join(lines[:-7]) + "\n" + lines[-7][:20], encoding="utf-8")
    (Path(c["out_dir"]) / "DONE").unlink()
    assert RN.run(c) == EXIT_OK
    rows = RN.read_preds(preds)
    keys = [(r["dataset"], r["item_id"], r["sample_k"]) for r in rows]
    assert len(keys) == len(set(keys)) == n_full


def test_repetition_loop_and_seed():
    assert RN.repetition_loop(" ".join(["a b c d e f g h i j k l m n o p q r s t"] * 3))
    assert not RN.repetition_loop("một hai ba " * 5)
    assert RN.sampling_seed("r", "d", "i") == RN.sampling_seed("r", "d", "i") != RN.sampling_seed("r", "d", "j")


def test_load_set_rejects_duplicates_and_missing_gold(tmp_path):
    p = tmp_path / "dup.jsonl"
    p.write_text('{"id":"a","question":"q","ground_truth":"1"}\n{"id":"a","question":"q","ground_truth":"1"}\n',
                 encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        RN.load_set({"name": "d", "path": str(p)})
    p.write_text('{"id":"a","question":"q"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="gold"):
        RN.load_set({"name": "d", "path": str(p)})


def test_rescore_never_regenerates(tmp_path, stub, monkeypatch):
    c = ecfg(tmp_path, n=4)
    RN.run(c)
    calls = len(stub.calls)
    # a stricter grader: nothing is correct any more
    monkeypatch.setattr(R, "_grader", lambda: types.SimpleNamespace(
        grade=lambda p, g, locale=None: {"correct": False, "span": None, "flag": None}, GRADER_VERSION="v2",
        BACKEND="stub"))
    out = tmp_path / "rescored.jsonl"
    flips = RN.rescore(Path(c["out_dir"]) / "preds.jsonl", c["eval"]["sets"], out)
    n_correct_before = sum(r["correct"] for r in RN.read_preds(Path(c["out_dir"]) / "preds.jsonl"))
    assert flips == n_correct_before > 0 and len(stub.calls) == calls
    assert all(r["correct"] == 0 and r["grader_version"] == "v2/stub" for r in RN.read_preds(out))


def test_fewshot_prefix_for_base_rows(tmp_path, stub):
    shots = tmp_path / "shots.jsonl"
    shots.write_text("\n".join(json.dumps({"question": f"Tính {i}?", "solution": "x", "answer": str(i)},
                                          ensure_ascii=False) for i in range(4)), encoding="utf-8")
    c = ecfg(tmp_path, n=2, shots=4)
    c["template"]["fewshot_path"] = str(shots)
    RN.run(c)
    p = stub.calls[0]["prompts"][0]
    assert p.count("Bài toán:") == 5 and p.endswith(T.prompt("Tính 0 cộng 0?"))


def test_y2_schema_fields(tmp_path, stub):
    """dataset = the row's language-neutral `dataset` when the eval file has one; eval_set keeps the file name."""
    p = tmp_path / "mixed.jsonl"
    lines = [json.dumps({"id": f"m{i}", "question": f"Tính {i} cộng {i}?",
                         "dataset": "gsm8k_platinum" if i % 2 else "math_algprealg",
                         "gold": {"answer_canonical": str(2 * i), "answer_type": "int"}}) for i in range(4)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    c = ecfg(tmp_path)
    c["eval"]["sets"] = [{"name": "vi_primary_1000", "path": str(p), "lang": "vi"}]
    RN.run(c)
    rows = RN.read_preds(Path(c["out_dir"]) / "preds.jsonl")
    assert {r["dataset"] for r in rows} == {"gsm8k_platinum", "math_algprealg"}
    assert {r["eval_set"] for r in rows} == {"vi_primary_1000"}
    assert {r["decode"] for r in rows} == {"sample", "greedy"}
    assert all((r["decode"] == "greedy") == (r["sample_k"] == GREEDY_K) for r in rows)
    assert RN.model_tag("Qwen/Qwen3-0.6B-Base") == "qwen3-0.6b"
    assert RN.model_tag("LiquidAI/LFM2.5-2.6B-Base") == "lfm2.5-2.6b"
