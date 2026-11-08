"""G5: an arm-Y pool row (built by arm_y.export from the stage stores, solution from D1's shared writer) must load
through the trainer's data path: train.launcher.check_data and train.data.prepare_pool with 0 malformed rows, and the
N layout (data/train/Y/<lang>/train_10k_{plus,minus}.jsonl). Pure CPU, no API."""
import json
from pathlib import Path

from vi_math_verified_aug.arm_y import export as EX
from vi_math_verified_aug.train import data as D
from vi_math_verified_aug.train.launcher import _CharTok, check_data

TRACE_OK = ("Số kẹo Lan có lúc đầu là 12 cái.\nLan cho em 5 cái nên còn 12 - 5 = 7 cái.\n"
            "Đáp số: \\boxed{7}")
TRACE_WRONG = "Lan còn 12 + 5 = 17 cái.\nĐáp số: \\boxed{17}"
TRACE_NOBOX = "Lan còn 7 cái."


def _w(p: Path, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _run(tmp: Path) -> Path:
    run = tmp / "run"
    run.mkdir()
    json.dump({"source": "pool", "n_seeds": 3, "variants": ["R2b"], "batch": True, "pi": {"M1": 1.0}, "J": 3},
              open(run / "config.json", "w"))
    seeds = [{"seed_id": f"s{i}", "text": "Lan có 12 cái kẹo, cho em 5 cái. Hỏi Lan còn bao nhiêu cái kẹo?",
              "en_text": "x", "gold": "7"} for i in range(3)]
    _w(run / "seeds.jsonl", seeds)
    _w(run / "slots.jsonl", [{"key": f"s{i}:1", "seed_id": f"s{i}", "slot": 1, "op": "M1"} for i in range(3)])
    # (a 4th slot with no screen / cycle verdict yet must come out as plus=None, never a placeholder False)
    q = "Lan có <q n=\"1\">12</q> cái kẹo, cho em <q n=\"2\">5</q> cái. Hỏi Lan còn bao nhiêu cái kẹo?"
    _w(run / "render_R2b.jsonl", [{"key": f"s{i}:1", "seed_id": f"s{i}", "slot": 1, "op": "M1", "tag_ok": True,
                                   "text_tagged": q, "text": "Lan có 12 cái kẹo, cho em 5 cái. Hỏi Lan còn bao nhiêu?",
                                   "target": "7", "ir_target": {"ops": [{}]}} for i in range(3)])
    _w(run / "gates.jsonl", [{"key": f"R2b:s{i}:1", "slot": f"s{i}:1", "variant": "R2b", "op": "M1", "leak_ok": True,
                              "answer_untagged": False, "chrf": 0.5, "chrf_ok": True, "cycle_ok": True,
                              "dedup_ok": True, "survivor": True} for i in range(3)])
    _w(run / "vgate.jsonl", [{"key": f"R2b:s{i}:1", "variant": "R2b", "slot": f"s{i}:1", "lite_ok": True,
                              "haiku_pass": True, "pass": True} for i in range(3)])
    _w(run / "screen_haiku.jsonl", [{"key": f"R2b:s{i}:1", "parsed": True, "hv3_pass": True} for i in range(3)])
    _w(run / "cycle_R2b.jsonl", [{"key": f"s{i}:1", "done": True, "cycle_ok": True, "cycle_model": "Qwen/Qwen3-32B-AWQ"}
                                 for i in range(3)])
    _w(run / "solve_R2b.jsonl", [{"key": "s0:1", "trace": TRACE_OK}, {"key": "s1:1", "trace": TRACE_WRONG},
                                 {"key": "s2:1", "trace": TRACE_NOBOX}])
    return run


def test_pending_verification_is_none_not_false(tmp_path, monkeypatch):
    run = _run(tmp_path)
    (run / "cycle_R2b.jsonl").unlink()
    monkeypatch.setattr(EX, "ROOT", tmp_path)
    EX.write(run, "R2b", "vi", out=tmp_path / "out", decontam=False)
    by = {r["slot"]: r for r in D.read_jsonl(tmp_path / "out" / "vi" / "pool.jsonl")}
    assert by["s0:1"]["plus"] is None            # screened, trace matches, cycle pending
    assert by["s1:1"]["plus"] is False           # trace answer != program answer: definitively not Y+


def test_y_rows_load_through_the_trainer(tmp_path, monkeypatch):
    run = _run(tmp_path)
    monkeypatch.setattr(EX, "ROOT", tmp_path)
    man = EX.write(run, "R2b", "vi", out=tmp_path / "out", decontam=False)
    d = tmp_path / "out" / "vi"
    assert {p.name for p in d.iterdir()} >= {"pool.jsonl", "train_10k_plus.jsonl", "train_10k_minus.jsonl",
                                             "SHA256SUMS", "manifest.json"}
    pool = D.read_jsonl(d / "pool.jsonl")
    # s2 has no boxed answer -> not a training row (format failure, as in N-)
    assert [r["slot"] for r in pool] == ["s0:1", "s1:1"]
    # the training answer is the trace's OWN boxed answer; + needs it to equal the program answer
    by = {r["slot"]: r for r in pool}
    assert by["s0:1"]["answer"] == "7" and by["s0:1"]["plus"] is True
    assert by["s1:1"]["answer"] == "17" and by["s1:1"]["plus"] is False and by["s1:1"]["program_answer"] == "7"
    assert by["s0:1"]["wave"] == 1 and by["s0:1"]["minus_eligible"] is True
    assert "\\boxed" not in by["s0:1"]["solution"]                        # clean_trace removed the answer line
    plus = D.read_jsonl(d / "train_10k_plus.jsonl")
    minus = D.read_jsonl(d / "train_10k_minus.jsonl")
    assert [r["slot"] for r in plus] == ["s0:1"] and len(minus) == man["subsample_k"] == 1
    for fn in ("pool.jsonl", "train_10k_plus.jsonl", "train_10k_minus.jsonl"):
        chk = check_data(d / fn)
        assert chk["ok"] and chk["malformed"] == 0, chk
        items, st = D.prepare_pool(D.read_jsonl(d / fn), _CharTok(), 10 ** 9)
        assert st["n_malformed"] == 0 and len(items) == st["n_pool"]
