"""common/apilog.py — append-only JSONL, safe under threads and processes; secrets scrubbed."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from vi_math_verified_aug.common.apilog import ApiLogger, append_line, iter_jsonl
from vi_math_verified_aug.common.env import parse_dotenv, secret_values, split_keys

REPO = Path(__file__).resolve().parents[1]


def test_record_fields_and_scrub(tmp_path):
    lg = ApiLogger("W1", run_id="r1", log_dir=tmp_path, secrets=["sk-ant-SECRET-1234567"])
    lg.log({"provider": "claude", "error": "bad key sk-ant-SECRET-1234567", "request": {"x": "é"}})
    recs = list(lg.records("claude"))
    assert len(recs) == 1
    r = recs[0]
    assert r["ws"] == "W1" and r["run_id"] == "r1" and r["ts_utc"].endswith("Z")
    assert "SECRET" not in json.dumps(r) and "REDACTED" in r["error"]
    assert r["request"]["x"] == "é"                                  # UTF-8 kept, not escaped


def test_threads_do_not_interleave(tmp_path):
    p = tmp_path / "x.jsonl"
    payload = "y" * 5000                                             # long lines make interleaving visible

    def work(t):
        for i in range(50):
            append_line(p, json.dumps({"t": t, "i": i, "p": payload}))
    ts = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 400
    recs = [json.loads(x) for x in lines]                            # every line parses
    assert len({(r["t"], r["i"]) for r in recs}) == 400


@pytest.mark.skipif(os.environ.get("VIMATH_MP_TESTS") != "1",
                    reason="spawns 4 Python processes; opt in with VIMATH_MP_TESTS=1 (laptop RAM rule)")
def test_processes_do_not_interleave(tmp_path):
    p = tmp_path / "mp.jsonl"
    code = ("import sys, json; sys.path.insert(0, sys.argv[3]);"
            "from vi_math_verified_aug.common.apilog import append_line;"
            "[append_line(sys.argv[1], json.dumps({'w': sys.argv[2], 'i': i, 'p': 'z' * 3000})) for i in range(60)]")
    procs = [subprocess.Popen([sys.executable, "-c", code, str(p), str(w), str(REPO)]) for w in range(4)]
    assert all(pr.wait(timeout=120) == 0 for pr in procs)
    recs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 240 and len({(r["w"], r["i"]) for r in recs}) == 240


def test_iter_jsonl_skips_torn_line(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"a": 1}\n{"a": 2\n{"a": 3}\n', encoding="utf-8")
    assert [r["a"] for r in iter_jsonl(p)] == [1, 3]


def test_dotenv_and_secret_values():
    env = parse_dotenv('# c\nA_KEY="abc12345678"\nexport B_TOKEN=tok-999999999 # note\nPLAIN=1\n'
                       "GEMINI_API_KEYS=k1aaaaaaaa, k2bbbbbbbb,,k1aaaaaaaa\n")
    assert env["A_KEY"] == "abc12345678" and env["B_TOKEN"] == "tok-999999999" and env["PLAIN"] == "1"
    assert split_keys(env["GEMINI_API_KEYS"]) == ["k1aaaaaaaa", "k2bbbbbbbb"]
    s = secret_values(env)
    assert {"abc12345678", "tok-999999999", "k1aaaaaaaa", "k2bbbbbbbb"} <= set(s) and "1" not in s
