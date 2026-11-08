"""D1 scale_gen: slot schedule, parse/format checks, lite labels, batch budget guard, finalize definitions."""
from __future__ import annotations

import asyncio
import json

import pytest

from vi_math_verified_aug.scale_gen import checks as C
from vi_math_verified_aug.scale_gen import lite as L
from vi_math_verified_aug.scale_gen import slots as SL


def _seeds(n_gsm=30, n_math=10):
    out = [{"seed_id": f"g{i:04d}", "source": "vi_gsm8k_train", "dataset": "gsm8k", "index": i} for i in range(n_gsm)]
    out += [{"seed_id": f"m{i:04d}", "source": "vi_math_train_algebra", "dataset": "math", "index": i}
            for i in range(n_math)]
    return out


def test_slots_deterministic_and_pool_independent():
    a = {s["key"]: s for s in SL.build_slots(_seeds())}
    b = {s["key"]: s for s in SL.build_slots(list(reversed(_seeds())))}
    assert a == b
    # op / card of a slot do not depend on which other seeds are in the pool
    c = {s["key"]: s for s in SL.build_slots(_seeds()[:20])}
    for k, s in c.items():
        assert (s["op"], s["card"]) == (a[k]["op"], a[k]["card"])
    assert len(a) == 40 * SL.J


def test_slots_rank_and_m6_rules():
    sl = SL.build_slots(_seeds(200, 100))
    assert [s["rank"] for s in sl] == list(range(len(sl)))
    assert all(s["slot"] == 1 for s in sl[:300]) and all(s["slot"] == 2 for s in sl[300:600])
    assert not any(s["op"] == "M6" for s in sl if s["dataset"] == "math")
    m6 = [s for s in sl if s["op"] == "M6"]
    assert m6 and all(s["partner"] and s["partner"] != s["seed_id"] and s["partner"].startswith("g") for s in m6)
    assert {s["card"] for s in sl} <= set(SL.CARD_IDS)


def test_m6_partner_stable_when_other_seeds_leave():
    full = {s["key"]: s for s in SL.build_slots(_seeds(200, 0))}
    m6 = [s for s in full.values() if s["op"] == "M6"]
    keep = {s["seed_id"] for s in m6} | {s["partner"] for s in m6}
    sub = [s for s in _seeds(200, 0) if s["seed_id"] in keep or int(s["seed_id"][1:]) % 3 == 0]
    part = {s["key"]: s for s in SL.build_slots(sub)}
    for s in m6:
        assert part[s["key"]]["partner"] == s["partner"]


def test_create_shared_never_overwrites(tmp_path):
    pool = tmp_path / "data" / "seeds" / "pool_v1.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text("".join(json.dumps(s) + "\n" for s in _seeds(5, 2)), encoding="utf-8")
    out = tmp_path / "data" / "train" / "slots_v1.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text('{"key": "x"}\n', encoding="utf-8")
    r = SL.create_shared(pool, out)
    assert r["created"] is False and out.read_text(encoding="utf-8") == '{"key": "x"}\n'


@pytest.mark.parametrize("ans,lang,exp", [("12.500", "vi", "12500"), ("1,250", "en", "1250"), ("3/4", "en", "3/4"),
                                          ("7 quả", "vi", None), ("2,5", "vi", "2.5"), ("", "vi", None),
                                          ("45 đồng", "vi", "45")])
def test_num_value(ans, lang, exp):
    assert C._num_value(ans, lang) == exp


def test_parse_gen_flags():
    ok = json.dumps({"problem": "Lan có 12 quả cam, Lan cho bạn 5 quả. Hỏi Lan còn lại bao nhiêu quả cam?",
                     "answer": "7", "op_applied": "M1", "solution": "12 - 5 = 7"}, ensure_ascii=False)
    p = C.parse_gen(ok, "vi", "M1")
    assert p["fmt_ok"] and p["answer"] == "7"
    assert C.parse_gen(ok, "vi", "M2")["fmt_flags"]["op_mismatch"]
    assert C.parse_gen(ok, "en", "M1")["fmt_flags"]["wrong_language"]
    leak = ok.replace("Hỏi Lan", "Đáp số là 7. Hỏi Lan")
    assert C.parse_gen(leak, "vi", "M1")["fmt_flags"]["leak_regex"]
    tail = json.dumps({"problem": "Mia has 120 stickers and gives away some. She now has 120 stickers left... how many?",
                       "answer": "120", "op_applied": "M2"})
    assert C.parse_gen(tail, "en", "M2")["fmt_flags"]["answer_in_text"]
    assert C.parse_gen("not json", "vi", "M1")["fmt_flags"]["parse_fail"]


def test_agree_and_trace():
    assert C.agree("3 + 4 = 7. \\boxed{7}", "7", "vi")
    assert not C.agree("\\boxed{8}", "7", "vi")
    assert C.agree("Total \\boxed{12,500}", "12500", "en")
    assert C.clean_trace("a = 3\nVậy \\boxed{3}") == "a = 3"
    assert C.clean_trace("a = 3\n\\boxed{3}") == "a = 3"
    assert C.clean_trace("a = 3\n**Đáp số: \\boxed{540}**\n") == "a = 3"
    assert C.clean_trace("a\n\n**Đáp số:** $\\boxed{45}$") == "a"
    assert C.clean_trace("x\nThe answer is \\boxed{12}.") == "x"
    assert C.clean_trace("a\n$$\\boxed{\\frac{1}{2}}$$") == "a"
    assert C.clean_trace("Vậy Lan còn \\boxed{7} quả cam.") == "Vậy Lan còn 7 quả cam."
    assert C.clean_trace("Tổng là \\boxed{7} và còn lại 3 bước.\nKết thúc") == "Tổng là 7 và còn lại 3 bước.\nKết thúc"


def test_lite_labels():
    v = {"contradiction": False, "ambiguous": False, "multi_question": False, "noninteger_count": False,
         "implausible": False, "incoherent": False, "leak": False, "natural": 4}
    assert L.labels(v)["lite_ok"]
    assert not L.labels({**v, "implausible": True})["lite_ok"]
    assert L.labels({**v, "implausible": True})["absurd"]
    assert not L.labels({**v, "natural": 3})["lite_ok"]
    assert not L.labels({**v, "natural": "x"})["lite_ok"]


def test_lite_pack_request_is_train_split():
    req, ids = L.pack_request("vi", [("vi|B|a:1", "Bài 1"), ("vi|B|b:1", "Bài 2")])
    assert req.split == "train" and req.model.startswith("gemini") and set(ids) == {"p1", "p2"}
    assert "[p2]\nBài 2" in req.messages[0]["content"] and "Vietnamese" in req.system


def test_batch_budget_guard(tmp_path, monkeypatch):
    from vi_math_verified_aug.scale_gen import batchrun as B

    class FakeCosts:
        def spent(self):
            return 11.9

    class FakeClient:
        costs = FakeCosts()
        cache = None

    monkeypatch.setattr(B, "client", lambda: FakeClient())
    bm = B.BatchManager(tmp_path / "b.jsonl", cap_usd=12.0)

    class R:
        def request_hash(self, p):
            return "h"
    with pytest.raises(B.BudgetStop):
        asyncio.run(bm.submit("gen", "vi", [("k", R())], est_per_item=0.2))
    assert not (tmp_path / "b.jsonl").exists()
