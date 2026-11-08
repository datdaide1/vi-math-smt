"""D5 joint gate audit: sampling is frozen/deterministic, plus/minus overlap is judged once, metrics + gate +
manipulation check are computed from judge rows (synthetic pools in a temp dir; no API calls)."""
import json

import pytest

from vi_math_verified_aug.analysis import c3_d5_audit as A
from vi_math_verified_aug.analysis.c3_data import lab_v2


def _w(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "TRAIN", tmp_path / "train")
    monkeypatch.setattr(A, "OUT", tmp_path / "audit")
    monkeypatch.setattr(A, "SEEDS", tmp_path / "seeds.jsonl")
    monkeypatch.setattr(A, "ROOT", tmp_path)
    _w(tmp_path / "seeds.jsonl", [{"seed_id": f"s{i}", "problem": f"seed {i}", "en": {"problem": f"seed en {i}"}}
                                 for i in range(300)])
    for arm in ("Y", "N"):
        rows = [{"id": f"{arm}/vi/s{i}:1", "question": f"{arm} q {i}", "seed_id": f"s{i}", "op": "M2" if i % 2 else "M1",
                 "plus": i % 3 != 0, "slot": f"s{i}:1"} for i in range(300)]
        _w(tmp_path / "train" / arm / "vi" / "train_10k_minus.jsonl", rows)
        _w(tmp_path / "train" / arm / "vi" / "train_10k_plus.jsonl", [r for r in rows if r["plus"]])
    return tmp_path


def test_sample_is_deterministic_and_frozen(env):
    m1 = A.draw("Y", "vi", "minus")
    a = (A.OUT / "sample_Y_vi_minus.jsonl").read_text(encoding="utf-8")
    assert m1["n_sample"] == A.N_PER_CELL and m1["n_population"] == 300
    assert A.draw("Y", "vi", "minus")["frozen"] is True                  # never redrawn
    (A.OUT / "sample_Y_vi_minus.jsonl").unlink()
    A.draw("Y", "vi", "minus")
    assert (A.OUT / "sample_Y_vi_minus.jsonl").read_text(encoding="utf-8") == a
    rows = A.load_jsonl(A.OUT / "sample_Y_vi_minus.jsonl")
    assert all(r["seed"].startswith("seed ") for r in rows)


def test_overlap_judged_once_and_metrics(env):
    for arm in ("Y", "N"):
        for v in ("plus", "minus"):
            A.draw(arm, "vi", v)
    its = A.judge_items("vi")
    n_rows = sum(len(A.load_jsonl(A.sample_path(a, "vi", v))) for a in ("Y", "N") for v in ("plus", "minus"))
    assert len(its) == len({i["key"] for i in its}) < n_rows                 # plus/minus overlap deduplicated
    # fake judge: Y items absurd when the source row is not plus; N items clean
    store = A.judge_store()
    minus_rows = {r["jkey"]: r for a in ("Y", "N") for r in A.load_jsonl(A.sample_path(a, "vi", "minus"))}
    plus_rows = {r["jkey"]: r for a in ("Y", "N") for r in A.load_jsonl(A.sample_path(a, "vi", "plus"))}
    for it in its:
        r = minus_rows.get(it["key"]) or plus_rows[it["key"]]
        bad = r["arm"] == "Y" and not r["plus_flag"]
        v = {"atomic": True, "no_conflict": True, "complete": True, "single_question": True, "realistic": not bad,
             "natural": 4, "teacher_would_write": not bad, "new_vs_seed": True, "op_followed": True,
             "codes": ["NL-ABSURD"] if bad else [], "severity": "major" if bad else "none", "note": ""}
        store.put(it["key"], {"verdict": v, "parsed": True, "labels": lab_v2(v), "api_ok": True, "cost": 0.0})
    R = A.report()
    yp, ym, nm = R["cells"]["Y|vi|plus"], R["cells"]["Y|vi|minus"], R["cells"]["N|vi|minus"]
    assert yp["absurd"]["k"] == 0 and ym["absurd"]["k"] > 0 and nm["absurd"]["k"] == 0
    assert yp["gate"]["absurd"]["pass_point"] and not ym["gate"]["absurd"]["pass_point"]
    man = R["manipulation"]["Y|vi|minus-plus"]["absurd"]
    assert man["d"] > 0 and man["lo"] > 0                                   # verification removed the errors
    assert R["manipulation"]["N|vi|minus-plus"]["absurd"]["d"] == 0
    assert (A.OUT / "tables.md").exists()
