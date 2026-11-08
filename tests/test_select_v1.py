"""D5 select_v1: matched selection invariants on synthetic pools (no real data, no API)."""
import json
import random

import pytest

from vi_math_verified_aug.analysis import select_v1 as S


def _w(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _pools(tmp, langs=("vi",), n_seeds=400, flip_plus=False, drop_en=()):
    rng = random.Random(0)
    seeds = []
    for i in range(n_seeds):
        ds = "gsm8k" if i % 3 else "math"
        seeds.append({"seed_id": f"s{i:04d}", "dataset": ds,
                      "screens": {"solvecheck": "FAIL" if i == 5 else ("UNINF" if i == 7 else
                                                                       ("PENDING2" if i == 9 else "PASS"))}, "subject": None if ds == "gsm8k" else
                      ("Prealgebra" if i % 2 else "Algebra"), "level": f"Level {1 + i % 5}",
                      "solution_en": "<<1+1=2>>" * (1 + i % 7)})
    _w(tmp / "seeds.jsonl", seeds)
    for lang in langs:
        for arm, ops in (("Y", ["M1", "M2", "M3", "M4", "M5"]), ("N", ["M1", "M2", "M3", "M4", "M5", "M6"])):
            rows = []
            r2 = random.Random(f"{arm}")                     # same plus pattern in every language
            for s in seeds:
                for j in (1, 2, 3):
                    if lang == "en" and s["seed_id"] in drop_en:
                        continue
                    if r2.random() < 0.15:                    # not generated
                        continue
                    plus = r2.random() < 0.55
                    rows.append({"id": f"{arm}/{lang}/{s['seed_id']}:{j}", "seed_id": s["seed_id"],
                                 "slot": f"{s['seed_id']}:{j}", "op": rng.choice(ops),
                                 "plus": (not plus) if flip_plus else plus,
                                 "question": " ".join(f"w{rng.randint(0, 999)}" for _ in range(rng.randint(20, 60))), "solution": "x " * rng.randint(30, 90)})
            _w(tmp / "train" / arm / lang / "pool.jsonl", rows)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "TRAIN", tmp_path / "train")
    monkeypatch.setattr(S, "FINAL", tmp_path / "train" / "final")
    monkeypatch.setattr(S, "NATURAL", tmp_path / "train" / "final_natural")
    monkeypatch.setattr(S, "SHARED", tmp_path / "train" / "final_shared")
    monkeypatch.setattr(S, "SEED_POOL", tmp_path / "seeds.jsonl")
    monkeypatch.setattr(S, "SEED_DEV", tmp_path / "dev.jsonl")
    monkeypatch.setattr(S, "ROOT", tmp_path)
    return tmp_path


def _seeds(sel, lang, arm):
    return [r["seed_id"] for r in sel["picks"][lang][arm]]


def test_matched_invariants(env):
    _pools(env)
    sel = S.select(["vi"], n=150)
    assert sel["n_effective"] == 150
    for a in S.ARMS:
        ss = _seeds(sel, "vi", a)
        assert len(ss) == 150 and len(set(ss)) == 150                 # <= 1 item per seed while seeds remain
        assert all(r["op"] != "M6" for r in sel["picks"]["vi"][a])
    assert sorted(_seeds(sel, "vi", "Y+")) == sorted(_seeds(sel, "vi", "N+"))
    assert sorted(_seeds(sel, "vi", "Y-")) == sorted(_seeds(sel, "vi", "N-"))
    assert sel["quota"]["plus"] == sel["quota"]["minus"]               # same stratum quotas
    assert all(r["plus"] for a in ("Y+", "N+") for r in sel["picks"]["vi"][a])
    assert S.select(["vi"], n=150)["alloc"] == sel["alloc"]            # deterministic


def test_minus_never_reads_plus(env):
    """Given the same seed allocation, the '-' item choice and the '-' candidate sets are identical whatever the
    plus flags are (the allocation itself shares the '+' stratum quotas by design, lead decision)."""
    _pools(env)
    a = S.select(["vi"], n=150)
    ca = S.candidates(S.load_pools("vi"))
    _pools(env, flip_plus=True)
    cb = S.candidates(S.load_pools("vi"))
    for arm in ("Y-", "N-"):
        assert {k: [r["id"] for r in v] for k, v in ca[arm].items()} == {k: [r["id"] for r in v] for k, v in cb[arm].items()}
        pa = S.pick_items(arm, a["alloc"]["minus"], ca[arm], a["strata"])
        pb = S.pick_items(arm, a["alloc"]["minus"], cb[arm], a["strata"])
        assert [r["id"] for r in pa] == [r["id"] for r in pb]
    assert {k: [r["id"] for r in v] for k, v in ca["Y+"].items()} != {k: [r["id"] for r in v] for k, v in cb["Y+"].items()}


def test_capacity_bound_and_second_round(env):
    _pools(env, n_seeds=60)
    sel = S.select(["vi"], n=3000)
    cap = sel["capacity"]["items_plus"]
    assert sel["n_effective"] == min(3000, cap)
    assert sel["n_effective"] > sel["capacity"]["seeds_plus"]          # rounds 2+ used once seeds run out


def test_joint_languages_share_seeds(env):
    _pools(env, langs=("vi", "en"), drop_en={"s0003", "s0010", "s0011"})
    sel = S.select(["vi", "en"], n=150)
    for a in S.ARMS:
        assert sorted(_seeds(sel, "vi", a)) == sorted(_seeds(sel, "en", a))
        assert not {"s0003", "s0010", "s0011"} & set(_seeds(sel, "vi", a))


def test_write_outputs_and_natural(env):
    _pools(env)
    sel = S.select(["vi"], n=100)
    S.write(sel)
    for x in ("Y", "N"):
        for f in ("train_plus.jsonl", "train_minus.jsonl", "SHA256SUMS"):
            assert (S.FINAL / x / "vi" / f).exists()
    man = json.load(open(S.FINAL / "manifest_vi.json", encoding="utf-8"))
    assert man["n_effective"] == 100 and "selection_effect" in man
    with pytest.raises(SystemExit):
        S.write(sel)                                                  # never silently overwritten
    S.write(sel, force=True)
    assert any((S.FINAL / "_history").iterdir())
    nat = S.natural("vi", 100)
    assert {len(v) for v in nat["picks"].values()} == {100}
    sh = S.select(["vi"], n=100, minus_seeds="shared")
    s = S.summarize(sh, "vi")
    assert sorted(_seeds(sh, "vi", "Y-")) == sorted(_seeds(sh, "vi", "Y+"))
    assert s["minus_items_that_are_plus"]["Y-"] > 0


def test_smd_ks_helpers():
    assert S.smd([1, 2, 3], [1, 2, 3]) == 0
    assert S.ks([1, 2, 3, 4], [1, 2, 3, 4])["D"] == 0
    r = S.ks(list(range(100)), list(range(50, 150)))
    assert r["D"] == pytest.approx(0.5) and r["p"] < 1e-6


def test_vendi_and_pairwise_distance():
    np = pytest.importorskip("numpy")
    from vi_math_verified_aug.analysis.select_v1_diversity import mean_pairwise_cos_distance, vendi
    I = np.eye(8, dtype=np.float32)
    assert vendi(I) == pytest.approx(8.0)                          # 8 orthogonal items -> 8 effective items
    same = np.tile(I[:1], (5, 1))
    assert vendi(same) == pytest.approx(1.0)
    assert mean_pairwise_cos_distance(same) == pytest.approx(0.0)
    assert mean_pairwise_cos_distance(I) == pytest.approx(1.0)


def test_dev_seeds_and_non_pool_seeds_excluded(env):
    _pools(env)
    _w(env / "dev.jsonl", [{"seed_id": f"s{i:04d}"} for i in range(0, 400, 2)])        # half the seeds are DEV
    sel = S.select(["vi"], n=300)
    for a in S.ARMS:
        assert all(int(s[1:]) % 2 == 1 for s in _seeds(sel, "vi", a))
    assert sel["seed_universe"]["n_dev_v1"] == 200


def test_solvecheck_fail_uninf_excluded(env):
    _pools(env)
    sel = S.select(["vi"], n=380)
    for a in S.ARMS:
        assert not {"s0005", "s0007"} & set(_seeds(sel, "vi", a))
    u = sel["seed_universe"]
    assert u["n_excluded_solvecheck"] == 2 and u["pending2_seeds"] == ["s0009"]


def _two_part_env(env, n_seeds=500):
    """Synthetic G5 work dir (wave-1 render/gates/solve) + Y pool with waves + N pool with slot_id/j."""
    rng = random.Random(1)
    seeds = [{"seed_id": f"s{i:04d}", "dataset": "gsm8k", "screens": {"solvecheck": "PASS"},
              "solution_en": "<<1+1=2>>" * (1 + i % 6)} for i in range(n_seeds)]
    _w(env / "seeds.jsonl", seeds)
    S.Y_WORK = env / "train" / "Y" / "_work"
    run = S.Y_WORK / "vi_scale"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({"variants": ["R2b"]}), encoding="utf-8")
    rend, gates, sol, ypool, npool = [], [], [], [], []
    for s in seeds:
        sid = s["seed_id"]
        ok = rng.random() < 0.8
        rend.append({"key": f"{sid}:1", "seed_id": sid, "slot": 1, "op": rng.choice(["M1", "M2", "M3"]), "tag_ok": ok})
        gates.append({"slot": f"{sid}:1", "leak_ok": True, "answer_untagged": False})
        sol.append({"key": f"{sid}:1", "trace": "x \\boxed{1}"})
        if ok:
            v1 = rng.random() < 0.4
            ypool.append({"id": f"Y/vi/{sid}:1", "seed_id": sid, "slot": f"{sid}:1", "op": "M2", "wave": 1, "plus": v1,
                          "question": f"Lan has {rng.randint(2, 99)} apples and buys {rng.randint(2, 99)} more pears today", "solution": "d"})
            if not v1:
                ypool.append({"id": f"Y/vi/{sid}:2", "seed_id": sid, "slot": f"{sid}:2", "op": "M1", "wave": 2,
                              "plus": rng.random() < 0.5, "question": f"Nam reads {rng.randint(2, 99)} pages every day for {rng.randint(2, 9)} weeks", "solution": "d"})
        for j in (1, 2):
            npool.append({"id": f"N/vi/{sid}:{j}", "seed_id": sid, "slot_id": f"{sid}:{j}",
                          "op": rng.choice(["M1", "M3", "M4", "M6"]), "minus_eligible": rng.random() < 0.9,
                          "verified": rng.choice([True, True, False, None]), "question": f"Hoa saves {rng.randint(2, 99)} coins each week during {rng.randint(2, 9)} months", "solution": "u"})
    for name, rows in (("render_R2b", rend), ("gates", gates), ("solve_R2b", sol)):
        _w(run / f"{name}.jsonl", rows)
    _w(env / "train" / "Y" / "vi" / "pool.jsonl", ypool)
    _w(env / "train" / "N" / "vi" / "pool.jsonl", npool)


def test_two_part_selection(env, monkeypatch):
    monkeypatch.setattr(S, "OPMATCH", env / "train" / "final_opmatched")
    old = S.Y_WORK
    try:
        _two_part_env(env)
        p1 = S.part1_minus("vi", 120)
        assert p1["n"] == 120 and len(p1["reserve_minus"]) == 12
        for sid in p1["seeds_minus"]:
            assert p1["slots"]["Y"][sid] == f"{sid}:1" and p1["slots"]["N"][sid] == f"{sid}:1"
        S.write_part1(p1)
        sel = S.select_part2(["vi"], p1)
        P = sel["picks"]["vi"]
        n = sel["n_effective"]
        assert all(len(P[a]) == n for a in S.ARMS)
        assert sorted(r["seed_id"] for r in P["Y+"]) == sorted(r["seed_id"] for r in P["N+"])
        assert sorted(r["seed_id"] for r in P["Y-"]) == sorted(r["seed_id"] for r in P["N-"])
        assert all(r["plus"] is True for a in ("Y+", "N+") for r in P[a])
        assert all(r["slot"].endswith(":1") for a in ("Y-", "N-") for r in P[a])        # first items only
        assert all(r["op"] != "M6" for a in S.ARMS for r in P[a])
        assert sel["quota"]["plus"] == sel["quota"]["minus"]                    # identical strata quotas, one n
        om = S.opmatched_from(sel, "vi")
        for sign in ("+", "-"):
            assert om["hist"][sign]["Y"] == om["hist"][sign]["N"]
        assert len({len(v) for v in om["picks"].values()}) == 1
        sh = S.shared_from(sel)
        assert {r["seed_id"] for r in sh["picks"]["vi"]["Y-"]} <= set(sel["alloc"]["plus"])
        # full publish path (as `select_v1 part2`): primary + shared + natural + op-matched, all validated
        man = S.write(sel)
        assert man["vi"]["n_effective"] == n
        S.write(sh, force=True, base=S.SHARED)
        S.write_natural("vi", S.natural("vi", 120, sel["pools"]["vi"]), sel["pools"]["vi"])
        S.write_opmatched(sel, "vi")
        for base in (S.FINAL, S.SHARED, S.NATURAL, S.OPMATCH):
            for x in ("Y", "N"):
                assert (base / x / "vi" / "SHA256SUMS").exists()
    finally:
        S.Y_WORK = old


def test_placeholder_set_is_never_written(env, monkeypatch):
    _pools(env)
    sel = S.select(["vi"], n=50)
    orig = S.Pool.full
    monkeypatch.setattr(S.Pool, "full", lambda self, rows: [{**r, "question": "w w w"} for r in orig(self, rows)])
    with pytest.raises(S.SelectionInvalid):
        S.write(sel)
    assert not (S.FINAL / "Y" / "vi" / "train_plus.jsonl").exists()
    assert not (S.FINAL.exists() and list(S.FINAL.rglob("*.staging")))


def test_changed_pool_is_detected(env):
    _pools(env)
    sel = S.select(["vi"], n=50)
    raw = [json.loads(l) for l in S.pool_path("Y", "vi").read_text(encoding="utf-8").splitlines() if l.strip()]
    _w(env / "train" / "Y" / "vi" / "pool.jsonl", raw[::-1])
    with pytest.raises(S.SelectionInvalid):
        S.write(sel)
    assert not (S.FINAL / "Y" / "vi" / "train_plus.jsonl").exists()
