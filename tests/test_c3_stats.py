"""D5 / C3: unit tests for the small-sample paired statistics (no data files needed)."""
import numpy as np
import pytest

from vi_math_verified_aug.analysis.c3_stats import (attributable_fraction, cluster_boot_rd, cluster_perm_test,
                                                    cochran_q, cochran_q_perm, newcombe_paired, newcombe_unpaired,
                                                    rate)


def test_rate_wilson():
    r = rate([1] * 8 + [0] * 2)
    assert r["k"] == 8 and r["n"] == 10 and r["p"] == 0.8
    assert 0.49 < r["lo"] < 0.5 and 0.94 < r["hi"] < 0.95       # Wilson 8/10: [0.490, 0.943]


def test_newcombe_unpaired_reference():
    # Newcombe (1998) Statist Med 17:873, example (a): 56/70 vs 48/80, method 10 -> D = 0.2, CI 0.0524 to 0.3339
    r = newcombe_unpaired(48, 80, 56, 70)
    assert r["d"] == pytest.approx(0.2)
    assert r["lo"] == pytest.approx(0.0524, abs=5e-4)
    assert r["hi"] == pytest.approx(0.3339, abs=5e-4)


def test_newcombe_paired_coverage():
    """Method 10 should have close-to-nominal coverage (Newcombe 1998: ~95%); checked by simulation on
    correlated pairs (cell probs a=.30, b=.10, c=.20, d=.40 -> true D = p_y - p_x = .10), n = 50."""
    rng = np.random.default_rng(7)
    probs = np.array([0.30, 0.10, 0.20, 0.40])
    hit, R = 0, 1500
    for _ in range(R):
        cells = rng.choice(4, size=50, p=probs)
        x = np.isin(cells, [0, 1]).astype(int)
        y = np.isin(cells, [0, 2]).astype(int)
        r = newcombe_paired(x, y)
        hit += r["lo"] <= 0.10 <= r["hi"]
    assert 0.925 <= hit / R <= 0.975


def test_newcombe_paired_properties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.integers(0, 2, 40)
        y = rng.integers(0, 2, 40)
        r = newcombe_paired(x, y)
        assert r["lo"] <= r["d"] <= r["hi"]
        assert -1 <= r["lo"] and r["hi"] <= 1
    r = newcombe_paired([1, 0, 1, 0], [1, 0, 1, 0])                   # all concordant
    assert r["d"] == 0 and r["lo"] < 0 < r["hi"] and r["p_mcnemar"] == 1.0
    r = newcombe_paired([0] * 20, [1] * 20)
    assert r["d"] == 1 and r["lo"] > 0.7 and r["p_mcnemar"] < 1e-5


def test_cluster_boot_and_perm():
    rng = np.random.default_rng(1)
    cl = np.repeat(np.arange(40), 3)
    e = np.repeat(rng.integers(0, 2, 40), 3)
    y_null = rng.integers(0, 2, 120)
    assert cluster_perm_test(y_null, e, cl, B=2000)["p_perm"] > 0.01
    y_eff = (e == 1).astype(int)
    r = cluster_perm_test(y_eff, e, cl, B=2000)
    assert r["rd"] == 1.0 and r["p_perm"] < 0.001
    d = cluster_boot_rd(np.ones(120), cl)
    assert d["d"] == 1.0 and d["lo"] == 1.0 and d["hi"] == 1.0


def test_attributable_fraction_extremes():
    cl = np.arange(60)
    e = np.array([1] * 30 + [0] * 30)
    y = e.copy()                                                         # every error is in an exposed unit
    assert attributable_fraction(y, e, cl, B=200)["paf"] == pytest.approx(1.0)
    y2 = np.tile([1, 0], 30)                                             # no association
    assert attributable_fraction(y2, e, cl, B=200)["paf"] == pytest.approx(0.0, abs=1e-9)


def test_cochran_q_hand_example():
    m = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    assert cochran_q(m) == pytest.approx(3.0)                           # 2*(3*14-36)/(18-14)
    r = cochran_q_perm(np.array([[1, 0, 0]] * 15), B=2000)
    assert r["p_perm"] < 0.001
