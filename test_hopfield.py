"""python -m pytest test_hopfield.py"""
import numpy as np
import pytest

import hopfield as H


def test_weights_are_hebbian_with_zero_diagonal():
    rng = np.random.default_rng(0)
    x = H.random_patterns(50, 4, rng)
    net = H.HopfieldNetwork(x)
    ref = sum(np.outer(x[m], x[m]) for m in range(4)) / 50.0
    np.fill_diagonal(ref, 0.0)
    assert np.allclose(net.W, ref)
    assert np.allclose(net.W, net.W.T)


def test_hamming_distance():
    a = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    assert H.hamming_distance(a, a) == 0.0
    assert H.hamming_distance(a, -a) == 1.0                  # the old code returned 0
    c = a.copy(); c[[0, 4]] *= -1                             # two opposite flips: the old code returned 0
    assert H.hamming_distance(a, c) == 0.25
    assert H.pattern_distance(a, -a) == 0.0                   # -a is the same memory


def test_stored_patterns_are_fixed_points_at_T0():
    rng = np.random.default_rng(1)
    x = H.random_patterns(300, 5, rng)
    net = H.HopfieldNetwork(x, temp=0.0)
    for mu in range(5):
        s, E, m = net.run(x[mu], 3, rng)
        assert np.array_equal(s, x[mu]) and m[-1][mu] == 1.0


@pytest.mark.parametrize("T, s1", [(0.7, -1.0), (0.7, 1.0), (0.3, -1.0)])
def test_glauber_probability_matches_the_paper(T, s1):
    """P(+1 | h) = 1/2 (1 + tanh(h/T)). Two units with W_01 = 1/2, so that the field on unit 0 is s_1 / 2."""
    rng = np.random.default_rng(2)
    net = H.HopfieldNetwork(np.array([[1, 1]]), temp=T)
    h = 0.5 * s1
    trials = 40000
    plus = 0
    for _ in range(trials):
        s = np.array([-1.0, s1])
        net.update_unit(s, 0, rng)
        plus += s[0] == 1
    assert plus / trials == pytest.approx(0.5 * (1 + np.tanh(h / T)), abs=0.01)


def test_sweep_is_asynchronous():
    """A sweep at T = 0 from a state next to a stored pattern: every unit sees the units updated before it."""
    rng = np.random.default_rng(3)
    x = H.random_patterns(100, 1, rng)
    net = H.HopfieldNetwork(x, temp=0.0)
    s = x[0].astype(float).copy()
    s[:10] *= -1                                              # 10 wrong bits
    net.sweep(s, rng)
    assert np.array_equal(s, x[0])


def test_energy_does_not_increase_at_T0():
    rng = np.random.default_rng(4)
    x = H.random_patterns(100, 8, rng)
    net = H.HopfieldNetwork(x, temp=0.0)
    s, E, m = net.run(H.random_patterns(100, 1, rng)[0], 15, rng)
    assert np.all(np.diff(E) <= 1e-12)


def test_energy_is_minus_half_N_overlap_squares_in_the_single_pattern_case():
    """E = -1/2 sum_ij W_ij S_i S_j = -N/2 (m^2 - 1/N) for one pattern (diagonal removed)."""
    rng = np.random.default_rng(5)
    x = H.random_patterns(80, 1, rng)
    net = H.HopfieldNetwork(x)
    s = rng.choice([-1.0, 1.0], size=80)
    m = float(net.overlaps(s)[0])
    assert net.energy(s) == pytest.approx(-0.5 * (m * m - 1.0 / 80))


def test_reproducible_with_a_seed():
    rng = np.random.default_rng(6)
    x = H.random_patterns(60, 3, rng)
    a = H.run_replicas((x, 0.5, 3, 4, "pattern", 0.1, np.random.SeedSequence(7)))
    b = H.run_replicas((x, 0.5, 3, 4, "pattern", 0.1, np.random.SeedSequence(7)))
    assert np.array_equal(a["E"], b["E"]) and np.array_equal(a["m"], b["m"])


@pytest.mark.parametrize("T", [0.5, 0.8])
def test_overlap_follows_mean_field_below_Tc(T):
    rng = np.random.default_rng(8)
    x = H.random_patterns(500, 3, rng)
    r = H.run_replicas((x, T, 8, 25, "pattern", 0.0, np.random.SeedSequence(9)))
    assert r["m"].mean() == pytest.approx(H.mean_field_m(T), abs=0.05)


def test_no_retrieval_above_Tc():
    rng = np.random.default_rng(10)
    x = H.random_patterns(500, 3, rng)
    r = H.run_replicas((x, 1.3, 8, 25, "pattern", 0.0, np.random.SeedSequence(11)))
    assert abs(r["m"].mean()) < 0.1


def test_capacity_below_and_above_alpha_c():
    low = H.run_capacity((400, 0.08, 3, 15, np.random.SeedSequence(12)))
    high = H.run_capacity((400, 0.30, 3, 15, np.random.SeedSequence(13)))
    assert low["m"].mean() > 0.95 and high["m"].mean() < 0.6


def test_pool_runs_a_scan_with_threads():
    rng = np.random.default_rng(14)
    x = H.random_patterns(40, 2, rng)
    res = H.temperature_scan(x, [0.2, 0.4], 2, 3, "pattern", 0.0, 2, 15)
    assert [r["T"] for r in res] == [0.2, 0.4]


def test_mean_field_capacity_is_the_known_value():
    a, m = H.mean_field_capacity()
    assert a == pytest.approx(0.138, abs=0.001) and m == pytest.approx(0.967, abs=0.002)
    assert H.mean_field_retrieval(0.2) is None and H.mean_field_retrieval(0.05) > 0.999


def test_relax_stops_at_a_fixed_point():
    rng = np.random.default_rng(15)
    x = H.random_patterns(200, 3, rng)
    net = H.HopfieldNetwork(x, temp=0.0)
    s, sweeps = net.relax(x[0], 50, rng)
    assert sweeps == 1 and np.array_equal(s, x[0])


def test_fit_extrapolation_recovers_a_known_law():
    Ns = np.array([100, 200, 400, 800, 1600, 3200], float)
    a, b, nu = 0.138, 5.0, 0.9
    est = a + b * Ns ** -nu
    err = np.full(6, 0.002)
    free = H.fit_extrapolation(Ns, est, err)
    assert free["a_inf"] == pytest.approx(a, abs=2e-3) and free["nu"] == pytest.approx(nu, abs=0.05)
    fixed = H.fit_extrapolation(Ns, est, err, nu=0.9)
    assert fixed["a_inf"] == pytest.approx(a, abs=1e-9) and fixed["chi2"] < 1e-12


def test_threshold_interpolation():
    al = [0.1, 0.2, 0.3]
    assert H.retrieval_threshold(al, np.array([1.0, 0.6, 0.0])) == pytest.approx(0.2 + 0.1 * 0.1 / 0.6)
    assert np.isnan(H.retrieval_threshold(al, np.array([1.0, 0.9, 0.8])))
