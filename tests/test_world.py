"""Vehikel: Koeffizientenmatrix, Wahrheit, Population, Stichprobe und binäres Beispiel."""

import numpy as np
import pytest

import frd_constants as C
import frd_world as W
from kausal_generator import SplitMix64


def test_coefficient_matrix_and_parents():
    b = W.coefficient_matrix(W.World(a=0.5, c=0.7, d=0.9, e=0.2, f=0.3, g=0.4))
    assert b[W.X, W.U] == b[W.Y, W.U] == 0.5 and b[W.M, W.X] == 0.7 and b[W.Y, W.M] == 0.9 and b[W.Y, W.X] == 0.2 and b[W.M, W.U] == 0.3
    assert b[W.M, W.U2] == b[W.Y, W.U2] == 0.4 and b.sum() == pytest.approx(0.5 * 2 + 0.7 + 0.9 + 0.2 + 0.3 + 0.8)
    assert W.parents(W.World()) == ((), (), (0,), (2,), (0, 3))
    assert W.parents(W.World(e=0.1, f=0.1, g=0.1)) == ((), (), (0,), (0, 1, 2), (0, 1, 2, 3))


def test_truth_is_the_sum_of_path_products():
    assert W.truth(W.World()) == pytest.approx(1.0)
    assert W.truth(W.World(c=0.4, d=0.5)) == pytest.approx(0.2)
    assert W.truth(W.World(e=0.3)) == pytest.approx(1.3)
    assert W.truth(W.World(f=0.3, g=0.5, a=2.0)) == pytest.approx(1.0)                  # Verborgene Ursachen ändern die do-Wirkung nicht


def test_population_by_hand(unit_world):
    cov, total = W.population(unit_world)
    assert cov[W.X, W.X] == pytest.approx(2.0) and cov[W.M, W.M] == pytest.approx(3.0) and cov[W.X, W.M] == pytest.approx(2.0)
    assert cov[W.X, W.Y] == pytest.approx(3.0) and cov[W.M, W.Y] == pytest.approx(4.0)
    assert total[W.Y, W.X] == pytest.approx(1.0) and total[W.M, W.X] == pytest.approx(1.0) and total[W.X, W.Y] == 0.0


def test_simulate_shapes_streams_and_measurement_noise():
    w = W.World(noise=0.0)
    v, m = W.simulate(w, 30, SplitMix64(3), reps=4)
    assert v.shape == (4, 30, 5) and m.shape == (4, 30) and (m == v[..., W.M]).all()
    again = W.simulate(w, 30, SplitMix64(3), reps=4)
    assert (again[0] == v).all() and not (W.simulate(w, 30, SplitMix64(4), reps=4)[0] == v).all()
    v2, m2 = W.simulate(W.World(noise=1.0), 20000, SplitMix64(5))
    assert (m2[0] - v2[0, :, W.M]).std() == pytest.approx(1.0, abs=0.02)


def test_simulation_has_the_population_covariance(unit_world):
    v, _ = W.simulate(unit_world, 200000, SplitMix64(6))
    emp = np.cov(v[0].T)
    cov, _ = W.population(unit_world)
    assert emp == pytest.approx(cov, abs=0.06)


def test_binary_truth_and_observation_by_hand():
    p1, p0 = W.binary_truth()
    assert p1 == pytest.approx(0.7) and p0 == pytest.approx(0.3)
    o1, o0 = W.binary_observed()
    assert o1 == pytest.approx(0.79) and o0 == pytest.approx(0.21)


def test_binary_simulation_matches_the_probabilities():
    u, x, m, y = (a[0] for a in W.simulate_binary(400000, SplitMix64(7)))
    assert u.mean() == pytest.approx(0.5, abs=0.005) and x[u == 1].mean() == pytest.approx(0.8, abs=0.005) and x[u == 0].mean() == pytest.approx(0.2, abs=0.005)
    assert m[x == 1].mean() == pytest.approx(0.9, abs=0.005) and m[x == 0].mean() == pytest.approx(0.1, abs=0.005)
    assert y[(m == 1) & (u == 1)].mean() == pytest.approx(0.9, abs=0.005) and y[(m == 0) & (u == 0)].mean() == pytest.approx(0.1, abs=0.005)
    assert y[x == 1].mean() - y[x == 0].mean() == pytest.approx(0.58, abs=0.006)
    assert set(np.unique(u)) <= {0, 1} and C.BIN_PU == 0.5


def test_total_effects_have_a_zero_diagonal(unit_world):
    """T = (I − B)⁻¹ − I: ein Knoten wirkt nicht auf sich selbst (die Diagonale von (I − B)⁻¹ ist 1)."""
    _, total = W.population(unit_world)
    assert np.diag(total) == pytest.approx(np.zeros(5)) and total[W.M, W.Y] == 0.0 and total[W.Y, W.U] == pytest.approx(1.0 + 1.0)   # U → Y direkt 1 plus U → X → M → Y gleich 1


def test_measurement_noise_is_added_with_the_second_block_of_the_stream():
    """Die Stichprobe zieht erst 5·n Normalwerte für die Knoten, dann n für den Messfehler; der Fehler wird addiert, nicht abgezogen."""
    w = W.World(noise=0.7)
    v, m = W.simulate(w, 25, SplitMix64(12), reps=2)
    rng = SplitMix64(12)
    rng.normal(2 * 25 * 5)
    assert m - v[..., W.M] == pytest.approx(0.7 * rng.normal(2 * 25).reshape(2, 25))


def test_binary_probabilities_follow_the_constants_not_a_symmetry(monkeypatch):
    """Mit P(U = 1) = 0,3 (statt 0,5): do(X = 1): 0,9 · 0,69 + 0,1 · 0,19 = 0,64; do(X = 0): 0,1 · 0,69 + 0,9 · 0,19 = 0,24.
    Beobachtet: P(U = 1 | X = 1) = 0,24 / 0,38, P(U = 1 | X = 0) = 0,06 / 0,62; P(Y = 1 | X) = 0,1 + 0,5 P(M = 1 | X) + 0,3 P(U = 1 | X)."""
    monkeypatch.setattr(C, "BIN_PU", 0.3)
    t1, t0 = W.binary_truth()
    assert t1 == pytest.approx(0.64) and t0 == pytest.approx(0.24)
    o1, o0 = W.binary_observed()
    assert o1 == pytest.approx(0.1 + 0.5 * 0.9 + 0.3 * 0.24 / 0.38) and o0 == pytest.approx(0.1 + 0.5 * 0.1 + 0.3 * 0.06 / 0.62)
